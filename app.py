from pathlib import Path
from tempfile import NamedTemporaryFile
import os
import uuid
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename

from core.classifier import ContentClassifier
from services.pptx_parser import PresentationParser
from services.rag_service import OllamaGenerationError, SlideRAGService
from services.document_utils import analyze_tone, extractive_summarize
from models.user import UserManager

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "neuroram_ai_content_secure_key_2026")

MAX_UPLOAD_BYTES = 20 * 1024 * 1024 # 20 MB limit (Section 4.11 & Appendix C)
ALLOWED_EXTENSIONS = {".pptx"}
TEMP_DIR = Path("instance/tmp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Initialize singletons
classifier = ContentClassifier()
parser = PresentationParser()
rag_service = SlideRAGService()
user_manager = UserManager()

# Pre-populate primary credentials for Rohit
user_manager.register_user("@Rohit", "rohit123", "rohit.sharma@novemcontrols.com")

# In-memory store for active session document metadata
session_documents = {}

# Active browser tab tracker for auto-terminating the server/batch job on tab close
import time
import threading

class TabSessionTracker:
    def __init__(self, heartbeat_timeout=15.0, close_grace_period=3.0):
        self.active_tabs = {}  # tab_id -> last_seen_time
        self.lock = threading.Lock()
        self.heartbeat_timeout = heartbeat_timeout
        self.close_grace_period = close_grace_period
        self.has_ever_connected = False
        self.shutdown_timer = None
        self.enabled = False

    def start(self):
        self.enabled = True
        t = threading.Thread(target=self._monitor_loop, daemon=True)
        t.start()

    def record_heartbeat(self, tab_id):
        if not self.enabled:
            return
        with self.lock:
            self.has_ever_connected = True
            self.active_tabs[tab_id] = time.time()
            if self.shutdown_timer:
                self.shutdown_timer.cancel()
                self.shutdown_timer = None

    def record_tab_closed(self, tab_id):
        if not self.enabled:
            return
        with self.lock:
            self.active_tabs.pop(tab_id, None)
            now = time.time()
            alive = [tid for tid, ts in self.active_tabs.items() if now - ts < self.heartbeat_timeout]
            if not alive and self.has_ever_connected:
                self._schedule_shutdown()

    def _schedule_shutdown(self):
        if self.shutdown_timer is None:
            self.shutdown_timer = threading.Timer(self.close_grace_period, self._terminate)
            self.shutdown_timer.daemon = True
            self.shutdown_timer.start()

    def _terminate(self):
        print("\n" + "="*65)
        print(" [*] Browser tab closed. Automatically terminating DocuSense AI server and batch job...")
        print("="*65 + "\n")
        os._exit(0)

    def _monitor_loop(self):
        while self.enabled:
            time.sleep(1.5)
            now = time.time()
            with self.lock:
                if not self.has_ever_connected:
                    continue
                stale = [tid for tid, ts in self.active_tabs.items() if now - ts > self.heartbeat_timeout]
                for tid in stale:
                    self.active_tabs.pop(tid, None)
                if not self.active_tabs:
                    self._schedule_shutdown()

tab_tracker = TabSessionTracker()

@app.post("/api/heartbeat")
def tab_heartbeat():
    data = request.get_json(silent=True) or {}
    tab_id = data.get("tab_id", "default")
    tab_tracker.record_heartbeat(tab_id)
    return jsonify(status="ok")

@app.post("/api/tab-closed")
def tab_closed():
    data = request.get_json(silent=True)
    if not data and request.data:
        try:
            import json
            data = json.loads(request.data.decode("utf-8", errors="ignore"))
        except Exception:
            data = {}
    tab_id = (data or {}).get("tab_id", "default")
    tab_tracker.record_tab_closed(tab_id)
    return jsonify(status="ok")

@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login_view"))
    return render_template("dashboard.html", user=session["user"])

@app.route("/login", methods=["GET", "POST"])
def login_view():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        auth = user_manager.authenticate_user(username, password)
        if auth:
            session["user"] = auth
            return redirect(url_for("index"))
        else:
            error = "Invalid credentials. (Hint: Use intern / novem2026 or register below)"
    return render_template("login.html", error=error)

@app.route("/register", methods=["POST"])
def register_view():
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    email = request.form.get("email", "")
    if user_manager.register_user(username, password, email):
        session["user"] = {"username": username, "email": email}
        return redirect(url_for("index"))
    return render_template("login.html", error="Registration failed. Username may already exist or password too short.")

@app.route("/logout")
def logout_view():
    session.clear()
    return redirect(url_for("login_view"))

@app.post("/api/presentation/analyse")
def analyse_presentation():
    """
    Core Presentation Analysis route directly from Appendix C (Section 10.4):
    Validates file, applies secure UUID naming, parses presentation, executes
    NLP ensemble sentence scoring, generates slide audit warnings, and cleans up.
    """
    if "file" not in request.files:
        return jsonify(error="Missing file field"), 400

    upload = request.files["file"]
    original = secure_filename(upload.filename or "")
    if not original or Path(original).suffix.lower() not in ALLOWED_EXTENSIONS:
        return jsonify(error="Only .pptx files are accepted"), 400

    if request.content_length and request.content_length > MAX_UPLOAD_BYTES:
        return jsonify(error="Upload exceeds 20 MB limit"), 413

    temp_path = None
    try:
        # Use random server-generated name; never trust client filename (Section 4.11)
        with NamedTemporaryFile(
            prefix=f"{uuid.uuid4().hex}_",
            suffix=".pptx",
            dir=TEMP_DIR,
            delete=False
        ) as tmp:
            temp_path = Path(tmp.name)
            upload.save(tmp)

        if temp_path.stat().st_size > MAX_UPLOAD_BYTES:
            return jsonify(error="Upload exceeds 20 MB limit"), 413

        # 1. Parse presentation hierarchy & audit rules
        slides = parser.parse(temp_path)
        extracted_text = "\n".join(s["text"] for s in slides)

        # 2. Score sentences with the 60/40 weighted ensemble
        sentence_results = classifier.highlight_sentences(extracted_text)

        # 3. Tone analysis & extractive summary
        tone = analyze_tone(extracted_text)
        summary = extractive_summarize(extracted_text, max_sentences=3)

        # 4. Index for Grounded RAG
        rag_service.index_presentation(slides)

        # Calculate high-level summary metrics
        total_words = sum(s["word_count"] for s in slides)
        scored_sentences = [s for s in sentence_results if s["ai_probability"] is not None]
        avg_ai_prob = (
            sum(s["ai_probability"] for s in scored_sentences) / len(scored_sentences)
            if scored_sentences else 0.0
        )
        total_warnings = sum(len(s["warnings"]) for s in slides)

        doc_summary = {
            "filename": original,
            "slide_count": len(slides),
            "total_words": total_words,
            "overall_ai_prob": round(avg_ai_prob, 4),
            "total_warnings": total_warnings,
            "tone": tone,
            "summary": summary
        }

        # Cache session doc for subsequent RAG questions
        session["current_doc"] = original

        return jsonify(
            success=True,
            doc_summary=doc_summary,
            slides=slides,
            sentences=sentence_results
        )

    except Exception as e:
        app.logger.exception("Presentation analysis failed")
        return jsonify(error=f"The presentation could not be analysed: {str(e)}"), 422

    finally:
        # Secure cleanup in finally block (Section 3.8 & Appendix C)
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                app.logger.exception("Temporary upload cleanup failed")

@app.post("/api/text/analyse")
def analyse_text():
    """Direct raw text analysis route for standalone paragraphs and essays."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not text.strip():
        return jsonify(error="Empty text supplied"), 400

    sentence_results = classifier.highlight_sentences(text)
    tone = analyze_tone(text)
    summary = extractive_summarize(text, max_sentences=2)

    scored_sentences = [s for s in sentence_results if s["ai_probability"] is not None]
    avg_ai_prob = (
        sum(s["ai_probability"] for s in scored_sentences) / len(scored_sentences)
        if scored_sentences else 0.0
    )

    return jsonify(
        success=True,
        overall_ai_prob=round(avg_ai_prob, 4),
        sentences=sentence_results,
        tone=tone,
        summary=summary
    )

@app.post("/api/presentation/chat")
def presentation_chat():
    """Grounded Retrieval-Augmented Generation (RAG) Q&A endpoint (Section 3.7 & 4.12)."""
    data = request.get_json(silent=True) or {}
    query = data.get("query", "")
    if not query.strip():
        return jsonify(error="Query cannot be empty"), 400

    try:
        result = rag_service.answer_query(query)
    except OllamaGenerationError as exc:
        app.logger.error("Local RAG generation failed: %s", exc)
        return jsonify(
            error=(
                "Local Phi generation failed. Check that Ollama is running "
                "and the configured model is installed."
            )
        ), 503
    return jsonify(result)

@app.get("/api/meta")
def get_metadata():
    """Returns model reporting specifications, hyperparameters, and benchmark figures."""
    return jsonify(
        ensemble_formula="P_final(AI|X) = 0.60 * P_xgb(X) + 0.40 * P_sgd(X)",
        vectorizer={
            "max_features": 10000,
            "ngram_range": [1, 2],
            "norm": "l2",
            "sublinear_tf": True
        },
        models={
            "xgboost": {"objective": "binary:logistic", "max_depth": 6, "eta": 0.05, "tree_method": "hist"},
            "sgd": {"loss": "log_loss", "penalty": "l2", "alpha": 0.0001}
        },
        reported_metrics={
            "accuracy": 0.952,
            "precision_ai": 0.96,
            "recall_ai": 0.93,
            "f1_ai": 0.945,
            "turnaround_seconds_20slides": 2.68
        },
        slide_rules=[
            {"rule": "Text Density", "trigger": "> 40 words", "action": "Suggest split or speaker notes"},
            {"rule": "Visual Balance", "trigger": "> 10 words and no picture", "action": "Suggest adding diagram/image"},
            {"rule": "Typography", "trigger": "Explicit font size < 18 pt", "action": "Increase font size"}
        ]
    )

if __name__ == "__main__":
    import webbrowser
    import threading

    # Enable browser tab close detection & auto-termination
    tab_tracker.start()

    def open_browser():
        try:
            webbrowser.open_new("http://127.0.0.1:5000/login")
        except Exception:
            pass

    # Automatically launch default browser 1.2s after server starts
    threading.Timer(1.2, open_browser).start()
    print(" * DocuSense AI running on http://127.0.0.1:5000")
    print(" * Opening browser automatically at http://127.0.0.1:5000/login ...")
    print(" * [Auto-Terminate] Batch job will automatically exit when you close the browser tab.")
    app.run(host="127.0.0.1", port=5000, debug=False)
