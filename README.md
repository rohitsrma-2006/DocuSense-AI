# DocuSense AI: AI Content Detection & PPTX Document Analysis System
### Industrial Training Project — Novem Controls Pvt. Ltd. (Registration ID: `5075/25-26`)

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask%203.1-green.svg)](https://flask.palletsprojects.com/)
[![ML Library](https://img.shields.io/badge/ML-scikit--learn%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic%20Evaluation-lightgrey.svg)]()

---

## 📌 Project Overview

**DocuSense AI** is an integrated Natural Language Processing (NLP) and document intelligence web platform designed to analyze Microsoft PowerPoint presentations (`.pptx`). It addresses two critical needs in academic and professional document inspection:

1. **Statistical AI Content Likelihood:** Sentence-by-sentence probabilistic classification to evaluate whether phrasing reflects human authorship or machine generation (ChatGPT, Claude, Gemini).
2. **Slide Quality & Pedagogical Formatting Audit:** Automated heuristic inspection of presentation layout, checking for text overcrowding, missing visuals, and illegible font sizes.
3. **Grounded Retrieval-Augmented Generation (RAG):** Contextual question-answering strictly bounded by slide content, complete with slide citations and hallucination-preventing abstention logic.

---

## 🎓 Academic & Industrial Context

* **Host Organisation:** Novem Controls Pvt. Ltd. (Industrial Training Program)
* **Registration ID:** `5075/25-26`
* **Student Name:** Rohit Sharma
* **Roll Number:** `23010603054`
* **Department:** Computer Science & Engineering
* **Institution:** Government Post Graduate College Dharamshala

### Administrative Scope & Title Reconciliation (Section 5.1 & 10.1 of Report)
While the administrative title recorded on the industrial training certificate is **"AI Smart RAM Management System"** (originating from initial C#/WPF systems training and addressing out-of-core memory management to avoid `MemoryError` when training on 500,000 records), the actual technical system, ML models, and full-stack implementation represent the **AI Content Detection and Document Analysis Smart System**. Both labels are maintained for academic traceability.

---

## 🚀 Key Features

### 1. 60/40 Weighted Ensemble Machine Learning Model
Classifies text using a calibrated linear opinion pool combining **XGBoost** and an incremental **Stochastic Gradient Descent (SGD)** linear classifier:
$$\mathcal{P}_{\text{final}}(\text{AI} \mid X) = 0.60 \times \mathcal{P}_{\text{xgb}}(X) + 0.40 \times \mathcal{P}_{\text{sgd}}(X)$$
* **Feature Extraction:** 10,000 max-features TF-IDF vectorizer (unigrams + bigrams, sublinear term frequency, L2 normalization).
* **Color-Coded Probability Bands:**
  * 🔴 **High AI Likeness:** $\mathcal{P} \ge 0.70$
  * 🟡 **Moderate / Ambiguous:** $0.40 \le \mathcal{P} < 0.70$
  * 🟢 **Likely Human:** $\mathcal{P} < 0.40$
  * ⚪ **Insufficient Text Guard:** Sentences $< 5$ words are marked as `NaN` to prevent false positives on titles or bullet fragments.

### 2. PowerPoint Slide Audit Heuristics (`services/pptx_parser.py`)
Traverses OpenXML shape hierarchies with dual-engine support (`python-pptx` with automatic fallback to native standard-library `zipfile` + `xml.etree.ElementTree`):
* **Rule 1 (Text Density):** Flags slides with **$> 40$ words** (*"Text exceeds 40 words. Consider splitting into multiple slides or speaker notes."*).
* **Rule 2 (Visual Balance):** Flags slides with **$> 10$ words and zero detected images** (*"Text-heavy slide has no detected picture. Consider adding a chart or image."*).
* **Rule 3 (Typography):** Flags explicit text runs with font size **$< 18\text{ pt}$** (*"Run uses explicit font size below 18 pt. Increase font size."*).

### 3. Grounded Retrieval-Augmented Generation (RAG)
* Indexes slide text into searchable passages.
* Answers user questions about the presentation with **mandatory slide citations** (e.g., `(Ref: Slide 3)`).
* **Strict Abstention:** If queried facts are not in the presentation, it replies:  
  *"The uploaded presentation does not contain sufficient information to answer this query."*
* Summary requests (for example, "summarize this presentation") use all indexed slides, chunked to fit the local model's context window.
* **Prompt Injection Defense:** Treats presentation text strictly as passive data.
* Uses a local Ollama model to synthesize answers from retrieved passages. Install and run Ollama with `phi:latest` available; configure `OLLAMA_MODEL` and `OLLAMA_BASE_URL` to use a different local model or Ollama endpoint. RAG questions return an error if Ollama is unavailable.

### 4. Document Tone & Extractive Summarization
* Lexicon-based tone analysis (Academic, Analytical, Promotional, Cautious).
* Extracts the top 2–3 most representative sentences as an executive summary.

### 5. Out-of-Core Batch Learning (`train.py`)
* Solves the `MemoryError` bottleneck when processing **500,000 records** on standard 16 GB RAM hardware.
* Streams **50,000-row chunks** via `pd.read_csv(chunksize=50_000)`, applying SGD `partial_fit` and continuation boosting.

---

## 📊 Benchmark & Reported Results

| Metric | Reported Value | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **95.2%** | Overall correct predictions across test set |
| **AI-Class Precision** | **0.96** | 96% of AI predictions were true AI examples |
| **AI-Class Recall** | **0.93** | 93% of all AI-labeled test examples detected |
| **AI-Class F1 Score** | **0.945** | Harmonic mean of precision (0.96) and recall (0.93) |
| **Processing Latency** | **~2.68 s** | Evaluated turnaround on a standard 20-slide presentation |

---

## 📂 Project Structure

```
Smart Ram Mangaement/
│
├── core/
│   ├── __init__.py
│   ├── preprocessing.py      # Unicode NFKC, regex, stop-words, sentence tokenizer
│   └── classifier.py         # 60/40 Weighted Ensemble model (XGBoost + SGD)
│
├── services/
│   ├── __init__.py
│   ├── pptx_parser.py        # OpenXML parser with dual-engine fallback & 3 audit rules
│   ├── rag_service.py        # Grounded RAG search with citations & abstention
│   └── document_utils.py     # Lexicon tone analyzer & extractive summarizer
│
├── models/
│   ├── __init__.py
│   └── user.py               # PBKDF2-SHA256 password salting & session user store
│
├── artifacts/                # Pre-calibrated serialized models
│   ├── vectorizer.joblib     # 10,000-feature TF-IDF model
│   ├── sgd_model.joblib      # Log-loss SGDClassifier
│   ├── xgb_model.json        # Binary logistic XGBoost Booster
│   └── metadata.json         # Manifest & benchmark metrics
│
├── templates/                # Responsive dark-mode dashboard UI (Appendix D)
│   ├── base.html             # Base layout and navigation
│   ├── login.html            # Figure D.1: Authentication form
│   └── dashboard.html        # Figures D.2 - D.6: Upload, highlights, warnings, RAG chat
│
├── static/
│   ├── css/style.css         # Modern glassmorphism cyberpunk theme
│   └── js/main.js            # Real-time dashboard controller
│
├── instance/tmp/             # Ephemeral temporary upload directory (cleaned up automatically)
├── train.py                  # Out-of-core chunked training pipeline (50k batches)
├── generate_demo_assets.py   # Model artifact generator & sample presentation builder
├── sample_presentation.pptx  # Test presentation demonstrating all 3 audit rules
├── test_pipeline.py          # Unit test verification suite
├── test_app_routes.py        # Flask endpoint test suite
├── app.py                    # Main Flask application
├── run.bat                   # One-click Windows desktop launcher
├── requirements.txt          # Pinned dependency requirements
├── .gitignore                # Git exclusion specifications
├── PROJECT_REPORT.md         # Exhaustive academic report (11 chapters)
└── README.md                 # Project documentation (this file)
```

---

## ⚡ Quickstart Guide

### Option 1: One-Click Desktop Launch (Windows)
Double-click **`run.bat`** in the project folder.  
It automatically:
1. Verifies all Python dependencies.
2. Generates pre-calibrated model artifacts if missing.
3. Launches the Flask server and opens **`http://127.0.0.1:5000`** in your default browser.

### Option 2: Manual Terminal Execution
```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Build demo assets & sample presentation (if first run)
python generate_demo_assets.py

# 3. Start the application
python app.py
```
Open **`http://127.0.0.1:5000`** in your web browser.

---

## 🔑 Login Credentials

* **Username:** **`@Rohit`** *(or `Rohit`)*
* **Password:** **`rohit123`**

*(You can also create a new account using the "Register" tab on the login screen).*

---

## 🧪 Testing & Verification

Run the automated test suites to verify all ML, parsing, and route components:
```bash
# Verify NLP pipeline, PPTX parser, RAG search, and tone analysis
python test_pipeline.py

# Verify Flask authentication, file upload, analysis API, and chat routes
python test_app_routes.py
```

---

## ⚖️ Responsible AI & Ethical Principles (Section 8.3)

* **Decision Support, Not Forensic Proof:** Statistical classifier outputs reflect probabilistic linguistic similarity to training data. They must never be used as sole or definitive proof of academic misconduct.
* **Human-in-the-Loop:** Automated scores must be reviewed alongside drafts, version histories, and student interviews.
* **Privacy by Design:** Uploaded files are processed in isolated temporary storage and immediately unlinked in Python `finally` blocks, ensuring zero permanent retention of private user documents.

---

## 👨‍💻 Author & Credits

* **Student:** Rohit Sharma (Roll: `23010603054`)
* **Department:** Computer Science & Engineering, Govt. Post Graduate College Dharamshala
* **Host Organization:** Novem Controls Pvt. Ltd. (Industrial Training, June–July 2026)
