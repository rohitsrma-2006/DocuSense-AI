from pathlib import Path
import os
import math
from typing import List, Dict, Any, Optional
from core.preprocessing import clean_text, tokenize_sentences

ARTIFACTS_DIR = Path("artifacts")

class ContentClassifier:
    """
    Weighted Ensemble Classifier implementing Section 4.7 & 5.6 of the specification:
    P_final(AI | X) = 0.60 * P_xgb(X) + 0.40 * P_sgd(X)
    """
    def __init__(self, artifacts_dir: Optional[Path] = None):
        self.artifacts_dir = artifacts_dir or ARTIFACTS_DIR
        self.vectorizer = None
        self.sgd_model = None
        self.xgb_model = None
        self.is_loaded = False
        self.load_models()

    def load_models(self) -> bool:
        vec_path = self.artifacts_dir / "vectorizer.joblib"
        sgd_path = self.artifacts_dir / "sgd_model.joblib"
        xgb_path = self.artifacts_dir / "xgb_model.json"

        if not (vec_path.exists() and sgd_path.exists() and xgb_path.exists()):
            return False

        try:
            import joblib
            import xgboost as xgb

            self.vectorizer = joblib.load(vec_path)
            self.sgd_model = joblib.load(sgd_path)
            self.xgb_model = xgb.Booster()
            self.xgb_model.load_model(str(xgb_path))
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Warning loading model artifacts: {e}")
            return False

    def score_text(self, text: str) -> float:
        """
        Computes the weighted ensemble probability for a given sentence.
        Returns float('nan') if cleaned text has fewer than 5 tokens.
        """
        cleaned = clean_text(text)
        tokens = cleaned.split()
        if len(tokens) < 5:
            return float("nan")

        if not self.is_loaded:
            # Fallback heuristic statistical estimator if model artifacts are not yet built [Assumption / External]
            # Uses vocabulary density, average token length, and characteristic syntax markers
            ai_markers = {"moreover", "furthermore", "crucial", "testament", "delve", "tapestry", "underscores", "paramount", "in summary", "overall"}
            marker_count = sum(1 for w in tokens if w in ai_markers)
            avg_word_len = sum(len(w) for w in tokens) / max(1, len(tokens))
            base_prob = 0.35 + (0.12 * marker_count) + (0.02 * (avg_word_len - 5.0))
            return max(0.05, min(0.95, base_prob))

        try:
            import xgboost as xgb
            matrix = self.vectorizer.transform([cleaned]).astype("float32")
            
            # Predict probabilities
            p_sgd = float(self.sgd_model.predict_proba(matrix)[0, 1])
            p_xgb = float(self.xgb_model.predict(xgb.DMatrix(matrix))[0])
            
            # Weighted formula: 60% XGBoost, 40% SGD
            final_p = 0.60 * p_xgb + 0.40 * p_sgd
            return float(final_p)
        except Exception:
            return float("nan")

    def highlight_sentences(self, text: str) -> List[Dict[str, Any]]:
        """
        Segments text into sentences and assigns probability scores and UI display bands:
        - probability >= 0.70: "high" (Red)
        - probability >= 0.40: "medium" (Amber)
        - probability < 0.40: "low" (Green)
        - tokens < 5: "insufficient"
        """
        results = []
        sentences = tokenize_sentences(text or "")
        
        for sentence in sentences:
            sentence_clean = sentence.strip()
            if not sentence_clean:
                continue
                
            prob = self.score_text(sentence_clean)
            
            if math.isnan(prob):
                band = "insufficient"
                prob_val = None
            elif prob >= 0.70:
                band = "high"
                prob_val = round(prob, 4)
            elif prob >= 0.40:
                band = "medium"
                prob_val = round(prob, 4)
            else:
                band = "low"
                prob_val = round(prob, 4)

            results.append({
                "sentence": sentence_clean,
                "ai_probability": prob_val,
                "band": band
            })

        return results
