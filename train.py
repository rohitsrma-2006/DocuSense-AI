"""
train.py: Chunked SGD training and XGBoost continuation
Directly implementing Appendix A (Section 10.2) of the Project Specification.

Key features:
1. Out-of-core data stream with pd.read_csv(chunksize=50_000)
2. Fixed 10,000 TF-IDF vocabulary fitted on a training-only sample
3. SGDClassifier with incremental partial_fit(classes=[0, 1])
4. XGBoost booster continuation with xgb_model=booster
5. Artifact persistence in artifacts/
"""

from pathlib import Path
import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from core.preprocessing import clean_text

DATA = Path("data/AI_Human.csv")
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)
CHUNK_SIZE = 50_000
TEXT_COL, LABEL_COL = "text", "label"

def run_training():
    if not DATA.exists():
        print(f"[!] Dataset file '{DATA}' not found.")
        print("To run on a full 500k dataset, place AI_Human.csv inside the data/ folder.")
        return

    print("Step 1: Fitting TF-IDF vocabulary on a training-only sample (up to 100,000 rows)...")
    sample = pd.read_csv(DATA, usecols=[TEXT_COL, LABEL_COL], nrows=100_000)
    sample[TEXT_COL] = sample[TEXT_COL].map(clean_text)

    vectorizer = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        dtype="float32"
    )
    vectorizer.fit(sample[TEXT_COL])
    joblib.dump(vectorizer, ARTIFACTS / "vectorizer.joblib")
    del sample
    print(f"  Vocabulary fitted ({len(vectorizer.vocabulary_)} terms) and saved to artifacts/vectorizer.joblib.")

    print("\nStep 2: Initializing SGDClassifier and XGBoost parameters...")
    sgd = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        random_state=42
    )
    classes = [0, 1]
    booster = None
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "eta": 0.05,
        "tree_method": "hist",
        "seed": 42,
        "nthread": 4
    }

    print("\nStep 3: Streaming chunks of 50,000 records...")
    for chunk_no, frame in enumerate(pd.read_csv(DATA, usecols=[TEXT_COL, LABEL_COL], chunksize=CHUNK_SIZE)):
        frame = frame.dropna(subset=[TEXT_COL, LABEL_COL])
        texts = frame[TEXT_COL].map(clean_text)
        y = frame[LABEL_COL].astype("int32").to_numpy()
        X = vectorizer.transform(texts).astype("float32")

        # SGD incremental update
        if chunk_no == 0:
            sgd.partial_fit(X, y, classes=classes)
        else:
            sgd.partial_fit(X, y)

        # XGBoost sequential continuation
        dtrain = xgb.DMatrix(X, label=y)
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=10,
            xgb_model=booster
        )
        print(f"  Processed chunk {chunk_no + 1} ({len(frame)} rows)")

    joblib.dump(sgd, ARTIFACTS / "sgd_model.joblib")
    booster.save_model(str(ARTIFACTS / "xgb_model.json"))
    print("\n[✓] Training complete. All model artifacts serialized in artifacts/.")

if __name__ == "__main__":
    run_training()
