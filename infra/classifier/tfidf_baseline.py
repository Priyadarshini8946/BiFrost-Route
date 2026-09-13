"""Layer 3: train the TF-IDF + LogisticRegression fallback classifier.

The plan's primary Layer-3 classifier is fine-tuned DistilBERT (see
model.py). This module trains the deterministic offline fallback that keeps
the pipe running on machines without torch/transformers, and also serves as
a cheap baseline to sanity-check that DistilBERT is actually learning
semantics (it must beat this baseline on the held-out EVAL split).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("layer3.tfidf")

from infra.classifier.model import CLASSES, DATASET_PATH, MODEL_DIR, load_dataset  # noqa: E402


def train_tfidf_lr() -> Path:
    """Fit TfidfVectorizer + LogisticRegression on the layer-3 train split."""
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    data = load_dataset()
    texts = [x["text"] for x in data["train"]]
    labels = [CLASSES.index(x["label"]) for x in data["train"]]

    vec = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, min_df=2, lowercase=True)
    X = vec.fit_transform(texts)
    lr = LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced")
    lr.fit(X, labels)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, MODEL_DIR / "tfidf_vec.joblib")
    joblib.dump(lr, MODEL_DIR / "lr.joblib")

    # quick sanity accuracy on train (memorization check is eval-time anyway)
    acc_train = float((lr.predict(X) == labels).mean())
    log.info("tfidf-lr fit ok (train acc %.3f)", acc_train)
    return MODEL_DIR / "lr.joblib"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_tfidf_lr()
    from infra.classifier.model import ComplexityClassifier, evaluate

    clf = ComplexityClassifier(prefer_onnx=False)
    print(json.dumps(evaluate(classifier=clf), indent=2))