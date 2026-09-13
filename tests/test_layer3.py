"""Layer 3 unit tests (offline — no torch/transformers required)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.classifier.dataset import build_dataset  # noqa: E402
from infra.classifier.model import CLASSES, load_dataset  # noqa: E402
from infra.classifier.tfidf_baseline import train_tfidf_lr  # noqa: E402


def test_dataset_is_deterministic_and_sized():
    a = build_dataset()
    b = build_dataset()
    assert len(a["train"]) >= 800
    assert len(a["eval"]) >= 50
    assert a["train"] == b["train"]
    assert a["eval"] == b["eval"]
    assert set(a["classes"]) == {"simple", "medium", "complex"}


def test_eval_is_held_out_from_train():
    d = load_dataset()
    train_texts = {x["text"] for x in d["train"]}
    overlap = [x for x in d["eval"] if x["text"] in train_texts]
    assert overlap == [], f"{len(overlap)} eval samples leaked into train"


def test_all_three_classes_present_in_both_splits():
    d = load_dataset()
    for split in ("train", "eval"):
        labels = {x["label"] for x in d[split]}
        assert labels == set(CLASSES)


def test_tfidf_fallback_trains_and_predicts():
    # No training in tests — use artifacts trained by the user's run.
    from infra.classifier.model import MODEL_DIR

    if not (MODEL_DIR / "lr.joblib").exists():
        pytest.skip("trained artifacts missing — run scripts\\train_layer3.py "
                    "(or the cloud trainer) first")
    from infra.classifier.model import ComplexityClassifier

    clf = ComplexityClassifier(prefer_onnx=False)
    assert clf.mode == "tfidf-lr"
    label, conf = clf.classify("how do I reset my password")
    assert label in CLASSES
    assert 0.0 < conf <= 1.0


def test_tfidf_baseline_beats_chance():
    from infra.classifier.model import MODEL_DIR

    if not (MODEL_DIR / "lr.joblib").exists():
        pytest.skip("trained artifacts missing — train first")
    from infra.classifier.model import ComplexityClassifier, evaluate

    clf = ComplexityClassifier(prefer_onnx=False)
    stats = evaluate(classifier=clf)
    assert stats["accuracy"] >= 0.70, "baseline collapsed"