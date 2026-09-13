"""Layer 3: DistilBERT query-complexity classifier (fine-tune + ONNX export).

Trains `distilbert-base-uncased` sequence classification head on the
layer-3 dataset (3 classes: simple/medium/complex), evaluates on the
hand-written held-out EVAL split, exports to ONNX (optimum) for fast
serving, and exposes a tiny `ComplexityClassifier` with the same API for
both the torch model and the ONNX artifact.

CPU-friendly: small dataset (~1k), 3 epochs, fp32, batch 16.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("layer3.classifier")

DATASET_PATH = Path("data/layer3_dataset.json")
MODEL_DIR = Path("models/layer3_distilbert")
ONNX_DIR = Path("models/layer3_onnx")
CLASSES = ["simple", "medium", "complex"]

try:  # torch/transformers are optional at import time
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        pipeline,
    )
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy (prevents majority collapse)."""

    def __init__(self, class_weights: Any = None,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss = torch.nn.functional.cross_entropy(
                logits, labels, weight=self.class_weights)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


def load_dataset() -> Dict[str, Any]:
    if not DATASET_PATH.exists():
        from infra.classifier.dataset import build_dataset

        build_dataset(DATASET_PATH)
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


class ComplexityClassifier:
    """Small wrapper: predict(str) -> (label, confidence). modes:
    'distilbert-torch' | 'distilbert-onnx' | 'tfidf-lr' (fallback).
    """

    def __init__(self, prefer_onnx: bool = True) -> None:
        self.mode = "none"
        self._pipeline = None
        self._onnx = None
        self._lr = None
        self._vec = None
        if prefer_onnx and ONNX_DIR.exists():
            self._load_onnx()
        if self.mode == "none" and _HAS_TORCH:
            self._load_torch()
        if self.mode == "none":
            self._load_lr()
        if self.mode == "none":
            raise RuntimeError("no classifier backend available")

    def _load_onnx(self) -> None:
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification

            tokenizer = AutoTokenizer.from_pretrained(str(ONNX_DIR))
            model = ORTModelForSequenceClassification.from_pretrained(
                str(ONNX_DIR))
            import torch as _t

            self._onnx = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                top_k=None,
                device=-1,
            )
            self.mode = "distilbert-onnx"
        except Exception as exc:  # noqa: BLE001
            log.warning("onnx load failed: %s", exc)
            self._onnx = None

    def _load_torch(self) -> None:
        try:
            if not (MODEL_DIR / "config.json").exists():
                return
            self._pipeline = pipeline(
                "text-classification",
                model=str(MODEL_DIR),
                tokenizer=str(MODEL_DIR),
                top_k=None,
                device=-1,
            )
            self.mode = "distilbert-torch"
        except Exception as exc:  # noqa: BLE001
            log.warning("torch load failed: %s", exc)
            self._pipeline = None

    def _load_lr(self) -> None:
        try:
            import joblib
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression

            self._vec = joblib.load(str(MODEL_DIR / "tfidf_vec.joblib"))
            self._lr = joblib.load(str(MODEL_DIR / "lr.joblib"))
            self.mode = "tfidf-lr"
        except Exception as exc:  # noqa: BLE001
            log.warning("tfidf-lr load failed: %s", exc)

    # ── prediction ─────────────────────────────────────────────────────────
    def _pred_onnx_torch(self, text: str) -> Tuple[str, float]:
        res = (self._onnx or self._pipeline)(text)[0]  # list of {label, score}
        top = max(res, key=lambda r: r["score"])
        return top["label"], float(top["score"])

    def _pred_lr(self, text: str) -> Tuple[str, float]:
        import numpy as np

        vec = self._vec.transform([text])
        probs = self._lr.predict_proba(vec)[0]
        idx = int(np.argmax(probs))
        label_idx = int(self._lr.classes_[idx])  # classes_ are int labels
        label = CLASSES[label_idx] if label_idx < len(CLASSES) else str(label_idx)
        return label, float(probs[idx])

    def classify(self, text: str) -> Tuple[str, float]:
        if self.mode == "tfidf-lr":
            return self._pred_lr(text)
        return self._pred_onnx_torch(text)


class ComplexityDataset:
    """HuggingFace datasets wrapper for the layer-3 train/eval split."""

    def __init__(self, encodings: Dict[str, Any], labels: List[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


def train(epochs: int = 3, batch_size: int = 16) -> Dict[str, Any]:
    """Fine-tune DistilBERT on the layer-3 train split; persist torch + ONNX."""
    if not _HAS_TORCH:
        raise RuntimeError("torch/transformers not installed")

    data = load_dataset()
    labels = [CLASSES.index(x["label"]) for x in data["train"]]
    texts = [x["text"] for x in data["train"]]

    # class weights: inverse frequency, so the loss does not let the
    # majority class (simple) dominate and crush the others
    from collections import Counter

    counts = Counter(labels)
    total = len(labels)
    w = [total / (len(CLASSES) * counts[i]) for i in range(len(CLASSES))]
    class_weights = torch.tensor(w, dtype=torch.float32)

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, id2label={i: c for i, c in enumerate(CLASSES)},
        label2id={c: i for i, c in enumerate(CLASSES)})

    # max_len 128: complex queries are long; 64 truncated their tails
    enc = tokenizer(texts, truncation=True, padding=True, max_length=128)
    ds = ComplexityDataset(enc, labels)

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=25,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        disable_tqdm=False,
        seed=11,
    )
    trainer = WeightedTrainer(
        class_weights=class_weights, model=model, args=args, train_dataset=ds)
    t0 = time.perf_counter()
    trainer.train()
    train_secs = time.perf_counter() - t0

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    # ONNX export (optimum) — the serving artifact
    onnx_ok = False
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification

        model.config.save_pretrained(str(MODEL_DIR))
        ort_model = ORTModelForSequenceClassification.from_pretrained(
            str(MODEL_DIR), export=True)
        ONNX_DIR.mkdir(parents=True, exist_ok=True)
        ort_model.save_pretrained(str(ONNX_DIR))
        tokenizer.save_pretrained(str(ONNX_DIR))
        onnx_ok = True
    except Exception as exc:  # noqa: BLE001
        log.warning("onnx export failed (serving falls back to torch): %s", exc)

    stats = evaluate(classes=CLASSES)
    stats["train_secs"] = round(train_secs, 1)
    stats["onnx_exported"] = onnx_ok
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def evaluate(classes: Optional[List[str]] = None,
             classifier: Optional[ComplexityClassifier] = None) -> Dict[str, Any]:
    """Evaluate on the held-out EVAL split: accuracy + per-class F1/recall."""
    from collections import Counter

    data = load_dataset()
    classes = classes or CLASSES
    own = classifier is None
    clf = classifier or ComplexityClassifier()
    correct = 0
    conf: Dict[str, Counter] = {c: Counter() for c in classes}
    lats: List[float] = []
    for sample in data["eval"]:
        t0 = time.perf_counter()
        pred, confv = clf.classify(sample["text"])
        lats.append((time.perf_counter() - t0) * 1000.0)
        if pred == sample["label"]:
            correct += 1
        conf[sample["label"]][pred] += 1
    acc = correct / len(data["eval"])
    rows = {}
    for label in classes:
        tp = conf[label][label]
        fp = sum(conf[other][label] for other in classes if other != label)
        fn = sum(conf[label][other] for other in classes if other != label)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows[label] = {"precision": round(prec, 4), "recall": round(rec, 4),
                       "f1": round(f1, 4)}
    if own:
        clf = None
    lats.sort()
    return {
        "accuracy": round(acc, 4),
        "per_class": rows,
        "latency_ms": {"mean": round(sum(lats) / len(lats), 2),
                       "p95": round(lats[int(len(lats) * 0.95)], 2)},
        "eval_n": len(data["eval"]),
        "mode": getattr(classifier, "mode", clf.mode if clf else "?"),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("training distilbert classifier…")
    stats = train()
    print(json.dumps(stats, indent=2))