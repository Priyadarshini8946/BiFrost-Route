#!/usr/bin/env python
"""Layer-3 TRAINING entrypoint — runs ONLY after explicit user approval.

Trains in this order:
  1. TF-IDF + LogisticRegression baseline (fallback + C4 comparator)
  2. DistilBERT fine-tune (3 classes) + ONNX export (serving artifact)

Persists artifacts under models/layer3_distilbert/ (+ models/layer3_onnx/)
and writes metrics.json. You run this script, not the gate script, when
you want to (re)train. The gate script never trains.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for stream in (sys.stdout, sys.stderr):
    if stream is not None and hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

EPOCHS = 5          # more epochs with weighted loss, still CPU-tractable (~1k samples)
BATCH = 16


def main() -> int:
    from infra.classifier.dataset import build_dataset  # noqa: E402
    from infra.classifier.model import load_dataset, train  # noqa: E402
    from infra.classifier.tfidf_baseline import train_tfidf_lr  # noqa: E402

    print("=" * 72)
    print("LAYER 3 TRAINING (run after explicit user approval)")
    print("=" * 72)
    d = build_dataset()
    print(f"dataset: {len(d['train'])} train / {len(d['eval'])} eval")

    t0 = time.perf_counter()
    print("\n[1/2] training TF-IDF + LogisticRegression baseline …")
    train_tfidf_lr()
    print(f"      done in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    print(f"\n[2/2] fine-tuning DistilBERT ({EPOCHS} epochs, CPU) …")
    stats = train(epochs=EPOCHS, batch_size=BATCH)
    print(f"      trained in {stats['train_secs']}s | accuracy {stats['accuracy']} | "
          f"onnx exported: {stats['onnx_exported']}")

    print("\nnext: scripts\\check_layer3_results.py  (the acceptance gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())