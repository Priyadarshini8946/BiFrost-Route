#!/usr/bin/env python
"""Layer-3 provisioning — dataset only. NO model training here, ever.

Model training (TF-IDF baseline + DistilBERT fine-tune) requires explicit
user approval and lives in scripts/train_layer3.py. This script only
builds the deterministic dataset and prints what you still need to do.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for stream in (sys.stdout, sys.stderr):
    if stream is not None and hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from infra.classifier.dataset import build_dataset  # noqa: E402
from infra.classifier.model import MODEL_DIR, load_dataset  # noqa: E402


def main() -> int:
    d = build_dataset()
    print(f"[layer3] dataset: {len(d['train'])} train / {len(d['eval'])} eval samples")
    print("[layer3] dataset persisted → data/layer3_dataset.json")

    if (MODEL_DIR / "config.json").exists():
        print("[layer3] distilbert artifacts exist (from a previous run).")
    else:
        print("[layer3] no trained distilbert model yet.")

    print("[layer3] MODEL TRAINING IS DISABLED BY POLICY — no model was trained.")
    print("[layer3] after you approve, run:  scripts\\train_layer3.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())