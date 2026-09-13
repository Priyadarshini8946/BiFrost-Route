#!/usr/bin/env python
"""Layer-3 acceptance harness — the "how do I check the results" gate.

  C1  eval accuracy            ≥ 0.90    (held-out rephrased EVAL split)
  C2  macro-F1                  ≥ 0.85    (semantics, not keyword matching)
  C3  per-class F1              ≥ 0.80    (no class collapses)
  C4  classifier beats the TF-IDF baseline (distilbert > tfidf baseline)
  C5  serving latency (mean)    < 150 ms  (ONNX or torch on CPU)

Also demonstrates the economic point of the layer: routing 100 real-shaped
queries with the classifier picks cheap/standard/frontier models — total
cost vs a frontier-only baseline, persisted for the dashboard layers.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for stream in (sys.stdout, sys.stderr):
    if stream is not None and hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from infra.classifier.model import (  # noqa: E402
    CLASSES,
    ComplexityClassifier,
    DATASET_PATH,
    evaluate,
    load_dataset,
)
from infra.classifier.tfidf_baseline import train_tfidf_lr  # noqa: E402
from infra.data_generation.trace_factory import (  # noqa: E402
    COMPLEXITY_TO_MODEL,
    MODEL_COSTS,
    generate_query_traces,
)

GATES: List[Dict[str, Any]] = []


def gate(name: str, target: str, measured: Any, passed: bool) -> None:
    GATES.append({"gate": name, "target": target, "measured": measured,
                  "passed": bool(passed)})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: target {target} "
          f"| measured {measured}")


def sim_routing(clf: ComplexityClassifier, n: int = 100) -> Dict[str, Any]:
    """Route n production-shaped traces by predicted complexity; cost math."""
    traces = generate_query_traces(count=n, seed=7)
    routed = []
    confusions = {"simple": 0, "medium": 0, "complex": 0}
    for t in traces:
        pred, conf = clf.classify(t["query_text"])
        model = COMPLEXITY_TO_MODEL.get(pred, "gemini-2.5-pro")
        cost = MODEL_COSTS[model]["in"] * t["prompt_tokens"] / 1000.0 \
            + MODEL_COSTS[model]["out"] * t["completion_tokens"] / 1000.0
        baseline = t["frontier_baseline_cost_usd"]
        routed.append({"trace_id": t["trace_id"], "true": t["complexity"],
                       "pred": pred, "conf": round(conf, 3),
                       "model": model, "cost_usd": round(cost, 6),
                       "baseline_usd": round(baseline, 6)})
        if pred != t["complexity"]:
            confusions[t["complexity"]] += 1
    total = sum(r["cost_usd"] for r in routed)
    baseline = sum(r["baseline_usd"] for r in routed)
    saved_pct = (baseline - total) / baseline * 100.0 if baseline else 0.0
    perfect = sum(1 for r in routed if r["pred"] == r["true"])
    return {
        "n": n,
        "routed_accuracy": round(perfect / n, 4),
        "total_cost_usd": round(total, 4),
        "frontier_only_usd": round(baseline, 4),
        "saved_usd": round(baseline - total, 4),
        "saved_pct": round(saved_pct, 1),
        "mismatches": confusions,
    }


def main() -> int:
    print("=" * 72)
    print("BIFROST ROUTE · LAYER 3 ACCEPTANCE CHECK "
          f"({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')})")
    print("=" * 72)

    data = load_dataset()
    print(f"\ndataset: train={len(data['train'])} eval={len(data['eval'])} "
          f"(classes {data['classes']})")
    if not (Path("models/layer3_distilbert/config.json").exists()):
        print("\n[BLOCKED] distilbert model is not trained yet.")
        print("Model training requires your explicit approval — this script will")
        print("never train. Run:  scripts\\train_layer3.py   after you approve.")
        return 2

    clf = ComplexityClassifier()
    stats = evaluate(classifier=clf)
    print(f"\n── classifier: {clf.mode} ──")

    acc = stats["accuracy"]; macro_f1 = sum(v["f1"] for v in stats["per_class"].values()) / len(stats["per_class"])
    per_class_min = min(v["f1"] for v in stats["per_class"].values())
    mean_lat = stats["latency_ms"]["mean"]
    print(f"  accuracy      : {acc:.3f}")
    print(f"  macro-F1      : {macro_f1:.3f}  (per-class { {k: round(v['f1'],3) for k,v in stats['per_class'].items()} })")
    print(f"  latency       : mean {mean_lat} ms (p95 {stats['latency_ms']['p95']} ms)")

    train_tfidf_lr = None  # no training inside the gate script

    if not Path("models/layer3_distilbert/lr.joblib").exists():
        print("\n[BLOCKED] tfidf baseline artifacts are missing — run "
              "scripts\\train_layer3.py first (after your approval).")
        return 2
    baseline_clf = ComplexityClassifier(prefer_onnx=False)
    baseline_stats = evaluate(classifier=baseline_clf)
    baseline_acc = baseline_stats["accuracy"]
    print(f"  tfidf baseline: accuracy {baseline_acc:.3f}")

    print("\n── gates ──")
    gate("C1 eval accuracy", ">= 0.90", f"{acc:.3f}", acc >= 0.90)
    gate("C2 macro-F1", ">= 0.85", f"{macro_f1:.3f}", macro_f1 >= 0.85)
    gate("C3 min per-class F1", ">= 0.80", f"{per_class_min:.3f}", per_class_min >= 0.80)
    gate("C4 beats TF-IDF baseline", f"> {baseline_acc:.3f}", f"{acc:.3f}", acc > baseline_acc)
    gate("C5 mean latency", "< 150 ms", f"{mean_lat} ms", mean_lat < 150.0)

    route = sim_routing(clf)
    print(f"\n── economics demo: {route['n']} production-shaped queries routed "
          f"by predicted complexity ──")
    print(f"  routing agreement: {route['routed_accuracy']:.1%} "
          f"| cost ${route['total_cost_usd']:.4f} vs frontier-only "
          f"${route['frontier_only_usd']:.4f} → saved ${route['saved_usd']:.4f} "
          f"({route['saved_pct']}%)")

    passed = sum(1 for g in GATES if g["passed"])
    print(f"\nRESULT: {passed}/{len(GATES)} gates passed")

    out = Path("data/layer3_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"train": len(data["train"]), "eval": len(data["eval"])},
        "classifier_mode": clf.mode,
        "metrics": {"accuracy": acc, "macro_f1": macro_f1,
                    "per_class": stats["per_class"], "latency_ms": stats["latency_ms"]},
        "baseline": {"mode": baseline_clf.mode, "accuracy": baseline_acc},
        "routing_demo": route,
        "gates": GATES,
        "all_passed": passed == len(GATES),
    }, indent=2), encoding="utf-8")
    print(f"persisted → {out}")
    return 0 if passed == len(GATES) else 1


if __name__ == "__main__":
    raise SystemExit(main())