#!/usr/bin/env python
"""Layer-4 provisioning: verify the full routing stack is ready and warm the
engine (reranker + corpus + cache) so gate measurements are steady-state.

Run `scripts/check_layer4_results.py` for the acceptance gates.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# UTF-8 console output on Windows
for stream in (sys.stdout, sys.stderr):
    if stream is not None and hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from infra.routing.router import RouteEngine  # noqa: E402


def main() -> int:
    t0 = time.perf_counter()
    print("== [1/4] infra health (MySQL/Redis/DynamoDB/Neo4j/Qdrant) ==")
    eng = RouteEngine(force_dry_run=True)
    print(f"  engine built in {time.perf_counter()-t0:.1f}s")
    print(f"  classifier mode : {eng.classifier.mode}  (trained artifacts "
          f"auto-selected when fresh)")
    print(f"  llm mode        : {eng.llm_mode}  (dry-run mock until API keys set)")
    print(f"  policy          : {eng.policy['model_mapping']}")
    print(f"  policy hops     : {eng.policy['max_escalation_hops']}")

    print("\n== [2/4] warm route (first call loads reranker + corpus) ==")
    t1 = time.perf_counter()
    s = eng.route("How do I reset my password?")
    print(f"  route: {s.complexity} -> {s.model} [{s.tier}] "
          f"faithful={s.faithfulness} esc={s.escalation_count} "
          f"status={s.status} in {time.perf_counter()-t1:.1f}s")

    print("\n== [3/4] semantic cache check ==")
    s2 = eng.route("How do I reset my password?")
    print(f"  repeat route: cache_hit={s2.cache_hit} sim={s2.cache_similarity}")

    print("\n== [4/4] API liveness (dry-run server check) ==")
    try:
        from fastapi.testclient import TestClient

        import api.app as api_app

        api_app._engine = None  # reuse a fresh engine in the app
        client = TestClient(api_app.app)
        h = client.get("/health")
        print(f"  GET /health -> {h.status_code} {h.json()}")
        r = client.post("/route", json={"query": "Where is my order?"})
        print(f"  POST /route -> {r.status_code} "
              f"model={r.json().get('model')} status={r.json().get('status')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  API check failed: {type(exc).__name__}: {exc}")

    print(f"\nlayer-4 setup complete in {time.perf_counter()-t0:.1f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())