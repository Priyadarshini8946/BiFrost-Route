#!/usr/bin/env python
"""Layer-1 acceptance harness — the "how do I check the results" gate.

Runs every Layer-1 acceptance target and FAILS (exit 1) on any miss:

  G0  MySQL schema seeded                      (3 tables have rows)
  G1  DynamoDB trace write latency  mean < 50 ms
  G2  Redis cache hit rate          > 30%  (semantic traffic benchmark)
      + LRU eviction proven (evicted_keys moves, policy = allkeys-lru)
  G3  "Total cost saved today"      < 2 s  (DuckDB warehouse, Redshift stand-in)

Results are also persisted to data/layer1_results.json.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabulate import tabulate  # noqa: E402

from infra.cache.cache_benchmark import (  # noqa: E402
    build_query_pool,
    rephrase,
    run_lru_eviction_demo,
    run_traffic,
)
from infra.cache.embedder import TfidfEmbedder  # noqa: E402
from infra.cache.semantic_cache import SemanticCache  # noqa: E402
from infra.db.mysql_client import table_counts  # noqa: E402
from infra.dynamodb.client import dynamodb_resource, table_name  # noqa: E402
from infra.dynamodb.seed_traces import seed_traces  # noqa: E402
from infra.warehouse.duckdb_warehouse import DuckDBWarehouse  # noqa: E402

GATES: List[Dict[str, Any]] = []


def gate(name: str, target: str, measured: Any, passed: bool, detail: str = "") -> None:
    GATES.append({"gate": name, "target": target, "measured": measured,
                  "passed": bool(passed), "detail": detail})
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {name}: target {target} | measured {measured} {detail}")


def check_mysql() -> None:
    print("\n── G0 · MySQL schema + seed data ──")
    counts = table_counts()
    for t, n in counts.items():
        print(f"  {t}: {n} rows")
    gate("G0 MySQL seeded", "3 tables > 0 rows", counts,
         all(n > 0 for n in counts.values()))


def check_dynamodb_write_latency() -> None:
    print("\n── G1 · DynamoDB trace write latency (< 50 ms mean) ──")
    items = seed_traces()  # returns existing 100 traces as plain dicts
    table = dynamodb_resource().Table(table_name())
    lats: List[float] = []
    for item in items:
        t0 = time.perf_counter()
        table.put_item(Item=item)
        lats.append((time.perf_counter() - t0) * 1000)  # ms
    lats.sort()
    def pct(p: float) -> float:
        return lats[min(len(lats) - 1, int(len(lats) * p))]
    stats = {
        "n": len(lats),
        "mean_ms": round(statistics.mean(lats), 2),
        "p50_ms": round(pct(0.50), 2),
        "p95_ms": round(pct(0.95), 2),
        "p99_ms": round(pct(0.99), 2),
        "max_ms": round(lats[-1], 2),
    }
    print(f"  {stats}")
    gate("G1 trace write latency", "mean < 50 ms", f"{stats['mean_ms']} ms",
         stats["mean_ms"] < 50.0)


def check_redis_cache() -> None:
    print("\n── G2 · Redis semantic cache hit rate (> 30%) + LRU eviction ──")
    pool = build_query_pool(n=800)
    # Paraphrase variants are part of the corpus so TF-IDF sees their tokens.
    corpus = pool + [rephrase(q, "exact") for q in pool[:80]]
    embedder = TfidfEmbedder().fit(corpus)
    cache = SemanticCache(embedder=embedder)
    cache.ensure_lru_config()
    cache.flush()

    traffic = run_traffic(cache, pool, traffic=1000)
    print(f"  traffic: {traffic['hits']}/{traffic['total_lookups']} hits "
          f"({traffic['traffic_hit_rate']*100:.1f}%) | exact {traffic['exact_hit_rate']*100:.1f}% | "
          f"rephrased {traffic['para_hit_rate']*100:.1f}% (mean sim {traffic['para_mean_similarity']})")

    lru = run_lru_eviction_demo(cache)
    policy = lru["maxmemory_policy"]
    print(f"  LRU: evicted_keys +{lru['eviction_delta']} | policy={policy} | "
          f"sampled cache keys surviving {lru['sampled_cache_keys_surviving']}/{lru['sampled_cache_keys_before']}")

    gate("G2 cache hit rate", "> 30%", f"{traffic['traffic_hit_rate']*100:.1f}%",
         traffic["traffic_hit_rate"] > 0.30)
    gate("G2 LRU eviction", "evicted_keys > 0 & policy=allkeys-lru",
         f"delta +{lru['eviction_delta']} ({policy})",
         lru["eviction_delta"] > 0 and policy == "allkeys-lru")


def check_warehouse() -> None:
    print("\n── G3 · Warehouse \"total cost saved today\" (< 2 s) ──")
    from infra.dynamodb.seed_traces import scan_all

    wh = DuckDBWarehouse()
    wh.init_schema()
    wh.con.execute("DELETE FROM fact_query_cost")
    traces = scan_all()
    wh.load_traces(traces)
    wh.export_parquet()

    elapsed, saved = wh.total_cost_saved_for()  # default: UTC today
    print(f"  query returned in {elapsed*1000:.1f} ms")
    print(f"  today: actual ${saved['actual_cost_usd']} vs frontier-only "
          f"${saved['frontier_baseline_usd']} → saved ${saved['cost_saved_usd']} "
          f"({saved['saved_pct']}%)")
    print("\n  cost breakdown by tier:")
    print(tabulate(wh.cost_breakdown_by_tier(), headers="keys"))
    gate("G3 cost-saved query latency", "< 2 s", f"{elapsed*1000:.1f} ms",
         elapsed < 2.0)
    gate("G3 cost saved > 0", "> $0", f"${saved['cost_saved_usd']}",
         saved["cost_saved_usd"] > 0)
    wh.close()


def main() -> int:
    print("=" * 72)
    print("BIFROST ROUTE · LAYER 1 ACCEPTANCE CHECK "
          f"({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')})")
    print("=" * 72)

    check_mysql()
    check_dynamodb_write_latency()
    check_redis_cache()
    check_warehouse()

    passed = sum(1 for g in GATES if g["passed"])
    print("\n" + "=" * 72)
    print(f"RESULT: {passed}/{len(GATES)} gates passed")

    out = Path("./data/layer1_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gates": GATES,
        "all_passed": passed == len(GATES),
    }, indent=2), encoding="utf-8")
    print(f"persisted → {out}")
    return 0 if passed == len(GATES) else 1


if __name__ == "__main__":
    raise SystemExit(main())