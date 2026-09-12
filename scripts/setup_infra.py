#!/usr/bin/env python
"""One-shot Layer-1 infrastructure setup:

  1. Redis    — configure maxmemory + allkeys-lru, clean cache
  2. DynamoDB — create query_traces (GSIs, on-demand), seed 100 traces
  3. MySQL    — apply schema.sql (idempotent) + seed data
  4. Warehouse— init DuckDB schema, load traces, export parquet, run rollups

Usage:  python scripts/setup_infra.py [--reseed-traces]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows consoles default to cp1252 — force UTF-8 so arrows/box drawing survive.
for stream in (sys.stdout, sys.stderr):
    if stream is not None and hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from tabulate import tabulate  # noqa: E402

from infra.cache.semantic_cache import SemanticCache, TfidfEmbedder  # noqa: E402
from infra.db.mysql_client import apply_schema, demo_queries, table_counts  # noqa: E402
from infra.dynamodb.seed_traces import demo_escalations_last_hour, seed_traces  # noqa: E402
from infra.dynamodb.setup_table import create_query_traces_table  # noqa: E402
from infra.warehouse.duckdb_warehouse import DuckDBWarehouse  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reseed-traces", action="store_true")
    parser.add_argument("--skip-mysql", action="store_true")
    args = parser.parse_args()

    # ── 1. Redis ─────────────────────────────────────────────────────────
    print("\n== [1/4] Redis semantic cache ==")
    tiny = TfidfEmbedder().fit(["seed corpus for index creation", "bifrost route layer 1"])
    cache = SemanticCache(embedder=tiny)
    cache.ensure_lru_config()
    cache.flush()
    print(f"  connected {cache.host}:{cache.port} | TTL {cache.ttl}s | "
          f"policy={cache.info()['maxmemory_policy']}")

    # ── 2. DynamoDB ───────────────────────────────────────────────────────
    print("\n== [2/4] DynamoDB query_traces ==")
    create_query_traces_table()
    items = seed_traces(count=100, reseed=args.reseed_traces)
    print(f"  seeded {len(items)} traces")
    esc = demo_escalations_last_hour()
    print(f"  demo GSI query — escalations in last hour: {len(esc)}")

    # ── 3. MySQL ──────────────────────────────────────────────────────────
    print("\n== [3/4] MySQL policies / budgets / registry ==")
    if not args.skip_mysql:
        apply_schema()
        for table, n in table_counts().items():
            print(f"  {table}: {n} rows")
        dq = demo_queries()
        policy = dq["policy"]
        print(f"  active policy {policy[0]} v{policy[1]} | medium→{policy[4]} | gate≥{policy[6]}")
        budget = dq["budget"]
        print(f"  eng-core budget: ${budget[1]} cap, ${budget[2]} spent, ${budget[3]} remaining")
    else:
        print("  skipped (--skip-mysql)")

    # ── 4. Warehouse ──────────────────────────────────────────────────────
    print("\n== [4/4] DuckDB warehouse ==")
    wh = DuckDBWarehouse()
    wh.init_schema()
    wh.con.execute("DELETE FROM fact_query_cost")
    loaded = wh.load_traces(items)
    parquet = wh.export_parquet()
    print(f"  loaded {loaded} fact rows | parquet export → {parquet.name}")

    elapsed, saved = wh.total_cost_saved_for()
    print(f"  ACCEPTANCE — cost saved today: ${saved['cost_saved_usd']} "
          f"(baseline ${saved['frontier_baseline_usd']} → actual ${saved['actual_cost_usd']}, "
          f"{saved['saved_pct']}% saved) in {elapsed*1000:.1f} ms")
    print("\n  cost breakdown by tier:")
    print(tabulate(wh.cost_breakdown_by_tier(), headers="keys"))
    wh.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())