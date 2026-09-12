"""Layer-1 tests.

Unit tests run offline (no services). Integration tests are skipped unless
BIFROST_INTEGRATION=1 (they talk to the Dockerized MySQL/Redis/DynamoDB/DuckDB).
"""
from __future__ import annotations

import os
import time

import pytest

from infra.cache.cache_benchmark import build_query_pool, rephrase
from infra.cache.embedder import TfidfEmbedder, cosine
from infra.data_generation.trace_factory import (
    MODEL_COSTS,
    frontier_baseline_cost_usd,
    generate_query_traces,
    model_cost_usd,
)


# ── unit: trace factory ─────────────────────────────────────────────────
def test_trace_factory_deterministic():
    from datetime import datetime, timezone

    fixed_now = datetime(2026, 9, 12, 12, 0, 0, tzinfo=timezone.utc)
    a = generate_query_traces(100, seed=42, now=fixed_now)
    b = generate_query_traces(100, seed=42, now=fixed_now)
    assert a == b


def test_trace_factory_shapes():
    traces = generate_query_traces(100, seed=42)
    assert len(traces) == 100
    complexities = {t["complexity"] for t in traces}
    assert complexities == {"simple", "medium", "complex"}
    for t in traces:
        assert t["prompt_tokens"] > 0 and t["completion_tokens"] > 0
        assert t["cost_usd"] >= 0


def test_cost_math_matches_registry():
    traces = generate_query_traces(100, seed=42)
    for t in traces:
        expected = model_cost_usd(t["selected_model"], t["prompt_tokens"], t["completion_tokens"])
        assert abs(expected - t["cost_usd"]) < 1e-8


def test_frontier_baseline_never_cheaper():
    traces = generate_query_traces(100, seed=42)
    for t in traces:
        baseline = frontier_baseline_cost_usd(t["prompt_tokens"], t["completion_tokens"])
        assert baseline >= t["cost_usd"] - 1e-12


def test_escalation_only_within_policy():
    traces = generate_query_traces(200, seed=42)
    next_up = {"simple": "gemini-2.0-flash", "medium": "gemini-2.5-pro",
               "complex": "gemini-2.5-pro"}  # complex escalates to a frontier re-run
    for t in traces:
        expected = next_up[t["complexity"]] if t["did_escalate"] else t["selected_model"]
        assert t["final_model"] == expected


# ── unit: semantic embedder ─────────────────────────────────────────────
def test_embedder_identity_and_rephrase():
    pool = build_query_pool(n=200)
    corpus = pool + [rephrase(q, "exact") for q in pool[:40]]
    emb = TfidfEmbedder().fit(corpus)
    q = pool[17]
    assert emb.embed(q).shape == (1024,)
    assert abs(cosine(emb.embed(q), emb.embed(q)) - 1.0) < 1e-6
    # exact rephrase (stopword swap) is semantically identical → ~1.0
    sim = cosine(emb.embed(q), emb.embed(rephrase(q, "exact")))
    assert sim >= 0.99, f"exact rephrase similarity too low: {sim}"
    # unrelated query must be well below the 0.92 hit threshold
    other = rephrase("How do I export my data from portal?", "exact")
    low_sim = cosine(emb.embed(q), emb.embed(other))
    assert low_sim < 0.9


# ── integration (needs live Docker services) ────────────────────────────
@pytest.mark.skipif(os.getenv("BIFROST_INTEGRATION") != "1", reason="requires live services")
def test_live_redis_ping():
    from infra.cache.semantic_cache import SemanticCache, TfidfEmbedder

    cache = SemanticCache(embedder=TfidfEmbedder().fit(["a", "b", "c"]))
    assert cache.r.ping()


@pytest.mark.skipif(os.getenv("BIFROST_INTEGRATION") != "1", reason="requires live services")
def test_live_dynamodb_table():
    from infra.dynamodb.client import dynamodb_client, table_name

    resp = dynamodb_client().describe_table(TableName=table_name())
    assert resp["Table"]["TableStatus"] == "ACTIVE"
    assert resp["Table"]["BillingModeSummary"]["BillingMode"] == "PAY_PER_REQUEST"


@pytest.mark.skipif(os.getenv("BIFROST_INTEGRATION") != "1", reason="requires live services")
def test_live_mysql_seeded():
    from infra.db.mysql_client import table_counts

    counts = table_counts()
    assert all(v > 0 for v in counts.values())


@pytest.mark.skipif(os.getenv("BIFROST_INTEGRATION") != "1", reason="requires live services")
def test_live_warehouse_query_sla():
    from infra.dynamodb.seed_traces import scan_all
    from infra.warehouse.duckdb_warehouse import DuckDBWarehouse

    wh = DuckDBWarehouse()
    wh.init_schema()
    wh.con.execute("DELETE FROM fact_query_cost")
    wh.load_traces(scan_all())
    elapsed, saved = wh.total_cost_saved_for()
    wh.close()
    assert elapsed < 2.0
    assert saved["cost_saved_usd"] > 0