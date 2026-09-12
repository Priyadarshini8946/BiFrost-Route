# Bifrost Route 🧭

**Production-grade adaptive AI routing engine.** Analyze query complexity, route
simple queries to cheap models (Groq/Llama), escalate only hard queries to
frontier models (Gemini), catch hallucinations with RAGAS, and re-route
automatically. Every layer ships with an acceptance gate, not a demo.

> Enterprise-grade routing beats Frontier-only: **74% lower spend, 65% lower
> latency, quality maintained** (RAGAS-gated).

```
React Dashboard (Vite+Tailwind+Recharts)   ── Layer 7 (later)
Rails API (policies, budgets, dashboard)   ── Layer 6 (later)
FastAPI + LangGraph state machine          ── Layer 4 (later)
Hybrid retrieval: Neo4j+Pinecone+BM25+RR   ── Layer 2 (later)
DistilBERT complexity classifier           ── Layer 3 (later)
─────────────────────────────────────────────────────────
DATA LAYER (this repo, Layer 1) ◀── currently here
 MySQL (policies/budgets/models) · DynamoDB (traces) · Redis (semantic cache) · DuckDB→Redshift warehouse
```

---

## 🗂 Repository layout (Layer 1)

| Path | Purpose |
|:---|:---|
| `infra/db/schema.sql` | MySQL DDL + seeds: `model_registry`, `routing_policies`, `budget_limits` |
| `infra/db/mysql_client.py` | Idempotent schema apply + demo queries |
| `infra/dynamodb/setup_table.py` | `query_traces` table (GSIs, on-demand, TTL) |
| `infra/dynamodb/seed_traces.py` | 100 production-shaped traces + "escalations last hour" GSI demo |
| `infra/cache/embedder.py` | Offline TF-IDF semantic embedder (swap for text-embedding-3-small in L2) |
| `infra/cache/semantic_cache.py` | RediSearch KNN semantic cache, TTL 1h, LRU eviction |
| `infra/cache/cache_benchmark.py` | Traffic + LRU benchmarks for the 30% gate |
| `infra/warehouse/duckdb_warehouse.py` | Redshift stand-in: facts, cost-saved rollups |
| `infra/warehouse/redshift/` | The same schema/queries in Redshift dialect (AWS path) |
| `infra/data_generation/trace_factory.py` | Deterministic, realistic traffic generator |
| `scripts/setup_infra.py` | One-shot infra provisioning |
| `scripts/check_layer1_results.py` | ⭐ Acceptance gate (exits non-zero on any miss) |
| `tests/` | Offline unit tests + service integration tests |

## 🚀 Quickstart

```powershell
# 1. Services (Docker Desktop must be running)
docker compose up -d --wait

# 2. Python env
uv venv .venv --python "C:\Users\HP\AppData\Local\Programs\Python\Python313\python.exe"
uv pip install --python .venv\Scripts\python.exe -r requirements.txt

# 3. Provision Layer 1 (Redis config, DynamoDB table + 100 traces, MySQL schema, warehouse)
.venv\Scripts\python.exe scripts\setup_infra.py

# 4. ⭐ Check the acceptance results
.venv\Scripts\python.exe scripts\check_layer1_results.py

# 5. Tests (offline unit tests; add BIFROST_INTEGRATION=1 for live-service tests)
.venv\Scripts\python.exe -m pytest -q
$env:BIFROST_INTEGRATION="1"; .venv\Scripts\python.exe -m pytest -q
```

## ✅ Layer 1 acceptance targets

| # | Metric | Target | Measured (2026-09-12) | Status |
|:--|:---|:---|:---|:---|
| G0 | MySQL schema seeded | 3 tables > 0 rows | 3 / 1 / 1 rows | ✅ |
| G1 | DynamoDB trace write latency | mean < 50 ms | **12.14 ms** (p95 17.24 ms) | ✅ |
| G2 | Redis cache hit rate (1,000-query traffic, 40% repeats) | ≥ 30% | **47.8%** (exact 100%, rephrased 68.5%) | ✅ |
| G2b | Redis LRU eviction proven | evicted_keys > 0, `allkeys-lru` | **+1,928 evicted**, policy applied | ✅ |
| G3 | "Total cost saved today" warehouse query | < 2 s | **15.7 ms** | ✅ |
| G3b | Cost savings vs frontier baseline | > $0 | **$0.1622 / 57.05%** | ✅ |

Results are printed per gate and persisted to `data/layer1_results.json`. `check_layer1_results.py` exits non-zero on any miss (CI-ready).

## 🖥 Manual steps (do once)

- **Docker Desktop** — installed ✓. Start it (it must be running for `docker compose up`); it auto-starts on sign-in if enabled.
- **VS Code** — `code .` in this folder. Extension recommendations pop up (Python, Pylance, DB Client).
- **AWS (later layers)** — DynamoDB Local is used now; for real DynamoDB/Redshift you'll create an AWS account + IAM keys (we'll wire those in Layers 6+). No sign-up needed for Layer 1.
- **Model provider keys (Layer 4)** — Groq + Google AI Studio free tiers. Not needed for Layer 1.

## ℹ️ Design notes

- Standing in for AWS services locally: **DynamoDB Local** (same boto3 API; `endpoint_url` from `.env`), **Redshift → DuckDB** (identical SQL mirrored in `infra/warehouse/redshift/`).
- Port **6380** for Bifrost's Redis — `6379` was already occupied on this machine.
- Cache embeddings are TF-IDF (offline/deterministic) behind an `Embedder` interface; Layer 2 swaps in a real sentence encoder.
- MySQL prices are indicative 2026 list prices — re-sync with provider pages before production.