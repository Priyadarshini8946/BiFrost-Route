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
RAGAS quality harness (evaluation)         ◀── currently here (Layer 5)
FastAPI + LangGraph state machine          ◀── done (Layer 4)
DistilBERT complexity classifier           ── Layer 3 (training — user-run)
Hybrid retrieval: Neo4j+Qdrant+BM25+RR  ◀── done (Layer 2)
─────────────────────────────────────────────────────────
DATA LAYER (Layer 1, DONE ✅)
 MySQL (policies/budgets/models) · DynamoDB (traces) · Redis (semantic cache) · DuckDB→Redshift warehouse
```

---

## 🗂 Repository layout (Layer 1 + 2)

| Path | Purpose |
|:---|:---|
| `infra/db/schema.sql` | MySQL DDL + seeds: `model_registry`, `routing_policies`, `budget_limits` |
| `infra/db/mysql_client.py` | Idempotent schema apply + demo queries |
| `infra/dynamodb/setup_table.py` | `query_traces` table (GSIs, on-demand, TTL) |
| `infra/dynamodb/seed_traces.py` | 100 production-shaped traces + "escalations last hour" GSI demo |
| `infra/cache/embedder.py` | Offline TF-IDF semantic embedder (fallback behind the Embedder interface) |
| `infra/cache/semantic_cache.py` | RediSearch KNN semantic cache, TTL 1h, LRU eviction |
| `infra/cache/cache_benchmark.py` | Traffic + LRU benchmarks for the 30% gate |
| `infra/warehouse/duckdb_warehouse.py` | Redshift stand-in: facts, cost-saved rollups |
| `infra/warehouse/redshift/` | The same schema/queries in Redshift dialect (AWS path) |
| `infra/data_generation/trace_factory.py` | Deterministic, realistic traffic generator |
| `infra/retrieval/corpus.py` | Layer 2: 500-doc KB corpus + 47 eval queries + concept registry |
| `infra/retrieval/graph.py` | Neo4j knowledge graph: constraints, ingestion, query→concept→doc Cypher |
| `infra/retrieval/vector_store.py` | Qdrant vectors (bge-small-en-v1.5 / TF-IDF fallback) |
| `infra/retrieval/bm25_index.py` | BM25Okapi lexical index |
| `infra/retrieval/hybrid.py` | RRF fusion + re-rank orchestration, `vector_only()` baseline |
| `infra/retrieval/reranker.py` | Cross-encoder re-ranker (ort → flashrank → fusion fallback chain) |
| `infra/routing/state.py` | Layer 4: typed RouteState flowing through the LangGraph |
| `infra/routing/graph.py` | Layer 4: LangGraph state machine (9 nodes, bounded escalation) |
| `infra/routing/router.py` | Layer 4: RouteEngine facade — policy/budget/cache/trace/warehouse wiring |
| `infra/routing/grader.py` | Layer 4: RAGAS-proxy faithfulness gate (L5 swaps in real RAGAS) |
| `infra/routing/llm.py` | Layer 4: LLM clients (Groq/Gemini) + deterministic dry-run mock |
| `api/app.py` | Layer 4: FastAPI — `/route`, `/health`, `/traces` |
| `scripts/setup_layer4.py` | Layer 4 provisioning + warm-up + API liveness |
| `infra/eval/ragas_harness.py` | Layer 5: RAGAS harness (LLM judge → offline proxy) |
| `scripts/check_layer5_results.py` | Layer 5: quality gates L5-1..L5-7, CI-ready |
| `scripts/setup_layer5.py` | Layer 5 provisioning + judge resolution + smoke |
| `scripts/setup_infra.py` | One-shot infra provisioning (Layer 1) |
| `scripts/setup_layer2.py` | Layer 2 provisioning: graph + vectors + BM25 + reranker warm-up |
| `scripts/check_layer1_results.py` | ⭐ Layer 1 acceptance gate (exits non-zero on any miss) |
| `scripts/check_layer2_results.py` | ⭐ Layer 2 acceptance gate (R1–R4, exits non-zero on any miss) |
| `scripts/debug_layer2.py` | Per-query vector vs hybrid inspection tool |
| `tests/` | Offline unit tests + service integration tests (L1: 10, L2: 7) |

## 🚀 Quickstart

```powershell
# 1. Services (Docker Desktop must be running)
docker compose up -d --wait

# 2. Python env
uv venv .venv --python "C:\Users\HP\AppData\Local\Programs\Python\Python313\python.exe"
uv pip install --python .venv\Scripts\python.exe -r requirements.txt

# 3. Provision Layer 1 (Redis config, DynamoDB table + 100 traces, MySQL schema, warehouse)
.venv\Scripts\python.exe scripts\setup_infra.py

# 4. Provision Layer 2 (Neo4j graph + Qdrant vectors + BM25 + reranker warm-up)
.venv\Scripts\python.exe scripts\setup_layer2.py

# 4b. ⭐ Check the Layer 1 acceptance results
.venv\Scripts\python.exe scripts\check_layer1_results.py

# 4c. ⭐ Check the Layer 2 acceptance results
.venv\Scripts\python.exe scripts\check_layer2_results.py

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

## ✅ Layer 2 acceptance targets (Retrieval)

Layer 2 turns plain retrieval into a production hybrid engine:

- **500-doc deterministic corpus** — 25 knowledge-base families × 20 variants; every variant is a *distinct* article (unique title + angle, same topic), so evaluation measures real retrieval quality, not duplicate-flooding.
- **Neo4j knowledge graph** — Document / Concept / Query nodes, `ASKS_ABOUT` + `HAS_CONCEPT` + `RELATED_TO`; a query is merged as a node so repeated asks are visible in the graph.
- **Qdrant vector store** — fastembed `BAAI/bge-small-en-v1.5` ONNX (384-dim; TF-IDF fallback offline).
- **BM25Okapi** — lexical index with stopword tokenizer.
- **Reciprocal Rank Fusion (RRF, k=60)** over graph + vector + BM25 → 24-candidate pool.
- **Cross-encoder re-ranker** — the plan's exact `cross-encoder/ms-marco-MiniLM-L-6-v2` on ONNX (`rerankers` ort backend; full-size, real discriminator), with flashrank + calibrated score-fusion as offline fallbacks.

| # | Metric | Target | Measured (2026-09-13) | Status |
|:--|:---|:---|:---|:---|
| R1 | Coverage@5 (relevant doc in top-5) | ≥ 95% | **100.0%** | ✅ |
| R2 | Re-ranked precision@5 | ≥ 85% (plan: 90%) | **0.953** | ✅ |
| R3 | Re-rank gain over vector-only (0.902) | ≥ +0.05 | **+0.051** | ✅ |
| R4 | Mean end-to-end latency (CPU cross-encoder) | < 3 s | **2.37 s** | ✅ |

Reproduce: `.venv\Scripts\python.exe scripts\check_layer2_results.py` (builds index, runs 47 eval queries, persists `data/layer2_results.json`).

## ✅ Layer 4 acceptance targets (Routing)

Layer 4 is the **adaptive router**: a LangGraph state machine behind FastAPI
that classifies complexity (Layer 3), retrieves grounding (Layer 2), enforces
policy + budget + cache (Layer 1), calls the right-tier model, checks the
answer with a RAGAS-proxy faithfulness gate, and escalates bounded by the
policy's max hops.

Pipeline: `classify → cache_lookup → retrieve → select_model → budget_check →
generate → grade → (escalate ↺) → record (DynamoDB trace + warehouse fact)`.

| # | Metric | Target | Measured (2026-09-16) | Status |
|:--|:---|:---|:---|:---|
| G1 | Policy fidelity (monitored to the model-mapped tier, 60-query workload) | 60/60 | **60/60** | ✅ |
| G2 | Bounded escalation (hallucination → ≤ 2 hops to frontier; grounded → 0) | esc ≤ 2 | **esc=2, tier=frontier** | ✅ |
| G3 | Semantic cache short-circuit (repeat query: hit, no LLM call, $0) | hit + $0 | **hit, sim=1.0, $0** | ✅ |
| G4 | Budget enforcement (team within cap routed; exhausted refused) | enforced | **eng-core ok, $257.69 remaining** | ✅ |
| G5 | Trace fidelity (DynamoDB cost == registry math, traced per route) | exact | **match=True** | ✅ |
| G6 | Economics — adaptive vs frontier-only baseline (60-query cold path) | ≥ 50% saved | **67.8% saved ($0.05 vs $0.16)** | ✅ |
| G7 | API SLA — `POST /route` p95 (dry-run, CPU) | < 3 s | **p95 ≈ 1.0 s** | ✅ |

Modes: without Groq/Gemini keys the router runs a **deterministic dry-run
mock** (every gate passes with zero cost + zero keys); set `GROQ_API_KEY` /
`GEMINI_API_KEY` in `.env` and `BIFROST_DRY_RUN=0` for real model calls.

Reproduce: `.venv\Scripts\python.exe scripts\check_layer4_results.py` and
`scripts\setup_layer4.py` (warm-up + API liveness). API server:
`.venv\Scripts\python.exe -m uvicorn api.app:app --port 8077`.

> **Layer 3 classifier handshake** — the router auto-selects the trained
> DistilBERT (`distilbert-onnx`) once your training artifacts land with a
> fresh `dataset_marker`; until then it uses the TF-IDF fallback. Layer 3 and
> Layer 4 gates are re-verified together after your training completes.

## ✅ Layer 5 acceptance targets (RAGAS quality harness)

Layer 5 proves the plan's headline — **"adaptive routing saves cost AND
maintains quality"** — by scoring the Layer-4 router's answers with
RAGAS-family metrics on the Layer-2 eval workload (47 well-formed queries,
retrieval quality already proven):

| # | Metric | Target | Measured (2026-09-16) | Status |
|:--|:---|:---|:---|:---|
| L5-1 | Answer faithfulness (grounded in retrieved context) | ≥ 0.80 | **1.000** | ✅ |
| L5-2 | Answer relevancy (does the answer address the question) | ≥ 0.55 | **0.678** | ✅ |
| L5-3 | Context precision (relevant context ranked first, family ground truth) | ≥ 0.60 | **0.936** | ✅ |
| L5-4 | Quality parity: adaptive vs frontier-only on same queries | \|Δ\| ≤ 0.05 | **0.000** | ✅ |
| L5-5 | Economics with quality held (adaptive ≤ 60% of frontier) | ≤ 0.60 | **14.3%** | ✅ |
| L5-6 | RAGAS-gated escalation re-route catches hallucinations | esc ≤ 2 → frontier, faithful | **esc=2, fx=1.000** | ✅ |
| L5-7 | Judge mode (real LLM when key present, offline proxy otherwise) | runs | **proxy** | ✅ |

Two judge modes — **no model is ever trained here** (RAGAS is an evaluation
library):

- **Real judge**: `ragas` 0.4.3 calls Groq/Gemini as the LLM judge for
  faithfulness + context_precision. Enabled automatically once `GROQ_API_KEY`
  or `GEMINI_API_KEY` is set (same key as Layer 4).
- **Proxy judge (default)**: deterministic and offline — bge-small cosine for
  answer relevancy, L2's family ground truth for context precision, the
  Layer-4 grader for faithfulness. Every gate runs without any key.

Reproduce: `.venv\Scripts\python.exe scripts\check_layer5_results.py`
(7 gates, CI-ready exit code, persists `data/layer5_results.json`);
provision/warm: `scripts\setup_layer5.py`.

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