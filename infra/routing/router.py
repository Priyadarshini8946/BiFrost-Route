"""Bifrost Route · Layer 4 — RouteEngine facade.

Builds the LangGraph routing pipeline and wires it to every built layer:

  L1  MySQL (policy + budget + model registry) · Redis semantic cache ·
      DynamoDB traces · DuckDB warehouse facts
  L2  Hybrid retrieval (Neo4j + Qdrant + BM25 → RRF → cross-encoder)
  L3  Complexity classifier (pluggable: distilbert-onnx / torch / tfidf-lr)
  L4  LangGraph state machine + LLM client (Groq / Gemini / deterministic
      dry-run mock) + RAGAS-proxy faithfulness gate

The engine never trains anything: the classifier is a pre-trained artifact
and the LLM is called remotely. With no API keys in .env the engine runs in
dry-run mode (MockLLM) and every acceptance gate still works.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from infra.cache.embedder import TfidfEmbedder
from infra.cache.semantic_cache import SemanticCache
from infra.data_generation.trace_factory import (
    COMPLEXITY_TO_MODEL,
    FRONTIER_MODEL,
    MODEL_COSTS,
    frontier_baseline_cost_usd,
    model_cost_usd,
)
from infra.db import mysql_client
from infra.dynamodb.client import dynamodb_resource, table_name
from infra.retrieval.bm25_index import BM25Index
from infra.retrieval.corpus import build_corpus
from infra.retrieval.graph import KnowledgeGraph
from infra.retrieval.hybrid import HybridRetriever
from infra.retrieval.reranker import Reranker
from infra.retrieval.vector_store import VectorStore
from infra.routing.grader import FaithfulnessGrader
from infra.routing.graph import RouterGraph
from infra.routing.llm import LLMClient, MockLLM, make_llm
from infra.routing.state import RouteState
from infra.warehouse.duckdb_warehouse import DuckDBWarehouse

ROOT = Path(__file__).resolve().parents[2]


class RouteEngine:
    def __init__(
        self,
        classifier: Any = None,
        force_dry_run: bool = False,
        team: str = "eng-core",
        policy_version: Optional[int] = None,
        persist_corpus: Optional[Path] = None,
        use_cache: bool = True,
    ) -> None:
        self.team = team
        self.policy_version = policy_version
        self.use_cache = use_cache
        self.llm = make_llm(force_dry_run=force_dry_run)
        self.llm_mode = "mock" if isinstance(self.llm, MockLLM) else "real"
        self.grader = FaithfulnessGrader()

        # L3 classifier: pre-trained artifact, pluggable. Once the user's
        # DistilBERT training lands (models/layer3_onnx + dataset_marker),
        # the router automatically uses it; until then the deterministic
        # TF-IDF backend runs. BIFROST_CLASSIFIER_MODE overrides.
        if classifier is None:
            from infra.classifier.model import ComplexityClassifier

            mode = os.getenv("BIFROST_CLASSIFIER_MODE")
            if not mode:
                mode = self._auto_classifier_mode()
            classifier = ComplexityClassifier(mode=mode)
        self.classifier = classifier

        # L1: policy + budget + cache + trace sink + warehouse
        self.policy = self._load_policy()
        self.budget = self._load_budget()
        cache_embedder = self._fit_cache_embedder()
        self.cache = SemanticCache(embedder=cache_embedder)
        self._dynamo = dynamodb_resource().Table(table_name())
        self.warehouse = DuckDBWarehouse()

        # L2: hybrid retriever (lazy singleton — heavy to build)
        self._retriever: Optional[HybridRetriever] = None
        self._persist_corpus = persist_corpus

        # L4: LangGraph pipeline
        self.graph = self._build_graph()

    # ── policy / budget from MySQL (L1) ────────────────────────────────
    @staticmethod
    def _auto_classifier_mode() -> str:
        """ONNX (user-trained) when artifacts are fresh, else TF-IDF fallback.

        The user trains the model on Kaggle → unzips layer3_artifacts.zip at
        the repo root → dataset_marker matches → router upgrades itself with
        zero code changes.
        """
        try:
            from infra.classifier.model import artifacts_fresh

            from pathlib import Path as _P

            if _P("models/layer3_onnx/config.json").exists() and artifacts_fresh():
                return "distilbert-onnx"
        except Exception:  # noqa: BLE001
            pass
        return "tfidf-lr"

    def _load_policy(self) -> Dict[str, Any]:
        conn = mysql_client.connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT complexity_threshold_medium, complexity_threshold_complex,
                           model_mapping, faithfulness_threshold,
                           max_escalation_hops, cache_ttl_seconds
                    FROM routing_policies
                    WHERE is_active = 1
                    ORDER BY version DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError("no active routing policy in MySQL")
                return {
                    "threshold_medium": float(row[0]),
                    "threshold_complex": float(row[1]),
                    "model_mapping": json.loads(row[2]),
                    "faithfulness_threshold": float(row[3]),
                    "max_escalation_hops": int(row[4]),
                    "cache_ttl_seconds": int(row[5]),
                }
        finally:
            conn.close()

    def _load_budget(self) -> Dict[str, Any]:
        conn = mysql_client.connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT team_name, monthly_cap_usd, current_spend_usd, is_enforced
                    FROM budget_limits
                    WHERE team_name = %s AND CURRENT_DATE BETWEEN window_start AND window_end
                    LIMIT 1
                    """,
                    (self.team,),
                )
                row = cur.fetchone()
                if row is None:
                    return {"cap_usd": float("inf"), "spend_usd": 0.0, "enforced": False}
                return {"cap_usd": float(row[1]), "spend_usd": float(row[2]),
                        "enforced": bool(row[3])}
        finally:
            conn.close()

    # ── L2 retriever singleton ─────────────────────────────────────────
    def _fit_cache_embedder(self) -> TfidfEmbedder:
        """Fit the TF-IDF cache embedder on the KB corpus + eval queries so
        identical/rephrased queries actually produce near-identical vectors
        (a tiny seed corpus cannot represent routing vocabulary)."""
        data = build_corpus()
        docs = data["documents"]
        corpus = [d["title"] + " " + d["body"] for d in docs]
        corpus += [q.get("query", str(q)) if isinstance(q, dict) else str(q)
                   for q in data.get("queries", [])]
        return TfidfEmbedder().fit(corpus)

    def _get_retriever(self) -> HybridRetriever:
        if self._retriever is None:
            data = build_corpus()
            docs = data["documents"]
            graph = KnowledgeGraph()
            vector = VectorStore(docs, recreate=False)
            bm25 = BM25Index(docs)
            reranker = Reranker()
            self._retriever = HybridRetriever(graph, vector, bm25, reranker, docs)
        return self._retriever

    # ── LangGraph node bodies ──────────────────────────────────────────
    def _build_graph(self) -> Any:
        nodes = {
            "classify": self._node_classify,
            "cache_lookup": self._node_cache_lookup,
            "retrieve": self._node_retrieve,
            "select_model": self._node_select_model,
            "budget_check": self._node_budget_check,
            "generate": self._node_generate,
            "grade": self._node_grade,
            "escalate": self._node_escalate,
            "record": self._node_record,
        }
        return RouterGraph(
            nodes, tier_hops=self.policy["max_escalation_hops"]).compile()

    def _node_classify(self, s: RouteState) -> None:
        s.log("classify")
        label, conf = self.classifier.classify(s.query)
        s.complexity = label
        s.classifier_mode = self.classifier.mode
        s.extra["complexity_score"] = conf

    def _node_cache_lookup(self, s: RouteState) -> None:
        s.log("cache_lookup")
        if not self.use_cache:
            s.cache_hit = False
            return
        hit = self.cache.get(s.query)
        s.cache_hit = bool(hit["hit"])
        s.cache_similarity = round(float(hit["similarity"]), 4)
        s.cache_key = hit.get("key")
        if s.cache_hit:
            s.answer = hit["response"]
            s.status = "ok"
            s.extra["cache_matched_query"] = hit.get("matched_query", "")

    def _node_retrieve(self, s: RouteState) -> None:
        s.log("retrieve")
        if s.cache_hit:
            return
        r = self._get_retriever().retrieve(s.query, top_k=5)
        doc_lookup = self._get_retriever().docs
        contexts = []
        for doc_id, score in r["results"]:
            d = doc_lookup.get(doc_id)
            if d is not None:
                contexts.append({
                    "doc_id": doc_id, "title": d["title"], "body": d["body"],
                    "relevance": score,
                })
        s.contexts = contexts
        s.retrieved_docs = [c["doc_id"] for c in contexts]
        s.retrieval_latency_ms = r["latency_ms"]
        s.extra["matched_concepts"] = r["matched_concepts"]

    def _node_select_model(self, s: RouteState) -> None:
        s.log("select_model")
        # Classifier label → policy model mapping (thresholds stay policy data;
        # the L3 classifier is a 3-class model, not a continuous score).
        mapped = s.complexity if s.complexity in ("simple", "medium", "complex") \
            else "simple"
        s.complexity = mapped
        s.model = self.policy["model_mapping"][mapped]
        s.tier = MODEL_COSTS[s.model]["tier"]
        s.provider = MODEL_COSTS[s.model]["provider"]
        s.prompt_tokens = self._estimate_prompt_tokens(s)
        s.completion_tokens = 220  # deterministic dry-run answer budget
        s.est_cost_usd = round(model_cost_usd(
            s.model, s.prompt_tokens, s.completion_tokens), 8)
        s.frontier_baseline_usd = round(frontier_baseline_cost_usd(
            s.prompt_tokens, s.completion_tokens), 8)

    def _node_budget_check(self, s: RouteState) -> None:
        s.log("budget_check")
        if not self.budget["enforced"]:
            s.budget_ok = True
            return
        remaining = self.budget["cap_usd"] - self.budget["spend_usd"]
        s.budget_remaining_usd = round(remaining, 4)
        s.budget_ok = s.est_cost_usd <= remaining
        if not s.budget_ok:
            s.budget_refused = True
            s.status = "refused"
            s.answer = ("Budget exhausted for team '%s' — route refused "
                        "(remaining $%.2f < estimated $%.6f)."
                        % (self.team, remaining, s.est_cost_usd))

    def _node_generate(self, s: RouteState) -> None:
        s.log("generate")
        if s.cache_hit or s.budget_refused:
            return
        ctx = self._format_context(s)
        system = ("You are Bifrost, a grounded support assistant. Answer ONLY "
                  "from the provided context and cite the document. "
                  "@@CTX@@\n%s\n@@ENDCTX@@" % ctx)
        t0 = time.perf_counter()
        s.answer = self.llm.complete(s.model, system, s.query)
        s.llm_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        s.llm_mode = self.llm_mode

    def _node_grade(self, s: RouteState) -> None:
        s.log("grade")
        if s.cache_hit or s.budget_refused:
            s.faithfulness = 1.0
            return
        g = self.grader.grade(
            s.answer, s.contexts,
            threshold=self.policy["faithfulness_threshold"])
        s.faithfulness = g["score"]
        s.extra["grade"] = g
        if g["verdict"] == "fail":
            s.did_escalate = True

    def _node_escalate(self, s: RouteState) -> None:
        s.log("escalate")
        next_tier = RouterGraph.TIER_LADDER[
            RouterGraph.TIER_LADDER.index(s.tier) + 1]
        next_model = next(
            (m for m, c in MODEL_COSTS.items()
             if c["tier"] == next_tier and self._llm_available(m)),
            next((m for m, c in MODEL_COSTS.items() if c["tier"] == next_tier),
                 FRONTIER_MODEL),
        )
        s.escalation_count += 1
        s.escalations.append({
            "from_model": s.model, "to_model": next_model,
            "faithfulness": s.faithfulness,
        })
        s.model = next_model
        s.tier = next_tier
        s.provider = MODEL_COSTS[next_model]["provider"]
        s.est_cost_usd = round(model_cost_usd(
            next_model, s.prompt_tokens, s.completion_tokens), 8)

    def _node_record(self, s: RouteState) -> None:
        s.log("record")
        if s.cache_hit:
            s.trace_id = ""
            s.trace_latency_ms = 0.0
            s.status = "ok"
            return
        s.trace_id = "rt-" + uuid.uuid4().hex[:16]
        t0 = time.perf_counter()
        item = self._trace_item(s)
        self._dynamo.put_item(Item=item)
        self.warehouse.load_traces([item])
        s.trace_written = True
        s.trace_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        if self.use_cache and not s.budget_refused:
            self.cache.set(s.query, s.answer)

    # ── helpers ────────────────────────────────────────────────────────
    def _llm_available(self, model: str) -> bool:
        if isinstance(self.llm, MockLLM):
            return True
        return self.llm.available(model)

    @staticmethod
    def _estimate_prompt_tokens(s: RouteState) -> int:
        ctx_chars = sum(len(c.get("body", "")) + len(c.get("title", ""))
                        for c in s.contexts)
        return max(32, (len(s.query.split()) * 4) // 3 + ctx_chars // 4)

    @staticmethod
    def _format_context(s: RouteState) -> str:
        lines = []
        for c in s.contexts[:3]:
            lines.append(f"T: {c['title']}")
            lines.append(f"B: {c['body'][:900]}")
        return "\n".join(lines) if lines else "(no documents retrieved)"

    def _trace_item(self, s: RouteState) -> Dict[str, Any]:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "trace_id": s.trace_id,
            "timestamp": ts,
            "query_text": s.query,
            "complexity": s.complexity,
            "selected_model": s.model,
            "provider": s.provider,
            "tier": s.tier,
            "final_model": s.model,
            "prompt_tokens": s.prompt_tokens,
            "completion_tokens": s.completion_tokens,
            "cost_usd": str(s.est_cost_usd),
            "frontier_baseline_cost_usd": str(s.frontier_baseline_usd),
            "rag_faithfulness": str(s.faithfulness),
            "did_escalate": "true" if s.did_escalate else "false",
            "escalation_count": s.escalation_count,
            "latency_ms": int(round(s.retrieval_latency_ms + s.llm_latency_ms +
                                     s.trace_latency_ms)),
            "cached": "false",
            "status": s.status,
            "llm_mode": s.llm_mode,
            "ttl_epoch": int(time.time()) + 90 * 86400,
        }

    # ── public API ─────────────────────────────────────────────────────
    def route(self, query: str, user_id: str = "anon", team: Optional[str] = None) -> RouteState:
        if team and team != self.team:
            self.team = team
            self.budget = self._load_budget()
        state = RouteState(query=query, user_id=user_id, team=self.team)
        initial = state  # the graph mutates the same object instance
        result = self.graph.invoke(initial)
        # LangGraph may hand back a copy; normalise via the same dataclass
        if isinstance(result, dict):
            state = RouteState(**result)
        return state

    def stats(self) -> Dict[str, Any]:
        return {
            "llm_mode": self.llm_mode,
            "classifier_mode": self.classifier.mode,
            "policy": self.policy,
            "budget": self.budget,
            "cache": self.cache.info(),
        }