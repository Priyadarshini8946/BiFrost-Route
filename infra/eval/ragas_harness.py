"""Bifrost Route · Layer 5 — RAGAS evaluation harness.

Layer 5 is the quality measurement layer. It answers the plan's headline
claim — "adaptive routing saves 65-74% cost AND maintains quality" — by
scoring the router's answers with RAGAS-family metrics:

  faithfulness        answers grounded in the retrieved context
  answer_relevancy    answers actually address the question
  context_precision   retrieved context is relevant, top-heavy

Two judge modes (RAGAS builds no model — nothing is trained here):

  • REAL judge — LLM-as-judge via Groq/Gemini API (ragas library). Needs
    the user's GROQ_API_KEY / GEMINI_API_KEY (manual step, ~2 minutes).
  • PROXY judge (default, offline, deterministic) — the Layer-2 cross-encoder
    (already loaded for retrieval) scores (query, answer) and (query, context)
    pairs: semantic relevance, same family of signal RAGAS uses.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from infra.routing.grader import FaithfulnessGrader  # noqa: E402
from infra.routing.router import RouteEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class EvalSample:
    question: str
    context: str = ""
    answer: str = ""
    context_scores: List[float] = field(default_factory=list)
    context_families: List[str] = field(default_factory=list)
    ground_truth_family: str = ""

    def as_dict(self) -> Dict[str, str]:
        return {"question": self.question, "contexts": [self.context],
                "answer": self.answer}


@dataclass
class RagasScore:
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    judge_mode: str = "proxy"


class JudgeConfig:
    """Resolve which LLM judge to use; None → proxy mode."""

    def __init__(self) -> None:
        self.groq_key = os.getenv("GROQ_API_KEY", "").strip()
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.groq_model = os.getenv("BIFROST_JUDGE_GROQ_MODEL",
                                    "llama-3.3-70b-versatile")
        self.gemini_model = os.getenv("BIFROST_JUDGE_GEMINI_MODEL",
                                      "gemini-2.5-pro")

    @property
    def mode(self) -> str:
        if self.groq_key:
            return "groq"
        if self.gemini_key:
            return "gemini"
        return "proxy"

    def available(self) -> bool:
        return self.mode in ("groq", "gemini")


class ProxyJudge:
    """Deterministic offline stand-in for the LLM judge.

    Reuses only artifacts already loaded by the running stack — the L2
    dense embedder (bge-small), the L2 corpus ground truth (query/doc
    families), and the L4 faithfulness grader. No downloads, no training,
    fully reproducible:

      faithfulness        claim-coverage of answer by context (L4 semantics)
      answer_relevancy    bge cosine(question, answer) — bounded 0..1
      context_precision   ground-truth share of top-3 contexts whose family
                          matches the query's family (RAGAS definition:
                          relevant context ranked first)
    """

    def __init__(self, engine: Optional[RouteEngine] = None,
                 family_map: Optional[Dict[str, str]] = None) -> None:
        self._grader = FaithfulnessGrader()
        self._engine = engine
        self._embedder = None
        self.family_map = family_map or {}

    def _get_embedder(self):
        if self._embedder is None and self._engine is not None:
            self._embedder = self._engine._get_retriever().vector.embedder
        return self._embedder

    def _cosine(self, a: str, b: str) -> float:
        emb = self._get_embedder()
        if emb is None or not a.strip() or not b.strip():
            return 0.0
        try:
            import numpy as np

            va = np.asarray(emb.embed(a), dtype=np.float32)
            vb = np.asarray(emb.embed(b), dtype=np.float32)
            na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
            if na == 0.0 or nb == 0.0:
                return 0.0
            return round(float(np.dot(va, vb) / (na * nb)), 4)
        except Exception:  # noqa: BLE001 — proxy must never crash a gate
            return 0.0

    def score(self, sample: EvalSample) -> RagasScore:
        fa = self._grader.grade(sample.answer,
                                [{"body": sample.context}])
        relevancy = self._cosine(sample.question, sample.answer)
        # context_precision: RAGAS definition — are relevant contexts ranked
        # top? Ground truth = matched family of query vs retrieved contexts.
        fam = sample.ground_truth_family
        if fam and sample.context_families:
            hits = [1 for f in sample.context_families
                    if f and f == fam]
            context_precision = round(len(hits) / len(sample.context_families),
                                      4)
        else:
            context_precision = 0.0
        return RagasScore(
            faithfulness=fa["score"],
            answer_relevancy=relevancy,
            context_precision=context_precision,
            judge_mode="proxy",
        )


class RagasHarness:
    """Routes a workload through the engine and scores answer quality."""

    def __init__(self, engine: Optional[RouteEngine] = None,
                 judge_mode: str = "auto") -> None:
        self.engine = engine or RouteEngine(force_dry_run=True)
        self.judge_cfg = JudgeConfig()
        self.preferred = judge_mode
        self.use_real = (judge_mode == "real" or
                         (judge_mode == "auto" and self.judge_cfg.available()))
        self.family_map: Dict[str, str] = {}
        try:
            from infra.retrieval.corpus import build_corpus

            data = build_corpus()
            self.family_map = {
                q["query"]: q["family"] for q in data["queries"]
            }
        except Exception:  # noqa: BLE001 — ground truth optional
            self.family_map = {}
        self._proxy = ProxyJudge(engine=self.engine,
                                 family_map=self.family_map)

    # ── scoring ───────────────────────────────────────────────────────
    def score(self, sample: EvalSample) -> RagasScore:
        pr = self._proxy.score(sample)
        if not self.use_real:
            return pr
        real = self._real_ragas([sample])
        if real:
            return real[0]
        return pr

    def _real_ragas(self, samples: List[EvalSample]) -> List[RagasScore]:
        """ragas library with the configured LLM judge (needs user's key).
        Returns [] on any failure so the gate degrades to the proxy."""
        try:
            from ragas import evaluate
            from ragas.metrics import context_precision, faithfulness
            from ragas.llms import LangchainLLMWrapper
            from datasets import Dataset

            rows = [s.as_dict() for s in samples]
            ds = Dataset.from_list(rows)

            llm = self._build_langchain_llm()
            if llm is None:
                return []
            wrapped = LangchainLLMWrapper(llm)

            result = evaluate(ds, metrics=[faithfulness, context_precision],
                              llm=wrapped, show_progress=False)
            out = []
            for i, row in enumerate(result):
                out.append(RagasScore(
                    faithfulness=float(row["faithfulness"]),
                    answer_relevancy=self._proxy.score(samples[i]).answer_relevancy,
                    context_precision=float(row["context_precision"]),
                    judge_mode=self.judge_cfg.mode,
                ))
            return out
        except Exception as e:  # noqa: BLE001 — offline fallback is the design
            import logging

            logging.getLogger("bifrost.l5").warning(
                "real RAGAS judge failed (%s) — falling back to proxy", e)
            return []

    def _build_langchain_llm(self):
        if self.judge_cfg.mode == "groq":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=self.judge_cfg.groq_model,
                api_key=self.judge_cfg.groq_key,
                base_url="https://api.groq.com/openai/v1",
                temperature=0,
            )
        if self.judge_cfg.mode == "gemini":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=self.judge_cfg.gemini_model,
                api_key=self.judge_cfg.gemini_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                temperature=0,
            )
        return None

    # ── workload ──────────────────────────────────────────────────────
    def collect(self, queries: List[str],
                frontier_only: bool = False) -> List[Dict[str, Any]]:
        """Route each query through the engine and return per-query quality
        + cost observations. frontier_only = True routes with the frontier
        model pinned (A/B parity baseline)."""
        eng = self.engine
        if frontier_only:
            eng = self._frontier_engine()
        rows = []
        for q in queries:
            t0 = time.perf_counter()
            s = eng.route(q)
            rows.append({
                "query": q,
                "answer": s.answer,
                "context": "\n".join(f"{c['title']}: {c['body']}"
                                     for c in s.contexts[:3]),
                "model": s.model,
                "tier": s.tier,
                "cost_usd": s.est_cost_usd,
                "faithfulness": s.faithfulness,
                "escalations": s.escalation_count,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "sample": self._make_sample(q, s),
            })
        return rows

    def _make_sample(self, q: str, s: Any) -> EvalSample:
        return EvalSample(
            q, answer=s.answer,
            context="\n".join(f"{c['title']}: {c['body']}"
                              for c in s.contexts[:3]),
            context_scores=[c.get("relevance", 0.0)
                            for c in s.contexts[:3]],
            context_families=[c.get("family", "")
                              for c in s.contexts[:3]],
            ground_truth_family=self.family_map.get(q, ""),
        )

    def _frontier_engine(self) -> RouteEngine:
        """Engine pinned to the frontier model (A/B parity baseline)."""
        from infra.data_generation.trace_factory import FRONTIER_MODEL

        return RouteEngine(force_dry_run=True, use_cache=False,
                           pin_model=FRONTIER_MODEL)

    def quality(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        samples = [r["sample"] for r in rows]
        if self.use_real:
            scored = self._real_ragas(samples)
        else:
            scored = [self._proxy.score(s) for s in samples]
        if not scored:
            scored = [self._proxy.score(s) for s in samples]
        agg = {
            "faithfulness": round(sum(s.faithfulness for s in scored) /
                                  len(scored), 4),
            "answer_relevancy": round(sum(s.answer_relevancy for s in scored) /
                                      len(scored), 4),
            "context_precision": round(sum(s.context_precision for s in scored) /
                                       len(scored), 4),
            "judge_mode": scored[0].judge_mode if scored else "proxy",
            "n": len(scored),
        }
        agg["total_cost_usd"] = round(sum(r["cost_usd"] for r in rows), 6)
        return agg