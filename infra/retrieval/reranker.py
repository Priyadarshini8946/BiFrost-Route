"""Cross-encoder re-ranker (plan: cross-encoder/ms-marco-MiniLM-L-6-v2).

Priority:
  1. `rerankers` ort backend — the plan's exact full-size cross-encoder
     ms-marco-MiniLM-L-6-v2 exported to ONNX (onnxruntime + optimum, no
     Python-side torch training stack). Full-size, real discriminator.
  2. flashrank — ms-marco-MiniLM-L-12-v2, a distilled cross-encoder, fully
     offline after a one-time ~3 MB download (fallback when optimum is
     missing or model download fails).
  3. Calibrated score fusion — deterministic, always available.

The re-ranker receives the fused candidate pool (vector + BM25 + graph
scores) and returns a fresh top-k ordering by cross-encoder relevance —
this is what pushes Layer 2 precision past vector-only retrieval.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("retrieval.rerank")

Passage = Dict[str, Any]
Passages = List[Dict[str, Any]]


class Reranker:
    def __init__(self) -> None:
        self.mode = "none"
        self._fn: Optional[Callable[..., List[Tuple[str, float]]]] = None
        self.last_ms: float = 0.0
        self._build()

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Reranker mode={self.mode!r}>"

    def rerank(self, query: str, candidates: Passages,
               top_k: int = 5) -> List[Tuple[str, float]]:
        if not candidates:
            return []
        assert self._fn is not None
        t0 = time.perf_counter()
        ranked = self._fn(query, candidates, top_k)
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return ranked

    # ── construction ──────────────────────────────────────────────────────
    def _build(self) -> None:
        if self._via_rerankers():
            return
        if self._via_flashrank():
            return
        self._fn = self._fusion
        self.mode = "calibrated score fusion (fallback)"
        log.warning("reranker mode: %s", self.mode)

    # ── rerankers ort (plan's exact ms-marco cross-encoder) ───────────────
    def _via_rerankers(self) -> bool:
        try:
            # model weights are cached locally — never probe HuggingFace at load
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from rerankers import Reranker as RR

            self._rr = RR(os.getenv(
                "RERANKER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            ), model_type="cross-encoder", backend="ort")
            # eager warm-up (downloading model weights happens here)
            self._rr.rank(query="warmup query", docs=["warmup doc"])
        except Exception as exc:  # noqa: BLE001
            log.warning("rerankers ort unavailable (%s)", exc)
            return False
        self._fn = self._run_rerankers
        self.mode = "rerankers ort cross-encoder/ms-marco-MiniLM-L-6-v2"
        log.info("reranker mode: %s", self.mode)
        return True

    # ── flashrank (distilled ms-marco cross-encoder, offline fallback) ───
    def _via_flashrank(self) -> bool:
        try:
            from flashrank import Ranker, RerankRequest

            self._fr = Ranker()  # default ms-marco-MiniLM-L-12-v2 ONNX
            probe = self._run_flashrank(
                "warmup query",
                [{"doc_id": "WARM", "title": "warmup", "body": "warm body",
                  "vector": 0.5, "bm25": 0.5, "graph": 0.0}],
                top_k=1)
            if not probe:
                raise RuntimeError("flashrank returned no candidates")
        except Exception as exc:  # noqa: BLE001
            log.warning("flashrank unavailable (%s)", exc)
            return False
        self._fn = self._run_flashrank
        self.mode = "flashrank ms-marco-MiniLM-L-12-v2 (distilled cross-encoder)"
        log.info("reranker mode: %s", self.mode)
        return True

    def _run_flashrank(self, query: str, candidates: Passages,
                       top_k: int) -> List[Tuple[str, float]]:
        from flashrank import RerankRequest

        passages = [{"id": c["doc_id"], "text": f"{c['title']}. {c['body']}"}
                    for c in candidates]
        out = self._fr.rerank(RerankRequest(query=query, passages=passages))
        items = out if isinstance(out, list) else getattr(out, "results", out)
        ranked: List[Tuple[str, float]] = []
        for item in items[:top_k]:
            if isinstance(item, dict):
                doc_id, score = item.get("id"), float(item.get("score") or 0.0)
            else:  # dataclass/object shape
                doc_id, score = getattr(item, "id", None), float(
                    getattr(item, "score", 0.0) or 0.0)
            if doc_id is not None:
                ranked.append((doc_id, round(score, 4)))
        return ranked

    # ── rerankers ort engine ───────────────────────────────────────────────
    def _run_rerankers(self, query: str, candidates: Passages,
                       top_k: int) -> List[Tuple[str, float]]:
        docs = [f"{c['title']}. {c['body']}" for c in candidates]
        out = self._rr.rank(query=query, docs=docs)
        results = out.results if hasattr(out, "results") else out
        text_to_id = {f"{c['title']}. {c['body']}": c["doc_id"] for c in candidates}
        ranked: List[Tuple[str, float]] = []
        for r in results[:top_k]:
            doc_text = r.document.text if hasattr(r, "document") else getattr(r, "text", "")
            doc_id = text_to_id.get(doc_text)
            if doc_id is not None:
                ranked.append((doc_id, round(float(getattr(r, "score", 0.0)), 4)))
        return ranked

    # ── calibrated fusion (final fallback, always available) ──────────────
    @staticmethod
    def _fusion(query: str, candidates: Passages,
                top_k: int) -> List[Tuple[str, float]]:
        def _norm(key: str) -> None:
            vals = [c.get(key, 0.0) for c in candidates]
            peak = max(vals) if vals else 0.0
            if peak > 0.0:
                for c in candidates:
                    c[key] = c.get(key, 0.0) / peak

        for key in ("vector", "bm25", "graph"):
            _norm(key)
        scored = sorted(
            ((c["doc_id"],
              round(0.45 * c.get("vector", 0.0) + 0.35 * c.get("bm25", 0.0)
                    + 0.20 * c.get("graph", 0.0), 4)) for c in candidates),
            key=lambda x: x[1], reverse=True)
        return scored[:top_k]