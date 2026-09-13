"""Hybrid retriever: Neo4j (graph) + Qdrant (vector) + BM25 (keyword),
merged by Reciprocal Rank Fusion (RRF), then re-ranked by the cross-encoder.

Why RRF instead of a weighted score sum: BM25 raw scores, Qdrant cosine and a
binary graph hit live on incompatible scales. A weighted sum lets the noisiest
signal (raw BM25 for rephrased queries) dominate the *pre-selection*, so the
cross-encoder never even sees the gold documents. RRF is rank-based:
score(doc) = Σ_sources 1/(K + rank), which is scale-free and robust to one
weak signal. The reranker then re-ranks the top of the RRF pool into top-k.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("retrieval.hybrid")


class HybridRetriever:
    def __init__(self, graph, vector_store, bm25, reranker, documents: List[Dict[str, Any]]) -> None:
        self.graph = graph
        self.vector = vector_store
        self.bm25 = bm25
        self.reranker = reranker
        self.docs = {d["doc_id"]: d for d in documents}

    def retrieve(self, query: str, top_k: int = 5, rrf_k: int = 60,
                 pool_size: int = 24) -> Dict[str, Any]:
        t0 = time.perf_counter()
        graph_hits, concepts = self.graph.query(query, limit=10)
        vec_hits = self.vector.search(query, k=20)
        bm_hits = self.bm25.search(query, k=20)

        # RRF over the three ranked lists (scale-free rank merge)
        rrf: Dict[str, float] = {}
        for rank, (doc_id, _score) in enumerate(vec_hits + bm_hits + graph_hits):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)

        candidates = []
        for doc_id in sorted(rrf, key=rrf.get, reverse=True)[:pool_size]:
            doc = self.docs.get(doc_id)
            if doc is None:
                continue
            vec_score = next((s for d, s in vec_hits if d == doc_id), 0.0)
            bm_score = next((s for d, s in bm_hits if d == doc_id), 0.0)
            is_graph = any(d == doc_id for d, _ in graph_hits)
            candidates.append({
                "doc_id": doc_id, "title": doc["title"], "body": doc["body"],
                "vector": vec_score, "bm25": bm_score, "graph": 1.0 if is_graph else 0.0,
            })

        ranked = self.reranker.rerank(query, candidates, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "query": query,
            "matched_concepts": concepts,
            "graph_hits": len(graph_hits),
            "vector_hits": len(vec_hits),
            "bm25_hits": len(bm_hits),
            "pool_size": len(candidates),
            "results": ranked,
            "latency_ms": round(elapsed_ms, 2),
            "reranker": self.reranker.mode,
        }

    def vector_only(self, query: str, top_k: int = 5) -> List[str]:
        return [doc_id for doc_id, _ in self.vector.search(query, k=top_k)]