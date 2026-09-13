#!/usr/bin/env python
"""Layer-2 provisioning: corpus → Neo4j graph + Qdrant vectors + BM25, then
the plan's demo ("How do I reset my password?" — graph concept, vector FAQ,
BM25 keyword catch). Run `scripts/check_layer2_results.py` for the gates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# UTF-8 console output on Windows (cp1252 default crashes on arrows/boxes)
for stream in (sys.stdout, sys.stderr):
    if stream is not None and hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from tabulate import tabulate  # noqa: E402

from infra.retrieval.bm25_index import BM25Index  # noqa: E402
from infra.retrieval.corpus import build_corpus  # noqa: E402
from infra.retrieval.graph import KnowledgeGraph  # noqa: E402
from infra.retrieval.hybrid import HybridRetriever  # noqa: E402
from infra.retrieval.reranker import Reranker  # noqa: E402
from infra.retrieval.vector_store import VectorStore  # noqa: E402


def main() -> int:
    data = build_corpus()
    docs = data["documents"]

    print(f"corpus: {len(docs)} documents | {len(data['queries'])} eval queries")

    print("\n== [1/3] Neo4j knowledge graph ==")
    graph = KnowledgeGraph()
    graph.reset()
    graph.ingest(docs)
    print(f"  graph stats: {graph.stats()}")

    print("\n== [2/3] Qdrant vectors + BM25 ==")
    vector = VectorStore(docs, recreate=True)
    vector.upsert()
    bm25 = BM25Index(docs)
    print(f"  vectors: {vector.embedder.dim}-dim, {len(docs)} points | bm25: {len(docs)} docs")

    print("\n== [3/3] hybrid retriever demo ==")
    reranker = Reranker()
    hybrid = HybridRetriever(graph, vector, bm25, reranker, docs)

    demos = [
        "How do I reset my password?",
        "I want a full copy of all our workspace data before we cancel.",
        "What compensation do we get when the service is down?",
        "How do I upgrade our subscription to the biggest plan?",
    ]
    rows = []
    for q in demos:
        r = hybrid.retrieve(q, top_k=5)
        rows.append({
            "query": q[:44],
            "concepts": ",".join(r["matched_concepts"]) or "-",
            "graph": r["graph_hits"], "vector": r["vector_hits"], "bm25": r["bm25_hits"],
            "top-5": "; ".join(doc_id for doc_id, _ in r["results"]),
            "ms": r["latency_ms"],
        })
    print(tabulate(rows, headers="keys", maxcolwidths=[46, 20, 6, 6, 6, 34, 8]))

    out = Path("data/layer2_corpus.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\npersisted corpus → {out}")
    graph.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())