"""Debug: per-query vector vs hybrid top-5 with families, to see exactly
where the reranker loses gold documents."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from infra.retrieval.bm25_index import BM25Index  # noqa: E402
from infra.retrieval.corpus import build_corpus  # noqa: E402
from infra.retrieval.graph import KnowledgeGraph  # noqa: E402
from infra.retrieval.hybrid import HybridRetriever  # noqa: E402
from infra.retrieval.reranker import Reranker  # noqa: E402
from infra.retrieval.vector_store import VectorStore  # noqa: E402

data = build_corpus()
docs = data["documents"]
queries = data["queries"]
registry = data["registry"]
by_id = {d["doc_id"]: d for d in docs}

graph = KnowledgeGraph()
vector = VectorStore(docs, recreate=False)
bm25 = BM25Index(docs)
reranker = Reranker()
hybrid = HybridRetriever(graph, vector, bm25, reranker, docs)

mismatches = 0
for eq in queries:
    gold = set(registry[eq["family"]])
    r = hybrid.retrieve(eq["query"], top_k=5)
    hyb = [d for d, _ in r["results"]]
    vec = hybrid.vector_only(eq["query"], top_k=5)
    hg = sum(1 for d in hyb if d in gold)
    vg = sum(1 for d in vec if d in gold)
    mismatch = hg < vg and vg >= 3  # reranker lost ground with a strong vector signal
    if mismatch or vg == 0:
        mismatches += 1
        print(f"\nQ: {eq['query']}  [gold={eq['family']}]")
        print(f"  vector top5 ({vg}/5): {[(d, by_id[d]['family']) for d in vec]}")
        print(f"  hybrid top5 ({hg}/5): {[(d, by_id[d]['family']) for d in hyb]}")

print(f"\n--- mismatches / lost-gold cases: {mismatches}")
graph.close()