"""BM25 keyword index (rank-bm25) over the Layer-2 corpus.

Provides the keyword fallback the plan requires: catches exact terminology
that vector search can miss, and contributes a normalized score to the hybrid
fusion in `hybrid.py`.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from rank_bm25 import BM25Okapi

STOP = {
    "the", "a", "an", "is", "are", "to", "of", "for", "on", "in", "and",
    "or", "with", "how", "what", "do", "does", "can", "my", "our", "your",
    "i", "we", "it", "as", "at", "by", "from", "that", "this", "there",
}


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


class BM25Index:
    def __init__(self, documents: List[Dict[str, Any]]) -> None:
        self._docs = {d["doc_id"]: d for d in documents}
        corpus = [tokenize(d["title"] + " " + d["body"]) for d in documents]
        self._bm25 = BM25Okapi(corpus)
        self._id_list = [d["doc_id"] for d in documents]

    def search(self, query: str, k: int = 20) -> List[Tuple[str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._id_list, scores), key=lambda x: x[1], reverse=True)
        ranked = [(doc_id, s) for doc_id, s in ranked if s > 0.0]
        top = ranked[:k]
        if not top:
            return []
        max_s = top[0][1]
        return [(doc_id, round(s / max_s, 4)) for doc_id, s in top]  # normalized