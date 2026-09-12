"""Semantic embedder for the cache.

Layer 1 uses a deterministic, offline TF-IDF sentence embedder (fixed DIM,
L2-normalized → cosine similarity == dot product). It plugs into the exact
same `Embedder` interface a real sentence encoder (e.g. text-embedding-3-small
or all-MiniLM-L6-v2) will satisfy in Layer 2 — swap `embed()` only.
"""
from __future__ import annotations

from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
    """L2-normalized TF-IDF embedder with fixed output dimension."""

    def __init__(self, max_features: int = 1024) -> None:
        self.max_features = max_features
        self._vec = TfidfVectorizer(
            max_features=max_features,
            sublinear_tf=True,
            stop_words="english",
            ngram_range=(1, 2),
            lowercase=True,
        )
        self._fitted = False

    @property
    def dim(self) -> int:
        return self.max_features

    def fit(self, documents: List[str]) -> "TfidfEmbedder":
        if not documents:
            raise ValueError("embedder.fit requires a non-empty corpus")
        self._vec.fit(documents)
        self._fitted = True
        return self

    def embed(self, text: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder must be fit() before embed()")
        raw = self._vec.transform([text]).toarray().astype(np.float32)[0]
        out = np.zeros(self.max_features, dtype=np.float32)
        n = min(len(raw), self.max_features)
        out[:n] = raw[:n]
        norm = float(np.linalg.norm(out))
        if norm > 0.0:
            out /= norm
        return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two L2-normalized vectors (= dot product)."""
    return float(np.dot(a, b))