"""Vector store (Qdrant) — the local stand-in for Pinecone.

Same pattern as Redshift→DuckDB: the API shape (collection + metadata +
cosine top-K) mirrors Pinecone; swapping to real Pinecone later only changes
the client import. Embeddings come from fastembed (ONNX bge-small, offline
after first download) with a TF-IDF fallback so the pipeline is always runnable.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from infra.cache.embedder import TfidfEmbedder  # noqa: E402

load_dotenv()
log = logging.getLogger("retrieval.vector")


def make_embedder(corpus_texts: List[str]):
    """fastembed TextEmbedding, falling back to the offline TF-IDF embedder."""
    try:
        from fastembed import TextEmbedding

        model = TextEmbedding(model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
        # validate once on a short string
        list(model.embed(["validation"]))
        log.info("embedder: fastembed %s", os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
        return EmbedderAdapter(model)
    except Exception as exc:  # noqa: BLE001 — offline/network fallback
        log.warning("fastembed unavailable (%s) → TF-IDF embedder", exc)
        tf = TfidfEmbedder(max_features=1024).fit(corpus_texts)
        return EmbedderAdapter(tf)


class EmbedderAdapter:
    """Uniform .embed(text)->np.ndarray over fastembed or TF-IDF."""

    def __init__(self, impl: Any) -> None:
        self._impl = impl
        self.dim = None
        probe = self.embed("probe")
        self.dim = int(probe.shape[0])

    def embed(self, text: str) -> np.ndarray:
        if hasattr(self._impl, "embed") and "TfidfEmbedder" not in type(self._impl).__name__:
            # fastembed-style: iterable of vectors
            vecs = list(self._impl.embed([text]))
            v = np.asarray(vecs[0], dtype=np.float32)
            n = float(np.linalg.norm(v))
            return v / n if n > 0 else v
        return self._impl.embed(text).astype(np.float32)  # TfidfEmbedder: single vector


class VectorStore:
    def __init__(self, documents: List[Dict[str, Any]], recreate: bool = True) -> None:
        self.host = os.getenv("QDRANT_HOST", "127.0.0.1")
        self.port = int(os.getenv("QDRANT_PORT", "6333"))
        self.collection = os.getenv("QDRANT_COLLECTION", "bifrost_docs")
        self.client = QdrantClient(host=self.host, port=self.port, timeout=60)
        self.documents = documents
        corpus_texts = [d["title"] + ". " + d["body"] for d in documents]
        self.embedder = make_embedder(corpus_texts)
        self._ensure_collection(recreate)

    def _ensure_collection(self, recreate: bool) -> None:
        if recreate and self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
            )

    def upsert(self) -> int:
        self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
        )
        batches = []
        for i, d in enumerate(self.documents):
            vec = self.embedder.embed(d["title"] + ". " + d["body"]).tolist()
            batches.append(PointStruct(
                id=i,
                vector=vec,
                payload={k: d.get(k) for k in ("doc_id", "title", "body", "family", "source")},
            ))
        self.client.upsert(collection_name=self.collection, points=batches, wait=True)
        return len(batches)

    def search(self, query: str, k: int = 20) -> List[Tuple[str, float]]:
        vec = self.embedder.embed(query).tolist()
        hits = self.client.query_points(
            collection_name=self.collection, query=vec, limit=k
        ).points
        return [(hit.payload["doc_id"], round(float(hit.score), 4)) for hit in hits]