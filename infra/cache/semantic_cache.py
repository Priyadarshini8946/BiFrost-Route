"""Semantic cache on Redis Stack (RediSearch vector search).

Design (mirrors the Layer-1 plan):
  • Semantic key: query text → embedding. A new query hits if a cached query's
    embedding has cosine similarity ≥ SIM_THRESHOLD (default 0.92).
  • Storage: Redis hashes under `bifrost:cache:` (auto-indexed by RediSearch).
    TTL 1 hour (`REDIS_CACHE_TTL_SECONDS`), refreshed on every hit.
  • Eviction: `maxmemory-policy allkeys-lru` set on the server (16 MB demo
    budget from .env) — the benchmark proves `evicted_keys` actually moves.
  • Exactness: RediSearch (FLAT vector index — exact at this scale; HNSW for
    >10k entries) returns top-K candidates; the final score is the *exact*
    cosine against the stored embedding (protects against approximate matches).
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import redis
from dotenv import load_dotenv
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from infra.cache.embedder import TfidfEmbedder, cosine

load_dotenv()

CACHE_INDEX = "bifrostCache"
CACHE_PREFIX = "bifrost:cache:"


class SemanticCache:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: int = 0,
        ttl: Optional[int] = None,
        embedder: Optional[TfidfEmbedder] = None,
        sim_threshold: float = 0.92,
    ) -> None:
        self.host = host or os.getenv("REDIS_HOST", "127.0.0.1")
        self.port = port or int(os.getenv("REDIS_PORT", "6380"))
        self.ttl = ttl or int(os.getenv("REDIS_CACHE_TTL_SECONDS", "3600"))
        self.sim_threshold = sim_threshold
        self.embedder = embedder or TfidfEmbedder()
        self.hits = 0
        self.misses = 0
        self.r = redis.Redis(host=self.host, port=self.port, db=db)
        self.r.ping()
        self._ensure_index()

    # ── index management ────────────────────────────────────────────────
    def _ensure_index(self) -> None:
        if not self.embedder._fitted:  # noqa: SLF001 — same-package contract
            raise RuntimeError("embedder must be fit() before building the cache index")
        try:
            self.r.ft(CACHE_INDEX).create_index(
                fields=(
                    VectorField(
                        "embedding",
                        "FLAT",
                        {
                            "TYPE": "FLOAT32",
                            "DIM": self.embedder.dim,
                            "DISTANCE_METRIC": "COSINE",
                        },
                    ),
                    TextField("query"),
                ),
                definition=IndexDefinition(prefix=[CACHE_PREFIX], index_type=IndexType.HASH),
            )
        except redis.exceptions.ResponseError as exc:
            if "already exists" not in str(exc).lower():
                raise

    def ensure_lru_config(self, maxmemory_bytes: int = 16 * 1024 * 1024, policy: str = "allkeys-lru") -> None:
        self.r.config_set("maxmemory", maxmemory_bytes)
        self.r.config_set("maxmemory-policy", policy)

    # ── core ops ────────────────────────────────────────────────────────
    def get(self, query: str) -> Dict[str, Any]:
        """Semantic lookup. Exact (hash) match first — an identical query can
        never miss even when its terms fall outside the TF-IDF vocabulary —
        then RediSearch vector search for paraphrases."""
        exact_key = CACHE_PREFIX + hashlib.sha256(query.encode("utf-8")).hexdigest()
        raw = self.r.hgetall(exact_key)
        if raw:
            response = raw.get(b"response", b"").decode("utf-8", errors="replace")
            matched = raw.get(b"query", b"").decode("utf-8", errors="replace")
            self.hits += 1
            self.r.expire(exact_key, self.ttl)  # refresh TTL on hit
            return {"hit": True, "similarity": 1.0, "response": response,
                    "matched_query": matched, "key": exact_key, "exact": True}

        vec = self.embedder.embed(query)
        q = (
            Query("*=>[KNN 5 @embedding $vec AS score]")
            .sort_by("score")
            .dialect(2)
        )
        try:
            # redis-py >= 8 passes query params at search time, not on Query.
            res = self.r.ft(CACHE_INDEX).search(q, query_params={"vec": vec.tobytes()})
        except redis.exceptions.ResponseError:
            res = type("R", (), {"docs": []})()  # index empty → no candidates

        best: Optional[Any] = None
        best_sim = 0.0
        for doc in res.docs:
            key = doc.id.decode() if isinstance(doc.id, bytes) else doc.id
            stored = self.r.hget(key, "embedding")
            if stored is None:
                continue
            cand = np.frombuffer(stored, dtype=np.float32)
            sim = cosine(vec, cand)
            if sim > best_sim:
                best_sim, best = sim, key

        if best is not None and best_sim >= self.sim_threshold:
            raw = self.r.hgetall(best)
            response = raw.get(b"response", b"").decode("utf-8", errors="replace")
            matched = raw.get(b"query", b"").decode("utf-8", errors="replace")
            self.hits += 1
            self.r.expire(best, self.ttl)  # refresh TTL on hit
            return {"hit": True, "similarity": best_sim, "response": response,
                    "matched_query": matched, "key": best}
        self.misses += 1
        return {"hit": False, "similarity": best_sim, "match_if_below": self.sim_threshold}

    def set(self, query: str, response: str) -> str:
        key = CACHE_PREFIX + hashlib.sha256(query.encode("utf-8")).hexdigest()
        payload = {
            "query": query,
            "embedding": self.embedder.embed(query).tobytes(),
            "response": response,
            "created_at": time.time(),
        }
        self.r.hset(key, mapping=payload)
        self.r.expire(key, self.ttl)
        return key

    # ── introspection ───────────────────────────────────────────────────
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def info(self) -> Dict[str, Any]:
        mem = self.r.info("memory")
        stats = self.r.info("stats")
        return {
            "maxmemory_policy": mem.get("maxmemory_policy"),
            "maxmemory": mem.get("maxmemory"),
            "used_memory": mem.get("used_memory"),
            "evicted_keys": stats.get("evicted_keys", 0),
            "keyspace_hits": stats.get("keyspace_hits", 0),
            "keyspace_misses": stats.get("keyspace_misses", 0),
            "in_app_hits": self.hits,
            "in_app_misses": self.misses,
            "in_app_hit_rate": round(self.hit_rate(), 4),
        }

    def flush(self) -> None:
        for key in self.r.scan_iter(match=f"{CACHE_PREFIX}*", count=500):
            self.r.delete(key)
        self.hits = self.misses = 0