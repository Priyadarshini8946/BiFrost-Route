"""Benchmark for the Layer-1 Redis acceptance targets.

  1. Traffic hit rate  > 30%   — 1,000 mixed lookups (60% new / 40% repeat;
     repeats are 75% exact + 25% rephrased) against a semantic cache.
  2. Semantic precision — of the *rephrased* lookups, how many genuinely
     matched (a rephrase that swaps a real token may rightly miss).
  3. LRU eviction      — after the traffic phase, load the Redis memory budget
     with filler data; `evicted_keys` must move (proves allkeys-lru is real).
"""
from __future__ import annotations

import random
from itertools import product
from typing import Any, Dict, List

from infra.cache.semantic_cache import SemanticCache

_POOL_TEMPLATES = [
    "What is the refund policy for order {n}?",
    "How do I reset my {s} password?",
    "What are the support hours for {s}?",
    "Where is my order {n} being shipped?",
    "How do I update my billing address on {s}?",
    "Can I change my plan from {p1} to {p2}?",
    "What is the cancellation deadline for {p1}?",
    "How do I invite a teammate to {s}?",
    "What file formats does {s} support?",
    "What is the SLA for enterprise accounts on {s}?",
    "How do I export my data from {s}?",
    "Is the {p1} plan billed monthly or yearly?",
    "How many seats does the {p1} plan include?",
    "Where can I download the mobile app for {s}?",
    "What happens to my data if I cancel {p1}?",
    "How do I enable two-factor auth on {s}?",
    "Does {s} offer a free trial for {p1}?",
    "What payment methods does {s} accept?",
    "How do I merge accounts between {s} and legacy {s2}?",
    "Are there usage limits on the {p1} plan?",
]

_STOPSWAP = {"the": "a", "a": "the", "your": "our", "my": "our", "for": "with", "?": ""}
_SYNSWAP = {"password": "credentials", "shipping": "delivery", "billing": "invoice",
            "app": "application", "plan": "subscription"}


def build_query_pool(n: int = 800, seed: int = 7) -> List[str]:
    """n distinct, realistic-looking cacheable queries."""
    rng = random.Random(seed)
    services = ["groq", "gemini", "redis", "docker", "mysql", "portal", "api", "web", "cli", "studio"]
    plans = ["Starter", "Pro", "Enterprise"]
    pool: List[str] = []
    filler = iter(range(10_000, 999_999))
    for tpl, svc in product(_POOL_TEMPLATES[:10], services):
        for _ in range(n // 100 + 1):
            q = tpl.format(n=next(filler), s=svc, s2=rng.choice(services),
                           p1=rng.choice(plans), p2=rng.choice(plans))
            if q not in pool:
                pool.append(q)
    rng.shuffle(pool)
    return pool[:n]


def rephrase(query: str, style: str) -> str:
    """style 'exact' → semantic-identical (stopwords swap);
    style 'loose' → real synonym swap (may legitimately miss)."""
    if style == "exact":
        out = query
        for a, b in _STOPSWAP.items():
            out = out.replace(a, b)
        return out.lower().strip()
    out = query
    for a, b in _SYNSWAP.items():
        if a in out.lower():
            out = out.replace(a, b)
            break
    return out.lower().strip()


def run_traffic(cache: SemanticCache, pool: List[str], traffic: int = 1000, seed: int = 7) -> Dict[str, Any]:
    rng = random.Random(seed)
    unseen = pool[:]
    rng.shuffle(unseen)
    seen: List[str] = []
    stats = {"exact_hits": 0, "exact_lookups": 0, "para_hits": 0, "para_lookups": 0,
             "misses": 0, "para_similarities": []}

    for _ in range(traffic):
        if rng.random() < 0.4 and seen:
            base = rng.choice(seen)
            if rng.random() < 0.25:
                style = "exact" if rng.random() < 0.5 else "loose"
                q = rephrase(base, style)
                stats["para_lookups"] += 1
                res = cache.get(q)
                if res["hit"]:
                    stats["para_hits"] += 1
                stats["para_similarities"].append(res["similarity"])
            else:
                stats["exact_lookups"] += 1
                res = cache.get(base)
                if res["hit"]:
                    stats["exact_hits"] += 1
        else:
            q = unseen.pop() if unseen else rng.choice(pool)
            seen.append(q)
            res = cache.get(q)
            if not res["hit"]:
                cache.set(q, f"[synthetic answer] {q}")

    total_lookups = sum(v for k, v in stats.items() if k != "para_similarities")
    hits = stats["exact_hits"] + stats["para_hits"]
    return {
        "total_lookups": total_lookups,
        "hits": hits,
        "misses": stats["misses"],
        "traffic_hit_rate": round(hits / total_lookups, 4) if total_lookups else 0.0,
        "exact_lookups": stats["exact_lookups"],
        "exact_hits": stats["exact_hits"],
        "exact_hit_rate": round(stats["exact_hits"] / stats["exact_lookups"], 4) if stats["exact_lookups"] else 0.0,
        "para_lookups": stats["para_lookups"],
        "para_hits": stats["para_hits"],
        "para_hit_rate": round(stats["para_hits"] / stats["para_lookups"], 4) if stats["para_lookups"] else 0.0,
        "para_mean_similarity": round(sum(stats["para_similarities"]) / len(stats["para_similarities"]), 4)
        if stats["para_similarities"] else 0.0,
    }


def run_lru_eviction_demo(cache: SemanticCache, filler_count: int = 3000, value_kb: int = 6) -> Dict[str, Any]:
    """Fill past the Redis memory budget; allkeys-lru must evict (old) keys."""
    r = cache.r
    before_info = cache.info()
    sample_keys = [k for k in r.scan_iter(match=f"{CACHE_PREFIX}*", count=500)][:30]
    before_survivors = sum(1 for k in sample_keys if r.exists(k))

    for i in range(filler_count):
        r.set(f"bifrost:fill:{i:06d}", b"X" * (value_kb * 1024), ex=3600)

    info = cache.info()
    after_survivors = sum(1 for k in sample_keys if r.exists(k))
    return {
        "evicted_keys_before": before_info["evicted_keys"],
        "evicted_keys_after": info["evicted_keys"],
        "eviction_delta": info["evicted_keys"] - before_info["evicted_keys"],
        "sampled_cache_keys_before": len(sample_keys),
        "sampled_cache_keys_surviving": after_survivors,
        "used_memory": info["used_memory"],
        "maxmemory_policy": info["maxmemory_policy"],
    }