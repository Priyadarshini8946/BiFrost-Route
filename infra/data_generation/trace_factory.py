"""Realistic query-trace generator for the Bifrost Route data layer.

Shapes deterministic synthetic traffic like a real production routing
workload — ~60% simple / ~25% medium / ~15% complex queries, escalation
events concentrated on hard queries, priced against the model_registry unit
economics. Keep MODEL_COSTS in sync with infra/db/schema.sql seeds.
"""
from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

# Unit economics (USD per 1K tokens) — must mirror the MySQL `model_registry` seed.
MODEL_COSTS: Dict[str, Dict[str, Any]] = {
    "llama-3.3-70b-versatile": {"provider": "groq",   "tier": "cheap",     "in": 0.000590, "out": 0.000790},
    "gemini-2.0-flash":        {"provider": "google", "tier": "standard",  "in": 0.000100, "out": 0.000400},
    "gemini-2.5-pro":          {"provider": "google", "tier": "frontier",  "in": 0.001250, "out": 0.010000},
}

COMPLEXITY_TO_MODEL: Dict[str, str] = {
    "simple": "llama-3.3-70b-versatile",
    "medium": "gemini-2.0-flash",
    "complex": "gemini-2.5-pro",
}

FRONTIER_MODEL = "gemini-2.5-pro"

# Query templates by complexity tier (factual / multi-step / multi-hop ambiguous).
_SIMPLE_TEMPLATES: List[str] = [
    "What is the refund policy?",
    "How do I reset my password?",
    "What are your support hours?",
    "Where is my order #{}?",
    "What shipping options are available?",
    "How do I update my billing address?",
    "Can I change my plan from {} to {}?",
    "What is the cancellation deadline?",
    "How do I invite a teammate to {}?",
    "What file formats does {} support?",
    "Is my payment method saved securely?",
    "How many seats does the {} plan include?",
    "What is the SLA for enterprise accounts?",
    "How do I export my data?",
    "Where can I download the mobile app?",
]
_MEDIUM_TEMPLATES: List[str] = [
    "Compare pricing between the {} and {} plans, including hidden fees.",
    "Why did my invoice go up from last month? Break down the line items.",
    "Summarize our quarterly report and list the top three risks.",
    "What happens when a team member leaves mid-cycle — prorating rules across {} and {}?",
    "Draft a response to this customer complaint and include next steps.",
    "Which features unlock at the {} tier and how do they affect our current workflow?",
    "Explain the SSO setup steps for {} and the common failure modes.",
    "Compare uptime guarantees of {} vs {} and map SLA credits to our contract.",
    "Identify which of our 20 integrations are deprecated and suggest replacements.",
    "Walk me through the onboarding checklist for a new {} workspace.",
]
_COMPLEX_TEMPLATES: List[str] = [
    "Reconcile the discrepancy between the sales report and finance ledger for {} "
    "considering FX effects.",
    "Given contract clause 4.2, the GDPR retention policy, and the customer's data "
    "deletion request, what is the compliant course of action?",
    "A production incident crossed the {} SLA: analyze sequence of events, blast "
    "radius, and whether the excluded-service clause applies.",
    "Our {} rollout failed in EMEA but not APAC. Hypothesize root causes across "
    "config drift, network topology, and regional compliance modes, then rank them.",
    "Merge the access-control matrices from {}, {} and {} into one policy and find "
    "conflicting grants.",
    "Interpret the {} benchmark results against our {} workload and adversarial "
    "prompts, and advise on model decommission.",
    "Audit the dispute between two departments over budget ownership in Q{}. "
    "Propose allocation rules that satisfy both compliance and finance.",
    "Our data retention timeline conflicts with customer contract {} in {} "
    "jurisdiction. Design a compliant data lifecycle.",
]


def _pick(rng: random.Random, templates: List[str]) -> str:
    t = rng.choice(templates)
    try:
        return t.format(*[rng.choice(["Pro", "Enterprise", "Starter"]) for _ in range(4)],
                        rng.randint(10000, 99999), rng.randint(1, 4))
    except (IndexError, KeyError):
        return t


def model_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Line-item cost for a call, from the registry's per-1K rates."""
    c = MODEL_COSTS[model]
    return c["in"] * input_tokens / 1000.0 + c["out"] * output_tokens / 1000.0


def frontier_baseline_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """What the SAME call would cost on the most expensive capable model."""
    return model_cost_usd(FRONTIER_MODEL, input_tokens, output_tokens)


def _next_model_up(complexity: str) -> str:
    order = ["simple", "medium", "complex"]
    idx = order.index(complexity)
    target = order[idx + 1] if idx + 1 < len(order) else complexity
    return COMPLEXITY_TO_MODEL[target]


def generate_query_traces(
    count: int = 100,
    seed: int = 42,
    now: datetime | None = None,
) -> List[Dict[str, Any]]:
    """Generate `count` deterministic production-shaped query traces.

    `now` is injectable so tests can pin the wall clock; when omitted the
    real UTC clock is used and ~70% of traces land inside the last hour
    (keeps the "escalations in the last hour" GSI demo populated).
    """
    rng = random.Random(seed)
    now = now or datetime.now(timezone.utc)
    traces: List[Dict[str, Any]] = []

    for _ in range(count):
        roll = rng.random()
        if roll < 0.60:
            complexity = "simple"
        elif roll < 0.85:
            complexity = "medium"
        else:
            complexity = "complex"

        # Token volumes scale with complexity.
        if complexity == "simple":
            prompt_tokens = rng.randint(60, 180)
            completion_tokens = rng.randint(40, 160)
            faithfulness = rng.uniform(0.88, 0.985)
            latency_ms = rng.randint(300, 900)
        elif complexity == "medium":
            prompt_tokens = rng.randint(180, 520)
            completion_tokens = rng.randint(160, 480)
            faithfulness = rng.uniform(0.80, 0.95)
            latency_ms = rng.randint(900, 2400)
        else:
            prompt_tokens = rng.randint(520, 1400)
            completion_tokens = rng.randint(480, 1200)
            faithfulness = rng.uniform(0.68, 0.93)
            latency_ms = rng.randint(1800, 5200)

        query_text = _pick(rng, {
            "simple": _SIMPLE_TEMPLATES, "medium": _MEDIUM_TEMPLATES, "complex": _COMPLEX_TEMPLATES,
        }[complexity])

        selected_model = COMPLEXITY_TO_MODEL[complexity]
        # Escalation: rail complexity + fails the RAGAS gate → hop to next tier.
        esc_roll = rng.random()
        escalate = {
            "simple": esc_roll < 0.03,
            "medium": esc_roll < 0.10,
            "complex": esc_roll < 0.30,
        }[complexity]

        final_model = _next_model_up(complexity) if escalate else selected_model
        # complexity has no tier above it: an escalated frontier query simply
        # re-runs on the same frontier model — `final_model` stays equal.
        escalation_count = (1 if escalate and complexity != "complex"
                            else (2 if escalate else 0))

        cost_usd = round(model_cost_usd(selected_model, prompt_tokens, completion_tokens), 8)
        baseline_usd = round(model_cost_usd(FRONTIER_MODEL, prompt_tokens, completion_tokens), 8)

        ts = now - timedelta(seconds=rng.randint(0, 3600) if rng.random() < 0.7
                             else rng.randint(3600, 24 * 3600))
        traces.append({
            "trace_id": f"trc-{uuid.UUID(int=rng.getrandbits(128))}",
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
            "query_text": query_text,
            "complexity": complexity,
            "selected_model": selected_model,
            "provider": MODEL_COSTS[selected_model]["provider"],
            "tier": MODEL_COSTS[selected_model]["tier"],
            "final_model": final_model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "frontier_baseline_cost_usd": baseline_usd,
            "rag_faithfulness": round(faithfulness, 4),
            "did_escalate": escalate,
            "escalation_count": escalation_count,
            "latency_ms": latency_ms,
            "cached": False,
            "status": "ok",
        })
    return traces