"""Bifrost Route · Layer 4 — LangGraph shared state.

One typed state object flows through the routing graph. Every node reads and
writes fields on it; the graph stays a pure function of (state → state) so it
is unit-testable without the LLM/cache/db layers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RouteState:
    # input
    query: str = ""
    user_id: str = "anon"
    team: str = "eng-core"

    # classifier (L3)
    complexity: Optional[str] = None        # simple|medium|complex
    classifier_mode: str = "none"

    # cache (L1)
    cache_hit: bool = False
    cache_similarity: float = 0.0
    cache_key: Optional[str] = None

    # retrieval (L2)
    contexts: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_docs: List[str] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0

    # economics (L1 registry)
    model: str = ""                          # selected by policy
    tier: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    est_cost_usd: float = 0.0
    frontier_baseline_usd: float = 0.0

    # budget (L1)
    budget_ok: bool = True
    budget_refused: bool = False
    budget_remaining_usd: float = 0.0

    # generation
    answer: str = ""
    llm_mode: str = "mock"                   # real provider or mock
    llm_latency_ms: float = 0.0

    # quality / escalation
    faithfulness: float = 0.0
    did_escalate: bool = False
    escalation_count: int = 0
    escalations: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"                       # ok | refused | failed

    # trace
    trace_id: str = ""
    trace_written: bool = False
    trace_latency_ms: float = 0.0

    # housekeeping
    hops: int = 0
    events: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def log(self, event: str) -> None:
        self.events.append(f"{len(self.events)}:{event}")