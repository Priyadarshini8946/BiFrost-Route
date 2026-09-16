"""Bifrost Route · Layer 4 — the LangGraph routing state machine.

Nodes (each a pure state→state step):

    classify → cache_lookup → retrieve → select_model → budget_check
            → generate → grade → (escalate → generate ↺) → record → END

  • classify     L3 complexity classifier (simple|medium|complex)
  • cache_lookup L1 semantic cache — hit short-circuits to record
  • retrieve     L2 hybrid retrieval (graph+vector+BM25 → RRF → re-rank)
  • select_model MySQL routing_policy → model tier mapping + token/cost math
  • budget_check budget_limits → refuse when remaining spend is insufficient
  • generate     LLM call (real provider or deterministic mock)
  • grade        RAGAS-proxy faithfulness gate (L5 replaces with real RAGAS)
  • escalate     bump tier (cheap→standard→frontier), bounded by max hops
  • record       DynamoDB trace + warehouse fact + semantic cache write

Escalation invariant: grade(fail) with hops < max → escalate; else record.
The graph never trains anything — the classifier is a pre-trained artifact.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from langgraph.graph import END, START, StateGraph

from infra.routing.state import RouteState


def _snapshot(st: RouteState) -> Dict[str, Any]:
    """Full-field state dump (LangGraph replaces each key wholesale)."""
    return {f: getattr(st, f) for f in RouteState.__dataclass_fields__}


class RouterGraph:
    """Builds a compiled LangGraph RoutingState from engine-bound node fns.

    Each node fn takes RouteState and mutates it; the graph wraps them so
    LangGraph sees state→state dict transitions.
    """

    TIER_LADDER = ["cheap", "standard", "frontier"]

    def __init__(self, nodes: Dict[str, Callable[[RouteState], None]],
                 tier_hops: int = 2) -> None:
        self._nodes = nodes
        self._tier_hops = tier_hops

    def compile(self):
        g = StateGraph(RouteState)

        def wrap(name: str) -> Callable[[RouteState], Dict[str, Any]]:
            fn = self._nodes[name]

            def step(state: RouteState) -> Dict[str, Any]:
                fn(state)
                return _snapshot(state)

            step.__name__ = name
            return step

        for name in self._nodes:
            g.add_node(name, wrap(name))

        g.add_edge(START, "classify")
        g.add_edge("classify", "cache_lookup")
        g.add_conditional_edges(
            "cache_lookup",
            lambda s: "record" if s.cache_hit else "retrieve",
            {"record": "record", "retrieve": "retrieve"},
        )
        g.add_edge("retrieve", "select_model")
        g.add_edge("select_model", "budget_check")
        g.add_conditional_edges(
            "budget_check",
            lambda s: "record" if s.budget_refused else "generate",
            {"record": "record", "generate": "generate"},
        )
        g.add_edge("generate", "grade")
        g.add_conditional_edges(
            "grade",
            lambda s: self._grade_route(s),
            {"escalate": "escalate", "record": "record"},
        )
        g.add_edge("escalate", "generate")
        g.add_edge("record", END)
        return g.compile()

    def _grade_route(self, s: RouteState) -> str:
        if s.faithfulness >= 0.850:  # policy faithfulness_threshold
            return "record"
        if s.escalation_count < self._tier_hops and \
                self._can_escalate(s.tier):
            return "escalate"
        return "record"

    @classmethod
    def _can_escalate(cls, tier: str) -> bool:
        return tier in cls.TIER_LADDER and \
            tier != cls.TIER_LADDER[-1]