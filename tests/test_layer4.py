"""Layer-4 tests — routing state machine, grader, cache, trace, economics.

Offline tests need no services. Integration tests (live MySQL/Redis/DynamoDB/
Neo4j/Qdrant + dry-run LLM) run only with BIFROST_INTEGRATION=1.
No test ever trains anything — the classifier is a pre-trained artifact.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.routing.state import RouteState  # noqa: E402

INTEGRATION = os.getenv("BIFROST_INTEGRATION", "0") == "1"


# ── offline unit tests ───────────────────────────────────────────────

def test_state_defaults():
    s = RouteState()
    assert s.complexity is None
    assert s.status == "ok"
    assert s.escalation_count == 0
    assert s.events == []
    s.log("x")
    assert s.events == ["0:x"]


def test_route_state_events_ordered():
    s = RouteState(query="q")
    s.log("classify")
    s.log("cache_lookup")
    assert s.events == ["0:classify", "1:cache_lookup"]


def test_grader_grounded_vs_generic():
    from infra.routing.grader import FaithfulnessGrader

    g = FaithfulnessGrader()
    ctx = [{"title": "Password reset",
            "body": "You can reset your password from Settings Security. "
                    "Click the reset link and choose a new password."}]
    grounded = g.grade("Password reset: You can reset your password from "
                       "Settings Security. Click the reset link and choose "
                       "a new password.", ctx)
    assert grounded["verdict"] == "pass"
    assert grounded["score"] >= 0.85
    generic = g.grade("I do not have specific information on that in the "
                      "knowledge base.", ctx)
    assert generic["verdict"] == "fail"
    assert generic["score"] < 0.85


def test_grader_no_context_fails():
    from infra.routing.grader import FaithfulnessGrader

    g = FaithfulnessGrader()
    r = g.grade("any answer at all", [])
    assert r["verdict"] == "fail"
    assert r["score"] == 0.0


def test_mock_llm_seam():
    from infra.routing.llm import MockLLM

    m = MockLLM()
    normal = m.complete("gemini-2.5-pro", "@@CTX@@\nT: X\nB: body words\n@@ENDCTX@@", "hi")
    assert "body words" in normal
    hall = m.complete("gemini-2.5-pro", "@@CTX@@\nT: X\nB: body words\n@@ENDCTX@@",
                      "fabricate __HALLUCINATE__")
    assert "quantum crystals" in hall or "moon" in hall


def test_router_graph_ladder():
    from infra.routing.graph import RouterGraph

    assert RouterGraph.TIER_LADDER == ["cheap", "standard", "frontier"]
    assert RouterGraph._can_escalate("cheap")
    assert RouterGraph._can_escalate("standard")
    assert not RouterGraph._can_escalate("frontier")


def test_router_graph_compiles_with_stub_nodes():
    from infra.routing.graph import RouterGraph

    calls: list = []

    def node_a(s: RouteState) -> None:
        calls.append("a")
        s.complexity = "simple"

    def node_b(s: RouteState) -> None:
        calls.append("b")
        s.model = "llama-3.3-70b-versatile"

    g = RouterGraph({
        "classify": node_a,
        "cache_lookup": lambda s: None,
        "retrieve": node_b,
        "select_model": lambda s: None,
        "budget_check": lambda s: None,
        "generate": lambda s: None,
        "grade": lambda s: None,
        "escalate": lambda s: None,
        "record": lambda s: None,
    })
    compiled = g.compile()
    out = compiled.invoke(RouteState(query="q"))
    # LangGraph hands back a state dict, not the dataclass — the engine's
    # route() normalises it the same way.
    if isinstance(out, dict):
        out = RouteState(**out)
    assert out.complexity == "simple"
    assert out.model == "llama-3.3-70b-versatile"


# ── integration tests (live stack, dry-run LLM) ─────────────────────

@pytest.fixture(scope="module")
def engine():
    if not INTEGRATION:
        pytest.skip("integration test — set BIFROST_INTEGRATION=1")
    from infra.routing.router import RouteEngine

    e = RouteEngine(force_dry_run=True, use_cache=False)
    e.route("warm up pipeline")
    return e


def test_intg_policy_mapping(engine):
    # a clearly-simple query must land on the cheapest model per policy
    s = engine.route("How do I reset my password?")
    assert s.status == "ok"
    assert s.model == engine.policy["model_mapping"]["simple"] or \
        engine.policy["model_mapping"]["simple"] in (s.model,)


def test_intg_escalation_bounded(engine):
    s = engine.route("fabricate an answer about payroll __HALLUCINATE__")
    assert s.did_escalate
    assert 0 < s.escalation_count <= engine.policy["max_escalation_hops"]
    assert s.tier == "frontier"


def test_intg_cache_short_circuit():
    if not INTEGRATION:
        pytest.skip("integration test — set BIFROST_INTEGRATION=1")
    from infra.routing.router import RouteEngine

    e = RouteEngine(force_dry_run=True)
    q = "What compensation do we get when the service is down?"
    first = e.route(q)
    assert first.status == "ok"
    second = e.route(q)
    assert second.cache_hit
    assert second.cache_similarity >= 0.92
    assert second.est_cost_usd == 0.0


def test_intg_trace_written_with_correct_cost(engine):
    from infra.data_generation.trace_factory import model_cost_usd

    s = engine.route("How do I invite a teammate?")
    assert s.trace_written
    items = engine._dynamo.scan()["Items"]
    item = next(i for i in items if i.get("trace_id") == s.trace_id)
    expected = model_cost_usd(s.model, int(item["prompt_tokens"]),
                              int(item["completion_tokens"]))
    assert abs(float(item["cost_usd"]) - expected) < 1e-6
    assert item["complexity"] == s.complexity
    assert item["status"] == "ok"


def test_intg_budget_not_refused_for_live_team(engine):
    s = engine.route("What shipping options are available?", team="eng-core")
    assert not s.budget_refused
    assert s.status == "ok"


def test_intg_economics_pays_for_itself(engine):
    from infra.data_generation.trace_factory import generate_query_traces

    traces = generate_query_traces(count=30, seed=11)
    routed = baseline = 0.0
    for t in traces:
        s = engine.route(t["query_text"])
        routed += s.est_cost_usd
        baseline += s.frontier_baseline_usd
    saving = 1.0 - routed / baseline
    assert saving >= 0.30, f"savings only {saving*100:.1f}%"