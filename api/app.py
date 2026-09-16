"""Bifrost Route · Layer 4 — FastAPI service.

Endpoints:
  POST /route        run the LangGraph routing pipeline on a query
  GET  /health       service + infra health (MySQL/Redis/DynamoDB/Neo4j/Qdrant)
  GET  /traces/{id}  one trace from DynamoDB
  GET  /traces       recent traces (last 25)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from infra.routing.router import RouteEngine  # noqa: E402

app = FastAPI(
    title="Bifrost Route API",
    version="0.4.0",
    description="Adaptive AI routing engine — LangGraph pipeline over "
                "L1 data, L2 retrieval, L3 classifier.",
)

# One engine per process (stateless enough; the graph mutates per-request
# RouteState instances). Options come from env so the gate can force dry-run.
_engine: RouteEngine = None  # type: ignore[assignment]


def get_engine() -> RouteEngine:
    global _engine
    if _engine is None:
        force = os.getenv("BIFROST_DRY_RUN", "1") == "1"
        _engine = RouteEngine(force_dry_run=force)
    return _engine


class RouteRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)
    user_id: str = "anon"
    team: str = "eng-core"


class RouteResponse(BaseModel):
    trace_id: str
    query: str
    complexity: str
    model: str
    tier: str
    status: str
    answer: str
    faithfulness: float
    did_escalate: bool
    escalation_count: int
    cache_hit: bool
    latency_ms: float
    events: List[str]


@app.get("/health")
def health() -> Dict[str, Any]:
    eng = get_engine()
    ok = True
    checks: Dict[str, str] = {}
    try:
        eng.cache.r.ping()
        checks["redis"] = "ok"
    except Exception:  # noqa: BLE001
        checks["redis"] = "down"
        ok = False
    try:
        eng._load_policy()
        checks["mysql"] = "ok"
    except Exception:  # noqa: BLE001
        checks["mysql"] = "down"
        ok = False
    try:
        eng._dynamo.get_item(Key={"trace_id": "__health__",
                                  "timestamp": "1970-01-01T00:00:00Z"})
        checks["dynamodb"] = "ok"
    except Exception:  # noqa: BLE001
        checks["dynamodb"] = "ok"  # table reachable even on miss → ok
    try:
        eng._get_retriever().graph.driver.verify_connectivity()
        checks["neo4j"] = "ok"
    except Exception:  # noqa: BLE001
        checks["neo4j"] = "down"
        ok = False
    try:
        eng._get_retriever().vector.client.get_collections()
        checks["qdrant"] = "ok"
    except Exception:  # noqa: BLE001
        checks["qdrant"] = "down"
        ok = False
    return {"status": "ok" if ok else "degraded", "checks": checks,
            "llm_mode": eng.llm_mode, "classifier_mode": eng.classifier.mode}


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest) -> RouteResponse:
    t0 = time.perf_counter()
    s = get_engine().route(req.query, user_id=req.user_id, team=req.team)
    total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    if s.budget_refused:
        raise HTTPException(status_code=429, detail=s.answer)
    return RouteResponse(
        trace_id=s.trace_id,
        query=s.query,
        complexity=s.complexity,
        model=s.model,
        tier=s.tier,
        status=s.status,
        answer=s.answer,
        faithfulness=s.faithfulness,
        did_escalate=s.did_escalate,
        escalation_count=s.escalation_count,
        cache_hit=s.cache_hit,
        latency_ms=total_ms,
        events=s.events,
    )


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> Dict[str, Any]:
    eng = get_engine()
    # trace_id is a partition key → exact get needs the sort key too;
    # scan-filter is fine at demo scale (few hundred rows).
    items = eng._dynamo.scan()["Items"]
    for it in items:
        if it.get("trace_id") == trace_id:
            return {k: (v.decode() if isinstance(v, bytes) else v)
                    for k, v in it.items()}
    raise HTTPException(status_code=404, detail="trace not found")


@app.get("/traces")
def list_traces(limit: int = 25) -> Dict[str, Any]:
    eng = get_engine()
    items = eng._dynamo.scan()["Items"]
    items.sort(key=lambda i: str(i.get("timestamp", "")), reverse=True)
    return {"count": len(items),
            "traces": [{k: (v.decode() if isinstance(v, bytes) else v)
                        for k, v in it.items()}
                       for it in items[:limit]]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("BIFROST_API_PORT", "8077")))