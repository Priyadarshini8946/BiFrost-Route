"""Seed `query_traces` with 100 production-shaped traces and run the plan's
demo queries ("escalations in the last hour" via GSI, cost breakdown by tier).
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from boto3.dynamodb.conditions import Attr, Key  # noqa: E402

from infra.data_generation.trace_factory import generate_query_traces  # noqa: E402
from infra.dynamodb.client import dynamodb_resource, table_name  # noqa: E402

TRACE_TTL_DAYS = 90


def _to_item(trace: Dict[str, Any]) -> Dict[str, Any]:
    item = {
        "trace_id": trace["trace_id"],
        "timestamp": trace["timestamp"],
        "query_text": trace["query_text"],
        "complexity": trace["complexity"],
        "selected_model": trace["selected_model"],
        "provider": trace["provider"],
        "tier": trace["tier"],
        "final_model": trace["final_model"],
        "prompt_tokens": trace["prompt_tokens"],
        "completion_tokens": trace["completion_tokens"],
        "cost_usd": str(trace["cost_usd"]),
        "frontier_baseline_cost_usd": str(trace["frontier_baseline_cost_usd"]),
        "rag_faithfulness": str(trace["rag_faithfulness"]),
        "did_escalate": "true" if trace["did_escalate"] else "false",
        "escalation_count": trace["escalation_count"],
        "latency_ms": trace["latency_ms"],
        "cached": "false",
        "status": "ok",
        "ttl_epoch": int(time.time()) + TRACE_TTL_DAYS * 86400,
    }
    return item


def seed_traces(count: int = 100, reseed: bool = False) -> List[Dict[str, Any]]:
    """Batch-write `count` traces; returns the full item list as written."""
    table = dynamodb_resource().Table(table_name())

    if not reseed:
        existing = table.scan(Select="COUNT")["Count"]
        if existing > 0:
            return [t for t in table.scan()["Items"]]

    traces = generate_query_traces(count=count)
    items = [_to_item(t) for t in traces]
    with table.batch_writer() as batch:
        for item in items:
            batch.put_item(Item=item)
    return items


def demo_escalations_last_hour() -> List[Dict[str, Any]]:
    """The plan's demo query: all escalations in the last hour (GSI query)."""
    table = dynamodb_resource().Table(table_name())
    now = datetime.now(timezone.utc)
    since = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    resp = table.query(
        IndexName="escalation-index",
        KeyConditionExpression=Key("did_escalate").eq("true") & Key("timestamp").gte(since),
    )
    return resp["Items"]


def scan_all() -> List[Dict[str, Any]]:
    return dynamodb_resource().Table(table_name()).scan()["Items"]


if __name__ == "__main__":
    items = seed_traces(reseed=False)
    print(f"[dynamodb] seeded {len(items)} traces into {table_name()}")
    escalations = demo_escalations_last_hour()
    print(f"[dynamodb] escalations in the last hour (GSI): {len(escalations)}")
    for e in escalations:
        print(
            f"  {e['trace_id']}  {e['complexity']:8s} {e['selected_model']:22s} "
            f"-> {e['final_model']:18s} faith={e['rag_faithfulness']}"
        )