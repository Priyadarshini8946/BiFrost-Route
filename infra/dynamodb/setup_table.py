"""Provision the `query_traces` table in DynamoDB (Local).

Design (mirrors the Layer-1 plan):
  Partition key  trace_id       (S)  — one item per routed query
  Sort key       timestamp      (S)  — ISO-8601 UTC (lexicographically sortable)
  Billing        PAY_PER_REQUEST    — the modern on-demand equivalent of the
                                      free-tier 25 WCU/RCU story
  GSIs           escalation-index   (did_escalate → timestamp)  ← "escalations
                                      in the last hour" demo query
                 model-index        (selected_model → timestamp)
                 complexity-index   (complexity → timestamp)
  TTL            ttl_epoch (unix)   — trace retention (90 days)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from infra.dynamodb.client import dynamodb_client, table_name  # noqa: E402

TRACE_TTL_DAYS = 90


def create_query_traces_table(force_recreate: bool = False) -> str:
    client = dynamodb_client()
    name = table_name()

    if force_recreate:
        try:
            client.delete_table(TableName=name)
            client.get_waiter("table_not_exists").wait(TableName=name)
            print(f"[dynamodb] recreated table {name}")
        except client.exceptions.ResourceNotFoundException:
            pass

    existing = client.list_tables()["TableNames"]  # list of plain strings
    if name not in existing:
        client.create_table(
            TableName=name,
            AttributeDefinitions=[
                {"AttributeName": "trace_id", "AttributeType": "S"},
                {"AttributeName": "timestamp", "AttributeType": "S"},
                {"AttributeName": "did_escalate", "AttributeType": "S"},
                {"AttributeName": "selected_model", "AttributeType": "S"},
                {"AttributeName": "complexity", "AttributeType": "S"},
            ],
            KeySchema=[
                {"AttributeName": "trace_id", "KeyType": "HASH"},
                {"AttributeName": "timestamp", "KeyType": "RANGE"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "escalation-index",
                    "KeySchema": [
                        {"AttributeName": "did_escalate", "KeyType": "HASH"},
                        {"AttributeName": "timestamp", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                },
                {
                    "IndexName": "model-index",
                    "KeySchema": [
                        {"AttributeName": "selected_model", "KeyType": "HASH"},
                        {"AttributeName": "timestamp", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                },
                {
                    "IndexName": "complexity-index",
                    "KeySchema": [
                        {"AttributeName": "complexity", "KeyType": "HASH"},
                        {"AttributeName": "timestamp", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                },
            ],
            BillingMode="PAY_PER_REQUEST",
            SSESpecification={"Enabled": False},
            Tags=[{"Key": "project", "Value": "bifrost-route"}, {"Key": "layer", "Value": "1"}],
        )
        print(f"[dynamodb] create_table issued for {name}")

    client.get_waiter("table_exists").wait(TableName=name)
    # TTL configuration for trace retention (apply after table is active).
    try:
        ttl_status = client.describe_time_to_live(TableName=name)[
            "TimeToLiveDescription"
        ]["TimeToLiveStatus"]
    except client.exceptions.ResourceNotFoundException:
        ttl_status = "DISABLED"
    if ttl_status != "ENABLED":
        client.update_time_to_live(
            TableName=name,
            TimeToLiveSpecification={"Enabled": True, "AttributeName": "ttl_epoch"},
        )

    while True:
        status = client.describe_table(TableName=name)["Table"]["TableStatus"]
        if status == "ACTIVE":
            break
        time.sleep(2)
    return status


if __name__ == "__main__":
    print(f"[dynamodb] table status: {create_query_traces_table()}")