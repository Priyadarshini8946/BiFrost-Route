"""MySQL access layer — routing_policies, budget_limits, model_registry.

All Layer 1 scripts go through this module so connection settings stay in one
place (.env) and schema application stays idempotent.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List

import pymysql
from dotenv import load_dotenv

load_dotenv()

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def connect() -> "pymysql.Connection":
    """Open a connection using .env config. Caller is responsible for closing."""
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "bifrost_route"),
        charset="utf8mb4",
        autocommit=True,
        connect_timeout=5,
    )


def apply_schema(max_attempts: int = 30, delay: float = 2.0) -> bool:
    """Apply schema.sql (DDL + idempotent seeds). Retries while MySQL boots."""
    import re

    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    # Strip `--` line comments BEFORE splitting on ';' — comments may legally
    # contain semicolons and would otherwise shred the statement stream.
    sql = re.sub(r"(?m)^\s*--.*$", "", sql)
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        conn = None
        try:
            conn = connect()
            with conn.cursor() as cur:
                for stmt in statements:
                    cur.execute(stmt)
            return True
        except Exception as exc:  # noqa: BLE001 — retry loop
            last_err = exc
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(delay)
    raise RuntimeError(f"MySQL schema apply failed after {max_attempts} attempts: {last_err}")


def table_counts() -> Dict[str, int]:
    """Row counts for the three Layer-1 tables."""
    conn = connect()
    try:
        with conn.cursor() as cur:
            out: Dict[str, int] = {}
            for table in ("model_registry", "routing_policies", "budget_limits"):
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                out[table] = cur.fetchone()[0]
            return out
    finally:
        conn.close()


def demo_queries() -> Dict[str, Any]:
    """Semi-real queries a business back-end would run against the policies DB."""
    conn = connect()
    try:
        with conn.cursor() as cur:
            # Active policy → what model does a "medium" query get today?
            cur.execute(
                """
                SELECT policy_name, version,
                       complexity_threshold_medium, complexity_threshold_complex,
                       JSON_EXTRACT(model_mapping, '$.medium') AS medium_model,
                       JSON_EXTRACT(model_mapping, '$.simple') AS simple_model,
                       faithfulness_threshold, max_escalation_hops, cache_ttl_seconds
                FROM routing_policies
                WHERE is_active = 1
                ORDER BY version DESC LIMIT 1
                """
            )
            policy = cur.fetchone()

            # Budget remaining for eng-core this month.
            cur.execute(
                """
                SELECT team_name, monthly_cap_usd, current_spend_usd,
                       monthly_cap_usd - current_spend_usd AS remaining_usd,
                       window_start, window_end
                FROM budget_limits
                WHERE team_name = 'eng-core' AND is_enforced = 1
                ORDER BY window_start DESC LIMIT 1
                """
            )
            budget = cur.fetchone()

            # Cheapest vs frontier unit economics for the cost story.
            cur.execute(
                "SELECT model_name, tier, cost_per_1k_input_tokens, cost_per_1k_output_tokens "
                "FROM model_registry WHERE is_active = 1 ORDER BY tier"
            )
            models = cur.fetchall()
            return {"policy": policy, "budget": budget, "models": models}
    finally:
        conn.close()