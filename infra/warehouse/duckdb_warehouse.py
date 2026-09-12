"""DuckDB analytics warehouse — the local stand-in for Redshift.

Layer 1 acceptance query: "total cost saved today" must return in < 2 s.

The warehouse ingests the DynamoDB `query_traces`, derives the frontier-only
baseline (`what would every query have cost on the most expensive model`),
and materializes `cost_saved_usd = frontier_baseline - actual`. The same SQL
is mirrored in infra/warehouse/redshift/ for the AWS migration (Layer 6+).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
from dotenv import load_dotenv

from infra.data_generation.trace_factory import (
    FRONTIER_MODEL,
    MODEL_COSTS,
    frontier_baseline_cost_usd,
)

load_dotenv()

WAREHOUSE_PATH = Path(os.getenv("WAREHOUSE_PATH", "./data/warehouse.duckdb"))
PARQUET_PATH = Path(os.getenv("WAREHOUSE_TRACES_PARQUET", "./data/traces.parquet"))

SCHEMA_SQL = """
CREATE OR REPLACE TABLE dim_model (
    model_name               VARCHAR PRIMARY KEY,
    provider                 VARCHAR,
    tier                     VARCHAR,
    cost_per_1k_input_tokens DOUBLE,
    cost_per_1k_output_tokens DOUBLE
);

CREATE OR REPLACE TABLE fact_query_cost (
    trace_date                 DATE,
    ts                         TIMESTAMP,
    trace_id                   VARCHAR,
    team                       VARCHAR,
    complexity                 VARCHAR,
    selected_model             VARCHAR,
    final_model                VARCHAR,
    provider                   VARCHAR,
    tier                       VARCHAR,
    input_tokens               BIGINT,
    output_tokens              BIGINT,
    cost_usd                   DOUBLE,
    frontier_baseline_cost_usd DOUBLE,
    cost_saved_usd             DOUBLE,
    rag_faithfulness           DOUBLE,
    did_escalate               BOOLEAN,
    escalation_count           BIGINT,
    latency_ms                 BIGINT,
    cache_hit                  BOOLEAN
);

CREATE OR REPLACE VIEW v_daily_cost_savings AS
SELECT trace_date,
       SUM(cost_usd)                   AS actual_cost_usd,
       SUM(frontier_baseline_cost_usd) AS frontier_baseline_usd,
       SUM(cost_saved_usd)             AS cost_saved_usd,
       ROUND(100.0 * SUM(cost_saved_usd) / NULLIF(SUM(frontier_baseline_cost_usd), 0), 2)
                                       AS saved_pct
FROM fact_query_cost
GROUP BY trace_date
ORDER BY trace_date;
"""


class DuckDBWarehouse:
    def __init__(self, path: Optional[Path] = None, read_only: bool = False) -> None:
        self.path = path or WAREHOUSE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path), read_only=read_only)

    def close(self) -> None:
        self.con.close()

    # ── DDL ─────────────────────────────────────────────────────────────
    def init_schema(self) -> None:
        self.con.execute(SCHEMA_SQL)
        rows = [(m, c["provider"], c["tier"], c["in"], c["out"])
                for m, c in MODEL_COSTS.items()]
        self.con.executemany(
            "INSERT OR REPLACE INTO dim_model VALUES (?, ?, ?, ?, ?)", rows
        )

    # ── ELT ─────────────────────────────────────────────────────────────
    def load_traces(self, traces: List[Dict[str, Any]]) -> int:
        """Insert traces with derived cost_saved vs frontier-only baseline."""
        rows = []
        for t in traces:
            pt, ct = int(t["prompt_tokens"]), int(t["completion_tokens"])
            actual = float(t["cost_usd"])
            baseline = frontier_baseline_cost_usd(pt, ct)
            # ISO-8601 UTC strings; DuckDB casts them to TIMESTAMP natively.
            ts_str = str(t["timestamp"]).replace("Z", "")
            trace_date = ts_str[:10]  # UTC date — matches "today" everywhere
            did_esc = str(t.get("did_escalate", "false")).lower() in ("true", "1")
            rows.append((
                trace_date,
                ts_str,
                t["trace_id"],
                t.get("team", "eng-core"),
                t.get("complexity", "simple"),
                t.get("selected_model", "llama-3.3-70b-versatile"),
                t.get("final_model", t.get("selected_model", "")),
                MODEL_COSTS.get(t.get("selected_model", ""), {}).get("provider", "?"),
                MODEL_COSTS.get(t.get("selected_model", ""), {}).get("tier", "?"),
                pt, ct,
                actual,
                baseline,
                max(baseline - actual, 0.0),
                float(t.get("rag_faithfulness", 0.0)),
                did_esc,
                int(t.get("escalation_count", 0)),
                int(t.get("latency_ms", 0)),
                str(t.get("cached", "false")).lower() in ("true", "1"),
            ))
        self.con.executemany("INSERT INTO fact_query_cost VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        return len(rows)

    def export_parquet(self, path: Optional[Path] = None) -> Path:
        out = path or PARQUET_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        self.con.execute(f"COPY (SELECT * FROM fact_query_cost ORDER BY ts) TO '{out.as_posix()}' (FORMAT PARQUET)")
        return out

    # ── Analytics queries (the plan's Redshift workloads, on DuckDB) ────
    def total_cost_saved_for(self, day: Optional[str] = None) -> Tuple[float, Dict[str, Any]]:
        """THE Layer-1 acceptance query: total cost saved today (< 2 s)."""
        if day is None:
            from datetime import datetime, timezone

            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = time.perf_counter()
        row = self.con.execute(
            "SELECT actual_cost_usd, frontier_baseline_usd, cost_saved_usd, saved_pct "
            "FROM v_daily_cost_savings WHERE trace_date = ?",
            [day],
        ).fetchone()
        elapsed = time.perf_counter() - start
        if row is None:
            row = (0.0, 0.0, 0.0, 0.0)
        return elapsed, {
            "day": day,
            "actual_cost_usd": round(row[0], 4),
            "frontier_baseline_usd": round(row[1], 4),
            "cost_saved_usd": round(row[2], 4),
            "saved_pct": row[3],
        }

    def cost_breakdown_by_tier(self) -> List[Dict[str, Any]]:
        rows = self.con.execute(
            """
            SELECT tier, COUNT(*) AS queries, ROUND(SUM(cost_usd), 4) AS cost_usd,
                   ROUND(AVG(rag_faithfulness), 4) AS avg_faithfulness,
                   SUM(CASE WHEN did_escalate THEN 1 ELSE 0 END) AS escalations
            FROM fact_query_cost GROUP BY tier ORDER BY cost_usd DESC
            """
        ).fetchall()
        return [{"tier": r[0], "queries": r[1], "cost_usd": r[2],
                 "avg_faithfulness": r[3], "escalations": r[4]} for r in rows]

    def escalation_summary_last_hour(self) -> Dict[str, int]:
        row = self.con.execute(
            """
            SELECT COUNT(*) FILTER (WHERE did_escalate),
                   COUNT(*) FILTER (WHERE did_escalate AND ts >= now() - INTERVAL '1 hour')
            FROM fact_query_cost
            """
        ).fetchone()
        return {"total_escalations": row[0], "last_hour": row[1]}

    def daily_savings_series(self) -> List[Dict[str, Any]]:
        rows = self.con.execute(
            "SELECT trace_date, cost_saved_usd, saved_pct FROM v_daily_cost_savings ORDER BY trace_date"
        ).fetchall()
        return [{"day": str(r[0]), "cost_saved_usd": round(r[1], 4), "saved_pct": r[2]} for r in rows]