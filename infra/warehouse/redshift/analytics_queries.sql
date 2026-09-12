-- ============================================================================
-- Bifrost Route · Layer 1 · Redshift analytics queries
-- Mirrors infra/warehouse/duckdb_warehouse.py — identical shape, Redshift SQL.
-- ============================================================================

-- [1] ACCEPTANCE QUERY — total cost saved today (< 2 s SLA)
WITH today AS (
    SELECT COALESCE(SUM(cost_usd), 0)                   AS actual_cost_usd,
           COALESCE(SUM(frontier_baseline_cost_usd), 0) AS frontier_baseline_usd,
           COALESCE(SUM(cost_saved_usd), 0)             AS cost_saved_usd
    FROM fact_query_cost
    WHERE trace_date = CURRENT_DATE
)
SELECT actual_cost_usd,
       frontier_baseline_usd,
       cost_saved_usd,
       ROUND(100.0 * cost_saved_usd / NULLIF(frontier_baseline_usd, 0), 2) AS saved_pct
FROM today;

-- [2] Cost breakdown by model tier (Layer 1 demo)
SELECT tier,
       COUNT(*)                                   AS queries,
       ROUND(SUM(cost_usd), 4)                    AS cost_usd,
       ROUND(AVG(rag_faithfulness), 4)            AS avg_faithfulness,
       SUM(did_escalate)                          AS escalations
FROM fact_query_cost
GROUP BY tier
ORDER BY cost_usd DESC;

-- [3] Escalations in the last hour (source of truth is DynamoDB GSI;
--     this is the warehouse-side rollup)
SELECT COUNT(*) AS escalations_last_hour
FROM fact_query_cost
WHERE did_escalate = 1
  AND ts >= DATEADD(hour, -1, GETDATE());

-- [4] Daily savings series (dashboard input for Layer 7)
SELECT trace_date,
       SUM(cost_saved_usd) AS cost_saved_usd,
       ROUND(100.0 * SUM(cost_saved_usd) / NULLIF(SUM(frontier_baseline_cost_usd), 0), 2) AS saved_pct
FROM fact_query_cost
GROUP BY trace_date
ORDER BY trace_date;