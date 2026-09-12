-- ============================================================================
-- Bifrost Route · Layer 1 · Redshift warehouse schema (AWS migration target)
-- Locally the same model runs on DuckDB (infra/warehouse/duckdb_warehouse.py);
-- these DDL statements are the COPY-able/SELECT-able Redshift equivalents.
-- ============================================================================

CREATE TABLE IF NOT EXISTS dim_model (
    model_name                VARCHAR(128) NOT NULL,
    provider                  VARCHAR(64)  NOT NULL,
    tier                      VARCHAR(16)  NOT NULL,
    cost_per_1k_input_tokens  DECIMAL(10,6) NOT NULL,
    cost_per_1k_output_tokens DECIMAL(10,6) NOT NULL,
    PRIMARY KEY (model_name)
)
DISTSTYLE ALL
SORTKEY (tier);

CREATE TABLE IF NOT EXISTS fact_query_cost (
    trace_date                 DATE          NOT NULL,
    ts                         TIMESTAMP     NOT NULL,
    trace_id                   VARCHAR(128)  NOT NULL,
    team                       VARCHAR(128),
    complexity                 VARCHAR(16),
    selected_model             VARCHAR(128),
    final_model                VARCHAR(128),
    provider                   VARCHAR(64),
    tier                       VARCHAR(16),
    input_tokens               BIGINT,
    output_tokens              BIGINT,
    cost_usd                   DECIMAL(12,6),
    frontier_baseline_cost_usd DECIMAL(12,6),
    cost_saved_usd             DECIMAL(12,6),
    rag_faithfulness           DECIMAL(6,4),
    did_escalate               SMALLINT,
    escalation_count           SMALLINT,
    latency_ms                 BIGINT,
    cache_hit                  SMALLINT
)
DISTKEY (trace_id)
SORTKEY (trace_date, ts);

-- ELT: from the DynamoDB export (traces.parquet) into Redshift.
-- COPY traces FROM 's3://bifrost-route-exports/traces.parquet'
-- IAM_ROLE 'arn:aws:iam::<acct>:role/bifrost-redshift-copy'
-- FORMAT AS PARQUET;