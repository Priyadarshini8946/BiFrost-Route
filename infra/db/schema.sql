-- ============================================================================
-- Bifrost Route · Layer 1 (Data Layer) · MySQL schema
-- Tables: model_registry · routing_policies · budget_limits
-- Idempotent: safe to re-apply (CREATE IF NOT EXISTS / INSERT IGNORE).
-- ============================================================================

CREATE DATABASE IF NOT EXISTS bifrost_route
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE bifrost_route;

-- ---------------------------------------------------------------------------
-- model_registry — the unit-economics catalog. Every model the router may
-- select, with provider, tier, and cost per 1K tokens (USD).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_registry (
    id                        INT UNSIGNED    NOT NULL AUTO_INCREMENT,
    model_name                VARCHAR(128)    NOT NULL,
    provider                  VARCHAR(64)     NOT NULL,
    tier                      ENUM('cheap','standard','frontier') NOT NULL,
    cost_per_1k_input_tokens  DECIMAL(10,6)   NOT NULL DEFAULT 0,
    cost_per_1k_output_tokens DECIMAL(10,6)   NOT NULL DEFAULT 0,
    max_context_tokens        INT UNSIGNED    NOT NULL DEFAULT 8192,
    is_active                 TINYINT(1)      NOT NULL DEFAULT 1,
    created_at                TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_model_name (model_name),
    KEY idx_tier (tier)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COMMENT='Model catalog with provider + tier + unit economics (USD / 1K tokens)';

-- ---------------------------------------------------------------------------
-- routing_policies — complexity thresholds, model mappings, quality gates.
-- model_mapping is JSON: {"simple": "<model>", "medium": "<model>", "complex": "<model>"}
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS routing_policies (
    id                          INT UNSIGNED  NOT NULL AUTO_INCREMENT,
    policy_name                 VARCHAR(128)  NOT NULL,
    description                 VARCHAR(512)  NULL,
    version                     INT UNSIGNED  NOT NULL DEFAULT 1,
    complexity_threshold_medium DECIMAL(4,3)  NOT NULL DEFAULT 0.500,
    complexity_threshold_complex DECIMAL(4,3) NOT NULL DEFAULT 0.800,
    model_mapping               JSON          NOT NULL,
    faithfulness_threshold      DECIMAL(4,3)  NOT NULL DEFAULT 0.850,
    max_escalation_hops         INT UNSIGNED  NOT NULL DEFAULT 2,
    cache_ttl_seconds           INT UNSIGNED  NOT NULL DEFAULT 3600,
    is_active                   TINYINT(1)    NOT NULL DEFAULT 1,
    created_at                  TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at                  TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP
                                          ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_policy_version (policy_name, version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COMMENT='Routing policy: complexity thresholds, model config, RAGAS gate, escalation';

-- ---------------------------------------------------------------------------
-- budget_limits — team-level spend caps (Layer 6 will enforce these).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS budget_limits (
    id                 INT UNSIGNED  NOT NULL AUTO_INCREMENT,
    team_name          VARCHAR(128)  NOT NULL,
    monthly_cap_usd    DECIMAL(12,4) NOT NULL,
    current_spend_usd  DECIMAL(12,4) NOT NULL DEFAULT 0,
    window_start       DATE          NOT NULL,
    window_end         DATE          NOT NULL,
    currency           CHAR(3)       NOT NULL DEFAULT 'USD',
    is_enforced        TINYINT(1)    NOT NULL DEFAULT 1,
    created_at         TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at         TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP
                                          ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_team_window (team_name, window_start, window_end)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COMMENT='Team-level spend caps — router must not exceed when enforced';

-- ============================================================================
-- SEED DATA (idempotent)
-- ============================================================================

-- Models. Prices are indicative 2026 list prices in USD per 1K tokens
-- (input/output). Re-sync with provider pricing pages before production.
INSERT IGNORE INTO model_registry
    (model_name, provider, tier, cost_per_1k_input_tokens, cost_per_1k_output_tokens, max_context_tokens, is_active)
VALUES
    ('llama-3.3-70b-versatile', 'groq',   'cheap',     0.000590, 0.000790, 131072, 1),
    ('gemini-2.0-flash',        'google', 'standard',  0.000100, 0.000400, 1048576, 1),
    ('gemini-2.5-pro',          'google', 'frontier',  0.001250, 0.010000, 1048576, 1);

-- Default routing policy (v1): simple → Groq Llama 70B, medium → Gemini Flash,
-- complex → Gemini Pro. Quality gate: faithfulness ≥ 0.85, max 2 escalation hops.
INSERT IGNORE INTO routing_policies
    (policy_name, description, version, complexity_threshold_medium,
     complexity_threshold_complex, model_mapping, faithfulness_threshold,
     max_escalation_hops, cache_ttl_seconds, is_active)
VALUES (
    'default-routing-v1',
    'Simple→Groq Llama 3.3 70B (cheap) · Medium→Gemini Flash · Complex→Gemini Pro (frontier)',
    1, 0.500, 0.800,
    JSON_OBJECT('simple','llama-3.3-70b-versatile',
                'medium','gemini-2.0-flash',
                'complex','gemini-2.5-pro'),
    0.850, 2, 3600, 1
);

-- Team budgets for the current calendar month. Spend pre-seeded so the
-- "budget remaining" demo query is meaningful; layer 6 resets/enforces live.
INSERT IGNORE INTO budget_limits
    (team_name, monthly_cap_usd, current_spend_usd, window_start, window_end, currency, is_enforced)
VALUES (
    'eng-core', 1000.0000, 742.3100,
    DATE_FORMAT(CURDATE(), '%Y-%m-01'), LAST_DAY(CURDATE()), 'USD', 1
);