-- DDL for ml_data_filters table
-- Purpose: Store exclusion lists (Soft Delete) for ML training with versioning support.

DROP TABLE IF EXISTS ml_data_filters CASCADE;

CREATE TABLE ml_data_filters (
    id SERIAL PRIMARY KEY,
    
    -- Versioning for A/B testing different cleaning strategies
    -- e.g., 'v1_correlation_strict', 'v2_error_analysis'
    version_tag VARCHAR(50) NOT NULL,
    
    -- Entity type being filtered
    -- e.g., 'keyword', 'product_id'
    filter_level VARCHAR(20) NOT NULL,
    
    -- The value to exclude. 
    -- Defined as TEXT to handle both Keywords (strings) and Product IDs (integers converted to string).
    filter_value TEXT NOT NULL,
    
    -- Logic/Metric used for exclusion
    -- e.g., 'low_correlation', 'prediction_error'
    reason VARCHAR(50),
    
    -- The metric value that triggered exclusion
    -- e.g., 0.02 (correlation score)
    score FLOAT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraint: Ensure unique exclusion items per version
    -- Prevents 'Tissue' from appearing twice in 'v1_correlation'
    CONSTRAINT uq_ml_filters_version_item UNIQUE (version_tag, filter_level, filter_value)
);

-- Index: Optimize Frequent Queries by Version
-- We will mostly select * where version_tag = '...'
CREATE INDEX idx_ml_filters_version ON ml_data_filters(version_tag);

-- Index: Optimize lookups by value (e.g. checking if a specific product is filtered)
CREATE INDEX idx_ml_filters_value ON ml_data_filters(filter_level, filter_value);
