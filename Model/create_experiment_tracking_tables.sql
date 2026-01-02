-- ==========================================
-- Experiment Tracking Schema
-- Purpose: 可追溯、可復現、可稽核的實驗記錄系統
-- Created: 2025-12-31
-- ==========================================

-- =========================
-- 1) experiment_runs
-- =========================
CREATE TABLE IF NOT EXISTS experiment_runs (
  run_id              TEXT PRIMARY KEY,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at         TIMESTAMPTZ,
  status              TEXT NOT NULL DEFAULT 'running'
    CHECK (status IN ('running','completed','failed')),

  -- Git & execution context
  git_commit          TEXT NOT NULL,
  git_branch          TEXT,
  git_dirty           BOOLEAN DEFAULT FALSE,
  git_diff_hash       TEXT,
  runner              TEXT,

  -- Command & config snapshot
  command             TEXT NOT NULL,
  config_json         JSONB NOT NULL,

  -- Data & label definition
  date_cutoff         DATE NOT NULL,
  label_strategy      TEXT NOT NULL
    CHECK (label_strategy IN ('absolute','hybrid','multiclass','custom')),
  label_params        JSONB,

  -- Split / CV definition
  split_strategy      TEXT NOT NULL
    CHECK (split_strategy IN (
      'holdout',
      'stratified_holdout',
      'group_holdout',
      'kfold',
      'stratified_kfold',
      'group_kfold',
      'timeseries_split',
      'custom'
    )),
  cv_params           JSONB,

  -- Preprocess leakage control
  preprocess_fit_scope TEXT
    CHECK (preprocess_fit_scope IN ('train_fold_only','train_only','full_train','custom')),

  -- Pipeline versioning / fingerprint
  pipeline_version    TEXT,
  code_fingerprint_hash TEXT,

  -- Feature versioning
  feature_set         TEXT NOT NULL,
  feature_hash        TEXT,

  -- Dataset fingerprint
  dataset_hash        TEXT NOT NULL,
  split_hash          TEXT NOT NULL,

  -- Model info
  model_type          TEXT NOT NULL
    CHECK (model_type IN ('xgboost','lightgbm','catboost','sklearn','custom')),
  model_params        JSONB,

  -- Metrics summary
  metrics_json        JSONB,

  -- Experiment grouping
  group_id            TEXT,
  parent_run_id       TEXT,

  -- Notes
  conclusion          TEXT,
  error_log           TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_group   ON experiment_runs(group_id);
CREATE INDEX IF NOT EXISTS idx_runs_commit  ON experiment_runs(git_commit);
CREATE INDEX IF NOT EXISTS idx_runs_status  ON experiment_runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_cutoff  ON experiment_runs(date_cutoff);
CREATE INDEX IF NOT EXISTS idx_runs_created ON experiment_runs(created_at DESC);

-- =========================
-- 2) experiment_samples
-- =========================
CREATE TABLE IF NOT EXISTS experiment_samples (
  run_id            TEXT NOT NULL,
  product_id        BIGINT NOT NULL,
  keyword           TEXT,

  y_true            SMALLINT NOT NULL CHECK (y_true IN (0, 1)),
  split             TEXT NOT NULL CHECK (split IN ('train_pool','test')),
  fold              INT NOT NULL DEFAULT -1,

  -- filter / inclusion semantics
  is_excluded        BOOLEAN NOT NULL DEFAULT FALSE,
  exclusion_reason   TEXT,
  is_included        BOOLEAN NOT NULL DEFAULT TRUE,

  PRIMARY KEY (run_id, product_id),
  FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_samples_split   ON experiment_samples(run_id, split, fold);
CREATE INDEX IF NOT EXISTS idx_samples_keyword ON experiment_samples(run_id, keyword);
CREATE INDEX IF NOT EXISTS idx_samples_incl    ON experiment_samples(run_id, is_included);

-- =========================
-- 3) experiment_predictions
-- =========================
CREATE TABLE IF NOT EXISTS experiment_predictions (
  run_id            TEXT NOT NULL,
  product_id        BIGINT NOT NULL,

  y_true            SMALLINT NOT NULL CHECK (y_true IN (0, 1)),
  y_prob            DOUBLE PRECISION NOT NULL,
  y_pred            SMALLINT NOT NULL CHECK (y_pred IN (0, 1)),

  split             TEXT NOT NULL CHECK (split IN ('train_pool','test')),
  fold              INT NOT NULL DEFAULT -1,
  threshold         DOUBLE PRECISION NOT NULL DEFAULT 0.5,

  PRIMARY KEY (run_id, product_id, split, fold),
  FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_preds_errors
  ON experiment_predictions(run_id, split, y_true, y_pred);

CREATE INDEX IF NOT EXISTS idx_preds_prob
  ON experiment_predictions(run_id, split, fold, y_prob DESC);

-- =========================
-- 4) experiment_features
-- =========================
CREATE TABLE IF NOT EXISTS experiment_features (
  run_id          TEXT NOT NULL,
  feature_name    TEXT NOT NULL,
  is_active       BOOLEAN NOT NULL,
  importance      DOUBLE PRECISION,
  meta_json       JSONB,

  PRIMARY KEY (run_id, feature_name),
  FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_features_active
  ON experiment_features(run_id, is_active);

-- =========================
-- 5) experiment_artifacts
-- =========================
CREATE TABLE IF NOT EXISTS experiment_artifacts (
  run_id          TEXT NOT NULL,
  artifact_type   TEXT NOT NULL,
  file_path       TEXT NOT NULL,
  file_hash       TEXT,
  meta_json       JSONB,

  PRIMARY KEY (run_id, artifact_type),
  FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_artifacts_run
  ON experiment_artifacts(run_id);

-- =========================
-- Comments for documentation
-- =========================
COMMENT ON TABLE experiment_runs IS '實驗執行記錄主表：每次 run 的完整 metadata';
COMMENT ON TABLE experiment_samples IS '實驗使用的樣本集：含 y_true、split、是否排除等';
COMMENT ON TABLE experiment_predictions IS '實驗預測結果：y_prob、y_pred、threshold';
COMMENT ON TABLE experiment_features IS '實驗使用的特徵清單與 importance';
COMMENT ON TABLE experiment_artifacts IS '實驗產出的檔案清單：model、vocab、report 等';

COMMENT ON COLUMN experiment_runs.git_dirty IS '代碼是否有未提交修改';
COMMENT ON COLUMN experiment_runs.git_diff_hash IS '若 dirty，則為 diff 的 hash';
COMMENT ON COLUMN experiment_runs.preprocess_fit_scope IS '前處理 fit 的範圍（避免 data leakage）';
COMMENT ON COLUMN experiment_runs.code_fingerprint_hash IS '關鍵代碼檔案的 fingerprint';

COMMENT ON COLUMN experiment_samples.is_excluded IS '是否被過濾器排除';
COMMENT ON COLUMN experiment_samples.is_included IS '是否納入 metrics 計算範圍';
