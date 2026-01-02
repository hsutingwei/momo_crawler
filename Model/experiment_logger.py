# -*- coding: utf-8 -*-
"""
experiment_logger.py
實驗資料庫記錄工具：寫入 experiment_runs, experiment_samples, experiment_predictions 等表
(Experiment Database Logger: Writes to experiment_runs, experiment_samples, experiment_predictions, etc.)

實現規格參考：implementation_plan.md Phase 5
"""

import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


# =============================================================================
# Database Connection
# =============================================================================

def get_db_connection():
    """
    獲取 PostgreSQL 連接
    
    環境變數：DATABASE_URL 或使用 config.database
    """
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        return psycopg2.connect(dsn)
    
    # Fallback: 使用專案的 database config
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config.database import DatabaseConfig
        
        db_config = DatabaseConfig()
        return db_config.get_connection()
    except Exception as e:
        raise RuntimeError(f"無法連接至資料庫: {e}")


# =============================================================================
# Experiment Runs Management
# =============================================================================

def insert_run_start(
    conn,
    run_id: str,
    git_commit: str,
    git_branch: Optional[str],
    git_dirty: bool,
    runner: Optional[str],
    command: str,
    config: Dict[str, Any],
    date_cutoff: str,
    label_strategy: str,
    label_params: Dict[str, Any],
    split_strategy: str,
    cv_params: Dict[str, Any],
    preprocess_fit_scope: str,
    pipeline_version: Optional[str],
    code_fingerprint_hash: Optional[str],
    feature_set: str,
    model_type: str,
    model_params: Dict[str, Any],
) -> None:
    """
    插入 run 開始記錄（status='running'）
    
    規格：implementation_plan.md #5.1
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO experiment_runs (
              run_id, status, created_at,
              git_commit, git_branch, git_dirty, runner,
              command, config_json,
              date_cutoff, label_strategy, label_params,
              split_strategy, cv_params,
              preprocess_fit_scope,
              pipeline_version, code_fingerprint_hash,
              feature_set,
              model_type, model_params,
              dataset_hash, split_hash
            )
            VALUES (
              %s, 'running', now(),
              %s, %s, %s, %s,
              %s, %s::jsonb,
              %s, %s, %s::jsonb,
              %s, %s::jsonb,
              %s,
              %s, %s,
              %s,
              %s, %s::jsonb,
              %s, %s
            )
            """,
            (
                run_id,
                git_commit, git_branch, git_dirty, runner,
                command, json.dumps(config, ensure_ascii=False),
                date_cutoff, label_strategy, json.dumps(label_params, ensure_ascii=False),
                split_strategy, json.dumps(cv_params, ensure_ascii=False),
                preprocess_fit_scope,
                pipeline_version, code_fingerprint_hash,
                feature_set,
                model_type, json.dumps(model_params, ensure_ascii=False),
                "PENDING", "PENDING",  # Will update later
            ),
        )
    conn.commit()
    print(f"[Experiment Logger] Run {run_id} 已啟動 (status=running)")


def update_run_finish(
    conn,
    run_id: str,
    status: str,  # 'completed' | 'failed'
    dataset_hash: Optional[str] = None,
    split_hash: Optional[str] = None,
    feature_hash: Optional[str] = None,
    metrics_json: Optional[Dict[str, Any]] = None,
    conclusion: Optional[str] = None,
    error_log: Optional[str] = None,
) -> None:
    """
    更新 run 完成狀態
    
    規格：implementation_plan.md #5.1
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE experiment_runs
            SET finished_at = now(),
                status = %s,
                dataset_hash = COALESCE(%s, dataset_hash),
                split_hash   = COALESCE(%s, split_hash),
                feature_hash = COALESCE(%s, feature_hash),
                metrics_json = COALESCE(%s::jsonb, metrics_json),
                conclusion   = COALESCE(%s, conclusion),
                error_log    = COALESCE(%s, error_log)
            WHERE run_id = %s
            """,
            (
                status,
                dataset_hash,
                split_hash,
                feature_hash,
                json.dumps(metrics_json, ensure_ascii=False) if metrics_json else None,
                conclusion,
                error_log,
                run_id,
            ),
        )
    conn.commit()
    print(f"[Experiment Logger] Run {run_id} 已結束 (status={status})")


# =============================================================================
# Samples & Predictions
# =============================================================================

def upsert_samples(conn, run_id: str, samples_df: pd.DataFrame) -> None:
    """
    批次插入/更新樣本記錄
    
    規格：implementation_plan.md #5.1
    
    Expected columns: product_id, keyword, y_true, split, fold, 
                      is_excluded, exclusion_reason, is_included
    """
    df = samples_df.copy()
    df["run_id"] = run_id
    
    # 確保必要欄位存在
    for col in ['fold', 'is_excluded', 'is_included']:
        if col not in df.columns:
            if col == 'fold':
                df['fold'] = -1
            elif col in ['is_excluded', 'is_included']:
                df[col] = False if col == 'is_excluded' else True
    
    df['fold'] = df['fold'].fillna(-1).astype(int)
    
    cols = ["run_id", "product_id", "keyword", "y_true", "split", "fold",
            "is_excluded", "exclusion_reason", "is_included"]
    
    # 填充缺失欄位
    for col in cols:
        if col not in df.columns:
            df[col] = None
    
    rows = [tuple(x) for x in df[cols].itertuples(index=False, name=None)]
    
    with conn.cursor() as cur:
        execute_values(
            cur,
            f"""
            INSERT INTO experiment_samples ({",".join(cols)})
            VALUES %s
            ON CONFLICT (run_id, product_id) DO UPDATE SET
              keyword = EXCLUDED.keyword,
              y_true = EXCLUDED.y_true,
              split = EXCLUDED.split,
              fold = EXCLUDED.fold,
              is_excluded = EXCLUDED.is_excluded,
              exclusion_reason = EXCLUDED.exclusion_reason,
              is_included = EXCLUDED.is_included
            """,
            rows,
            page_size=5000,
        )
    conn.commit()
    print(f"[Experiment Logger] 已插入 {len(rows)} 個樣本 (run {run_id})")


def upsert_predictions(conn, run_id: str, preds_df: pd.DataFrame) -> None:
    """
    批次插入/更新預測記錄
    
    規格：implementation_plan.md #5.2
    
    Expected columns: product_id, y_true, y_prob, y_pred, split, fold, threshold, is_oof
    """
    df = preds_df.copy()
    df["run_id"] = run_id
    
    # 確保 fold 存在
    if "fold" not in df.columns:
        df["fold"] = -1
    df["fold"] = df["fold"].fillna(-1).astype(int)
    
    # 確保 is_oof 存在
    if "is_oof" not in df.columns:
        df["is_oof"] = 0  # Default to test
    
    # 確保 threshold 存在
    if "threshold" not in df.columns:
        df["threshold"] = 0.5
    
    cols = ["run_id", "product_id", "y_true", "y_prob", "y_pred", "split", "fold", "threshold"]
    rows = [tuple(x) for x in df[cols].itertuples(index=False, name=None)]
    
    with conn.cursor() as cur:
        execute_values(
            cur,
            f"""
            INSERT INTO experiment_predictions ({",".join(cols)})
            VALUES %s
            ON CONFLICT (run_id, product_id, split, fold) DO UPDATE SET
              y_true = EXCLUDED.y_true,
              y_prob = EXCLUDED.y_prob,
              y_pred = EXCLUDED.y_pred,
              threshold = EXCLUDED.threshold
            """,
            rows,
            page_size=5000,
        )
    conn.commit()
    print(f"[Experiment Logger] 已插入 {len(rows)} 條預測 (run {run_id})")


# =============================================================================
# Features & Artifacts
# =============================================================================

def upsert_features(
    conn,
    run_id: str,
    features: List[str],
    active: bool = True,
    importances: Optional[Dict[str, float]] = None
) -> None:
    """
    批次插入/更新特徵記錄
    
    規格：implementation_plan.md #5.3
    """
    rows = []
    for f in features:
        imp = None if not importances else importances.get(f)
        rows.append((run_id, f, active, imp, None))
    
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO experiment_features (run_id, feature_name, is_active, importance, meta_json)
            VALUES %s
            ON CONFLICT (run_id, feature_name) DO UPDATE SET
              is_active = EXCLUDED.is_active,
              importance = EXCLUDED.importance,
              meta_json = EXCLUDED.meta_json
            """,
            rows,
            page_size=5000,
        )
    conn.commit()
    print(f"[Experiment Logger] 已插入 {len(rows)} 個特徵 (run {run_id})")


def upsert_artifact(
    conn,
    run_id: str,
    artifact_type: str,
    file_path: str,
    file_hash: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None
) -> None:
    """
    插入/更新單一 artifact 記錄
    
    規格：implementation_plan.md #5.3, #5.4, #5.5
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO experiment_artifacts (run_id, artifact_type, file_path, file_hash, meta_json)
            VALUES (%s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (run_id, artifact_type) DO UPDATE SET
              file_path = EXCLUDED.file_path,
              file_hash = EXCLUDED.file_hash,
              meta_json = EXCLUDED.meta_json
            """,
            (run_id, artifact_type, file_path, file_hash,
             json.dumps(meta, ensure_ascii=False) if meta else None),
        )
    conn.commit()


# =============================================================================
# Utility Functions
# =============================================================================

def make_run_id(prefix: str = "") -> str:
    """生成唯一的 run_id"""
    import uuid
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    short_uuid = uuid.uuid4().hex[:8]
    if prefix:
        return f"{prefix}_{timestamp}_{short_uuid}"
    return f"{timestamp}_{short_uuid}"


if __name__ == "__main__":
    print("experiment_logger.py 加載成功")
    
    # Test database connection
    try:
        conn = get_db_connection()
        print("✅ 資料庫連接成功")
        conn.close()
    except Exception as e:
        print(f"❌ 資料庫連接失敗: {e}")
