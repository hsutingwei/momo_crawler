# -*- coding: utf-8-sig -*-
"""
將 feature_analysis.py 的輸出匯入資料庫：
- JSON: feature_analysis_db_ready_*.json
- 圖檔: feature_visualization_*.{svg|png}

用法示例：
python ingest_feature_analysis.py \
  --json Model/analysis_outputs/feature_analysis_db_ready_20250823_164614.json \
  --viz  Model/analysis_outputs/feature_visualization_20250823_164614.svg \
  --codes product_level,d0dca4c6-aad9-47b8-b5f7-3671208c31d6,dc7dd570-0446-41e3-8511-fdc83ea1af69,8b0f585b-642a-402d-a7f3-c3363a40614b
"""

from pathlib import Path
import sys
import os
# 專案根目錄 (…/momo_crawler-main)
REPO_ROOT = Path(__file__).resolve().parent.parent  # .../Model -> 上一層 = 專案根目錄
sys.path.insert(0, str(REPO_ROOT))

import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

import psycopg2
import psycopg2.extras as pgx

# 沿用 csv_to_db.py 的連線方式
from config.database import DatabaseConfig  # type: ignore

def parse_args():
    ap = argparse.ArgumentParser(description="Ingest feature_analysis outputs into DB")
    ap.add_argument("--json", required=True, help="feature_analysis_db_ready_*.json 路徑")
    ap.add_argument("--viz", required=False, default=None, help="feature_visualization_*.svg/png 路徑（可不填）")
    ap.add_argument("--codes", required=True, help="以逗號分隔的 code 清單（如：product_level,xxxx,yyyy）")
    return ap.parse_args()

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def to_date(d: str) -> datetime.date:
    d = d.replace("/", "-")
    return datetime.strptime(d, "%Y-%m-%d").date()

# === 新增：由 code 反查 ml_* 對應的 run/algorithm/summary ===
def fetch_runs_for_code(cur, code: str) -> List[Dict[str, Any]]:
    """
    回傳此 code 對應的所有 runs + algorithms + summary 指標
    若你只想拿「最新一個 run」，可在這裡加 ORDER BY 並 LIMIT 1。
    """
    cur.execute("""
        SELECT
          mr.run_id,
          mra.id AS algorithm_id,
          mrs.auc_mean,
          mrs.precision_1_mean AS pr1_mean,
          mrs.recall_1_mean    AS rc1_mean,
          mrs.f1_1_mean        AS f1_1_mean
        FROM ml_modes mm
        JOIN ml_runs mr
          ON mr.mode_id = mm.id
        LEFT JOIN ml_run_algorithms mra
          ON mra.run_id = mr.run_id
        LEFT JOIN ml_run_summary mrs
          ON mrs.run_id = mr.run_id AND mrs.algorithm_id = mra.id
        WHERE mm.code = %s
        -- 想只取最新 run，可改成：
        -- ORDER BY COALESCE(mrs.auc_mean, 0) DESC NULLS LAST
    """, (code,))
    return list(cur.fetchall())

def main():
    args = parse_args()
    payload = load_json(args.json)
    codes = [c.strip() for c in args.codes.split(",") if c.strip()]

    analysis_id = payload["analysis_id"]  # UUID
    analysis_ts = datetime.fromisoformat(payload["analysis_timestamp"])
    cfg = payload.get("analysis_config", {})
    conclusions = payload.get("conclusions") or {}
    artifacts = payload.get("artifacts") or {}
    if args.viz:
        artifacts = dict(artifacts or {})
        artifacts["visualization_path"] = args.viz

    data_summary = payload.get("data_summary") or {}
    # 修正：JSON 結構中特徵統計在 feature_analysis.feature_analysis
    feature_analysis = payload.get("feature_analysis", {})
    dense_stats = feature_analysis.get("feature_analysis", [])
    # 修正：JSON 結構中視覺化在 visualization_analysis.visualization_results
    viz_analysis = payload.get("visualization_analysis", {})
    viz_results = viz_analysis.get("visualization_results", {})
    # 將字典格式轉換為列表格式，每個視覺化方法作為一個項目
    viz = []
    for viz_type, viz_data in viz_results.items():
        viz_data['type'] = viz_type  # 添加類型標識
        viz.append(viz_data)

    mode_code = cfg.get("mode", "product_level")
    date_cutoff = to_date(cfg["date_cutoff"])
    pipeline_version = cfg.get("pipeline_version")
    vocab_mode = cfg.get("vocab_mode")
    top_n = int(cfg["top_n"]) if cfg.get("top_n") is not None else None
    exclude_products = cfg.get("exclude_products")
    keyword = cfg.get("keyword")

    db = DatabaseConfig()
    conn = db.get_connection()
    conn.autocommit = False
    cur = conn.cursor()

    try:
        # 1) fa_batches upsert
        cur.execute("""
            INSERT INTO fa_batches (
                analysis_id, analysis_timestamp, mode_code, date_cutoff, pipeline_version,
                vocab_mode, top_n, exclude_products, keyword, config, conclusions, artifacts
            ) VALUES (
                %(analysis_id)s, %(analysis_timestamp)s, %(mode_code)s, %(date_cutoff)s, %(pipeline_version)s,
                %(vocab_mode)s, %(top_n)s, %(exclude_products)s, %(keyword)s, %(config)s, %(conclusions)s, %(artifacts)s
            )
            ON CONFLICT (analysis_id) DO UPDATE SET
                analysis_timestamp = EXCLUDED.analysis_timestamp,
                mode_code = EXCLUDED.mode_code,
                date_cutoff = EXCLUDED.date_cutoff,
                pipeline_version = EXCLUDED.pipeline_version,
                vocab_mode = EXCLUDED.vocab_mode,
                top_n = EXCLUDED.top_n,
                exclude_products = EXCLUDED.exclude_products,
                keyword = EXCLUDED.keyword,
                config = EXCLUDED.config,
                conclusions = EXCLUDED.conclusions,
                artifacts = EXCLUDED.artifacts,
                updated_at = now()
        """, {
            "analysis_id": analysis_id,
            "analysis_timestamp": analysis_ts,
            "mode_code": mode_code,
            "date_cutoff": date_cutoff,
            "pipeline_version": pipeline_version,
            "vocab_mode": vocab_mode,
            "top_n": top_n,
            "exclude_products": exclude_products,
            "keyword": keyword,
            "config": json.dumps(cfg, ensure_ascii=False),
            "conclusions": json.dumps(conclusions, ensure_ascii=False),
            "artifacts": json.dumps(artifacts, ensure_ascii=False),
        })

        # 2) fa_data_summary（replace）
        cur.execute("""DELETE FROM fa_data_summary WHERE analysis_id = %s""", (analysis_id,))
        cur.execute("""
            INSERT INTO fa_data_summary (
                analysis_id, total_samples, positive_samples, negative_samples,
                imbalance_ratio, dense_features_count, tfidf_features_count, summary_json
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            analysis_id,
            data_summary.get("total_samples"),
            data_summary.get("positive_samples"),
            data_summary.get("negative_samples"),
            data_summary.get("imbalance_ratio"),
            data_summary.get("dense_features_count"),
            data_summary.get("tfidf_features_count"),
            json.dumps(data_summary, ensure_ascii=False),
        ))

        # 3) fa_feature_stats（先刪後插）
        cur.execute("""DELETE FROM fa_feature_stats WHERE analysis_id = %s""", (analysis_id,))
        if dense_stats:
            rows = []
            for fs in dense_stats:
                name = fs.get("feature_name")
                sep = fs.get("separation_metrics", {})
                sig = fs.get("significance", {})
                y0 = fs.get("y0_stats", {}) or {}
                y1 = fs.get("y1_stats", {}) or {}
                rows.append((
                    analysis_id, name,
                    y0.get("mean"), y1.get("mean"),
                    sep.get("cohens_d"), sep.get("mutual_info"),
                    sep.get("p_value"), sep.get("overlap_coefficient"),
                    bool(sig.get("is_significant")), bool(sig.get("has_high_separation")),
                    json.dumps(fs, ensure_ascii=False)
                ))
            pgx.execute_values(cur, """
                INSERT INTO fa_feature_stats (
                    analysis_id, feature_name, y0_mean, y1_mean, cohens_d,
                    mutual_info, p_value, overlap_coefficient, is_significant,
                    has_high_separation, raw_json
                ) VALUES %s
            """, rows)

        # 4) fa_visualizations（先刪後插）
        cur.execute("""DELETE FROM fa_visualizations WHERE analysis_id = %s""", (analysis_id,))
        if viz:
            viz_rows = []
            for v in viz:
                vtype = (v.get("type") or "").lower()  # 'pca'|'tsne'|'umap'
                meta = dict(v)
                sep_score = v.get("separation_score")
                
                # 修正：explained_variance_ratio 是列表，不是字典
                explained_variance_ratio = v.get("explained_variance_ratio", [])
                explained_1 = explained_variance_ratio[0] if len(explained_variance_ratio) > 0 else None
                explained_2 = explained_variance_ratio[1] if len(explained_variance_ratio) > 1 else None
                
                # 修正：cumulative_variance_ratio 是列表，取第一個值
                cumulative_variance_ratio = v.get("cumulative_variance_ratio", [])
                cumulative = cumulative_variance_ratio[0] if len(cumulative_variance_ratio) > 0 else None
                
                plot_path = v.get("plot_path") or (artifacts.get("visualization_path") if isinstance(artifacts, dict) else None) or args.viz
                viz_rows.append((
                    analysis_id, vtype, sep_score, explained_1, explained_2, cumulative,
                    plot_path, json.dumps(meta, ensure_ascii=False)
                ))
            pgx.execute_values(cur, """
                INSERT INTO fa_visualizations (
                    analysis_id, viz_type, separation_score, explained_var_1, explained_var_2,
                    cumulative_var_2, plot_path, meta
                ) VALUES %s
            """, viz_rows)

        # 5) fa_analysis_codes（由 --codes 提供）
        cur.execute("""DELETE FROM fa_analysis_codes WHERE analysis_id = %s""", (analysis_id,))
        if codes:
            pgx.execute_values(cur, """
                INSERT INTO fa_analysis_codes (analysis_id, code) VALUES %s
            """, [(analysis_id, c) for c in codes])

        # 6) fa_related_runs（由 --codes 自動查 DB → 相當於原先的 --runs-map）
        cur.execute("""DELETE FROM fa_related_runs WHERE analysis_id = %s""", (analysis_id,))
        rr_rows = []
        for code in codes:
            rows = fetch_runs_for_code(cur, code)
            for r in rows:
                # r is a tuple: (run_id, algorithm_id, auc_mean, pr1_mean, rc1_mean, f1_1_mean)
                # Convert Decimal to float for JSON serialization
                def decimal_to_float(value):
                    if value is None:
                        return None
                    return float(value)
                
                rr_rows.append((
                    analysis_id, code, r[0], r[1],  # run_id, algorithm_id
                    r[2], r[3], r[4], r[5],  # auc_mean, pr1_mean, rc1_mean, f1_1_mean
                    json.dumps({
                        "auc_mean": decimal_to_float(r[2]), 
                        "pr1_mean": decimal_to_float(r[3]), 
                        "rc1_mean": decimal_to_float(r[4]), 
                        "f1_1_mean": decimal_to_float(r[5])
                    }, ensure_ascii=False)
                ))
        if rr_rows:
            pgx.execute_values(cur, """
                INSERT INTO fa_related_runs (
                    analysis_id, code, run_id, algorithm_id,
                    auc_mean, pr1_mean, rc1_mean, f1_1_mean, metrics_json
                ) VALUES %s
            """, rr_rows)

        print(f"[OK] analysis_id={analysis_id} 匯入完成（含 codes→runs 關聯）。")
        conn.commit()  # 提交事務
    finally:
        conn.close()

if __name__ == "__main__":
    main()