# -*- coding: utf-8-sig -*-
"""
將 feature_analysis.py 的輸出匯入資料庫：
- JSON: feature_analysis_db_ready_*.json
- 圖檔: feature_visualization_*.{svg|png}

用法示例：
python ingest_feature_analysis.py \
  --json Model/analysis_outputs/feature_analysis_db_ready_20250823_164614.json \
  --viz Model/analysis_outputs/feature_visualization_20250823_164614.svg \
  --codes product_level,d0dca4c6-aad9-47b8-b5f7-3671208c31d6,dc7dd570-0446-41e3-8511-fdc83ea1af69,8b0f585b-642a-402d-a7f3-c3363a40614b \
  --runs-map Model/analysis_outputs/runs_map.json

runs_map.json (optional) 範例：
{
  "product_level": [
    {"run_id": "8632cd22-...-d9a", "algorithm_id": 101},
    {"run_id": "ffe61f55-...-5c5", "algorithm_id": 102}
  ],
  "d0dca4c6-aad9-...": [{"run_id": "...", "algorithm_id": 201}]
}
"""

import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

import psycopg2
import psycopg2.extras as pgx

# ---- 你也可以改為沿用既有的連線工具，例如 config.database.get_db_connection() ----
def get_pg_conn():
    # 讀環境變數，找不到就用預設（與你專案一致可自行修改）
    host = os.getenv("PGHOST", "localhost")
    port = int(os.getenv("PGPORT", "5432"))
    user = os.getenv("PGUSER", "postgres")
    password = os.getenv("PGPASSWORD", "postgres")
    dbname = os.getenv("PGDATABASE", "postgres")
    return psycopg2.connect(host=host, port=port, user=user, password=password, dbname=dbname)

def parse_args():
    ap = argparse.ArgumentParser(description="Ingest feature_analysis outputs into DB")
    ap.add_argument("--json", required=True, help="feature_analysis_db_ready_*.json 路徑")
    ap.add_argument("--viz", required=False, default=None, help="feature_visualization_*.svg/png 路徑（可不填）")
    ap.add_argument("--codes", required=True, help="以逗號分隔的 code 清單（product_level 與各不平衡處理 code）")
    ap.add_argument("--runs-map", required=False, help="(可選) JSON 檔，描述 code 與 run/algorithm_id 對應")
    return ap.parse_args()

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def load_runs_map(path: Optional[str]) -> Dict[str, List[Dict[str, Any]]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def to_date(d: str) -> datetime.date:
    # 允許 'YYYY-MM-DD' 或 'YYYY/MM/DD'
    d = d.replace("/", "-")
    return datetime.strptime(d, "%Y-%m-%d").date()

def main():
    args = parse_args()
    payload = load_json(args.json)
    runs_map = load_runs_map(args.runs_map)
    codes = [c.strip() for c in args.codes.split(",") if c.strip()]

    analysis_id = payload["analysis_id"]  # UUID
    analysis_ts = datetime.fromisoformat(payload["analysis_timestamp"])  # ISO 8601
    cfg = payload.get("analysis_config", {})
    conclusions = payload.get("conclusions") or {}
    artifacts = payload.get("artifacts") or {}

    # 若有帶圖檔路徑，就登記在 artifacts 以利入庫
    if args.viz:
        artifacts = dict(artifacts or {})
        artifacts["visualization_path"] = args.viz

    # data_summary
    data_summary = payload.get("data_summary") or {}
    # dense feature stats
    dense_stats = payload.get("dense_feature_stats") or []
    # visualizations
    viz = payload.get("visualizations") or []

    # 取 config 常用欄位
    mode_code = cfg.get("mode", "product_level")
    date_cutoff = to_date(cfg["date_cutoff"])
    pipeline_version = cfg.get("pipeline_version")
    vocab_mode = cfg.get("vocab_mode")
    top_n = int(cfg["top_n"]) if cfg.get("top_n") is not None else None
    exclude_products = cfg.get("exclude_products")
    keyword = cfg.get("keyword")

    conn = get_pg_conn()
    try:
        with conn:
            with conn.cursor(cursor_factory=pgx.RealDictCursor) as cur:

                # 1) fa_batches（upsert）
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

                # 3) fa_feature_stats（先刪後插，或做 upsert 也行）
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
                        # 從 JSON 取常用欄位
                        sep_score = v.get("separation_score")
                        explained_1 = v.get("explained_variance_ratio", {}).get("pc1")
                        explained_2 = v.get("explained_variance_ratio", {}).get("pc2")
                        cumulative = v.get("explained_variance_ratio", {}).get("pc1_pc2_cumulative")
                        # 圖檔路徑（若 JSON 未給，改用命令列傳入的 --viz）
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

                # 5) fa_analysis_codes
                cur.execute("""DELETE FROM fa_analysis_codes WHERE analysis_id = %s""", (analysis_id,))
                pgx.execute_values(cur, """
                    INSERT INTO fa_analysis_codes (analysis_id, code) VALUES %s
                """, [(analysis_id, c) for c in codes])

                # 6) （可選）fa_related_runs：若提供 runs_map，就一起建
                if runs_map:
                    cur.execute("""DELETE FROM fa_related_runs WHERE analysis_id = %s""", (analysis_id,))
                    rr_rows = []
                    for code, items in runs_map.items():
                        for it in items:
                            rr_rows.append((
                                analysis_id, code, it["run_id"], it.get("algorithm_id"),
                                it.get("auc_mean"), it.get("pr1_mean"), it.get("rc1_mean"),
                                it.get("f1_1_mean"), json.dumps(it, ensure_ascii=False)
                            ))
                    if rr_rows:
                        pgx.execute_values(cur, """
                            INSERT INTO fa_related_runs (
                              analysis_id, code, run_id, algorithm_id,
                              auc_mean, pr1_mean, rc1_mean, f1_1_mean, metrics_json
                            ) VALUES %s
                        """, rr_rows)

        print(f"[OK] analysis_id={analysis_id} 已匯入。")

    finally:
        conn.close()

if __name__ == "__main__":
    main()