# -*- coding: utf-8 -*-
"""
Model/ingest_run_to_db.py
把 train.py 產生的輸出(報表/manifest)寫入資料庫。
- 只寫入 metadata / 指標 / 產物路徑；不寫入資料集矩陣。
- 對應你提供的 schema：ml_modes / ml_runs / ml_run_features / ml_run_algorithms /
  ml_fold_metrics / ml_run_summary / ml_run_artifacts

用法：
  python Model/ingest_run_to_db.py \
    --outdir Model/outputs \
    --manifest Model/outputs/run_manifest.json \
    --mode-code product_level \
    --mode-short "商品層級：用 cutoff 前特徵預測 cutoff 後是否發生銷售增加" \
    --mode-long  "<放詳細說明>" \
    [--dry-run]

# 方式一：指定 manifest
python Model/ingest_run_to_db.py \
--outdir Model/outputs \
--manifest Model/outputs/run_manifest.json \
--mode-code product_level \
--mode-short "商品層級：用 cutoff 前特徵預測 cutoff 後是否發生銷售增加" \
--mode-long "以商品為單位建模；特徵含 Dense 欄位（價格、關鍵字、媒體旗標、互動數…）及 cutoff 前彙整評論的 TF-IDF Top-N 詞彙(1/0)，目標 y=在 cutoff 以後任一批次銷售數量是否有『增加』。支援排除商品、10 折交叉驗證、特徵選擇等設定。"

# 方式二：不指定 manifest，會從 outdir 找最新 run_manifest*.json
python Model/ingest_run_to_db.py --outdir Model/outputs --mode-code product_level --mode-short "..." --mode-long "..."
"""
import os
import json
import glob
import argparse
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values, Json

# ---------- helpers ----------

def find_manifest(outdir: str, given: str | None):
    if given and os.path.exists(given):
        return given
    # fallback: 找 outdir 下最新的 run_manifest*.json
    cands = sorted(glob.glob(os.path.join(outdir, "**", "run_manifest*.json"), recursive=True),
                   key=lambda p: os.path.getmtime(p), reverse=True)
    if not cands:
        raise FileNotFoundError("找不到 run manifest，請用 --manifest 指定或確認 outdir。")
    return cands[0]

def read_csv_if_exists(path: str) -> pd.DataFrame | None:
    return pd.read_csv(path) if (path and os.path.exists(path)) else None

def load_env():
    load_dotenv()
    return dict(
        host=os.getenv("PGHOST", "127.0.0.1"),
        port=int(os.getenv("PGPORT", "5432")),
        db=os.getenv("PGDATABASE", "postgres"),
        user=os.getenv("PGUSER", "postgres"),
        pwd=os.getenv("PGPASSWORD", "")
    )

def connect_db(cfg: dict):
    return psycopg2.connect(
        host=cfg["host"], port=cfg["port"], dbname=cfg["db"],
        user=cfg["user"], password=cfg["pwd"]
    )

def upsert_mode(cur, code: str, short: str, long: str) -> int:
    cur.execute("""
        INSERT INTO ml_modes(code, mode_desc_short, mode_desc_long)
        VALUES (%s, %s, %s)
        ON CONFLICT (code) DO UPDATE
          SET mode_desc_short = EXCLUDED.mode_desc_short,
              mode_desc_long  = EXCLUDED.mode_desc_long,
              updated_at = now()
        RETURNING id;
    """, (code, short, long))
    mode_id = cur.fetchone()[0]
    return mode_id

def insert_run(cur, run_obj: dict, mode_id: int) -> None:
    # run_obj 由 manifest 組：包含 date_cutoff / cv / config / target_definition
    cur.execute("""
        INSERT INTO ml_runs(run_id, mode_id, date_cutoff, cv_splits, target_definition, config)
        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (run_id) DO NOTHING;
    """, (
        run_obj["run_id"], mode_id, run_obj["date_cutoff"],
        run_obj["cv_splits"], run_obj["target_definition"],
        json.dumps(run_obj["config"], ensure_ascii=False)
    ))

def insert_run_features(cur, run_id: str, date_cutoff: str,
                        dataset_manifest: dict | None,
                        vocab_mode: str | None, topn: int | None) -> None:
    if not dataset_manifest:
        return
    # Dense
    dense_cols = dataset_manifest.get("dense_features", [])
    dense_art = dict(Xdense_csv=dataset_manifest["files"].get("Xdense_csv"))
    cur.execute("""
        INSERT INTO ml_run_features(run_id, family, name, params, source_window, version, artifacts)
        VALUES (%s, 'dense', %s, %s::jsonb, %s::jsonb, %s, %s::jsonb)
    """, (
        run_id,
        "dense_default",
        json.dumps({"columns": dense_cols}, ensure_ascii=False),
        json.dumps({"type":"pre_cutoff","start": None, "end": date_cutoff}),
        None,
        json.dumps(dense_art, ensure_ascii=False)
    ))

    # TF-IDF
    vocab_len = dataset_manifest.get("shapes", {}).get("vocab")
    tf_art = dict(
        vocab_file=dataset_manifest["files"].get("vocab_txt"),
        Xtfidf_npz=dataset_manifest["files"].get("Xtfidf_npz")
    )
    tf_params = {"vocab_mode": vocab_mode, "top_n": topn, "vocab_size": vocab_len}
    cur.execute("""
        INSERT INTO ml_run_features(run_id, family, name, params, source_window, version, artifacts)
        VALUES (%s, 'tfidf', %s, %s::jsonb, %s::jsonb, %s, %s::jsonb)
    """, (
        run_id,
        f"tfidf_{vocab_mode}_top{topn}",
        json.dumps(tf_params, ensure_ascii=False),
        json.dumps({"type":"pre_cutoff","start": None, "end": date_cutoff}),
        None,
        json.dumps(tf_art, ensure_ascii=False)
    ))

def upsert_algorithms(cur, run_id: str, alg_fs_pairs: list[tuple[str, str]]) -> dict:
    """
    回傳 {(algorithm, fs_method) -> algorithm_id}
    """
    alg2id = {}
    for algo, fs in alg_fs_pairs:
        cur.execute("""
            INSERT INTO ml_run_algorithms(run_id, algorithm, fs_method, hyperparams)
            VALUES (%s, %s, %s, '{}'::jsonb)
            ON CONFLICT (run_id, algorithm, fs_method) DO UPDATE
              SET updated_at = now()
            RETURNING id;
        """, (run_id, algo, fs))
        alg2id[(algo, fs)] = cur.fetchone()[0]
    return alg2id

def insert_fold_metrics(cur, run_id: str, alg2id: dict, df: pd.DataFrame) -> None:
    if df is None or df.empty: 
        return
    # 預期欄位：fold, algorithm, fs_method, n_products, pos_ratio, accuracy, f1_macro, f1_weighted,
    # precision_macro, recall_macro, auc, precision_0, recall_0, f1_0, auc_0, support_0,
    # precision_1, recall_1, f1_1, auc_1, support_1
    rows = []
    for _, r in df.iterrows():
        algo = str(r["algorithm"])
        fs   = str(r.get("fs_method", "no_fs"))
        alg_id = alg2id.get((algo, fs))
        if not alg_id:
            # 若還沒註冊演算法，先補上
            cur.execute("""
                INSERT INTO ml_run_algorithms(run_id, algorithm, fs_method, hyperparams)
                VALUES (%s, %s, %s, '{}'::jsonb)
                ON CONFLICT (run_id, algorithm, fs_method) DO UPDATE SET updated_at=now()
                RETURNING id;
            """, (run_id, algo, fs))
            alg_id = cur.fetchone()[0]
            alg2id[(algo, fs)] = alg_id

        rows.append((
            run_id, alg_id, int(r["fold"]),
            int(r.get("n_products", 0)),
            float(r.get("pos_ratio", 0)),
            float(r.get("accuracy", 0)),
            float(r.get("f1_macro", 0)),
            float(r.get("f1_weighted", 0)),
            float(r.get("precision_macro", 0)),
            float(r.get("recall_macro", 0)),
            float(r.get("auc", 0)),
            float(r.get("precision_0", 0)), float(r.get("recall_0", 0)),
            float(r.get("f1_0", 0)), float(r.get("auc_0", 0)), int(r.get("support_0", 0)),
            float(r.get("precision_1", 0)), float(r.get("recall_1", 0)),
            float(r.get("f1_1", 0)), float(r.get("auc_1", 0)), int(r.get("support_1", 0))
        ))
    execute_values(cur, """
        INSERT INTO ml_fold_metrics(
            run_id, algorithm_id, fold, n_products, pos_ratio,
            accuracy, f1_macro, f1_weighted, precision_macro, recall_macro, auc,
            precision_0, recall_0, f1_0, auc_0, support_0,
            precision_1, recall_1, f1_1, auc_1, support_1
        ) VALUES %s
    """, rows)

def insert_summary(cur, run_id: str, alg2id: dict, summary_df: pd.DataFrame | None,
                   folds: int) -> None:
    """
    若有 model_results.csv（已含 mean/std），直接寫入；
    否則由 fold_metrics 先 groupby 算出來再寫入（此處假設你已先呼叫 insert_fold_metrics）。
    """
    if summary_df is not None and not summary_df.empty and \
       {"algorithm","fs_method"}.issubset(summary_df.columns):
        for _, r in summary_df.iterrows():
            algo = str(r["algorithm"])
            fs   = str(r.get("fs_method", "no_fs"))
            alg_id = alg2id.get((algo, fs))
            if not alg_id:
                cur.execute("""
                    INSERT INTO ml_run_algorithms(run_id, algorithm, fs_method, hyperparams)
                    VALUES (%s, %s, %s, '{}'::jsonb)
                    ON CONFLICT (run_id, algorithm, fs_method) DO UPDATE SET updated_at=now()
                    RETURNING id;
                """, (run_id, algo, fs))
                alg_id = cur.fetchone()[0]
                alg2id[(algo, fs)] = alg_id

            cur.execute("""
                INSERT INTO ml_run_summary(
                  run_id, algorithm_id, folds,
                  accuracy_mean, f1_macro_mean, f1_weighted_mean,
                  precision_macro_mean, recall_macro_mean, auc_mean,
                  precision_0_mean, recall_0_mean, f1_0_mean, auc_0_mean,
                  precision_1_mean, recall_1_mean, f1_1_mean, auc_1_mean,
                  accuracy_std, f1_macro_std, f1_weighted_std,
                  precision_macro_std, recall_macro_std, auc_std,
                  precision_0_std, recall_0_std, f1_0_std, auc_0_std,
                  precision_1_std, recall_1_std, f1_1_std, auc_1_std
                ) VALUES (
                  %s,%s,%s,
                  %s,%s,%s,
                  %s,%s,%s,
                  %s,%s,%s,%s,
                  %s,%s,%s,%s,
                  %s,%s,%s,
                  %s,%s,%s,
                  %s,%s,%s,%s,
                  %s,%s,%s,%s
                )
            """, (
                run_id, alg_id, folds,
                r.get("accuracy_mean"), r.get("f1_macro_mean"), r.get("f1_weighted_mean"),
                r.get("precision_macro_mean"), r.get("recall_macro_mean"), r.get("auc_mean"),
                r.get("precision_0_mean"), r.get("recall_0_mean"), r.get("f1_0_mean"), r.get("auc_0_mean"),
                r.get("precision_1_mean"), r.get("recall_1_mean"), r.get("f1_1_mean"), r.get("auc_1_mean"),
                r.get("accuracy_std"), r.get("f1_macro_std"), r.get("f1_weighted_std"),
                r.get("precision_macro_std"), r.get("recall_macro_std"), r.get("auc_std"),
                r.get("precision_0_std"), r.get("recall_0_std"), r.get("f1_0_std"), r.get("auc_0_std"),
                r.get("precision_1_std"), r.get("recall_1_std"), r.get("f1_1_std"), r.get("auc_1_std"),
            ))
    else:
        # 若沒有 summary_df，你也可以在這裡用 SQL 由 ml_fold_metrics 聚合產出，再塞回 ml_run_summary
        pass

def insert_artifacts(cur, run_id: str, artifacts_json: dict) -> None:
    cur.execute("""
        INSERT INTO ml_run_artifacts(run_id, artifacts)
        VALUES (%s, %s::jsonb)
    """, (run_id, json.dumps(artifacts_json, ensure_ascii=False)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="train.py 的輸出資料夾")
    ap.add_argument("--manifest", help="run_manifest.json 路徑（可略，會自動尋找最新）")
    ap.add_argument("--mode-code", required=True, help="如 product_level")
    ap.add_argument("--mode-short", required=True, help="模式簡述")
    ap.add_argument("--mode-long", required=True, help="模式詳細說明")
    ap.add_argument("--dry-run", action="store_true", help="不寫入 DB, 僅顯示解析內容")
    args = ap.parse_args()

    manifest_path = find_manifest(args.outdir, args.manifest)
    with open(manifest_path, "r", encoding="utf-8") as f:
        run_manifest = json.load(f)

    run_id        = run_manifest.get("run_id")
    date_cutoff   = run_manifest.get("date_cutoff")
    cv_splits     = int(run_manifest.get("cv", run_manifest.get("cv_splits", 10)))
    vocab_mode    = run_manifest.get("vocab_mode")
    topn_list     = run_manifest.get("top_n_list") or []
    algorithms    = run_manifest.get("algorithms") or []
    fs_methods    = run_manifest.get("fs_methods") or []
    dataset_man   = run_manifest.get("dataset_manifest")  # 我們前面在 train.py 有寫進去
    target_def    = run_manifest.get("y_definition", "post-cutoff any increase")

    # 報表位置（盡量容錯抓路徑）
    fold_csv = None
    summary_csv = None
    # 優先抓與 manifest 同資料夾
    base = os.path.dirname(manifest_path)
    cand_fold = glob.glob(os.path.join(base, "**", "fold_metrics.csv"), recursive=True)
    cand_sum  = glob.glob(os.path.join(base, "**", "model_results.csv"), recursive=True)
    if cand_fold: fold_csv = cand_fold[0]
    if cand_sum:  summary_csv = cand_sum[0]

    df_fold = read_csv_if_exists(fold_csv)
    df_sum  = read_csv_if_exists(summary_csv)

    # 整理所有 artifacts
    artifacts = {
        "manifest": manifest_path,
        "fold_metrics_csv": fold_csv,
        "summary_csv": summary_csv,
        "dataset_manifest": dataset_man and dataset_man.get("path"),
    }

    # 演算法/FS 組合（若 manifest.children 列出了更細者，也可從 children 合併）
    alg_fs_pairs = []
    if algorithms and fs_methods:
        for a in algorithms:
            for fs in fs_methods:
                alg_fs_pairs.append((a, fs))
    else:
        # 從 fold_metrics 推斷
        if df_fold is not None and not df_fold.empty:
            alg_fs_pairs = sorted({(str(x["algorithm"]), str(x.get("fs_method","no_fs")))
                                  for _, x in df_fold.iterrows()})

    # 準備 run 記錄
    cfg_obj = {
        "excluded_products": run_manifest.get("exclude_products"),
        "pipeline_version": run_manifest.get("pipeline_version"),
        "vocab_mode": vocab_mode,
        "top_n_list": topn_list,
        "algorithms": algorithms,
        "fs_methods": fs_methods
    }
    run_obj = {
        "run_id": run_id,
        "date_cutoff": date_cutoff,
        "cv_splits": cv_splits,
        "target_definition": target_def,
        "config": cfg_obj
    }

    if args.dry_run:
        print("[DRY-RUN] manifest:", json.dumps(run_manifest, ensure_ascii=False, indent=2))
        print("[DRY-RUN] fold_metrics head:\n", df_fold.head() if df_fold is not None else None)
        print("[DRY-RUN] summary head:\n", df_sum.head() if df_sum is not None else None)
        print("[DRY-RUN] alg_fs_pairs:", alg_fs_pairs)
        return

    # --- 寫入 DB ---
    dbc = connect_db(load_env())
    dbc.autocommit = False
    try:
        with dbc.cursor() as cur:
            # 1) 模式 upsert
            mode_id = upsert_mode(cur, args.mode_code, args.mode_short, args.mode_long)

            # 2) 這次 run 基本資料
            insert_run(cur, run_obj, mode_id)

            # 3) 特徵家族（dense / tfidf）
            topn = (topn_list[0] if topn_list else None)
            insert_run_features(cur, run_id, date_cutoff, dataset_man, vocab_mode, topn)

            # 4) 演算法列表
            alg2id = upsert_algorithms(cur, run_id, alg_fs_pairs)

            # 5) 每折指標
            insert_fold_metrics(cur, run_id, alg2id, df_fold)

            # 6) 摘要
            insert_summary(cur, run_id, alg2id, df_sum, cv_splits)

            # 7) 產物路徑
            insert_artifacts(cur, run_id, artifacts)

        dbc.commit()
        print(f"[OK] run_id={run_id} 已寫入資料庫。")
    except Exception as e:
        dbc.rollback()
        raise
    finally:
        dbc.close()

if __name__ == "__main__":
    main()