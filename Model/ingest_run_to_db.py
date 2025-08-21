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
# --- 放在檔案最上面（原本 import 之前）---
from pathlib import Path
import sys
import os
# 專案根目錄 (…/momo_crawler-main)
REPO_ROOT = Path(__file__).resolve().parent.parent  # .../Model -> 上一層 = 專案根目錄
sys.path.insert(0, str(REPO_ROOT))
print(f"[ingest] repo root = {REPO_ROOT}")

import re
import json
import glob
import uuid
import csv
import argparse
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# 沿用 csv_to_db.py 的連線方式
from config.database import DatabaseConfig  # type: ignore


def upsert_ml_mode(cur, code: str, short: str, long: str) -> int:
    cur.execute("""
        INSERT INTO ml_modes(code, mode_desc_short, mode_desc_long)
        VALUES (%s, %s, %s)
        ON CONFLICT (code) DO UPDATE
        SET mode_desc_short = EXCLUDED.mode_desc_short,
            mode_desc_long  = EXCLUDED.mode_desc_long
        RETURNING id;
    """, (code, short, long))
    mode_id = cur.fetchone()[0]
    return mode_id


def insert_ml_run(cur, run_manifest: Dict[str, Any], mode_id: int) -> str:
    # run_id 來自 run manifest
    run_id = run_manifest.get("run_id")
    if not run_id:
        run_id = str(uuid.uuid4())

    date_cutoff = run_manifest.get("date_cutoff")
    if isinstance(date_cutoff, str):
        date_cutoff = dt.datetime.strptime(date_cutoff, "%Y-%m-%d").date()

    cv_splits = int(run_manifest.get("cv", 10))
    target_definition = run_manifest.get("target_definition", "post-cutoff any increase")

    # 統一把訓練設定存到 config JSONB
    config = {
        "vocab_mode": run_manifest.get("vocab_mode"),
        "top_n": run_manifest.get("top_n"),
        "pipeline_version": run_manifest.get("pipeline_version"),
        "excluded_products": run_manifest.get("excluded_products", []),
        "notes": run_manifest.get("notes"),
        "fs_methods": run_manifest.get("fs_methods", []),  # 總 manifest 若沒有，也沒關係
        "algorithms": run_manifest.get("algorithms", [])
    }

    cur.execute("""
        INSERT INTO ml_runs(run_id, mode_id, date_cutoff, cv_splits, target_definition, config)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (run_id) DO NOTHING;
    """, (run_id, mode_id, date_cutoff, cv_splits, target_definition, json.dumps(config)))
    return run_id


def insert_ml_run_feature(cur, run_id: str, family: str, name: str, params: Dict[str, Any],
                          source_window: Optional[Dict[str, Any]] = None,
                          version: Optional[str] = None, artifacts: Optional[Dict[str, Any]] = None) -> int:
    cur.execute("""
        INSERT INTO ml_run_features(run_id, family, name, params, source_window, version, artifacts)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """, (run_id, family, name, json.dumps(params),
          json.dumps(source_window) if source_window else None,
          version, json.dumps(artifacts) if artifacts else None))
    return cur.fetchone()[0]


def insert_ml_run_algorithm(cur, run_id: str, algorithm: str, fs_method: str,
                            hyperparams: Dict[str, Any], notes: Optional[str] = None) -> int:
    # 以 upsert 方式，不論新插入或衝突都回傳該列 id
    cur.execute("""
        INSERT INTO ml_run_algorithms (run_id, algorithm, fs_method, hyperparams, notes)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (run_id, algorithm, fs_method)
        DO UPDATE
           SET hyperparams = EXCLUDED.hyperparams,
               notes       = EXCLUDED.notes
        RETURNING id;
    """, (run_id, algorithm, fs_method, json.dumps(hyperparams), notes))
    alg_id = cur.fetchone()[0]
    return alg_id


def insert_fold_metrics_csv(cur, run_id: str, algorithm_id: int, csv_path: str):
    # 期待欄位：fold, n_products, pos_ratio, accuracy, f1_macro, f1_weighted, precision_macro, recall_macro, auc,
    # precision_0, recall_0, f1_0, auc_0, support_0, precision_1, recall_1, f1_1, auc_1, support_1
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        cur.execute("""
            INSERT INTO ml_fold_metrics(
              run_id, algorithm_id, fold, n_products, pos_ratio,
              accuracy, f1_macro, f1_weighted, precision_macro, recall_macro, auc,
              precision_0, recall_0, f1_0, auc_0, support_0,
              precision_1, recall_1, f1_1, auc_1, support_1
            )
            VALUES (
              %s, %s, %s, %s, %s,
              %s, %s, %s, %s, %s, %s,
              %s, %s, %s, %s, %s,
              %s, %s, %s, %s, %s
            );
        """, (
            run_id, algorithm_id,
            int(r.get("fold", 0)),
            int(r.get("n_products", 0)),
            float(r.get("pos_ratio", 0) or 0),
            float(r.get("accuracy", 0) or 0),
            float(r.get("f1_macro", 0) or 0),
            float(r.get("f1_weighted", 0) or 0),
            float(r.get("precision_macro", 0) or 0),
            float(r.get("recall_macro", 0) or 0),
            float(r.get("auc", 0) or 0),
            float(r.get("precision_0", 0) or 0),
            float(r.get("recall_0", 0) or 0),
            float(r.get("f1_0", 0) or 0),
            float(r.get("auc_0", 0) or 0),
            int(r.get("support_0", 0) or 0),
            float(r.get("precision_1", 0) or 0),
            float(r.get("recall_1", 0) or 0),
            float(r.get("f1_1", 0) or 0),
            float(r.get("auc_1", 0) or 0),
            int(r.get("support_1", 0) or 0)
        ))


def insert_summary_csv(cur, run_id: str, algorithm_id: int, csv_path: str):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        r = next(reader)  # 只有一列
    
    # 檢查 CSV 欄位並提供預設值
    def safe_get_float(key: str, default: float = 0.0) -> float:
        val = r.get(key, default)
        try:
            return float(val) if val is not None else default
        except (ValueError, TypeError):
            return default
    
    cur.execute("""
        INSERT INTO ml_run_summary(
          run_id, algorithm_id, folds,
          accuracy_mean, f1_macro_mean, f1_weighted_mean, precision_macro_mean, recall_macro_mean, auc_mean,
          precision_0_mean, recall_0_mean, f1_0_mean, auc_0_mean,
          precision_1_mean, recall_1_mean, f1_1_mean, auc_1_mean,
          accuracy_std, f1_macro_std, f1_weighted_std, precision_macro_std, recall_macro_std, auc_std,
          precision_0_std, recall_0_std, f1_0_std, auc_0_std,
          precision_1_std, recall_1_std, f1_1_std, auc_1_std
        )
        VALUES (
          %s, %s, %s,
          %s, %s, %s, %s, %s, %s,
          %s, %s, %s, %s,
          %s, %s, %s, %s,
          %s, %s, %s, %s, %s, %s,
          %s, %s, %s, %s,
          %s, %s, %s, %s
        );
    """, (
        run_id, algorithm_id, int(r.get("folds", 0)),
        safe_get_float("accuracy_mean"),
        safe_get_float("f1_macro_mean"),
        safe_get_float("f1_weighted_mean"),
        safe_get_float("precision_macro_mean"),
        safe_get_float("recall_macro_mean"),
        safe_get_float("auc_mean"),
        safe_get_float("precision_0_mean"),
        safe_get_float("recall_0_mean"),
        safe_get_float("f1_0_mean"),
        safe_get_float("auc_0_mean"),
        safe_get_float("precision_1_mean"),
        safe_get_float("recall_1_mean"),
        safe_get_float("f1_1_mean"),
        safe_get_float("auc_1_mean"),
        safe_get_float("accuracy_std"),
        safe_get_float("f1_macro_std"),
        safe_get_float("f1_weighted_std"),
        safe_get_float("precision_macro_std"),
        safe_get_float("recall_macro_std"),
        safe_get_float("auc_std"),
        safe_get_float("precision_0_std"),
        safe_get_float("recall_0_std"),
        safe_get_float("f1_0_std"),
        safe_get_float("auc_0_std"),
        safe_get_float("precision_1_std"),
        safe_get_float("recall_1_std"),
        safe_get_float("f1_1_std"),
        safe_get_float("auc_1_std")
    ))


def insert_run_artifacts(cur, run_id: str, artifacts: Dict[str, Any]):
    cur.execute("""
        INSERT INTO ml_run_artifacts(run_id, artifacts)
        VALUES (%s, %s)
        ON CONFLICT (run_id) DO UPDATE
        SET artifacts = EXCLUDED.artifacts,
            created_at = CURRENT_TIMESTAMP;
    """, (run_id, json.dumps(artifacts)))


# ---------- FS helpers / manifest helpers ----------

def find_manifest(outdir: str, manifest_path: Optional[str]) -> str:
    """
    若指定 --manifest 就用該檔；否則從 outdir 找「最新的 *_RUN_manifest.json」。
    """
    if manifest_path:
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"指定的 manifest 不存在: {manifest_path}")
        return manifest_path

    patterns = [
        os.path.join(outdir, "*_RUN_manifest.json"),
        os.path.join(outdir, "RUN_manifest.json"),
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        raise FileNotFoundError("找不到 run manifest，請用 --manifest 指定或確認 outdir。")

    # 取最後修改時間最新的一個
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def safe_get_fs_methods(run_manifest: Dict[str, Any], child_manifests: List[Dict[str, Any]]) -> List[str]:
    """
    run manifest 假如沒有 fs_methods（複數），就從子 manifest 的 fs_method（單數）蒐集。
    """
    vals = run_manifest.get("fs_methods")
    if isinstance(vals, list) and vals:
        return list(dict.fromkeys(vals))  # 去重保持順序
    # fallback: collect from children
    acc = []
    for cm in child_manifests:
        m = cm.get("fs_method") or cm.get("fs_methods")
        if isinstance(m, list):
            acc.extend(m)
        elif isinstance(m, str):
            acc.append(m)
    return list(dict.fromkeys(acc))


def find_child_manifests(outdir: str, run_id: str) -> List[str]:
    """
    尋找子 manifest：命名通常像 run_..._manifest.json（但不含 RUN_manifest）
    """
    pats = [
        os.path.join(outdir, f"*{run_id}*manifest.json"),
        os.path.join(outdir, "run_*_manifest.json"),
        os.path.join(outdir, "*_manifest.json"),
    ]
    allc = []
    for p in pats:
        allc.extend(glob.glob(p))
    # 過濾 RUN_manifest
    allc = [p for p in allc if "_RUN_manifest.json" not in p and os.path.basename(p) != "RUN_manifest.json"]
    # 去重
    uniq = list(dict.fromkeys(allc))
    return uniq


# ---------- main flow ----------

def main():
    ap = argparse.ArgumentParser(
        description="把 train.py 的輸出 ingest 進資料庫"
    )
    ap.add_argument("--outdir", required=True, help="train.py 寫出檔案的資料夾")
    ap.add_argument("--manifest", required=False, help="指定 RUN manifest 路徑（可省略，會自動尋找最新）")
    ap.add_argument("--mode-code", required=True, help="模式代碼（如 product_level）")
    ap.add_argument("--mode-short", required=True, help="模式簡述")
    ap.add_argument("--mode-long", required=True, help="模式詳述")
    args = ap.parse_args()

    outdir = args.outdir
    run_manifest_path = find_manifest(outdir, args.manifest)
    run_manifest = load_json(run_manifest_path)

    # 連 DB
    db = DatabaseConfig()
    conn = db.get_connection()
    conn.autocommit = False
    cur = conn.cursor()

    try:
        # 1) ml_modes
        mode_id = upsert_ml_mode(cur, args.mode_code, args.mode_short, args.mode_long)

        # 2) ml_runs
        run_id = insert_ml_run(cur, run_manifest, mode_id)

        # 3) 子 manifest 列出來
        child_paths = find_child_manifests(outdir, run_id)
        child_manifests = [load_json(p) for p in child_paths]

        # 4) ml_run_features：從 dataset_manifest 提取 dense 與 tfidf 參數
        dataset_manifest = run_manifest.get("dataset_manifest", {})
        if dataset_manifest:
            # dense features
            dense_features = dataset_manifest.get("dense_features", [])
            if dense_features:
                insert_ml_run_feature(
                    cur, run_id,
                    family="dense",
                    name="dense_basic",
                    params={"features": dense_features},
                    source_window={"type": "pre_cutoff", "end": str(run_manifest.get("date_cutoff"))},
                    version=run_manifest.get("pipeline_version"),
                    artifacts=None
                )

            # tfidf features
            vocab_mode = dataset_manifest.get("vocab_mode")
            top_n_built = dataset_manifest.get("top_n_built")
            if vocab_mode and top_n_built:
                tfidf_name = f"tfidf_{vocab_mode}_top{top_n_built}"
                art = {}
                vocab_file = dataset_manifest.get("files", {}).get("vocab_txt")
                if vocab_file:
                    art["vocab_file"] = vocab_file
                insert_ml_run_feature(
                    cur, run_id,
                    family="tfidf",
                    name=tfidf_name,
                    params={
                        "vocab_mode": vocab_mode,
                        "top_n": top_n_built
                    },
                    source_window={"type": "pre_cutoff", "end": str(run_manifest.get("date_cutoff"))},
                    version=run_manifest.get("pipeline_version"),
                    artifacts=art or None
                )

        # 5) ml_run_algorithms + 指標
        artifacts_all: Dict[str, Any] = {"children": []}
        for cpath, cm in zip(child_paths, child_manifests):
            algo = cm.get("algorithm") or cm.get("alg") or "unknown"
            # ★ 兼容 fs_method 與 fs_methods
            fs_method = None
            if "fs_method" in cm:
                fs_method = cm["fs_method"]
            elif "fs_methods" in cm:
                # 若是 list 取第一個，或轉成字串
                val = cm["fs_methods"]
                fs_method = val[0] if isinstance(val, list) and val else str(val)
            else:
                fs_method = "no_fs"

            hyper = cm.get("hyperparams", {})
            
            # 為 notes 欄位提供描述性文字
            notes = f"Algorithm: {algo}, FS Method: {fs_method}, Top-N: {cm.get('top_n', 'N/A')}, CV: {cm.get('cv', 'N/A')}"
            
            alg_id = insert_ml_run_algorithm(cur, run_id, algo, fs_method, hyper, notes)

            # fold_metrics.csv / summary.csv - 修正欄位名稱匹配
            files = cm.get("files", {})
            fcsv = files.get("fold_metrics") or cm.get("fold_metrics_csv")
            scsv = files.get("summary") or cm.get("summary_csv")

            if fcsv and os.path.isfile(fcsv):
                insert_fold_metrics_csv(cur, run_id, alg_id, fcsv)
            if scsv and os.path.isfile(scsv):
                insert_summary_csv(cur, run_id, alg_id, scsv)

            artifacts_all["children"].append({
                "manifest": cpath,
                "algorithm": algo,
                "fs_method": fs_method,
                "fold_metrics_csv": fcsv,
                "summary_csv": scsv
            })

        # 6) ml_run_artifacts：把 RUN manifest、子 manifest 等都記錄
        artifacts_all["run_manifest"] = run_manifest_path
        artifacts_all["child_manifests"] = child_paths
        insert_run_artifacts(cur, run_id, artifacts_all)

        conn.commit()
        print(f"[OK] run_id={run_id} 已寫入資料庫。")

    except Exception as e:
        conn.rollback()
        print(f"[ERROR] 交易失敗：{e}")
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()