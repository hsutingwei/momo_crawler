# -*- coding: utf-8 -*-
"""
ingest_embeddings_dataset.py

將 emb_build.py 產出的 dataset（manifest + 檔案們）寫入資料庫：
- upsert 到 emb_models（provider, model_name, version 唯一）
- upsert 到 emb_datasets（model_id, dataset_prefix 唯一）

用法範例：
python ingest_embeddings_dataset.py \
  --base-dir Model/embeddings/train_output/datasets \
  --dataset-prefix dataset_product_level_20250625_global_top100_20250827-222350

或直接給 manifest 路徑：
python ingest_embeddings_dataset.py --manifest Model/embeddings/train_output/datasets/dataset_product_level_20250625_global_top100_20250827-222350_manifest.json

資料庫連線：
- 優先嘗試 import config.database.get_pg_conn() / connect_postgres()
- 不存在則用環境變數 PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE 連 psycopg2
"""

from pathlib import Path
import sys
import os
# 專案根目錄 (…/momo_crawler-main)
REPO_ROOT = Path(__file__).resolve().parent.parent  # .../Model -> 上一層 = 專案根目錄
sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import os
import sys
from datetime import datetime, date
from typing import Optional, Tuple

# 沿用 csv_to_db.py 的連線方式
from config.database import DatabaseConfig  # type: ignore

def parse_args():
    ap = argparse.ArgumentParser(
        description="Ingest embedding dataset (from emb_build.py) into emb_models / emb_datasets"
    )
    # 兩種指定方式擇一：manifest 或 base-dir + dataset-prefix
    ap.add_argument("--manifest", type=str, default=None,
                    help="Full path to *_manifest.json produced by emb_build.py")
    ap.add_argument("--base-dir", type=str, default=None,
                    help="Base dir containing dataset files (same as emb_build --outdir)")
    ap.add_argument("--dataset-prefix", type=str, default=None,
                    help="Dataset prefix (filename stem without suffix); e.g. dataset_product_level_20250625_...")

    # 若 manifest 裡沒有或要覆蓋，可補充以下欄位
    ap.add_argument("--provider", type=str, default=None)
    ap.add_argument("--model-name", type=str, default=None)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--tokenization", type=str, default=None)  # none | ckip | bpe …
    ap.add_argument("--version", type=str, default=None)       # 你的 pipeline tag
    ap.add_argument("--notes", type=str, default=None)
    ap.add_argument("--mode", type=str, default=None)          # product_level | comment_level
    ap.add_argument("--text-source", type=str, default=None)   # raw | norm
    ap.add_argument("--pooling", type=str, default=None)       # mean | max | sif …
    ap.add_argument("--date-cutoff", type=str, default=None)   # YYYY-MM-DD
    ap.add_argument("--run-name", type=str, default=None)      # optional

    return ap.parse_args()


def _str_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def locate_files(base_dir: str, dataset_prefix: str) -> dict:
    """
    依據 base_dir + dataset_prefix 推導出各檔案的路徑。
    回傳字典包含 manifest/meta/x/y/vocab/xdense/xembed paths（有些可能不存在）。
    """
    p = os.path.join
    paths = {
        "manifest_path": p(base_dir, f"{dataset_prefix}_manifest.json"),
        "meta_path":     p(base_dir, f"{dataset_prefix}_meta.csv"),
        "y_csv_path":    p(base_dir, f"{dataset_prefix}_y.csv"),
        "y_npy_path":    p(base_dir, f"{dataset_prefix}_y.npy"),
        "x_embed_npy":   p(base_dir, f"{dataset_prefix}_Xembed.npy"),
        "x_embed_npz":   p(base_dir, f"{dataset_prefix}_X_embed.npz"),   # 另一種命名
        "x_dense_csv":   p(base_dir, f"{dataset_prefix}_Xdense.csv"),
        "vocab_path":    p(base_dir, f"{dataset_prefix}_vocab.txt"),
    }
    return paths


def safe_load_manifest(manifest_path: str) -> dict:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_model(cur, provider: str, model_name: str, dim: int,
                 tokenization: Optional[str], version: Optional[str], notes: Optional[str]) -> int:
    """
    UPSERT 到 emb_models，回傳 model_id
    """
    sql = """
    INSERT INTO emb_models (provider, model_name, dim, tokenization, version, notes)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (provider, model_name, version)
    DO UPDATE SET dim = EXCLUDED.dim,
                  tokenization = EXCLUDED.tokenization,
                  notes = EXCLUDED.notes
    RETURNING id;
    """
    cur.execute(sql, (provider, model_name, dim, tokenization, version, notes))
    model_id = cur.fetchone()[0]
    cur.connection.commit()
    return model_id


def upsert_dataset(cur, model_id: int, payload: dict) -> int:
    """
    UPSERT 到 emb_datasets，回傳 emb_dataset.id
    唯一鍵：(model_id, dataset_prefix)
    """
    sql = """
    INSERT INTO emb_datasets
      (model_id, run_name, mode, text_source, pooling, date_cutoff,
       dataset_prefix, base_dir, manifest_path, x_path, y_path, meta_path,
       vocab_path, dense_path, dim, n_rows)
    VALUES
      (%(model_id)s, %(run_name)s, %(mode)s, %(text_source)s, %(pooling)s, %(date_cutoff)s,
       %(dataset_prefix)s, %(base_dir)s, %(manifest_path)s, %(x_path)s, %(y_path)s, %(meta_path)s,
       %(vocab_path)s, %(dense_path)s, %(dim)s, %(n_rows)s)
    ON CONFLICT (model_id, dataset_prefix)
    DO UPDATE SET
      run_name = EXCLUDED.run_name,
      mode = EXCLUDED.mode,
      text_source = EXCLUDED.text_source,
      pooling = EXCLUDED.pooling,
      date_cutoff = EXCLUDED.date_cutoff,
      base_dir = EXCLUDED.base_dir,
      manifest_path = EXCLUDED.manifest_path,
      x_path = EXCLUDED.x_path,
      y_path = EXCLUDED.y_path,
      meta_path = EXCLUDED.meta_path,
      vocab_path = EXCLUDED.vocab_path,
      dense_path = EXCLUDED.dense_path,
      dim = EXCLUDED.dim,
      n_rows = EXCLUDED.n_rows
    RETURNING id;
    """
    cur.execute(sql, payload | {"model_id": model_id})
    dataset_id = cur.fetchone()[0]
    cur.connection.commit()
    return dataset_id


def main():
    args = parse_args()

    # 解析來源：manifest 優先；否則 base_dir + dataset_prefix
    if args.manifest:
        manifest_path = args.manifest
        if not os.path.exists(manifest_path):
            print(f"[ERROR] manifest 路徑不存在：{manifest_path}")
            sys.exit(1)
        base_dir = os.path.dirname(manifest_path)
        # 由檔名推 dataset_prefix（去掉尾端 _manifest.json）
        fname = os.path.basename(manifest_path)
        if not fname.endswith("_manifest.json"):
            print("[ERROR] manifest 檔名必須以 _manifest.json 結尾")
            sys.exit(1)
        dataset_prefix = fname[:-len("_manifest.json")]
        paths = locate_files(base_dir, dataset_prefix)
        paths["manifest_path"] = manifest_path
    else:
        if not (args.base_dir and args.dataset_prefix):
            print("[ERROR] 請提供 --manifest，或同時提供 --base-dir 與 --dataset-prefix")
            sys.exit(1)
        base_dir = args.base_dir
        dataset_prefix = args.dataset_prefix
        paths = locate_files(base_dir, dataset_prefix)
        manifest_path = paths["manifest_path"]
        if not os.path.exists(manifest_path):
            print(f"[ERROR] 找不到 manifest：{manifest_path}")
            sys.exit(1)

    manifest = safe_load_manifest(manifest_path)

    # --- 從 manifest / 參數彙整模型資訊 ---
    provider = args.provider or manifest.get("provider") or manifest.get("embed_provider") \
               or manifest.get("embedding", {}).get("provider")
    model_name = args.model_name or manifest.get("model_name") or manifest.get("embed_model") \
                 or manifest.get("embedding", {}).get("model_name")
    dim = args.dim or manifest.get("dim") or manifest.get("embed_dim") \
          or manifest.get("embedding", {}).get("dim")
    tokenization = args.tokenization or manifest.get("tokenization")
    version = args.version or manifest.get("pipeline_version") or manifest.get("version")
    notes = args.notes or manifest.get("notes")

    if not provider or not model_name or not dim:
        print("[ERROR] 缺少模型資訊（provider / model_name / dim）。可用 --provider/--model-name/--dim 覆蓋。")
        sys.exit(1)

    # --- 一般資料集欄位 ---
    mode = args.mode or manifest.get("mode")
    text_source = args.text_source or manifest.get("text_source") or manifest.get("data", {}).get("text_source")
    pooling = args.pooling or manifest.get("pooling") or manifest.get("embedding", {}).get("pooling")
    date_cutoff = _str_to_date(args.date_cutoff or manifest.get("date_cutoff") or manifest.get("data", {}).get("date_cutoff"))
    run_name = args.run_name or manifest.get("run_name")

    # 檔案路徑：x_path 以 Xembed 為主；若同時存在 .npy / .npz，取 .npy（或你想反過來）
    x_path = None
    if os.path.exists(paths["x_embed_npy"]):
        x_path = paths["x_embed_npy"]
    elif os.path.exists(paths["x_embed_npz"]):
        x_path = paths["x_embed_npz"]
    else:
        # 也支援 emb_build 另一組命名 X_embed.npz
        # 若仍找不到，就從 manifest 讀
        x_path = manifest.get("paths", {}).get("X_embed") or manifest.get("X_embed_path")

    # y：CSV 優先（你現在用 CSV）
    y_path = None
    if os.path.exists(paths["y_csv_path"]):
        y_path = paths["y_csv_path"]
    elif os.path.exists(paths["y_npy_path"]):
        y_path = paths["y_npy_path"]
    else:
        y_path = manifest.get("paths", {}).get("y") or manifest.get("y_path")

    meta_path = paths["meta_path"] if os.path.exists(paths["meta_path"]) else manifest.get("paths", {}).get("meta")
    vocab_path = paths["vocab_path"] if os.path.exists(paths["vocab_path"]) else manifest.get("paths", {}).get("vocab")
    dense_path = paths["x_dense_csv"] if os.path.exists(paths["x_dense_csv"]) else manifest.get("paths", {}).get("Xdense")

    # 估 n_rows：manifest 若有記錄就用；否則從 meta.csv 估
    n_rows = manifest.get("n_rows") or manifest.get("num_samples")
    if not n_rows and meta_path and os.path.exists(meta_path):
        try:
            import pandas as pd
            n_rows = int(sum(1 for _ in open(meta_path, "r", encoding="utf-8")) - 1)  # 粗略數行數（扣 header）
        except Exception:
            n_rows = None

    db = DatabaseConfig()
    conn = db.get_connection()
    conn.autocommit = False
    cur = conn.cursor()

    try:
        model_id = ensure_model(
            cur,
            provider=provider,
            model_name=model_name,
            dim=int(dim),
            tokenization=tokenization,
            version=version,
            notes=notes,
        )

        payload = {
            "run_name": run_name,
            "mode": mode,
            "text_source": text_source,
            "pooling": pooling,
            "date_cutoff": date_cutoff,
            "dataset_prefix": dataset_prefix,
            "base_dir": base_dir,
            "manifest_path": manifest_path,
            "x_path": x_path,
            "y_path": y_path,
            "meta_path": meta_path,
            "vocab_path": vocab_path,
            "dense_path": dense_path,
            "dim": int(dim),
            "n_rows": n_rows,
        }

        dataset_id = upsert_dataset(cur, model_id=model_id, payload=payload)

        print("\n=== EMBEDDING DATASET INGESTED ===")
        print(f"  model_id      : {model_id}")
        print(f"  dataset_id    : {dataset_id}")
        print(f"  provider/model: {provider} / {model_name} (dim={dim}, tokenization={tokenization})")
        print(f"  version       : {version}")
        print(f"  mode/textSrc  : {mode} / {text_source}  pooling={pooling}")
        print(f"  prefix        : {dataset_prefix}")
        print(f"  base_dir      : {base_dir}")
        print(f"  X path        : {x_path}")
        print(f"  y path        : {y_path}")
        print(f"  meta path     : {meta_path}")
        if vocab_path:
            print(f"  vocab path    : {vocab_path}")
        if dense_path:
            print(f"  Xdense path   : {dense_path}")
        print(f"  n_rows        : {n_rows}")
        print("==================================\n")

    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()