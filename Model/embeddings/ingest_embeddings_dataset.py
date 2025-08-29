# -*- coding: utf-8-sig -*-
"""
ingest_embeddings_dataset.py

將 train_with_embeddings.py 產出的 dataset（manifest + 檔案們）寫入資料庫：
- upsert 到 emb_models（provider, model_name, version 唯一）
- upsert 到 emb_datasets（model_id, dataset_prefix 唯一）

用法（示例）：
  python Model/ingest_embeddings_dataset.py \
    --manifest Model/embeddings/train_output/datasets/dataset_product_level_20250625_..._manifest.json \
    --ingest-vectors --target auto --batch-size 1000

也可透過 --base-dir + --dataset-prefix 指定一批資料，或 --dataset-id（先插入 emb_datasets 後再補向量）。
"""

from __future__ import annotations
import os, sys, json, argparse, math, uuid
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras as pg_extras
# 專案根目錄 (…/momo_crawler-main)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # .../Model -> 上一層 = 專案根目錄
sys.path.insert(0, str(REPO_ROOT))

# 沿用 csv_to_db.py 的連線方式
from config.database import DatabaseConfig  # type: ignore

def parse_args():
    ap = argparse.ArgumentParser(description="Ingest embedding dataset metadata + vectors")
    # A. 來源指定（擇一）
    ap.add_argument("--manifest", type=str, help="Path to *_manifest.json produced by emb_build/train_with_embeddings")
    ap.add_argument("--base-dir", type=str, help="Base dir for dataset files")
    ap.add_argument("--dataset-prefix", type=str, help="Dataset prefix (without suffix)")
    ap.add_argument("--dataset-id", type=int, help="Use an existing emb_datasets.id to fetch paths")

    # B. 基本欄位（原本就有的：寫 emb_models / emb_datasets 時用）
    ap.add_argument("--provider", type=str, help="Provider: sentence-transformers|openai|jina|word2vec|fasttext")
    ap.add_argument("--model-name", type=str, help="Model name, e.g., jinaai/jina-embeddings-v2-base-zh")
    ap.add_argument("--dim", type=int, help="Embedding dimension")
    ap.add_argument("--tokenization", type=str, default="none", help="none|ckip|bpe|... pipeline tokenization label")
    ap.add_argument("--notes", type=str, help="free-form notes")

    # C. 控制是否也匯入向量
    ap.add_argument("--ingest-vectors", action="store_true", help="After metadata, also ingest vectors into emb_* tables")
    ap.add_argument("--target", type=str, default="auto", choices=["auto", "product", "comment", "tokens"], help="Where to write vectors. 'auto' derived from manifest.mode")
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")

    return ap.parse_args()

# --------------------------
# 工具
# --------------------------
def first_exists(*paths) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def locate_files(base_dir: str, dataset_prefix: str) -> dict:
    p = os.path.join
    return {
        "manifest": p(base_dir, f"{dataset_prefix}_manifest.json"),
        "meta":     p(base_dir, f"{dataset_prefix}_meta.csv"),
        "X_npz":    p(base_dir, f"{dataset_prefix}_X_embed.npz"),
        "X_npy":    p(base_dir, f"{dataset_prefix}_Xembed.npy"),
        "y_csv":    p(base_dir, f"{dataset_prefix}_y.csv"),
        "y_npy":    p(base_dir, f"{dataset_prefix}_y.npy"),
        # 可選 token 輸出（若你將來支援）
        "tokens_csv": p(base_dir, f"{dataset_prefix}_tokens.csv"),
        "tokens_npz": p(base_dir, f"{dataset_prefix}_tokens.npz"),
    }

def to_timestamp(s: str) -> Optional[str]:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s[:len(fmt)], fmt).isoformat(sep=" ")
        except Exception:
            continue
    return s or None

def l2_norm(vec: np.ndarray) -> float:
    v = float(np.linalg.norm(vec)) if vec is not None else 0.0
    return 0.0 if math.isnan(v) else v

# --------------------------
# DB helpers
# --------------------------
def upsert_emb_model(cur, provider, model_name, dim, tokenization, version, notes) -> int:
    cur.execute("""
        INSERT INTO emb_models (provider, model_name, dim, tokenization, version, notes)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (provider, model_name, version)
        DO UPDATE SET dim = EXCLUDED.dim, tokenization = EXCLUDED.tokenization, notes = EXCLUDED.notes
        RETURNING id
    """, (provider, model_name, dim, tokenization, version, notes))
    return cur.fetchone()[0]

def insert_emb_dataset(cur, model_id: int, meta: dict, files: dict) -> int:
    # 這裡把你 manifest 的重要欄位填進 emb_datasets
    run_name = meta.get("run_name")
    mode = meta.get("mode")
    text_source = meta.get("text_source")
    pooling = (meta.get("embeddings") or {}).get("product_aggregation") or meta.get("pooling")
    date_cutoff = meta.get("date_cutoff")
    dataset_prefix = meta.get("dataset_prefix")
    base_dir = meta.get("base_dir")

    # 檔案
    manifest_path = files["manifest"]
    x_path = first_exists(files["X_npz"], files["X_npy"])
    y_path = first_exists(files["y_csv"], files["y_npy"])
    meta_path = files["meta"]
    vocab_path = None
    dense_path = None

    dim = int(meta.get("dim") or (meta.get("embeddings") or {}).get("dim") or 0)
    n_rows = int(meta.get("n_rows") or 0)

    cur.execute("""
        INSERT INTO emb_datasets
            (model_id, run_name, mode, text_source, pooling, date_cutoff,
             dataset_prefix, base_dir, manifest_path, x_path, y_path, meta_path,
             vocab_path, dense_path, dim, n_rows)
        VALUES
            (%s, %s, %s, %s, %s, %s,
             %s, %s, %s, %s, %s, %s,
             %s, %s, %s, %s)
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
             dim = EXCLUDED.dim,
             n_rows = EXCLUDED.n_rows
        RETURNING id
    """, (model_id, run_name, mode, text_source, pooling, date_cutoff,
          dataset_prefix, base_dir, manifest_path, x_path, y_path, meta_path,
          vocab_path, dense_path, dim, n_rows))
    return cur.fetchone()[0]

def upsert_emb_product_batch(cur, rows: List[tuple], batch_size: int = 1000):
    sql = """
    INSERT INTO emb_product_batch
      (product_id, batch_time, model_id, emb, num_comments, weights_meta)
    VALUES %s
    ON CONFLICT (product_id, batch_time, model_id)
    DO UPDATE SET emb = EXCLUDED.emb,
                  num_comments = EXCLUDED.num_comments,
                  weights_meta = EXCLUDED.weights_meta
    """
    for i in range(0, len(rows), batch_size):
        pg_extras.execute_values(cur, sql, rows[i:i+batch_size], template=None, page_size=batch_size)

def upsert_emb_comment(cur, rows: List[tuple], batch_size: int = 1000):
    sql = """
    INSERT INTO emb_comment
      (comment_id, model_id, emb, emb_norm)
    VALUES %s
    ON CONFLICT (comment_id, model_id)
    DO UPDATE SET emb = EXCLUDED.emb,
                  emb_norm = EXCLUDED.emb_norm
    """
    for i in range(0, len(rows), batch_size):
        pg_extras.execute_values(cur, sql, rows[i:i+batch_size], template=None, page_size=batch_size)

def upsert_emb_tokens(cur, rows: List[tuple], batch_size: int = 2000):
    sql = """
    INSERT INTO emb_tokens (model_id, token, emb)
    VALUES %s
    ON CONFLICT (model_id, token)
    DO UPDATE SET emb = EXCLUDED.emb
    """
    for i in range(0, len(rows), batch_size):
        pg_extras.execute_values(cur, sql, rows[i:i+batch_size], template=None, page_size=batch_size)

# --------------------------
# 主流程
# --------------------------
def main():
    args = parse_args()

    # 解析資料來源（manifest / base+prefix / dataset-id）
    if args.dataset_id:
        # 從 emb_datasets 反查檔案與 model_id
        db = DatabaseConfig()
        conn = db.get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT ed.model_id, ed.base_dir, ed.dataset_prefix, ed.manifest_path,
                   ed.x_path, ed.y_path, ed.meta_path, ed.dim
              FROM emb_datasets ed
             WHERE ed.id = %s
        """, (args.dataset_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            raise ValueError(f"emb_datasets.id={args.dataset_id} not found.")
        model_id, base_dir, dataset_prefix, manifest_path, x_path, y_path, meta_path, dim = row
        files = locate_files(base_dir, dataset_prefix)
        files["manifest"] = manifest_path
        files["meta"] = meta_path
        # 嘗試保留原先記錄的 x_path/y_path，但若不存在則用預設命名
        if x_path and os.path.exists(x_path):
            files["X_npz"] = x_path if x_path.endswith(".npz") else files["X_npz"]
            files["X_npy"] = x_path if x_path.endswith(".npy") else files["X_npy"]
        if y_path and os.path.exists(y_path):
            files["y_csv"] = y_path if y_path.endswith(".csv") else files["y_csv"]
            files["y_npy"] = y_path if y_path.endswith(".npy") else files["y_npy"]
        manifest = load_manifest(files["manifest"])
        conn.close()

    else:
        # manifest path 或 base-dir + dataset-prefix
        if args.manifest:
            manifest_path = args.manifest
            if not os.path.exists(manifest_path):
                raise FileNotFoundError(f"manifest not found: {manifest_path}")
            manifest = load_manifest(manifest_path)
            base_dir = os.path.dirname(manifest_path)
            dataset_prefix = os.path.basename(manifest_path)[:-len("_manifest.json")]
            files = locate_files(base_dir, dataset_prefix)
        else:
            if not (args.base_dir and args.dataset_prefix):
                raise ValueError("Provide --manifest OR (--base-dir AND --dataset-prefix) OR --dataset-id")
            base_dir = args.base_dir
            dataset_prefix = args.dataset_prefix
            files = locate_files(base_dir, dataset_prefix)
            manifest = load_manifest(files["manifest"])

        # 先處理 emb_models / emb_datasets（跟你原本相同精神）
        db = DatabaseConfig()
        conn = db.get_connection()
        conn.autocommit = False
        cur = conn.cursor()

        # 若沒手動給 provider/model/dim，就從 manifest 推測
        provider = args.provider or (manifest.get("provider") or "sentence-transformers")
        model_name = args.model_name or (manifest.get("embed_model") or manifest.get("model_name"))
        dim = args.dim or int(manifest.get("dim") or (manifest.get("embeddings") or {}).get("dim") or 0)
        tokenization = args.tokenization or (manifest.get("tokenization") or "none")
        version = str(uuid.uuid4())
        notes = args.notes or manifest.get("notes")

        if not model_name or not dim:
            conn.close()
            raise ValueError("model_name / dim 未提供且 manifest 也無法推得，請補參數。")

        model_id = upsert_emb_model(cur, provider, model_name, dim, tokenization, version, notes)

        # 填 emb_datasets：將 manifest 的 metadata 與實際檔案路徑紀錄進來
        # 先把 manifest 裡缺少的欄位補齊（run_name/mode/text_source/pooling/date_cutoff 等）
        meta_for_ds = {
            "run_name": manifest.get("run_name"),
            "mode": manifest.get("mode"),
            "text_source": manifest.get("text_source"),
            "pooling": (manifest.get("embeddings") or {}).get("product_aggregation"),
            "date_cutoff": manifest.get("date_cutoff"),
            "dataset_prefix": dataset_prefix,
            "base_dir": base_dir,
            "dim": dim,
            "n_rows": manifest.get("n_rows"),
        }
        dataset_id = insert_emb_dataset(cur, model_id, meta_for_ds, files)

        conn.commit()
        conn.close()

    # -------------------------
    # （新增）匯入向量 emb_*
    # -------------------------
    if not args.ingest_vectors:
        print("Metadata ingest done. (Vectors not ingested; use --ingest-vectors to enable)")
        return

    # 讀取 meta 與 X
    meta_path = files["meta"]
    X_path = first_exists(files["X_npz"], files["X_npy"])
    if not meta_path or not X_path:
        raise FileNotFoundError(f"meta or X_embed file not found. meta={meta_path}, X={X_path}")

    meta_df = pd.read_csv(meta_path, encoding="utf-8-sig")
    if X_path.endswith(".npz"):
        npz = np.load(X_path)
        X = npz[npz.files[0]]
    else:
        X = np.load(X_path)

    if args.max_rows:
        meta_df = meta_df.iloc[:args.max_rows].copy()
        X = X[:args.max_rows]

    if len(meta_df) != len(X):
        raise ValueError(f"meta rows ({len(meta_df)}) != X rows ({len(X)})")

    # 目標表
    mode = manifest.get("mode") or "product_level"
    target = args.target if args.target != "auto" else ("product" if mode == "product_level" else "comment")

    # 欄位對齊
    db = DatabaseConfig()
    conn2 = db.get_connection()
    conn2.autocommit = False
    cur2 = conn2.cursor()

    inserted = 0
    if target == "product":
        # 需要 product_id 與批次欄位（容錯找欄名）
        if "product_id" not in meta_df.columns:
            raise ValueError("meta.csv must contain product_id for product_level ingestion")
        pid_col = "product_id"
        bt_col = None
        for c in ["batch_time", "batch_capture_time", "aligned_batch_time", "capture_time"]:
            if c in meta_df.columns:
                bt_col = c
                break

        emb_info = manifest.get("embeddings") or {}
        weights_meta = {
            "pooling": emb_info.get("product_aggregation") or manifest.get("pooling") or "mean",
            "text_source": manifest.get("text_source"),
        }
        numc_col = None
        for c in ["num_comments", "comment_count", "n_comments"]:
            if c in meta_df.columns:
                numc_col = c
                break

        rows = []
        for i, row in meta_df.iterrows():
            pid = int(row[pid_col])
            if bt_col:
                bt = to_timestamp(row[bt_col])
            else:
                cutoff = manifest.get("date_cutoff")
                bt = f"{cutoff} 00:00:00" if cutoff else datetime.utcnow().isoformat(sep=" ")
            vec = X[i].astype(np.float32)
            numc = int(row[numc_col]) if numc_col else None
            rows.append((pid, bt, model_id, vec, numc, json.dumps(weights_meta)))

        if args.dry_run:
            print(f"[DRY RUN] Would insert emb_product_batch rows: {len(rows)}")
        else:
            upsert_emb_product_batch(cur2, rows, batch_size=args.batch_size)
            conn2.commit()
            inserted = len(rows)

    elif target == "comment":
        if "comment_id" not in meta_df.columns:
            raise ValueError("meta.csv must contain comment_id for comment_level ingestion")
        rows = []
        for i, row in meta_df.iterrows():
            cid = str(row["comment_id"])
            vec = X[i].astype(np.float32)
            rows.append((cid, model_id, vec, l2_norm(vec)))
        if args.dry_run:
            print(f"[DRY RUN] Would insert emb_comment rows: {len(rows)}")
        else:
            upsert_emb_comment(cur2, rows, batch_size=args.batch_size)
            conn2.commit()
            inserted = len(rows)

    else:  # tokens（可選）
        tokens_path = first_exists(files.get("tokens_csv"), files.get("tokens_npz"))
        if not tokens_path:
            print("Token vectors file not found; skip emb_tokens.")
        else:
            rows = []
            if tokens_path.endswith(".csv"):
                tdf = pd.read_csv(tokens_path, encoding="utf-8-sig")
                # 期待欄位：token, emb_0 ... emb_{dim-1}
                emb_cols = [c for c in tdf.columns if c.startswith("emb_")]
                for _, r in tdf.iterrows():
                    token = str(r["token"])
                    vec = r[emb_cols].values.astype(np.float32)
                    rows.append((model_id, token, vec))
            else:
                # 自行定義 npz 格式：keys = tokens, embeddings
                tnpz = np.load(tokens_path, allow_pickle=True)
                tokens = tnpz["tokens"]
                embs = tnpz["embeddings"]
                for token, vec in zip(tokens, embs):
                    rows.append((model_id, str(token), vec.astype(np.float32)))

            if args.dry_run:
                print(f"[DRY RUN] Would insert emb_tokens rows: {len(rows)}")
            else:
                upsert_emb_tokens(cur2, rows, batch_size=max(1000, args.batch_size))
                conn2.commit()
                inserted = len(rows)

    try:
        conn2.close()
    except Exception:
        pass

    print("\n=== INGEST COMPLETED ===")
    print(f"  mode/target : {mode} / {target}")
    print(f"  dataset     : {files.get('manifest')}")
    print(f"  meta        : {meta_path}")
    print(f"  X_embed     : {X_path} shape={X.shape}")
    print(f"  model_id    : {model_id}")
    print(f"  inserted    : {inserted}")
    print("========================\n")

if __name__ == "__main__":
    import json  # used above
    main()