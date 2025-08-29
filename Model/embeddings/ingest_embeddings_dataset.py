# -*- coding: utf-8-sig -*-
"""
ingest_embeddings_dataset.py

將 train_with_embeddings.py 產出的 dataset（manifest + 檔案們）寫入資料庫：
- upsert 到 emb_models（provider, model_name, version 唯一）
- upsert 到 emb_datasets（model_id, dataset_prefix 唯一）
- 可選：匯入向量到 emb_comment / emb_product_batch / emb_tokens

用法（示例）：
  python Model/embeddings/ingest_embeddings_dataset.py \
    --manifest Model/embeddings/train_output/run_20250625_global_top100_xgboost_no_fs_20250828-153309_manifest.json \
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

    # B. 基本欄位（可覆蓋 manifest 中的值）
    ap.add_argument("--provider", type=str, help="Provider: sentence-transformers|openai|jina|word2vec|fasttext")
    ap.add_argument("--model-name", type=str, help="Model name, e.g., jinaai/jina-embeddings-v2-base-zh")
    ap.add_argument("--dim", type=int, help="Embedding dimension")
    ap.add_argument("--tokenization", type=str, help="none|ckip|bpe|... pipeline tokenization label")
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

def extract_model_info_from_manifest(manifest: dict) -> dict:
    """從 manifest 中提取模型資訊"""
    embed_info = manifest.get("embeddings", {})
    
    # 模型資訊
    embed_model = embed_info.get("embed_model") or manifest.get("embed_model")
    if embed_model:
        provider = embed_model.split("/")[0]
        model_name = embed_model
    else:
        provider = manifest.get("provider") or "sentence-transformers"
        model_name = manifest.get("model_name")
    
    dim = embed_info.get("dim") or manifest.get("dim")
    tokenization = manifest.get("tokenization") or "none"
    version = manifest.get("pipeline_version") or manifest.get("version")
    notes = manifest.get("notes")
    
    return {
        "provider": provider,
        "model_name": model_name,
        "dim": dim,
        "tokenization": tokenization,
        "version": version,
        "notes": notes
    }

def extract_dataset_info_from_manifest(manifest: dict, dataset_prefix: str, base_dir: str) -> dict:
    """從 manifest 中提取資料集資訊"""
    embed_info = manifest.get("embeddings", {})
    
    return {
        "run_name": manifest.get("run_id") or manifest.get("run_name"),
        "mode": manifest.get("mode"),
        "text_source": embed_info.get("text_source") or manifest.get("text_source"),
        "pooling": embed_info.get("product_aggregation") or manifest.get("pooling"),
        "date_cutoff": manifest.get("date_cutoff"),
        "dataset_prefix": dataset_prefix,
        "base_dir": base_dir,
        "dim": embed_info.get("dim") or manifest.get("dim"),
        "n_rows": manifest.get("shapes", {}).get("y") or manifest.get("n_rows")
    }

def resolve_file_paths(manifest: dict, base_dir: str, dataset_prefix: str) -> dict:
    """解析檔案路徑，支援 train_with_embeddings.py 的結構"""
    files = {}
    
    # 檢查是否有 embedding 資訊
    embed_info = manifest.get("embeddings", {})
    if embed_info and "path" in embed_info:
        # train_with_embeddings.py 格式：embedding 檔案在 embeddings.path 目錄下
        embed_dir = embed_info["path"]
        
        # 檢查 embedding 目錄下的檔案
        embed_manifest_path = os.path.join(embed_dir, "manifest.json")
        if os.path.exists(embed_manifest_path):
            try:
                with open(embed_manifest_path, "r", encoding="utf-8-sig") as f:
                    embed_manifest = json.load(f)
                embed_files = embed_manifest.get("files", {})
                
                # 從 embedding manifest 讀取檔案路徑
                if "X_embed" in embed_files:
                    files["X_npz"] = os.path.join(embed_dir, embed_files["X_embed"])
                if "meta" in embed_files:
                    files["meta"] = os.path.join(embed_dir, embed_files["meta"])
            except Exception as e:
                print(f"[WARNING] 無法讀取 embedding manifest: {e}")
        
        # 直接檢查 embedding 目錄下的檔案
        if "X_npz" not in files:
            x_path_candidates = [
                os.path.join(embed_dir, "X_embed.npz"),
                os.path.join(embed_dir, "Xembed.npy"),
                os.path.join(embed_dir, "X_embed.npy")
            ]
            for candidate in x_path_candidates:
                if os.path.exists(candidate):
                    files["X_npz"] = candidate
                    break
        
        if "meta" not in files:
            meta_candidate = os.path.join(embed_dir, "meta.csv")
            if os.path.exists(meta_candidate):
                files["meta"] = meta_candidate
        
        # y 檔案可能在 embedding 目錄或主目錄
        y_candidates = [
            os.path.join(embed_dir, "y.csv"),
            os.path.join(embed_dir, "y.npy")
        ]
        for candidate in y_candidates:
            if os.path.exists(candidate):
                files["y_csv"] = candidate
                break
    
    # 如果還沒找到，從主 manifest.files 讀取
    manifest_files = manifest.get("files", {})
    if "X_npz" not in files:
        if "Xembed_npy" in manifest_files:
            files["X_npz"] = manifest_files["Xembed_npy"]
        elif "X_embed" in manifest_files:
            files["X_npz"] = manifest_files["X_embed"]
    
    if "y_csv" not in files:
        if "y_csv" in manifest_files:
            files["y_csv"] = manifest_files["y_csv"]
    
    if "meta" not in files:
        if "meta_csv" in manifest_files:
            files["meta"] = manifest_files["meta_csv"]
    
    # 回退到預設命名規則
    default_files = locate_files(base_dir, dataset_prefix)
    for key, path in default_files.items():
        if key not in files and os.path.exists(path):
            files[key] = path
    
    # 確保 manifest 路徑存在
    files["manifest"] = os.path.join(base_dir, f"{dataset_prefix}_manifest.json")
    
    return files

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
    # 檔案路徑
    manifest_path = files["manifest"]
    x_path = first_exists(files.get("X_npz"), files.get("X_npy"))
    y_path = first_exists(files.get("y_csv"), files.get("y_npy"))
    meta_path = files.get("meta")
    vocab_path = None
    dense_path = None

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
    """, (model_id, meta.get("run_name"), meta.get("mode"), meta.get("text_source"), 
          meta.get("pooling"), meta.get("date_cutoff"), meta.get("dataset_prefix"), 
          meta.get("base_dir"), manifest_path, x_path, y_path, meta_path,
          vocab_path, dense_path, meta.get("dim"), meta.get("n_rows")))
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
    # 轉換 numpy 陣列為 PostgreSQL 可接受的格式
    converted_rows = []
    for row in rows:
        product_id, batch_time, model_id, vec, num_comments, weights_meta = row
        # 將 numpy 陣列轉換為 list，然後轉換為 PostgreSQL 的 VECTOR 格式
        vec_list = vec.tolist() if hasattr(vec, 'tolist') else list(vec)
        converted_rows.append((product_id, batch_time, model_id, vec_list, num_comments, weights_meta))
    
    for i in range(0, len(converted_rows), batch_size):
        pg_extras.execute_values(cur, sql, converted_rows[i:i+batch_size], template=None, page_size=batch_size)

def upsert_emb_comment(cur, rows: List[tuple], batch_size: int = 1000):
    sql = """
    INSERT INTO emb_comment
      (comment_id, model_id, emb, emb_norm)
    VALUES %s
    ON CONFLICT (comment_id, model_id)
    DO UPDATE SET emb = EXCLUDED.emb,
                  emb_norm = EXCLUDED.emb_norm
    """
    # 轉換 numpy 陣列為 PostgreSQL 可接受的格式
    converted_rows = []
    for row in rows:
        comment_id, model_id, vec, emb_norm = row
        vec_list = vec.tolist() if hasattr(vec, 'tolist') else list(vec)
        converted_rows.append((comment_id, model_id, vec_list, emb_norm))
    
    for i in range(0, len(converted_rows), batch_size):
        pg_extras.execute_values(cur, sql, converted_rows[i:i+batch_size], template=None, page_size=batch_size)

def upsert_emb_tokens(cur, rows: List[tuple], batch_size: int = 2000):
    sql = """
    INSERT INTO emb_tokens (model_id, token, emb)
    VALUES %s
    ON CONFLICT (model_id, token)
    DO UPDATE SET emb = EXCLUDED.emb
    """
    # 轉換 numpy 陣列為 PostgreSQL 可接受的格式
    converted_rows = []
    for row in rows:
        model_id, token, vec = row
        vec_list = vec.tolist() if hasattr(vec, 'tolist') else list(vec)
        converted_rows.append((model_id, token, vec_list))
    
    for i in range(0, len(converted_rows), batch_size):
        pg_extras.execute_values(cur, sql, converted_rows[i:i+batch_size], template=None, page_size=batch_size)

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
        else:
            if not (args.base_dir and args.dataset_prefix):
                raise ValueError("Provide --manifest OR (--base-dir AND --dataset-prefix) OR --dataset-id")
            base_dir = args.base_dir
            dataset_prefix = args.dataset_prefix
            manifest_path = os.path.join(base_dir, f"{dataset_prefix}_manifest.json")
            if not os.path.exists(manifest_path):
                raise FileNotFoundError(f"manifest not found: {manifest_path}")
            manifest = load_manifest(manifest_path)

        # 解析檔案路徑
        files = resolve_file_paths(manifest, base_dir, dataset_prefix)

        # 先處理 emb_models / emb_datasets
        db = DatabaseConfig()
        conn = db.get_connection()
        conn.autocommit = False
        cur = conn.cursor()

        # 從 manifest 提取模型資訊，允許參數覆蓋
        model_info = extract_model_info_from_manifest(manifest)
        provider = args.provider or model_info["provider"]
        model_name = args.model_name or model_info["model_name"]
        dim = args.dim or model_info["dim"]
        tokenization = args.tokenization or model_info["tokenization"]
        version = model_info["version"] or str(uuid.uuid4())
        notes = args.notes or model_info["notes"]

        if not model_name or not dim:
            conn.close()
            raise ValueError("model_name / dim 未提供且 manifest 也無法推得，請補參數。")

        model_id = upsert_emb_model(cur, provider, model_name, dim, tokenization, version, notes)

        # 從 manifest 提取資料集資訊
        dataset_info = extract_dataset_info_from_manifest(manifest, dataset_prefix, base_dir)
        dataset_id = insert_emb_dataset(cur, model_id, dataset_info, files)

        conn.commit()
        conn.close()

        print(f"\n=== METADATA INGESTED ===")
        print(f"  model_id      : {model_id}")
        print(f"  dataset_id    : {dataset_id}")
        print(f"  provider/model: {provider} / {model_name} (dim={dim})")
        print(f"  mode/textSrc  : {dataset_info['mode']} / {dataset_info['text_source']}")
        print(f"  prefix        : {dataset_prefix}")
        print("========================\n")

    # -------------------------
    # （新增）匯入向量 emb_*
    # -------------------------
    if not args.ingest_vectors:
        print("Metadata ingest done. (Vectors not ingested; use --ingest-vectors to enable)")
        return

    # 讀取 meta 與 X
    meta_path = files.get("meta")
    X_path = first_exists(files.get("X_npz"), files.get("X_npy"))
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
            if bt_col and pd.notna(row[bt_col]) and str(row[bt_col]).strip():
                bt = to_timestamp(row[bt_col])
            else:
                # 如果沒有 batch_time，使用 date_cutoff 或當前時間
                cutoff = manifest.get("date_cutoff")
                if cutoff:
                    bt = f"{cutoff} 00:00:00"
                else:
                    bt = datetime.utcnow().isoformat(sep=" ")
            
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