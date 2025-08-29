# -*- coding: utf-8-sig -*-
"""
ingest_embeddings_dataset.py

將 train_with_embeddings.py 產出的 dataset（manifest + 檔案們）寫入資料庫：
- upsert 到 emb_models（provider, model_name, version 唯一）
- upsert 到 emb_datasets（model_id, dataset_prefix 唯一）

用法範例：
python ingest_embeddings_dataset.py \
  --base-dir Model/embeddings/train_output/datasets \
  --dataset-prefix dataset_product_level_20250625_global_top100_20250827-222350

或直接給 manifest 路徑：
python ingest_embeddings_dataset.py --manifest Model/embeddings/train_output/datasets/dataset_product_level_20250625_global_top100_20250827-222350_manifest.json
"""

from pathlib import Path
import sys
import os
import uuid
# 專案根目錄 (…/momo_crawler-main)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # .../Model -> 上一層 = 專案根目錄
sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
from datetime import datetime, date
from typing import Optional, Tuple

# 沿用 csv_to_db.py 的連線方式
from config.database import DatabaseConfig  # type: ignore

def parse_args():
    ap = argparse.ArgumentParser(
        description="Ingest embedding dataset (from train_with_embeddings.py) into emb_models / emb_datasets"
    )
    # 兩種指定方式擇一：manifest 或 base-dir + dataset-prefix
    ap.add_argument("--manifest", type=str, default=None,
                    help="Full path to *_manifest.json produced by train_with_embeddings.py")
    ap.add_argument("--base-dir", type=str, default=None,
                    help="Base dir containing dataset files (same as train_with_embeddings --outdir)")
    ap.add_argument("--dataset-prefix", type=str, default=None,
                    help="Dataset prefix (filename stem without suffix); e.g. run_20250625_global_top100_xgboost_no_fs_20250828-153309")

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
    with open(manifest_path, "r", encoding="utf-8-sig") as f:
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
        if not fname.endswith("manifest.json"):
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
    # 支援 train_with_embeddings.py 的格式：embeddings.embed_model
    embed_info = manifest.get("embeddings", {})
    provider = None
    model_name = None
    dim = None
    
    if embed_info:
        # train_with_embeddings.py 格式
        embed_model = embed_info.get("embed_model")
        if embed_model:
            provider = embed_model.split("/")[0]
            model_name = embed_model
        dim = embed_info.get("dim")
    else:
        # 原本 emb_build.py 格式
        provider = manifest.get("embed_model").split("/")[0] if manifest.get("embed_model") else None
        model_name = manifest.get("model_name") or manifest.get("embed_model") \
                     or manifest.get("embedding", {}).get("model_name")
        dim = manifest.get("dim") or manifest.get("embed_dim") \
              or manifest.get("embedding", {}).get("dim")
    
    tokenization = manifest.get("tokenization")
    version = str(uuid.uuid4())
    notes = manifest.get("notes") or None

    if not provider or not model_name or not dim:
        print("[ERROR] 缺少模型資訊（provider / model_name / dim）。")
        print(f"  provider: {provider}")
        print(f"  model_name: {model_name}")
        print(f"  dim: {dim}")
        print(f"  embed_info: {embed_info}")
        sys.exit(1)

    # --- 一般資料集欄位 ---
    mode = manifest.get("mode")
    text_source = None
    pooling = None
    
    if embed_info:
        # train_with_embeddings.py 格式
        text_source = embed_info.get("text_source")
        pooling = embed_info.get("product_aggregation")  # 這裡可能是 product_aggregation
    else:
        # 原本 emb_build.py 格式
        text_source = manifest.get("text_source") or manifest.get("data", {}).get("text_source")
        pooling = manifest.get("pooling") or manifest.get("embedding", {}).get("pooling")
    
    date_cutoff = _str_to_date(manifest.get("date_cutoff") or manifest.get("data", {}).get("date_cutoff"))
    run_name = manifest.get("run_id") or manifest.get("run_name")

    # 檔案路徑：從 manifest.files 讀取
    files = manifest.get("files", {})
    x_path = None
    y_path = None
    meta_path = None
    vocab_path = None
    dense_path = None
    
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
                    x_path = os.path.join(embed_dir, embed_files["X_embed"])
                elif "meta" in embed_files:
                    meta_path = os.path.join(embed_dir, embed_files["meta"])
            except Exception as e:
                print(f"[WARNING] 無法讀取 embedding manifest: {e}")
        
        # 直接檢查 embedding 目錄下的檔案
        if not x_path:
            x_path_candidates = [
                os.path.join(embed_dir, "X_embed.npz"),
                os.path.join(embed_dir, "Xembed.npy"),
                os.path.join(embed_dir, "X_embed.npy")
            ]
            for candidate in x_path_candidates:
                if os.path.exists(candidate):
                    x_path = candidate
                    break
        
        if not meta_path:
            meta_candidate = os.path.join(embed_dir, "meta.csv")
            if os.path.exists(meta_candidate):
                meta_path = meta_candidate
        
        # y 檔案可能在 embedding 目錄或主目錄
        y_candidates = [
            os.path.join(embed_dir, "y.csv"),
            os.path.join(embed_dir, "y.npy")
        ]
        for candidate in y_candidates:
            if os.path.exists(candidate):
                y_path = candidate
                break
    
    # 如果還沒找到，從主 manifest.files 讀取
    if not x_path:
        if "Xembed_npy" in files:
            x_path = files["Xembed_npy"]
        elif "X_embed" in files:
            x_path = files["X_embed"]
        else:
            # 回退到檔案系統檢查
            if os.path.exists(paths["x_embed_npy"]):
                x_path = paths["x_embed_npy"]
            elif os.path.exists(paths["x_embed_npz"]):
                x_path = paths["x_embed_npz"]

    if not y_path:
        if "y_csv" in files:
            y_path = files["y_csv"]
        else:
            # 回退到檔案系統檢查
            if os.path.exists(paths["y_csv_path"]):
                y_path = paths["y_csv_path"]
            elif os.path.exists(paths["y_npy_path"]):
                y_path = paths["y_npy_path"]

    if not meta_path:
        meta_path = files.get("meta_csv") or (paths["meta_path"] if os.path.exists(paths["meta_path"]) else None)
    
    vocab_path = files.get("vocab_txt") or (paths["vocab_path"] if os.path.exists(paths["vocab_path"]) else None)
    dense_path = files.get("Xdense_csv") or (paths["x_dense_csv"] if os.path.exists(paths["x_dense_csv"]) else None)

    # 估 n_rows：從 manifest.shapes 讀取
    shapes = manifest.get("shapes", {})
    n_rows = None
    if "y" in shapes:
        n_rows = shapes["y"]
    elif "X_embed" in shapes and len(shapes["X_embed"]) > 0:
        n_rows = shapes["X_embed"][0]
    elif "Xembed" in shapes and len(shapes["Xembed"]) > 0:
        n_rows = shapes["Xembed"][0]

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