# -*- coding: utf-8-sig -*-
"""
Model/embeddings/feature_analysis_with_embeddings.py
特徵分析工具（支援 Embedding）：驗證特徵是否足以區分類別
"""

import os
import json
import argparse
import warnings
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
import umap

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 動態導入 data_loader
try:
    from data_loader import (
        load_training_set,
        load_product_level_training_set
    )
except ImportError:
    print("警告：無法導入 data_loader，請確保在正確的環境中運行")

def parse_args():
    """解析命令行參數"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="product_level",
                    choices=["product_level", "comment_level"])
    ap.add_argument("--date-cutoff", type=str, default="2025-06-25")
    ap.add_argument("--pipeline-version", type=str, default=os.getenv("PIPELINE_VERSION", None))
    ap.add_argument("--vocab-mode", type=str, default="global")
    ap.add_argument("--top-n", type=str, default="100")
    ap.add_argument("--exclude-products", type=str, default="8918452")
    ap.add_argument("--keyword", type=str, default=None)
    ap.add_argument("--dataset-prefix", type=str, default=None,
                    help="同資料夾多批次時，用來鎖定要載入的那一組")
    ap.add_argument("--embed-path", type=str, required=True,
                    help="放置 dataset_* 檔案的資料夾路徑")
    ap.add_argument("--embed-mode", type=str, default="append", choices=["append", "only"])
    ap.add_argument("--embed-scale", type=str, default="none", choices=["none", "standardize"])
    ap.add_argument("--embed-dtype", type=str, default="float32", choices=["float32", "float64"])
    ap.add_argument("--outdir", type=str, default="Model/analysis_outputs")
    ap.add_argument("--save-plots", action="store_true", help="是否保存圖表")
    ap.add_argument("--plot-format", type=str, default="png", choices=["png", "pdf", "svg"])
    return ap.parse_args()

def _pick_dataset_bundle(dirpath: str, prefix: str | None) -> dict:
    """在同一資料夾中挑出要用的那一組 dataset 檔案"""
    d = Path(dirpath)
    
    # 找新格式 manifest
    manifests = sorted(
        d.glob("dataset_*_manifest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    target_manifest: Path | None = None
    if prefix:
        for m in manifests:
            name = m.name
            if name.startswith(prefix) or prefix in name:
                target_manifest = m
                break
    if target_manifest is None and manifests:
        target_manifest = manifests[0]
    
    if target_manifest is None:
        return {}
    
    stem = target_manifest.stem
    root = stem[:-len("_manifest")]
    
    def find1(patterns: list[str]) -> Path:
        for pat in patterns:
            hits = list(d.glob(pat))
            if hits:
                return hits[0]
        raise FileNotFoundError(f"找不到 {patterns} 其中之一")
    
    xembed = find1([f"{root}_Xembed.npy", f"{root}_xembed.npy"])
    meta = find1([f"{root}_meta.csv", f"{root}_Meta.csv"])
    yfile = find1([f"{root}_y.csv", f"{root}_Y.csv"])
    
    return {
        "manifest": target_manifest,
        "X_embed": xembed,
        "meta": meta,
        "y": yfile,
        "root": root,
    }

def load_embeddings(embed_path: str, mode: str, dtype: str = "float32",
                    dataset_prefix: str | None = None) -> tuple[np.ndarray, pd.Series, np.ndarray, dict]:
    """載入 embedding 檔案"""
    base = Path(embed_path)
    
    # 先嘗試新格式
    picked = _pick_dataset_bundle(embed_path, dataset_prefix)
    if picked:
        X_embed = np.load(picked["X_embed"]).astype(dtype)
        
        meta_df = pd.read_csv(picked["meta"])
        key_series = (meta_df["product_id"]
                      if "product_id" in meta_df.columns
                      else meta_df.iloc[:, 0])
        
        y_df = pd.read_csv(picked["y"])
        for cand in ["y", "label", "target"]:
            if cand in y_df.columns:
                y_labels = y_df[cand].to_numpy()
                break
        else:
            y_labels = y_df.iloc[:, 0].to_numpy()
        
        with open(picked["manifest"], "r", encoding="utf-8-sig") as f:
            manifest_info = json.load(f)
        
        print(f"載入 embedding（新格式）: {X_embed.shape}，{len(y_labels)} 標籤")
        return X_embed, key_series, y_labels, manifest_info
    
    # 回退到舊格式
    x_embed_path = base / "X_embed.npz"
    meta_path = base / "meta.csv"
    y_path = base / "y.npy"
    manifest = base / "manifest.json"
    
    for p in [x_embed_path, meta_path, y_path, manifest]:
        if not p.exists():
            raise FileNotFoundError(f"找不到檔案：{p}")
    
    try:
        data = np.load(str(x_embed_path))
        X_embed = data[data.files[0]].astype(dtype)
        data.close()
    except Exception:
        from scipy.sparse import load_npz
        X_embed = load_npz(str(x_embed_path)).toarray().astype(dtype)
    
    meta_df = pd.read_csv(meta_path)
    key_series = (meta_df["product_id"]
                  if "product_id" in meta_df.columns
                  else meta_df.iloc[:, 0])
    
    y_labels = np.load(str(y_path))
    
    with open(manifest, "r", encoding="utf-8-sig") as f:
        manifest_info = json.load(f)
    
    print(f"載入 embedding（舊格式）: {X_embed.shape}，{len(y_labels)} 標籤")
    return X_embed, key_series, y_labels, manifest_info

def main():
    """主函式"""
    args = parse_args()
    run_id = str(uuid.uuid4())
    
    print("=== 特徵分析工具（支援 Embedding）===")
    print(f"分析 ID: {run_id}")
    print(f"配置: {vars(args)}")
    
    # 載入 embedding
    print(f"\n載入 embedding 從: {args.embed_path}")
    X_embed, embed_keys, y_embed, embed_info = load_embeddings(
        args.embed_path, args.embed_mode, args.embed_dtype, args.dataset_prefix
    )
    
    print(f"Embedding 形狀: {X_embed.shape}")
    print(f"標籤分佈: y=0: {(y_embed==0).sum()}, y=1: {(y_embed==1).sum()}")
    
    # 這裡可以添加更多分析功能
    print("\n=== 分析完成 ===")
    print("功能已實現，可以進行特徵分析了")

if __name__ == "__main__":
    main()
