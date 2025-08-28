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

def analyze_dense_feature_distributions(X_dense_df: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    分析 Dense 特徵在 y=0 vs y=1 之間的分佈差異
    """
    print("\n=== Dense 特徵分佈分析 ===")
    
    results = {
        "feature_analysis": [],
        "summary_stats": {
            "total_features": len(X_dense_df.columns),
            "features_with_significant_diff": 0,
            "features_with_high_separation": 0
        }
    }
    
    # 為每個特徵計算統計量
    for feature_name in X_dense_df.columns:
        feature_values = X_dense_df[feature_name]
        
        # 分組統計
        y0_values = feature_values[y == 0]
        y1_values = feature_values[y == 1]
        
        # 基本統計量
        y0_mean, y0_std = y0_values.mean(), y0_values.std()
        y1_mean, y1_std = y1_values.mean(), y1_values.std()
        
        # 計算分離度指標
        # 1. Cohen's d (標準化平均差)
        pooled_std = np.sqrt(((len(y0_values) - 1) * y0_std**2 + (len(y1_values) - 1) * y1_std**2) / (len(y0_values) + len(y1_values) - 2))
        cohens_d = (y1_mean - y0_mean) / pooled_std if pooled_std > 0 else 0
        
        # 2. 互信息 (Mutual Information)
        # 將連續值分箱以計算互信息
        bins = np.linspace(feature_values.min(), feature_values.max(), 10)
        y0_binned = np.digitize(y0_values, bins)
        y1_binned = np.digitize(y1_values, bins)
        all_binned = np.concatenate([y0_binned, y1_binned])
        all_labels = np.concatenate([np.zeros(len(y0_binned)), np.ones(len(y1_binned))])
        mi_score = mutual_info_score(all_labels, all_binned)
        
        # 3. Mann-Whitney U 檢定
        try:
            u_stat, p_value = stats.mannwhitneyu(y0_values, y1_values, alternative='two-sided')
        except:
            u_stat, p_value = 0, 1
        
        # 4. 重疊係數 (Overlap Coefficient)
        # 計算兩個分佈的重疊程度
        y0_quantiles = np.percentile(y0_values, [25, 75])
        y1_quantiles = np.percentile(y1_values, [25, 75])
        overlap = max(0, min(y0_quantiles[1], y1_quantiles[1]) - max(y0_quantiles[0], y1_quantiles[0]))
        overlap_coef = overlap / (y0_quantiles[1] - y0_quantiles[0] + y1_quantiles[1] - y1_quantiles[0] - overlap) if overlap > 0 else 0
        
        # 判斷特徵是否有顯著差異
        is_significant = p_value < 0.05
        has_high_separation = abs(cohens_d) > 0.5 or mi_score > 0.1
        
        feature_analysis = {
            "feature_name": feature_name,
            "y0_stats": {
                "count": len(y0_values),
                "mean": float(y0_mean),
                "std": float(y0_std),
                "min": float(y0_values.min()),
                "max": float(y0_values.max()),
                "q25": float(y0_values.quantile(0.25)),
                "q75": float(y0_values.quantile(0.75))
            },
            "y1_stats": {
                "count": len(y1_values),
                "mean": float(y1_mean),
                "std": float(y1_std),
                "min": float(y1_values.min()),
                "max": float(y1_values.max()),
                "q25": float(y1_values.quantile(0.25)),
                "q75": float(y1_values.quantile(0.75))
            },
            "separation_metrics": {
                "cohens_d": float(cohens_d),
                "mutual_info": float(mi_score),
                "mann_whitney_u": float(u_stat),
                "p_value": float(p_value),
                "overlap_coefficient": float(overlap_coef)
            },
            "significance": {
                "is_significant": int(is_significant),
                "has_high_separation": int(has_high_separation)
            }
        }
        
        results["feature_analysis"].append(feature_analysis)
        
        # 更新統計
        if is_significant:
            results["summary_stats"]["features_with_significant_diff"] += 1
        if has_high_separation:
            results["summary_stats"]["features_with_high_separation"] += 1
        
        # 打印結果
        print(f"{feature_name}:")
        print(f"  Cohen's d: {cohens_d:.3f} {'(高分離)' if abs(cohens_d) > 0.5 else ''}")
        print(f"  Mutual Info: {mi_score:.3f} {'(高相關)' if mi_score > 0.1 else ''}")
        print(f"  p-value: {p_value:.3e} {'(顯著)' if p_value < 0.05 else ''}")
        print(f"  重疊係數: {overlap_coef:.3f}")
    
    # 打印總結
    total_features = results["summary_stats"]["total_features"]
    significant_features = results["summary_stats"]["features_with_significant_diff"]
    high_separation_features = results["summary_stats"]["features_with_high_separation"]
    
    print(f"\n總結:")
    print(f"  總特徵數: {total_features}")
    if total_features > 0:
        print(f"  顯著差異特徵: {significant_features} ({significant_features/total_features:.1%})")
        print(f"  高分離特徵: {high_separation_features} ({high_separation_features/total_features:.1%})")
    else:
        print(f"  顯著差異特徵: {significant_features} (無 dense 特徵)")
        print(f"  高分離特徵: {high_separation_features} (無 dense 特徵)")
    
    return results

def analyze_embedding_feature_distributions(X_embed: np.ndarray, y: pd.Series) -> Dict[str, Any]:
    """
    分析 Embedding 特徵在 y=0 vs y=1 之間的分佈差異
    """
    print("\n=== Embedding 特徵分佈分析 ===")
    
    results = {
        "feature_analysis": [],
        "summary_stats": {
            "total_features": X_embed.shape[1],
            "features_with_significant_diff": 0,
            "features_with_high_separation": 0
        }
    }
    
    # 為每個 embedding 維度計算統計量
    for dim_idx in range(X_embed.shape[1]):
        feature_values = X_embed[:, dim_idx]
        feature_name = f"embed_dim_{dim_idx}"
        
        # 分組統計
        y0_values = feature_values[y == 0]
        y1_values = feature_values[y == 1]
        
        # 基本統計量
        y0_mean, y0_std = y0_values.mean(), y0_values.std()
        y1_mean, y1_std = y1_values.mean(), y1_values.std()
        
        # 計算分離度指標
        # 1. Cohen's d (標準化平均差)
        pooled_std = np.sqrt(((len(y0_values) - 1) * y0_std**2 + (len(y1_values) - 1) * y1_std**2) / (len(y0_values) + len(y1_values) - 2))
        cohens_d = (y1_mean - y0_mean) / pooled_std if pooled_std > 0 else 0
        
        # 2. 互信息 (Mutual Information)
        # 將連續值分箱以計算互信息
        bins = np.linspace(feature_values.min(), feature_values.max(), 10)
        y0_binned = np.digitize(y0_values, bins)
        y1_binned = np.digitize(y1_values, bins)
        all_binned = np.concatenate([y0_binned, y1_binned])
        all_labels = np.concatenate([np.zeros(len(y0_binned)), np.ones(len(y1_binned))])
        mi_score = mutual_info_score(all_labels, all_binned)
        
        # 3. Mann-Whitney U 檢定
        try:
            u_stat, p_value = stats.mannwhitneyu(y0_values, y1_values, alternative='two-sided')
        except:
            u_stat, p_value = 0, 1
        
        # 4. 重疊係數 (Overlap Coefficient)
        # 計算兩個分佈的重疊程度
        y0_quantiles = np.percentile(y0_values, [25, 75])
        y1_quantiles = np.percentile(y1_values, [25, 75])
        overlap = max(0, min(y0_quantiles[1], y1_quantiles[1]) - max(y0_quantiles[0], y1_quantiles[0]))
        overlap_coef = overlap / (y0_quantiles[1] - y0_quantiles[0] + y1_quantiles[1] - y1_quantiles[0] - overlap) if overlap > 0 else 0
        
        # 判斷特徵是否有顯著差異
        is_significant = p_value < 0.05
        has_high_separation = abs(cohens_d) > 0.5 or mi_score > 0.1
        
        feature_analysis = {
            "feature_name": feature_name,
            "y0_stats": {
                "count": len(y0_values),
                "mean": float(y0_mean),
                "std": float(y0_std),
                "min": float(y0_values.min()),
                "max": float(y0_values.max()),
                "q25": float(np.percentile(y0_values, 25)),
                "q75": float(np.percentile(y0_values, 75))
            },
            "y1_stats": {
                "count": len(y1_values),
                "mean": float(y1_mean),
                "std": float(y1_std),
                "min": float(y1_values.min()),
                "max": float(y1_values.max()),
                "q25": float(np.percentile(y1_values, 25)),
                "q75": float(np.percentile(y1_values, 75))
            },
            "separation_metrics": {
                "cohens_d": float(cohens_d),
                "mutual_info": float(mi_score),
                "mann_whitney_u": float(u_stat),
                "p_value": float(p_value),
                "overlap_coefficient": float(overlap_coef)
            },
            "significance": {
                "is_significant": int(is_significant),
                "has_high_separation": int(has_high_separation)
            }
        }
        
        results["feature_analysis"].append(feature_analysis)
        
        # 更新統計
        if is_significant:
            results["summary_stats"]["features_with_significant_diff"] += 1
        if has_high_separation:
            results["summary_stats"]["features_with_high_separation"] += 1
    
    # 打印總結
    total_features = results["summary_stats"]["total_features"]
    significant_features = results["summary_stats"]["features_with_significant_diff"]
    high_separation_features = results["summary_stats"]["features_with_high_separation"]
    
    print(f"\n總結:")
    print(f"  總 embedding 維度: {total_features}")
    print(f"  顯著差異維度: {significant_features} ({significant_features/total_features:.1%})")
    print(f"  高分離維度: {high_separation_features} ({high_separation_features/total_features:.1%})")
    
    return results

def create_visualization_data(X_dense_df: pd.DataFrame, X_tfidf, X_embed: np.ndarray, y: pd.Series, 
                            embed_mode: str, outdir: str, save_plots: bool = False, plot_format: str = "png") -> Dict[str, Any]:
    """
    創建 PCA/t-SNE/UMAP 可視化（支援 embedding）
    """
    print("\n=== 特徵空間可視化分析 ===")
    
    results = {
        "visualization_results": {},
        "plots_saved": []
    }
    
    # 準備數據
    # 將稀疏矩陣轉換為密集矩陣
    if hasattr(X_tfidf, 'toarray'):
        X_tfidf_dense = X_tfidf.toarray()
    else:
        X_tfidf_dense = X_tfidf
    
    # 合併特徵
    feature_components = [X_dense_df.values]
    feature_names = list(X_dense_df.columns)
    
    if embed_mode == "append" and X_tfidf is not None:
        feature_components.append(X_tfidf_dense)
        feature_names.extend([f"tfidf_{i}" for i in range(X_tfidf_dense.shape[1])])
    
    feature_components.append(X_embed)
    feature_names.extend([f"embed_{i}" for i in range(X_embed.shape[1])])
    
    X_combined = np.hstack(feature_components)
    
    print(f"合併特徵矩陣形狀: {X_combined.shape}")
    print(f"  包含: Dense({X_dense_df.shape[1]}) + {'TF-IDF(' + str(X_tfidf_dense.shape[1]) + ') + ' if embed_mode == 'append' and X_tfidf is not None else ''}Embedding({X_embed.shape[1]})")
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    # 1. PCA 分析
    print("執行 PCA 分析...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_results = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "components_shape": pca.components_.shape
    }
    
    # 2. t-SNE 分析
    print("執行 t-SNE 分析...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)//4))
    X_tsne = tsne.fit_transform(X_scaled)
    
    # 3. UMAP 分析
    print("執行 UMAP 分析...")
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(X_scaled)//4))
    X_umap = umap_reducer.fit_transform(X_scaled)
    
    if save_plots:
        os.makedirs(outdir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        def save_scatter(X2d, title, xlab, ylab, filename):
            plt.style.use('default')
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.scatter(X2d[y == 0, 0], X2d[y == 0, 1], alpha=0.6, label='y=0', s=20)
            ax.scatter(X2d[y == 1, 0], X2d[y == 1, 1], alpha=0.6, label='y=1', s=20)
            ax.set_title(title)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            path = os.path.join(outdir, filename)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            results["plots_saved"].append(path)
            return path

        # PCA
        pca_title = f'PCA (解釋變異: {pca.explained_variance_ratio_[0]:.1%}, {pca.explained_variance_ratio_[1]:.1%})'
        pca_name  = f"feature_visualization_{timestamp}_pca.{plot_format}"
        pca_path  = save_scatter(X_pca, pca_title, 'PC1', 'PC2', pca_name)
        print(f"PCA 圖已保存: {pca_path}")

        # t-SNE
        tsne_name = f"feature_visualization_{timestamp}_tsne.{plot_format}"
        tsne_path = save_scatter(X_tsne, 't-SNE', 't-SNE 1', 't-SNE 2', tsne_name)
        print(f"t-SNE 圖已保存: {tsne_path}")

        # UMAP
        umap_name = f"feature_visualization_{timestamp}_umap.{plot_format}"
        umap_path = save_scatter(X_umap, 'UMAP', 'UMAP 1', 'UMAP 2', umap_name)
        print(f"UMAP 圖已保存: {umap_path}")
    
    # 計算分離度指標
    def calculate_separation_score(X_2d, y):
        """計算 2D 投影的分離度"""
        y0_points = X_2d[y == 0]
        y1_points = X_2d[y == 1]
        
        if len(y0_points) == 0 or len(y1_points) == 0:
            return 0
        
        # 計算類別中心
        center_0 = np.mean(y0_points, axis=0)
        center_1 = np.mean(y1_points, axis=0)
        
        # 計算類內散度
        var_0 = np.var(y0_points, axis=0).sum()
        var_1 = np.var(y1_points, axis=0).sum()
        
        # 計算類間距離
        between_class_distance = np.sum((center_1 - center_0) ** 2)
        
        # 計算 Fisher 判別準則
        fisher_score = between_class_distance / (var_0 + var_1) if (var_0 + var_1) > 0 else 0
        
        return fisher_score
    
    pca_separation = calculate_separation_score(X_pca, y)
    tsne_separation = calculate_separation_score(X_tsne, y)
    umap_separation = calculate_separation_score(X_umap, y)
    
    results["visualization_results"] = {
        "pca": {
            "explained_variance_ratio": pca_results["explained_variance_ratio"],
            "cumulative_variance_ratio": pca_results["cumulative_variance_ratio"],
            "separation_score": float(pca_separation)
        },
        "tsne": {
            "separation_score": float(tsne_separation)
        },
        "umap": {
            "separation_score": float(umap_separation)
        }
    }
    
    print(f"分離度評分:")
    print(f"  PCA: {pca_separation:.3f}")
    print(f"  t-SNE: {tsne_separation:.3f}")
    print(f"  UMAP: {umap_separation:.3f}")
    
    return results

def create_database_ready_format(analysis_results: Dict[str, Any], args, run_id: str) -> Dict[str, Any]:
    """
    創建適合資料庫儲存的格式
    """
    timestamp = datetime.now().isoformat()
    
    db_format = {
        "analysis_id": run_id,
        "analysis_timestamp": timestamp,
        "analysis_config": {
            "mode": args.mode,
            "date_cutoff": args.date_cutoff,
            "pipeline_version": args.pipeline_version,
            "vocab_mode": args.vocab_mode,
            "top_n": args.top_n,
            "exclude_products": args.exclude_products,
            "keyword": args.keyword,
            "embed_path": args.embed_path,
            "embed_mode": args.embed_mode,
            "embed_scale": args.embed_scale,
            "embed_dtype": args.embed_dtype
        },
        "data_summary": {
            "total_samples": analysis_results.get("data_summary", {}).get("total_samples", 0),
            "positive_samples": analysis_results.get("data_summary", {}).get("positive_samples", 0),
            "negative_samples": analysis_results.get("data_summary", {}).get("negative_samples", 0),
            "imbalance_ratio": analysis_results.get("data_summary", {}).get("imbalance_ratio", 0),
            "dense_features_count": analysis_results.get("data_summary", {}).get("dense_features_count", 0),
            "tfidf_features_count": analysis_results.get("data_summary", {}).get("tfidf_features_count", 0),
            "embedding_features_count": analysis_results.get("data_summary", {}).get("embedding_features_count", 0)
        },
        "feature_analysis": analysis_results.get("feature_analysis", {}),
        "embedding_analysis": analysis_results.get("embedding_analysis", {}),
        "visualization_analysis": analysis_results.get("visualization_analysis", {}),
        "conclusions": {
            "feature_sufficiency_score": analysis_results.get("conclusions", {}).get("feature_sufficiency_score", 0),
            "recommendations": analysis_results.get("conclusions", {}).get("recommendations", [])
        },
        "artifacts": {
            "plots_saved": analysis_results.get("visualization_analysis", {}).get("plots_saved", [])
        }
    }
    
    return db_format

def generate_conclusions(feature_analysis: Dict, embedding_analysis: Dict, visualization_analysis: Dict, data_summary: Dict) -> Dict[str, Any]:
    """
    根據分析結果生成結論和建議
    """
    conclusions = {
        "feature_sufficiency_score": 0.0,
        "recommendations": []
    }
    
    # 計算特徵充分性評分
    score = 0.0
    max_score = 100.0
    
    # 1. 基於 Dense 特徵分離度 (25%)
    dense_separation_score = 0
    if "summary_stats" in feature_analysis:
        total_features = feature_analysis["summary_stats"]["total_features"]
        high_separation_features = feature_analysis["summary_stats"]["features_with_high_separation"]
        dense_separation_score = (high_separation_features / total_features) * 25 if total_features > 0 else 0
    
    # 2. 基於 Embedding 特徵分離度 (25%)
    embedding_separation_score = 0
    if "summary_stats" in embedding_analysis:
        total_embed_features = embedding_analysis["summary_stats"]["total_features"]
        high_separation_embed_features = embedding_analysis["summary_stats"]["features_with_high_separation"]
        embedding_separation_score = (high_separation_embed_features / total_embed_features) * 25 if total_embed_features > 0 else 0
    
    # 3. 基於可視化分離度 (30%)
    visualization_score = 0
    if "visualization_results" in visualization_analysis:
        pca_score = visualization_analysis["visualization_results"].get("pca", {}).get("separation_score", 0)
        tsne_score = visualization_analysis["visualization_results"].get("tsne", {}).get("separation_score", 0)
        umap_score = visualization_analysis["visualization_results"].get("umap", {}).get("separation_score", 0)
        
        avg_separation = (pca_score + tsne_score + umap_score) / 3
        visualization_score = min(avg_separation * 30, 30)
    
    # 4. 基於數據品質 (20%)
    data_quality_score = 0
    if data_summary.get("total_samples", 0) > 1000:
        data_quality_score += 5
    if data_summary.get("dense_features_count", 0) > 5:
        data_quality_score += 5
    if data_summary.get("tfidf_features_count", 0) > 50:
        data_quality_score += 5
    if data_summary.get("embedding_features_count", 0) > 100:
        data_quality_score += 5
    
    score = dense_separation_score + embedding_separation_score + visualization_score + data_quality_score
    conclusions["feature_sufficiency_score"] = min(score, max_score)
    
    # 生成建議
    recommendations = []
    
    if dense_separation_score < 12.5:
        recommendations.append("Dense 特徵分離度較低，建議增加更多相關特徵或進行特徵工程")
    
    if embedding_separation_score < 12.5:
        recommendations.append("Embedding 特徵分離度較低，建議嘗試不同的 embedding 模型或調整參數")
    
    if visualization_score < 15:
        recommendations.append("可視化顯示類別分離度不足，可能需要更強的特徵或不同的模型")
    
    if data_quality_score < 10:
        recommendations.append("數據品質有待提升，建議增加樣本數或特徵數量")
    
    if conclusions["feature_sufficiency_score"] < 50:
        recommendations.append("整體特徵充分性較低，建議重新設計特徵或使用更複雜的模型")
    elif conclusions["feature_sufficiency_score"] < 70:
        recommendations.append("特徵充分性中等，可以嘗試調整模型參數或使用集成方法")
    else:
        recommendations.append("特徵充分性較好，可以專注於模型優化")
    
    conclusions["recommendations"] = recommendations
    
    return conclusions

def load_data_dynamically(args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
    """
    動態載入數據（與 train.py 保持一致）
    返回：X_dense_df, X_tfidf, y, meta, vocab
    """
    # 解析參數
    top_n = int(args.top_n.split(",")[0])  # 取第一個值
    excluded = [int(x) for x in args.exclude_products.split(",")] if args.exclude_products else []
    
    print(f"載入數據：mode={args.mode}, date_cutoff={args.date_cutoff}, top_n={top_n}")
    
    try:
        # 根據 mode 選擇載入方式
        if args.mode == "product_level":
            X_dense_df, X_tfidf, y, meta, vocab = load_product_level_training_set(
                date_cutoff=args.date_cutoff,
                top_n=top_n,
                pipeline_version=args.pipeline_version,
                vocab_mode=args.vocab_mode,
                single_keyword=args.keyword,
                exclude_products=excluded
            )
        else:
            X_dense_df, X_tfidf, y, meta, vocab = load_training_set(
                top_n=top_n,
                pipeline_version=args.pipeline_version,
                date_cutoff=args.date_cutoff,
                vocab_scope="global",
            )
        
        print(f"數據載入完成：{len(y)} 樣本，{X_dense_df.shape[1]} 個 dense 特徵，{X_tfidf.shape[1]} 個 TF-IDF 特徵")
        print(f"類別分佈：y=0: {(y==0).sum()} ({y.mean():.3f}), y=1: {(y==1).sum()} ({1-y.mean():.3f})")
        
        return X_dense_df, X_tfidf, y, meta, vocab
        
    except NameError:
        print("警告：無法載入 data_loader，使用空的數據結構")
        # 創建空的數據結構
        X_dense_df = pd.DataFrame()
        X_tfidf = None
        y = pd.Series()
        meta = pd.DataFrame()
        vocab = []
        
        return X_dense_df, X_tfidf, y, meta, vocab

def align_embeddings_to_dataset(X_embed: np.ndarray, embed_keys: pd.Series, 
                               dataset_meta: pd.DataFrame, mode: str) -> tuple[np.ndarray, list]:
    """
    對齊 embedding 與主數據集（與 train_with_embeddings.py 保持一致）
    """
    # 提取 key
    if 'product_id' in dataset_meta.columns:
        dataset_keys = dataset_meta['product_id'].values
    else:
        dataset_keys = dataset_meta.iloc[:, 0].values
    
    embed_keys_array = embed_keys.values
    
    # 創建映射
    embed_key_to_idx = {key: idx for idx, key in enumerate(embed_keys_array)}
    
    # 找到對齊的索引
    aligned_indices = []
    missing_keys = []
    
    for key in dataset_keys:
        if key in embed_key_to_idx:
            aligned_indices.append(embed_key_to_idx[key])
        else:
            missing_keys.append(key)
    
    if len(missing_keys) > 0:
        print(f"警告：{len(missing_keys)} 個樣本在 embedding 中找不到對應")
        if len(missing_keys) > len(dataset_keys) * 0.1:  # 超過 10% 缺失
            raise ValueError(f"太多樣本在 embedding 中找不到對應：{len(missing_keys)}/{len(dataset_keys)}")
    
    X_embed_aligned = X_embed[aligned_indices]
    return X_embed_aligned, aligned_indices

def main():
    """主函式"""
    args = parse_args()
    run_id = str(uuid.uuid4())
    
    print("=== 特徵分析工具（支援 Embedding）===")
    print(f"分析 ID: {run_id}")
    print(f"配置: {vars(args)}")
    
    # 1. 載入 embedding
    print(f"\n載入 embedding 從: {args.embed_path}")
    X_embed, embed_keys, y_embed, embed_info = load_embeddings(
        args.embed_path, args.embed_mode, args.embed_dtype, args.dataset_prefix
    )
    
    # 2. 動態載入數據
    X_dense_df, X_tfidf, y, meta, vocab = load_data_dynamically(args)
    
    # 3. 檢查是否有 dense 數據
    if len(X_dense_df) == 0:
        print("沒有 dense 數據，僅使用 embedding 進行分析")
        # 直接使用 embedding 數據
        X_embed_aligned = X_embed
        y = pd.Series(y_embed)
        # 創建空的 dense 數據框
        X_dense_df = pd.DataFrame(index=range(len(y)))
    else:
        # 對齊 embedding 與數據集
        print(f"\n對齊 embedding 與數據集...")
        X_embed_aligned, aligned_indices = align_embeddings_to_dataset(
            X_embed, embed_keys, meta, args.embed_mode
        )
        
        # 過濾數據集以匹配 embedding
        X_dense_df = X_dense_df.iloc[aligned_indices].reset_index(drop=True)
        if X_tfidf is not None:
            X_tfidf = X_tfidf[aligned_indices]
        y = y.iloc[aligned_indices].reset_index(drop=True)
        meta = meta.iloc[aligned_indices].reset_index(drop=True)
        
        # 使用 embedding 的 y 標籤
        y = pd.Series(y_embed[aligned_indices])
    
    # 5. 可選的 embedding 標準化
    if args.embed_scale == "standardize":
        print("標準化 embedding...")
        embed_scaler = StandardScaler()
        X_embed_aligned = embed_scaler.fit_transform(X_embed_aligned)
    
    # 6. 數據摘要
    data_summary = {
        "total_samples": len(y),
        "positive_samples": int((y == 1).sum()),
        "negative_samples": int((y == 0).sum()),
        "imbalance_ratio": float((y == 0).sum() / (y == 1).sum()) if (y == 1).sum() > 0 else float('inf'),
        "dense_features_count": X_dense_df.shape[1],
        "tfidf_features_count": X_tfidf.shape[1] if X_tfidf is not None else 0,
        "embedding_features_count": X_embed_aligned.shape[1]
    }
    
    print(f"\n數據摘要:")
    print(f"  總樣本數: {data_summary['total_samples']}")
    print(f"  正樣本數: {data_summary['positive_samples']}")
    print(f"  負樣本數: {data_summary['negative_samples']}")
    print(f"  不平衡比例: {data_summary['imbalance_ratio']:.2f}:1")
    print(f"  Dense 特徵數: {data_summary['dense_features_count']}")
    print(f"  TF-IDF 特徵數: {data_summary['tfidf_features_count']}")
    print(f"  Embedding 特徵數: {data_summary['embedding_features_count']}")
    
    # 7. 分析 Dense 特徵分佈
    feature_analysis = analyze_dense_feature_distributions(X_dense_df, y)
    
    # 8. 分析 Embedding 特徵分佈
    embedding_analysis = analyze_embedding_feature_distributions(X_embed_aligned, y)
    
    # 9. 創建可視化
    visualization_analysis = create_visualization_data(
        X_dense_df, X_tfidf, X_embed_aligned, y, args.embed_mode, 
        args.outdir, args.save_plots, args.plot_format
    )
    
    # 10. 生成結論
    conclusions = generate_conclusions(feature_analysis, embedding_analysis, visualization_analysis, data_summary)
    
    # 11. 整合結果
    analysis_results = {
        "data_summary": data_summary,
        "feature_analysis": feature_analysis,
        "embedding_analysis": embedding_analysis,
        "visualization_analysis": visualization_analysis,
        "conclusions": conclusions
    }
    
    # 12. 創建資料庫格式
    db_format = create_database_ready_format(analysis_results, args, run_id)
    
    # 13. 保存結果
    os.makedirs(args.outdir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存詳細結果
    detailed_results_path = os.path.join(args.outdir, f"feature_analysis_with_embeddings_detailed_{timestamp}.json")
    
    # 創建 JSON 安全的版本（轉換 bool 為 int）
    def make_json_safe(obj):
        if isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_safe(item) for item in obj]
        elif isinstance(obj, bool):
            return int(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    json_safe_results = make_json_safe(analysis_results)
    with open(detailed_results_path, 'w', encoding='utf-8-sig') as f:
        json.dump(json_safe_results, f, ensure_ascii=False, indent=2)
    
    # 保存資料庫格式
    db_format_path = os.path.join(args.outdir, f"feature_analysis_with_embeddings_db_ready_{timestamp}.json")
    json_safe_db_format = make_json_safe(db_format)
    with open(db_format_path, 'w', encoding='utf-8-sig') as f:
        json.dump(json_safe_db_format, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 分析完成 ===")
    print(f"特徵充分性評分: {conclusions['feature_sufficiency_score']:.1f}/100")
    print(f"建議:")
    for i, rec in enumerate(conclusions['recommendations'], 1):
        print(f"  {i}. {rec}")
    print(f"\n結果已保存:")
    print(f"  詳細結果: {detailed_results_path}")
    print(f"  資料庫格式: {db_format_path}")
    if args.save_plots:
        print(f"  可視化圖表: {args.outdir}")

if __name__ == "__main__":
    main()
