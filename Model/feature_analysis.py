# -*- coding: utf-8-sig-sig -*-
"""
Model/feature_analysis.py
特徵分析工具：驗證特徵是否足以區分類別
- 分析 Dense 特徵在 y=0 vs y=1 之間的分佈差異
- 使用 PCA/t-SNE/UMAP 可視化特徵空間
- 動態讀取數據（與 train.py 保持一致）
- 輸出分析結果並準備資料庫儲存格式

用法：
python Model/feature_analysis.py \
  --mode product_level \
  --date-cutoff 2025-06-25 \
  --vocab-mode global \
  --top-n 100 \
  --pipeline-version v1 \
  --exclude-products 8918452 \
  --outdir Model/analysis_outputs
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

# 動態導入 data_loader（與 train.py 保持一致）
try:
    from data_loader import (
        load_training_set,
        load_product_level_training_set
    )
except ImportError:
    print("警告：無法導入 data_loader，請確保在正確的環境中運行")

def parse_args():
    """解析命令行參數（與 train.py 保持一致）"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="product_level",
                    choices=["product_level", "comment_level"])
    ap.add_argument("--date-cutoff", type=str, default="2025-06-25")
    ap.add_argument("--pipeline-version", type=str, default=os.getenv("PIPELINE_VERSION", None))
    ap.add_argument("--vocab-mode", type=str, default="global")
    ap.add_argument("--top-n", type=str, default="100")
    ap.add_argument("--exclude-products", type=str, default="8918452")
    ap.add_argument("--keyword", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="Model/analysis_outputs")
    ap.add_argument("--save-plots", action="store_true", help="是否保存圖表")
    ap.add_argument("--plot-format", type=str, default="png", choices=["png", "pdf", "svg"])
    return ap.parse_args()

def load_data_dynamically(args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
    """
    動態載入數據（與 train.py 保持一致）
    返回：X_dense_df, X_tfidf, y, meta, vocab
    """
    # 解析參數
    top_n = int(args.top_n.split(",")[0])  # 取第一個值
    excluded = [int(x) for x in args.exclude_products.split(",")] if args.exclude_products else []
    
    print(f"載入數據：mode={args.mode}, date_cutoff={args.date_cutoff}, top_n={top_n}")
    
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
    print(f"  顯著差異特徵: {significant_features} ({significant_features/total_features:.1%})")
    print(f"  高分離特徵: {high_separation_features} ({high_separation_features/total_features:.1%})")
    
    return results

def create_visualization_data(X_dense_df: pd.DataFrame, X_tfidf, y: pd.Series, 
                            outdir: str, save_plots: bool = False, plot_format: str = "png") -> Dict[str, Any]:
    """
    創建 PCA/t-SNE/UMAP 可視化
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
    X_combined = np.hstack([X_dense_df.values, X_tfidf_dense])
    feature_names = list(X_dense_df.columns) + [f"tfidf_{i}" for i in range(X_tfidf_dense.shape[1])]
    
    print(f"合併特徵矩陣形狀: {X_combined.shape}")
    
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
    
    # 創建可視化
    if save_plots:
        os.makedirs(outdir, exist_ok=True)
        
        # 設置圖表樣式
        plt.style.use('default')
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # PCA 圖
        scatter0 = axes[0].scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], alpha=0.6, label='y=0', s=20)
        scatter1 = axes[0].scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], alpha=0.6, label='y=1', s=20)
        axes[0].set_title(f'PCA (解釋變異: {pca.explained_variance_ratio_[0]:.1%}, {pca.explained_variance_ratio_[1]:.1%})')
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # t-SNE 圖
        axes[1].scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], alpha=0.6, label='y=0', s=20)
        axes[1].scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], alpha=0.6, label='y=1', s=20)
        axes[1].set_title('t-SNE')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # UMAP 圖
        axes[2].scatter(X_umap[y == 0, 0], X_umap[y == 0, 1], alpha=0.6, label='y=0', s=20)
        axes[2].scatter(X_umap[y == 1, 0], X_umap[y == 1, 1], alpha=0.6, label='y=1', s=20)
        axes[2].set_title('UMAP')
        axes[2].set_xlabel('UMAP 1')
        axes[2].set_ylabel('UMAP 2')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存圖表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"feature_visualization_{timestamp}.{plot_format}"
        plot_path = os.path.join(outdir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        results["plots_saved"].append(plot_path)
        print(f"可視化圖表已保存: {plot_path}")
    
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
            "keyword": args.keyword
        },
        "data_summary": {
            "total_samples": analysis_results.get("data_summary", {}).get("total_samples", 0),
            "positive_samples": analysis_results.get("data_summary", {}).get("positive_samples", 0),
            "negative_samples": analysis_results.get("data_summary", {}).get("negative_samples", 0),
            "imbalance_ratio": analysis_results.get("data_summary", {}).get("imbalance_ratio", 0),
            "dense_features_count": analysis_results.get("data_summary", {}).get("dense_features_count", 0),
            "tfidf_features_count": analysis_results.get("data_summary", {}).get("tfidf_features_count", 0)
        },
        "feature_analysis": analysis_results.get("feature_analysis", {}),
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

def generate_conclusions(feature_analysis: Dict, visualization_analysis: Dict, data_summary: Dict) -> Dict[str, Any]:
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
    
    # 1. 基於特徵分離度 (40%)
    feature_separation_score = 0
    if "summary_stats" in feature_analysis:
        total_features = feature_analysis["summary_stats"]["total_features"]
        high_separation_features = feature_analysis["summary_stats"]["features_with_high_separation"]
        feature_separation_score = (high_separation_features / total_features) * 40 if total_features > 0 else 0
    
    # 2. 基於可視化分離度 (30%)
    visualization_score = 0
    if "visualization_results" in visualization_analysis:
        pca_score = visualization_analysis["visualization_results"].get("pca", {}).get("separation_score", 0)
        tsne_score = visualization_analysis["visualization_results"].get("tsne", {}).get("separation_score", 0)
        umap_score = visualization_analysis["visualization_results"].get("umap", {}).get("separation_score", 0)
        
        avg_separation = (pca_score + tsne_score + umap_score) / 3
        visualization_score = min(avg_separation * 30, 30)
    
    # 3. 基於數據品質 (30%)
    data_quality_score = 0
    if data_summary.get("total_samples", 0) > 1000:
        data_quality_score += 10
    if data_summary.get("dense_features_count", 0) > 5:
        data_quality_score += 10
    if data_summary.get("tfidf_features_count", 0) > 50:
        data_quality_score += 10
    
    score = feature_separation_score + visualization_score + data_quality_score
    conclusions["feature_sufficiency_score"] = min(score, max_score)
    
    # 生成建議
    recommendations = []
    
    if feature_separation_score < 20:
        recommendations.append("特徵分離度較低，建議增加更多相關特徵或進行特徵工程")
    
    if visualization_score < 15:
        recommendations.append("可視化顯示類別分離度不足，可能需要更強的特徵或不同的模型")
    
    if data_quality_score < 20:
        recommendations.append("數據品質有待提升，建議增加樣本數或特徵數量")
    
    if conclusions["feature_sufficiency_score"] < 50:
        recommendations.append("整體特徵充分性較低，建議重新設計特徵或使用更複雜的模型")
    elif conclusions["feature_sufficiency_score"] < 70:
        recommendations.append("特徵充分性中等，可以嘗試調整模型參數或使用集成方法")
    else:
        recommendations.append("特徵充分性較好，可以專注於模型優化")
    
    conclusions["recommendations"] = recommendations
    
    return conclusions

def main():
    """主函式"""
    args = parse_args()
    run_id = str(uuid.uuid4())
    
    print("=== 特徵分析工具 ===")
    print(f"分析 ID: {run_id}")
    print(f"配置: {vars(args)}")
    
    # 1. 動態載入數據
    X_dense_df, X_tfidf, y, meta, vocab = load_data_dynamically(args)
    
    # 2. 數據摘要
    data_summary = {
        "total_samples": len(y),
        "positive_samples": int((y == 1).sum()),
        "negative_samples": int((y == 0).sum()),
        "imbalance_ratio": float((y == 0).sum() / (y == 1).sum()) if (y == 1).sum() > 0 else float('inf'),
        "dense_features_count": X_dense_df.shape[1],
        "tfidf_features_count": X_tfidf.shape[1]
    }
    
    print(f"\n數據摘要:")
    print(f"  總樣本數: {data_summary['total_samples']}")
    print(f"  正樣本數: {data_summary['positive_samples']}")
    print(f"  負樣本數: {data_summary['negative_samples']}")
    print(f"  不平衡比例: {data_summary['imbalance_ratio']:.2f}:1")
    print(f"  Dense 特徵數: {data_summary['dense_features_count']}")
    print(f"  TF-IDF 特徵數: {data_summary['tfidf_features_count']}")
    
    # 3. 分析 Dense 特徵分佈
    feature_analysis = analyze_dense_feature_distributions(X_dense_df, y)
    
    # 4. 創建可視化
    visualization_analysis = create_visualization_data(
        X_dense_df, X_tfidf, y, args.outdir, args.save_plots, args.plot_format
    )
    
    # 5. 生成結論
    conclusions = generate_conclusions(feature_analysis, visualization_analysis, data_summary)
    
    # 6. 整合結果
    analysis_results = {
        "data_summary": data_summary,
        "feature_analysis": feature_analysis,
        "visualization_analysis": visualization_analysis,
        "conclusions": conclusions
    }
    
    # 7. 創建資料庫格式
    db_format = create_database_ready_format(analysis_results, args, run_id)
    
    # 8. 保存結果
    os.makedirs(args.outdir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存詳細結果
    detailed_results_path = os.path.join(args.outdir, f"feature_analysis_detailed_{timestamp}.json")
    
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
    db_format_path = os.path.join(args.outdir, f"feature_analysis_db_ready_{timestamp}.json")
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
