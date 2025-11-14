# -*- coding: utf-8-sig -*-
"""
Model/train.py
三模型( SVM / LightGBM / XGBoost ) × 兩種特徵選取(無 / 基於 LGBM 重要度)
- cuML 存在就用 GPU SVM，否則自動退回 sklearn.SVC（CPU）
- LightGBM / XGBoost 預設走 GPU，缺套件會自動略過或退回
- 以 date_cutoff 切訓練/測試；TF-IDF top_n 可切換
- 輸出結果到 model_results.csv / model_results.json

python Model/train.py \
  --mode product_level \
  --date-cutoff 2025-06-25 \
  --vocab-mode global \
  --top-n 100,200 \
  --algorithms xgboost,svm,lightgbm \
  --fs-methods no_fs,lgbm_fs \
  --cv 10 \
  --exclude-products 8918452 \
  --outdir Model/outputs \
  --oversample random
"""

# -*- coding: utf-8-sig -*-
from __future__ import annotations
import os, json, argparse, warnings, uuid
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from scipy.sparse import hstack, csr_matrix, save_npz
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, precision_recall_fscore_support,
    average_precision_score, brier_score_loss
)

# LightGBM (optional)
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
    LGBMClassifier = None
    warnings.warn("LightGBM 未安裝，將略過 LightGBM。")

# imbalanced-learn
try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    HAS_IMB = True
except Exception:
    HAS_IMB = False
    RandomOverSampler = None
    SMOTE = None

# XGBoost
import xgboost as xgb

# SVM (GPU/CPU)
try:
    from cuml.svm import SVC as cuSVC
    HAS_CUML = True
except Exception:
    HAS_CUML = False
    cuSVC = None
    warnings.warn("cuML 未安裝，SVM 將退回 sklearn。")
from sklearn.svm import SVC as skSVC

# data loaders
from data_loader import (
    load_training_set,                  # comment-level
    load_product_level_training_set     # product-level (剛剛新增)
)

def parse_args():
    """解析命令行參數"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="product_level",
                    choices=["product_level", "comment_level"],
                    help="訓練模式：product_level（商品級別）或 comment_level（評論級別）")
    ap.add_argument("--date-cutoff", type=str, default="2025-06-25",
                    help="資料切分日期：此日期之前的資料用於訓練，之後用於標籤生成")
    ap.add_argument("--pipeline-version", type=str, default=os.getenv("PIPELINE_VERSION", None),
                    help="資料處理流程版本號（從環境變數 PIPELINE_VERSION 讀取）")
    ap.add_argument("--vocab-mode", type=str, default="global",
                    help="詞彙表模式：目前支援 global（全局詞彙表）")
    ap.add_argument("--top-n", type=str, default="100",
                    help="TF-IDF 詞彙表大小，可給多個值（逗號分隔，如 \"100,200\"）")
    ap.add_argument("--algorithms", type=str, default="xgboost,svm",
                    help="使用的算法列表（逗號分隔，如 \"xgboost,svm,lightgbm\"）")
    ap.add_argument("--fs-methods", type=str, default="no_fs,lgbm_fs",
                    help="特徵選擇方法（逗號分隔）：no_fs（無特徵選擇）、lgbm_fs（基於 LightGBM 重要度）")
    ap.add_argument("--cv", type=int, default=10,
                    help="交叉驗證折數（StratifiedKFold）")
    ap.add_argument("--exclude-products", type=str, default="8918452",
                    help="排除的商品 ID 列表（逗號分隔）")
    ap.add_argument("--keyword", type=str, default=None,
                    help="單一關鍵詞篩選：只使用指定關鍵詞的資料")
    ap.add_argument("--outdir", type=str, default="Model/outputs",
                    help="輸出目錄：所有結果檔案將保存在此目錄")
    ap.add_argument("--oversample", type=str, default="none",
                choices=["none", "random", "smote", "xgb_scale_pos_weight"],
                help="類別不平衡處理：none/random/smote/xgb_scale_pos_weight（僅作用於訓練折）")
    ap.add_argument("--label-delta-threshold", type=float, default=10.0,
                    help="y=1 的條件：銷售增量絕對值閾值（單位：件數）")
    ap.add_argument("--label-ratio-threshold", type=float, default=None,
                    help="y=1 的條件：銷售增量比例閾值（可選，需同時滿足 delta 和 ratio）")
    ap.add_argument("--label-max-gap-days", type=float, default=14.0,
                    help="標籤定義：sales snapshot 之間的最大間隔天數（天）")
    ap.add_argument("--min-comments", type=int, default=0,
                    help="資料篩選：時間窗口內至少需要 N 條評論的商品才會被納入訓練")
    ap.add_argument("--keyword-blacklist", type=str, default=None,
                    help="資料篩選：排除的關鍵詞列表（逗號分隔）")
    ap.add_argument("--keyword-whitelist", type=str, default=None,
                    help="資料篩選：只保留的關鍵詞列表（逗號分隔）")
    ap.add_argument("--topk", type=str, default="20,50,100",
                    help="Top-K precision/recall 評估的 k 值列表（逗號分隔）")
    ap.add_argument("--threshold-grid", type=str,
                    default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5",
                    help="threshold 搜尋的候選值列表（逗號分隔）")
    ap.add_argument("--threshold-target", type=str, default="f1_1",
                    choices=["f1_1", "recall_1", "precision_1"],
                    help="選擇 threshold 時要優化的目標指標")
    ap.add_argument("--threshold-min-recall", type=float, default=0.0,
                    help="threshold search 時 recall(y=1) 的最低要求")
    return ap.parse_args()

def std_scaler_to_sparse(Xdf: pd.DataFrame) -> csr_matrix:
    """
    將 DataFrame 標準化後轉換為稀疏矩陣
    
    Args:
        Xdf: 輸入的 DataFrame（dense 特徵）
    
    Returns:
        標準化後的稀疏矩陣（CSR 格式）
    """
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(Xdf.values)
    return csr_matrix(X_scaled)

def select_features_by_lgbm(X_train, y_train, X_valid, top_k=100):
    """
    使用 LightGBM 進行特徵選擇：根據特徵重要度選取前 top_k 個特徵
    
    Args:
        X_train: 訓練集特徵
        y_train: 訓練集標籤
        X_valid: 驗證集特徵
        top_k: 要選取的特徵數量
    
    Returns:
        (X_train_selected, X_valid_selected, selected_indices)
    """
    if not HAS_LGBM:
        warnings.warn("未安裝 LightGBM，FS 略過。")
        return X_train, X_valid, None
    clf = LGBMClassifier(device="gpu", n_estimators=400, learning_rate=0.05,
                         num_leaves=64, random_state=42)
    clf.fit(X_train, y_train)
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1]  # 按重要度降序排列
    k = max(1, min(top_k, (imp > 0).sum() or top_k))
    idx = order[:k]
    return X_train[:, idx], X_valid[:, idx], idx

def get_positive_score(model, X):
    """
    獲取模型對正類（y=1）的預測分數
    
    優先使用 predict_proba，其次使用 decision_function，最後使用 predict
    
    Args:
        model: 訓練好的模型
        X: 輸入特徵
    
    Returns:
        正類的預測分數（機率或決策函數值）
    """
    try:
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] >= 2 else proba[:, -1]
    except Exception:
        try:
            return model.decision_function(X)
        except Exception:
            return model.predict(X)

def eval_metrics(y_true, y_pred, y_score) -> dict:
    """
    計算模型評估指標
    
    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        y_score: 預測分數（用於計算 AUC）
    
    Returns:
        包含各項評估指標的字典
    """
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["auc"] = float("nan")
    # 計算每個類別的詳細指標
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0,1], zero_division=0)
    out.update({
        "precision_0": float(p[0]), "recall_0": float(r[0]), "f1_0": float(f1[0]), "support_0": int(sup[0]),
        "precision_1": float(p[1]), "recall_1": float(r[1]), "f1_1": float(f1[1]), "support_1": int(sup[1]),
    })
    try:
        auc1 = roc_auc_score(y_true, y_score); auc0 = 1.0 - auc1
    except Exception:
        auc0, auc1 = float("nan"), float("nan")
    out.update({"auc_0": float(auc0), "auc_1": float(auc1)})
    return out

def parse_str_list(raw: Optional[str]) -> Optional[List[str]]:
    """
    解析逗號分隔的字串列表
    
    Args:
        raw: 逗號分隔的字串（如 "a,b,c"）
    
    Returns:
        解析後的列表，如果輸入為 None 則返回 None
    """
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",")]
    values = [item for item in values if item]
    return values or None

def build_model(alg_name: str, args):
    """
    根據算法名稱構建對應的模型實例
    
    Args:
        alg_name: 算法名稱（"xgboost", "svm", "lightgbm"）
        args: 命令行參數對象
    
    Returns:
        (model, needs_dense): 模型實例和是否需要 dense 格式的標記
    """
    alg = (alg_name or "").lower()
    if alg == "xgboost":
        model = xgb.XGBClassifier(
            tree_method="gpu_hist", predictor="gpu_predictor",
            n_estimators=800, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
            scale_pos_weight=args.suggested_spw if getattr(args, "oversample", "none") == "xgb_scale_pos_weight" else None
        )
        return model, False  # XGBoost 支援稀疏矩陣
    if alg == "svm":
        if HAS_CUML and cuSVC is not None:
            return cuSVC(probability=True), True  # cuML SVM 需要 dense
        return skSVC(probability=True, kernel="rbf"), True  # sklearn SVM 需要 dense
    if alg == "lightgbm" and HAS_LGBM:
        return LGBMClassifier(device="gpu", n_estimators=800, learning_rate=0.05,
                               num_leaves=127, subsample=0.8, colsample_bytree=0.8, random_state=42), False  # LightGBM 支援稀疏矩陣
    warnings.warn(f"Unsupported algorithm: {alg_name}")
    return None, None

def compute_topk_metrics(df_preds: pd.DataFrame, ks: List[int]) -> List[Dict[str, float]]:
    """
    計算 Top-K 指標：選取預測分數最高的 k 個樣本，計算其 precision 和 recall
    
    Args:
        df_preds: 包含 y_true 和 y_score 的預測結果 DataFrame
        ks: k 值列表（如 [20, 50, 100]）
    
    Returns:
        每個 k 值對應的指標列表
    """
    if df_preds is None or df_preds.empty:
        return []
    total_pos = int((df_preds["y_true"] == 1).sum())
    results = []
    for k in ks:
        if k <= 0:
            continue
        k_eff = min(int(k), len(df_preds))
        if k_eff <= 0:
            continue
        topk = df_preds.nlargest(k_eff, "y_score")  # 選取分數最高的 k 個
        hits = int(topk["y_true"].sum())  # 其中有多少個是真正的正類
        precision = hits / k_eff if k_eff else 0.0
        recall = hits / total_pos if total_pos else 0.0
        results.append({
            "k": int(k),
            "precision": float(precision),
            "recall": float(recall),
            "hits": hits
        })
    return results

def sweep_thresholds(y_true: np.ndarray,
                     y_score: np.ndarray,
                     thresholds: List[float],
                     min_recall: float,
                     target_metric: str) -> Dict[str, Any]:
    """
    掃描不同閾值，選擇最佳閾值
    
    在滿足最小 recall 要求的前提下，選擇使目標指標（f1_1/recall_1/precision_1）最大的閾值
    
    Args:
        y_true: 真實標籤
        y_score: 預測分數
        thresholds: 要掃描的閾值列表
        min_recall: 最小 recall 要求
        target_metric: 目標優化指標（"f1_1", "recall_1", "precision_1"）
    
    Returns:
        包含掃描結果和選定閾值的字典
    """
    if thresholds is None or len(thresholds) == 0:
        thresholds = [0.5]
    rows = []
    total = len(y_true)
    for th in thresholds:
        preds = (y_score >= th).astype(int)  # 根據閾值進行二分類
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        tn = int(((preds == 0) & (y_true == 0)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        specificity = (tn / (tn + fp)) if (tn + fp) else 0.0
        rows.append({
            "threshold": float(th),
            "precision_1": float(precision),
            "recall_1": float(recall),
            "f1_1": float(f1),
            "specificity": float(specificity),
            "support": total
        })
    # 篩選滿足最小 recall 要求的候選
    valid = [r for r in rows if r["recall_1"] >= min_recall]
    pool = valid if valid else rows  # 如果沒有滿足條件的，則從全部候選中選擇
    chosen = max(pool, key=lambda r: r.get(target_metric, 0.0))
    return {
        "scanned": rows,
        "chosen_for": target_metric,
        "min_recall": float(min_recall),
        "chosen_threshold": float(chosen["threshold"]),
        "metrics_at_chosen": chosen
    }

def build_metrics_v1(df_preds: pd.DataFrame,
                     ks: List[int],
                     thresholds: List[float],
                     min_recall: float,
                     target_metric: str) -> Optional[Dict[str, Any]]:
    """
    構建完整的評估指標（v1 格式）
    
    包含：整體二分類指標、各類別指標、Top-K 指標、閾值搜尋結果
    
    Args:
        df_preds: 預測結果 DataFrame（需包含 y_true, y_pred, y_score）
        ks: Top-K 評估的 k 值列表
        thresholds: 閾值搜尋的候選值列表
        min_recall: 最小 recall 要求
        target_metric: 閾值選擇的目標指標
    
    Returns:
        完整的評估指標字典，如果輸入為空則返回 None
    """
    if df_preds is None or df_preds.empty:
        return None
    y_true = df_preds["y_true"].to_numpy()
    y_score = df_preds["y_score"].to_numpy()
    y_pred = df_preds["y_pred"].to_numpy()
    # 計算整體指標
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float("nan")
    try:
        pr_auc = average_precision_score(y_true, y_score)  # PR-AUC
    except Exception:
        pr_auc = float("nan")
    try:
        brier = brier_score_loss(y_true, y_score)  # Brier 分數（校準度）
    except Exception:
        brier = float("nan")
    binary_overall = {
        "roc_auc": float(auc),
        "pr_auc": float(pr_auc),
        "accuracy": float((y_pred == y_true).mean()),
        "brier": float(brier)
    }
    # 計算各類別指標
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    by_class = {
        "0": {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f1[0]), "support": int(sup[0])},
        "1": {"precision": float(p[1]), "recall": float(r[1]), "f1": float(f1[1]), "support": int(sup[1])},
    }
    topk_metrics = compute_topk_metrics(df_preds, ks)
    threshold_info = sweep_thresholds(y_true, y_score, thresholds, min_recall, target_metric)
    return {
        "binary_overall": binary_overall,
        "by_class": by_class,
        "topk": topk_metrics,
        "threshold_search": threshold_info
    }

def extract_feature_importance(model, feature_names: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
    """
    提取模型的特徵重要度（前 top_k 個）
    
    支援樹模型（feature_importances_）和線性模型（coef_）
    
    Args:
        model: 訓練好的模型
        feature_names: 特徵名稱列表
        top_k: 要返回的前 k 個重要特徵
    
    Returns:
        特徵重要度列表（按重要度降序排列）
    """
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = getattr(model, "feature_importances_")
    elif hasattr(model, "coef_"):
        coefs = getattr(model, "coef_")
        importances = np.abs(coefs).ravel()  # 線性模型使用係數絕對值
    if importances is None or len(importances) != len(feature_names):
        return []
    idx = np.argsort(importances)[::-1][:top_k]  # 按重要度降序排列，取前 k 個
    results = []
    for i in idx:
        results.append({
            "feature": feature_names[i],
            "importance": float(importances[i]),
            "feature_type": "tfidf" if feature_names[i].startswith("tfidf::") else "dense"
        })
    return results

def collect_error_examples(df_preds: pd.DataFrame, top_n: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """
    收集錯誤分類的典型案例
    
    - False Positives: 真實為負類但預測分數最高的樣本
    - False Negatives: 真實為正類但預測分數最低的樣本
    
    Args:
        df_preds: 預測結果 DataFrame
        top_n: 每種類型要收集的樣本數
    
    Returns:
        包含 false_positives 和 false_negatives 的字典
    """
    if df_preds is None or df_preds.empty:
        return {"false_positives": [], "false_negatives": []}
    # False Positives: 真實為 0 但預測分數最高的（最容易被誤判為正類的負類樣本）
    fp = df_preds[df_preds["y_true"] == 0].nlargest(top_n, "y_score")
    # False Negatives: 真實為 1 但預測分數最低的（最容易被誤判為負類的正類樣本）
    fn = df_preds[df_preds["y_true"] == 1].nsmallest(top_n, "y_score")
    cols = ["product_id", "keyword", "y_score"]
    return {
        "false_positives": fp[cols].to_dict(orient="records"),
        "false_negatives": fn[cols].to_dict(orient="records")
    }

def build_model_run_payload(run_id: str,
                            args,
                            alg_name: str,
                            fs_method: str,
                            dataset_art: dict,
                            files_dict: dict,
                            feature_blocks: Dict[str, Any],
                            metrics_v1: dict,
                            feature_importance: List[Dict[str, Any]],
                            error_examples: Dict[str, Any]) -> dict:
    y_definition = f"next batch sales delta >= {args.label_delta_threshold}"
    if args.label_max_gap_days:
        y_definition += f" within {args.label_max_gap_days} days"
    if args.label_ratio_threshold:
        y_definition += f" and ratio >= {args.label_ratio_threshold}"
    data_summary = dataset_art.get("class_distribution", {})
    return {
        "problem_spec": {
            "name": f"{args.mode}_sales_change_detection",
            "experiment_type": "predictive_classification",
            "target_focus": "positive_class_detection",
            "y_definition": y_definition,
            "labeling_config": args.labeling_config,
            "sampling_config": args.sampling_config
        },
        "data_spec": {
            "dataset_manifest": dataset_art.get("path"),  # path to manifest
            "n_samples": int(data_summary.get("total", 0)),
            "pos_samples": int(data_summary.get("pos", 0)),
            "neg_samples": int(data_summary.get("neg", 0)),
            "pos_rate": float(data_summary.get("pos_rate", 0.0)),
            "split_strategy": f"StratifiedKFold (n_splits={args.cv})"
        },
        "model_spec": {
            "algorithm": alg_name,
            "fs_method": fs_method,
            "hyperparams": feature_blocks.get("params", {}),
            "feature_blocks": {
                "dense": feature_blocks.get("dense", []),
                "tfidf": {
                    "mode": args.vocab_mode,
                    "vocab_size": feature_blocks.get("tfidf_vocab_size", 0)
                }
            },
            "imbalance_strategy": args.oversample
        },
        "metrics": metrics_v1,
        "feature_importance": feature_importance,
        "error_analysis": error_examples,
        "artifacts": {
            **files_dict,
            "dataset_manifest": dataset_art.get("path")
        },
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
def save_dataset_artifacts(outdir: str,
                           mode: str,
                           date_cutoff: str,
                           vocab_mode: str,
                           top_n_built: int,
                           X_dense_df: pd.DataFrame,
                           X_tfidf: csr_matrix,
                           y: pd.Series,
                           meta: pd.DataFrame,
                           vocab: list,
                           labeling_config: Optional[dict] = None,
                           sampling_config: Optional[dict] = None) -> dict:
    """
    把目前載入好的資料集落地成暫存檔，回傳各檔案路徑（之後可寫進 DB）。
    """
    os.makedirs(outdir, exist_ok=True)
    ds_dir = os.path.join(outdir, "datasets")
    os.makedirs(ds_dir, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"dataset_{mode}_{date_cutoff.replace('-','')}_{vocab_mode}_top{top_n_built}_{stamp}"

    p_Xdense = os.path.join(ds_dir, f"{base}_Xdense.csv")
    p_Xtfidf = os.path.join(ds_dir, f"{base}_Xtfidf.npz")
    p_y      = os.path.join(ds_dir, f"{base}_y.csv")
    p_meta   = os.path.join(ds_dir, f"{base}_meta.csv")
    p_vocab  = os.path.join(ds_dir, f"{base}_vocab.txt")
    p_dsman  = os.path.join(ds_dir, f"{base}_manifest.json")

    # 1) Dense features
    X_dense_df.to_csv(p_Xdense, index=False, encoding="utf-8-sig")

    # 2) 稀疏矩陣
    save_npz(p_Xtfidf, X_tfidf)

    # 3) y
    y.to_frame("y").to_csv(p_y, index=False, encoding="utf-8-sig")

    # 4) meta（建議保留 product_id / name / keyword 等辨識欄）
    meta.to_csv(p_meta, index=False, encoding="utf-8-sig")

    # 5) vocab
    with open(p_vocab, "w", encoding="utf-8-sig") as f:
        for t in vocab:
            f.write(f"{t}\n")

    # 6) dataset manifest
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    total = pos + neg
    class_distribution = {
        "pos": pos,
        "neg": neg,
        "total": total,
        "pos_rate": float(pos / total) if total else 0.0
    }

    ds_manifest = {
        "mode": mode,
        "date_cutoff": date_cutoff,
        "vocab_mode": vocab_mode,
        "top_n_built": int(top_n_built),
        "files": {
            "Xdense_csv": p_Xdense,
            "Xtfidf_npz": p_Xtfidf,
            "y_csv": p_y,
            "meta_csv": p_meta,
            "vocab_txt": p_vocab
        },
        "shapes": {
            "Xdense": list(X_dense_df.shape),
            "Xtfidf": [int(X_tfidf.shape[0]), int(X_tfidf.shape[1])],
            "y": int(y.shape[0]),
            "vocab": int(len(vocab))
        },
        "dense_features": list(X_dense_df.columns),
        "meta_columns": list(meta.columns),
        "class_distribution": class_distribution,
        "labeling_config": labeling_config,
        "sampling_config": sampling_config
    }
    with open(p_dsman, "w", encoding="utf-8-sig") as f:
        json.dump(ds_manifest, f, ensure_ascii=False, indent=2)

    ds_manifest["path"] = p_dsman
    return ds_manifest

def run_one_setting(run_id: str, args, top_n: int, alg_name: str, fs_method: str,
                    X_dense_df, X_tfidf, y, meta, vocab, outdir: str, dataset_art: dict):
    # 組特徵：將 dense 特徵標準化並轉為稀疏矩陣，與 TF-IDF 特徵合併
    X_dense = std_scaler_to_sparse(X_dense_df)
    dense_cols = list(X_dense_df.columns)
    feature_names = dense_cols + [f"tfidf::{tok}" for tok in vocab]
    X_all = hstack([X_dense, X_tfidf], format="csr")

    # 構建模型：根據算法名稱創建對應的模型實例
    model, needs_dense = build_model(alg_name, args)
    if model is None:
        return None, None, None

    # 10 折
    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
    fold_rows = []

    # 收集所有預測結果
    all_predictions = []
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        Xtr, Xva = X_all[tr_idx], X_all[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        meta_va = meta.iloc[va_idx]  # 驗證集的 meta 資料

        # 特徵選擇（Feature Selection）
        if fs_method == "lgbm_fs":
            Xtr_fs, Xva_fs, idx = select_features_by_lgbm(Xtr, ytr, Xva, top_k=100)
            sel_len = None if idx is None else int(len(idx))
        else:
            Xtr_fs, Xva_fs, sel_len = Xtr, Xva, None

        # 轉換為 Dense 格式（某些模型需要，如 SVM）
        if needs_dense:
            Xtr_fit = Xtr_fs.toarray()
            Xva_fit = Xva_fs.toarray()
        else:
            Xtr_fit, Xva_fit = Xtr_fs, Xva_fs

        # ====== Oversampling（只對訓練資料做，用於處理類別不平衡）======
        if args.oversample != "none":
            if not HAS_IMB:
                warnings.warn("未安裝 imbalanced-learn，oversample 選項將被略過。請安裝 imbalanced-learn。")
            else:
                if args.oversample == "random":
                    sampler = RandomOverSampler(random_state=42)
                    # RandomOverSampler 可處理稀疏矩陣（保持原格式）
                    Xtr_fit, ytr = sampler.fit_resample(Xtr_fit, ytr)
                elif args.oversample == "smote":
                    # SMOTE 需 dense
                    Xtr_dense = Xtr_fit.toarray() if hasattr(Xtr_fit, "toarray") else Xtr_fit
                    sampler = SMOTE(random_state=42)
                    Xtr_dense, ytr = sampler.fit_resample(Xtr_dense, ytr)
                    Xtr_fit = Xtr_dense

        model.fit(Xtr_fit, ytr)
        y_pred = model.predict(Xva_fit)
        y_score = get_positive_score(model, Xva_fit)
        m = eval_metrics(yva, y_pred, y_score)
        
        # 收集這一折的預測結果
        for i, (true_val, pred_val, score_val) in enumerate(zip(yva, y_pred, y_score)):
            pred_row = {
                "run_id": run_id,
                "fold": fold,
                "product_id": int(meta_va.iloc[i].get("product_id", 0)),
                "keyword": meta_va.iloc[i].get("keyword", ""),
                "y_true": int(true_val),
                "y_pred": int(pred_val),
                "y_score": float(score_val)
            }
            all_predictions.append(pred_row)
        
        row = {
            "run_id": run_id,
            "fold": fold,
            "algorithm": alg_name,
            "fs_method": fs_method,
            "n_products": int(len(va_idx)),
            "pos_ratio": float(yva.mean()),
            **m
        }
        if sel_len is not None:
            row["selected_features"] = sel_len
        fold_rows.append(row)

    df_folds = pd.DataFrame(fold_rows)

    # 計算跨折的統計摘要（平均值和標準差）
    agg = {}
    for col in [c for c in df_folds.columns if c not in {"run_id","fold","algorithm","fs_method","n_products","pos_ratio","selected_features"}]:
        agg[f"{col}_mean"] = df_folds[col].mean()
        agg[f"{col}_std"]  = df_folds[col].std(ddof=0)
    df_summary = pd.DataFrame([{
        "run_id": run_id,
        "algorithm": alg_name,
        "fs_method": fs_method,
        "folds": args.cv,
        **agg
    }])

    # 保存結果檔案
    os.makedirs(outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"run_{args.date_cutoff.replace('-','')}_{args.vocab_mode}_top{top_n}_{alg_name}_{fs_method}_{stamp}"
    p_folds   = os.path.join(outdir, f"{base}_fold_metrics.csv")
    p_summary = os.path.join(outdir, f"{base}_summary.csv")
    p_predictions = os.path.join(outdir, f"{base}_predictions.csv")
    
    df_folds.to_csv(p_folds, index=False, encoding="utf-8-sig")
    df_summary.to_csv(p_summary, index=False, encoding="utf-8-sig")
    
    # 保存所有折的預測結果（用於後續分析）
    df_predictions = None
    if all_predictions:
        df_predictions = pd.DataFrame(all_predictions)
        df_predictions.to_csv(p_predictions, index=False, encoding="utf-8-sig")
    else:
        p_predictions = None

    # 構建完整的評估指標和模型資訊（方便之後寫入 DB）
    metrics_v1 = build_metrics_v1(
        df_predictions,
        getattr(args, "topk_values", [20, 50, 100]),
        getattr(args, "threshold_grid_values", [0.5]),
        getattr(args, "threshold_min_recall", 0.0),
        getattr(args, "threshold_target", "f1_1")
    ) if df_predictions is not None else None
    # 在完整資料集上訓練模型以提取特徵重要度
    feature_model = None
    feature_importance = []
    error_examples = {"false_positives": [], "false_negatives": []}
    if metrics_v1:
        feature_model, needs_dense_final = build_model(alg_name, args)
        if feature_model is not None:
            X_full = X_all
            X_full_fit = X_full.toarray() if needs_dense_final else X_full
            feature_model.fit(X_full_fit, y)  # 在完整資料集上訓練
            feature_importance = extract_feature_importance(feature_model, feature_names, top_k=20)
        error_examples = collect_error_examples(df_predictions)
    # 提取模型參數（優先使用完整資料集訓練的模型）
    base_params_model = feature_model if feature_model is not None else model
    if hasattr(base_params_model, "get_params"):
        # XGBoost 有特殊的 get_xgb_params 方法
        params_dict = getattr(base_params_model, "get_xgb_params", base_params_model.get_params)()
    else:
        params_dict = {}
    feature_block_info = {
        "params": params_dict,
        "dense": dense_cols,
        "tfidf_vocab_size": len(vocab)
    }
    files_dict = {
        "fold_metrics": p_folds,
        "summary": p_summary
    }
    if p_predictions:
        files_dict["predictions"] = p_predictions
    
    manifest = {
        "run_id": run_id,
        "date_cutoff": args.date_cutoff,
        "mode": args.mode,
        "vocab_mode": args.vocab_mode,
        "top_n": top_n,
        "algorithms": [alg_name],
        "fs_method": fs_method,
        "cv": args.cv,
        "pipeline_version": args.pipeline_version,
        "excluded_products": [int(x) for x in (args.exclude_products.split(",") if args.exclude_products else [])],
        "files": files_dict,
        "shapes": {
            "X_dense": list(X_dense_df.shape),
            "X_tfidf": [int(X_tfidf.shape[0]), int(X_tfidf.shape[1])],
            "y": int(y.shape[0]),
            "vocab": int(len(vocab))
        },
        "dense_features": list(X_dense_df.columns),
        "vocab_preview": vocab[:10],
        "dataset_manifest": dataset_art,
        "labeling_config": args.labeling_config,
        "sampling_config": args.sampling_config,
        "target_focus": "positive_class_detection",
        "imbalance": {
            "oversample": args.oversample,          # none / random / smote
            "applied_on": "train_folds"
        },
        "model_params": {
            "algorithm": alg_name,
            "fs_method": fs_method,
            # 實際丟進去的參數（像 xgb 有開 scale_pos_weight 就會出現在這裡）
            "params": params_dict
        }
    }
    if metrics_v1:
        manifest["metrics_v1"] = metrics_v1
        manifest["feature_importance"] = feature_importance
        manifest["error_analysis"] = error_examples
    model_run_path = None
    if metrics_v1:
        model_run_path = os.path.join(outdir, f"{base}_model_run_v1.json")
        manifest["model_run_path"] = model_run_path
    p_manifest = os.path.join(outdir, f"{base}_manifest.json")
    with open(p_manifest, "w", encoding="utf-8-sig") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if model_run_path:
        model_run_payload = build_model_run_payload(
            run_id, args, alg_name, fs_method, dataset_art, files_dict, feature_block_info, metrics_v1, feature_importance, error_examples
        )
        with open(model_run_path, "w", encoding="utf-8-sig") as f:
            json.dump(model_run_payload, f, ensure_ascii=False, indent=2)

    return df_folds, df_summary, p_manifest

def main():
    args = parse_args()
    args.keyword_blacklist = parse_str_list(args.keyword_blacklist)
    args.keyword_whitelist = parse_str_list(args.keyword_whitelist)
    topk_values = [int(x) for x in args.topk.split(",") if x.strip()] if args.topk else [20, 50, 100]
    threshold_grid_values = [float(x) for x in args.threshold_grid.split(",") if x.strip()] if args.threshold_grid else [0.5]
    args.topk_values = topk_values
    args.threshold_grid_values = threshold_grid_values
    labeling_config = {
        "definition": "delta_abs",
        "delta_threshold": args.label_delta_threshold,
        "ratio_threshold": args.label_ratio_threshold,
        "max_gap_days": args.label_max_gap_days
    }
    sampling_config = {
        "min_comments": args.min_comments,
        "keyword_whitelist": args.keyword_whitelist,
        "keyword_blacklist": args.keyword_blacklist,
        "single_keyword": args.keyword,
        "exclude_products": [int(x) for x in args.exclude_products.split(",")] if args.exclude_products else []
    }
    args.labeling_config = labeling_config
    args.sampling_config = sampling_config
    run_id = str(uuid.uuid4())

    # 解析多值參數（逗號分隔的列表）
    topn_list = [int(x) for x in args.top_n.split(",")]
    alg_list  = [s.strip().lower() for s in args.algorithms.split(",")]
    fs_list   = [s.strip() for s in args.fs_methods.split(",")]
    excluded  = [int(x) for x in args.exclude_products.split(",")] if args.exclude_products else []

    # 取得資料（依 mode）
    if args.mode == "product_level":
        X_dense_df, X_tfidf, y, meta, vocab = load_product_level_training_set(
            date_cutoff=args.date_cutoff,
            top_n=topn_list[0],                 # vocab 以第一個 top_n 建立（若要每個 topN都建 vocab，可外層迭代再呼叫一次 loader）
            pipeline_version=args.pipeline_version,
            vocab_mode=args.vocab_mode,
            single_keyword=args.keyword,
            exclude_products=excluded,
            label_delta_threshold=args.label_delta_threshold,
            label_ratio_threshold=args.label_ratio_threshold,
            label_max_gap_days=args.label_max_gap_days,
            min_comments=args.min_comments,
            keyword_whitelist=args.keyword_whitelist,
            keyword_blacklist=args.keyword_blacklist
        )
    else:
        # 保留原 comment-level（以便回溯）——只用單一 topN
        from data_loader import load_training_set
        X_dense_df, X_tfidf, y, meta, vocab = load_training_set(
            top_n=topn_list[0],
            pipeline_version=args.pipeline_version,
            date_cutoff=args.date_cutoff,
            vocab_scope="global",
        )

    dataset_art = save_dataset_artifacts(
        outdir=args.outdir,
        mode=args.mode,
        date_cutoff=args.date_cutoff,
        vocab_mode=args.vocab_mode,
        top_n_built=len(vocab),
        X_dense_df=X_dense_df,
        X_tfidf=X_tfidf,
        y=y,
        meta=meta,
        vocab=vocab,
        labeling_config=labeling_config,
        sampling_config=sampling_config
    )

    # === 類別分佈 & 建議權重 ===
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    total = int(len(y))
    pos_ratio = (pos / total) if total else 0.0
    suggested_spw = (neg / pos) if pos > 0 else None  # XGBoost 推薦 scale_pos_weight

    args.suggested_spw = suggested_spw  # 讓子函式可用
    args.pos = pos; args.neg = neg; args.pos_ratio = pos_ratio

    # 遍歷所有參數組合進行訓練
    # 注意：目前簡化實現，所有 topN 共用同一份 vocab 和 X_tfidf
    # 若要對每個 topN 都重建 vocab+X_tfidf，需要在外層迭代並重新呼叫 loader
    os.makedirs(args.outdir, exist_ok=True)
    all_folds, all_summaries, manifests = [], [], []

    for topn in topn_list:
        # 簡化實現：當 topn != 建立時的 topN，只是記錄 metadata，不重建 TF-IDF
        # 如需完整重建，可在這裡重新呼叫 load_product_level_training_set
        for alg in alg_list:
            for fs in fs_list:
                out = run_one_setting(run_id, args, topn, alg, fs, X_dense_df, X_tfidf, y, meta, vocab, args.outdir, dataset_art)
                if out[0] is None:
                    continue
                df_folds, df_summary, p_manifest = out
                all_folds.append(df_folds)
                all_summaries.append(df_summary)
                manifests.append(p_manifest)

    # 合併所有參數組合的結果，便於整體檢視
    if all_folds:
        folds_cat = pd.concat(all_folds, ignore_index=True)
        folds_cat.to_csv(os.path.join(args.outdir, f"{run_id}_ALL_fold_metrics.csv"),
                         index=False, encoding="utf-8-sig")
    if all_summaries:
        sum_cat = pd.concat(all_summaries, ignore_index=True)
        sum_cat.to_csv(os.path.join(args.outdir, f"{run_id}_ALL_summary.csv"),
                       index=False, encoding="utf-8-sig")

    # 寫入 run-level 總 manifest（包含所有子實驗的 manifest 路徑和配置）
    run_manifest = {
        "run_id": run_id,
        "date_cutoff": args.date_cutoff,
        "mode": args.mode,
        "vocab_mode": args.vocab_mode,
        "top_n_list": topn_list,
        "algorithms": alg_list,
        "fs_methods": fs_list,
        "cv": args.cv,
        "pipeline_version": args.pipeline_version,
        "exclude_products": excluded,
        "dataset_manifest": dataset_art,
        "children": manifests,
        "labeling_config": labeling_config,
        "sampling_config": sampling_config,
        "target_focus": "positive_class_detection",
        "imbalance_config": {
            "oversample": args.oversample,            # none / random / smote
            "target_distribution": {
                "pos": pos, "neg": neg,
                "total": total, "pos_ratio": pos_ratio
            },
            "suggested_weights": {
                "xgboost": {"scale_pos_weight": suggested_spw if args.oversample == "xgb_scale_pos_weight" else None},
                # 如果你之後要在 SVM/LightGBM 啟用 class_weight 也能在這裡寫下建議值
                # "svm": {"class_weight": "balanced"},
                # "lightgbm": {"is_unbalance": True}
            }
        }
    }

    with open(os.path.join(args.outdir, f"{run_id}_RUN_manifest.json"), "w", encoding="utf-8-sig") as f:
        json.dump(run_manifest, f, ensure_ascii=False, indent=2)

    print("[DONE] Outputs saved to:", args.outdir)

if __name__ == "__main__":
    main()
