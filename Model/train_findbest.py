# -*- coding: utf-8-sig -*-
"""
Model/train_findbest.py
基於 train.py，新增最佳閾值尋找與視覺化功能
- 訓練流程與 train.py 相同
- 訓練完成後，針對驗證集計算不同 threshold 下的 precision、recall、F1-score
- 找出使 F1-score 最大化的最佳閾值
- 繪製 threshold vs metrics 圖表

python Model/train_findbest.py \
  --mode product_level \
  --date-cutoff 2025-06-25 \
  --vocab-mode global \
  --top-n 100 \
  --algorithms xgboost \
  --fs-methods no_fs \
  --cv 10 \
  --exclude-products 8918452 \
  --outdir Model/outputs \
  --oversample random
"""

# -*- coding: utf-8-sig -*-
from __future__ import annotations
import os, json, argparse, warnings, uuid
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非互動式後端
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from scipy.sparse import hstack, csr_matrix, save_npz
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, precision_recall_fscore_support,
    precision_recall_curve
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
    load_training_set,
    load_product_level_training_set
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="product_level",
                    choices=["product_level", "comment_level"])
    ap.add_argument("--date-cutoff", type=str, default="2025-06-25")
    ap.add_argument("--pipeline-version", type=str, default=os.getenv("PIPELINE_VERSION", None))
    ap.add_argument("--vocab-mode", type=str, default="global")
    ap.add_argument("--top-n", type=str, default="100")
    ap.add_argument("--algorithms", type=str, default="xgboost,svm")
    ap.add_argument("--fs-methods", type=str, default="no_fs,lgbm_fs")
    ap.add_argument("--cv", type=int, default=10)
    ap.add_argument("--exclude-products", type=str, default="8918452")
    ap.add_argument("--keyword", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="Model/outputs")
    ap.add_argument("--oversample", type=str, default="none",
                choices=["none", "random", "smote", "xgb_scale_pos_weight"],
                help="類別不平衡處理：none/random/smote/xgb_scale_pos_weight（僅作用於訓練折）")
    return ap.parse_args()

def std_scaler_to_sparse(Xdf: pd.DataFrame) -> csr_matrix:
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(Xdf.values)
    return csr_matrix(X_scaled)

def select_features_by_lgbm(X_train, y_train, X_valid, top_k=100):
    if not HAS_LGBM:
        warnings.warn("未安裝 LightGBM，FS 略過。")
        return X_train, X_valid, None
    clf = LGBMClassifier(device="gpu", n_estimators=400, learning_rate=0.05,
                         num_leaves=64, random_state=42)
    clf.fit(X_train, y_train)
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1]
    k = max(1, min(top_k, (imp > 0).sum() or top_k))
    idx = order[:k]
    return X_train[:, idx], X_valid[:, idx], idx

def get_positive_score(model, X):
    try:
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] >= 2 else proba[:, -1]
    except Exception:
        try:
            return model.decision_function(X)
        except Exception:
            return model.predict(X)

def eval_metrics(y_true, y_pred, y_score) -> dict:
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

def find_best_threshold(y_true, y_score, average='macro'):
    """
    找出使 F1-score 最大化的最佳閾值
    
    Args:
        y_true: 真實標籤
        y_score: 預測機率分數
        average: 'macro' 或 'weighted'，用於計算平均 metrics（目前未使用，保留以備未來擴充）
    
    Returns:
        best_threshold: 最佳閾值
        best_precision: 該閾值下的 precision
        best_recall: 該閾值下的 recall
        best_f1: 該閾值下的 F1-score
        thresholds: 所有閾值（長度比 precision/recall 少 1）
        precisions: 所有閾值對應的 precision
        recalls: 所有閾值對應的 recall
        f1_scores: 所有閾值對應的 F1-score
    """
    # 使用 precision_recall_curve 計算 precision 和 recall
    # 注意：返回的 thresholds 長度比 precision/recall 少 1
    # precision/recall 的最後一個值對應 threshold=0（所有樣本都預測為正類）
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    
    # 計算 F1-score: f1 = 2 * (precision * recall) / (precision + recall)
    # 避免除零
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # 找出 F1-score 最大值的索引
    best_idx = np.argmax(f1_scores)
    
    # 對應的閾值
    # 如果 best_idx < len(thresholds)，直接使用對應的 threshold
    # 如果 best_idx == len(thresholds)（最後一個點），對應 threshold=0
    if best_idx < len(thresholds):
        best_threshold = thresholds[best_idx]
    else:
        # 如果最佳值在最後一個點，對應 threshold=0
        best_threshold = 0.0
    
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    best_f1 = f1_scores[best_idx]
    
    return best_threshold, best_precision, best_recall, best_f1, thresholds, precision, recall, f1_scores

def find_best_threshold_class1(y_true, y_score):
    """
    針對 y=1 類別找出最佳閾值（這實際上就是標準的二分類問題）
    與 find_best_threshold 相同，但命名更明確
    """
    return find_best_threshold(y_true, y_score, average='macro')

def plot_threshold_metrics(y_true, y_score, save_path, title_suffix=""):
    """
    繪製 threshold vs metrics 圖表
    
    Args:
        y_true: 真實標籤
        y_score: 預測機率分數
        save_path: 儲存路徑
        title_suffix: 標題後綴
    """
    # 計算最佳閾值與所有 metrics
    best_thresh, best_prec, best_rec, best_f1, thresholds, precisions, recalls, f1_scores = \
        find_best_threshold(y_true, y_score)
    
    # 建立完整的 thresholds 陣列
    # precision_recall_curve 返回的 thresholds 長度比 precision/recall 少 1
    # precision/recall 的最後一個值對應 threshold=0（所有樣本都預測為正類）
    # thresholds 是按降序排列的，從高到低
    if len(thresholds) > 0:
        # 補上最後一個點對應的 threshold=0
        all_thresholds = np.concatenate([thresholds, [0.0]])
    else:
        # 如果沒有 thresholds（所有樣本預測相同），使用單一 threshold=0
        all_thresholds = np.array([0.0])
    
    # 確保長度一致（應該已經一致了，但為了安全起見）
    min_len = min(len(all_thresholds), len(precisions), len(recalls), len(f1_scores))
    if len(all_thresholds) != min_len or len(precisions) != min_len:
        warnings.warn(f"長度不一致: thresholds={len(all_thresholds)}, precisions={len(precisions)}, recalls={len(recalls)}, f1_scores={len(f1_scores)}，將截斷至 {min_len}")
        all_thresholds = all_thresholds[:min_len]
        precisions = precisions[:min_len]
        recalls = recalls[:min_len]
        f1_scores = f1_scores[:min_len]
    
    # 按照 threshold 排序（從大到小，這樣繪圖時從左到右顯示）
    # precision_recall_curve 返回的 thresholds 已經是按降序排列的，所以不需要再排序
    # 但為了確保，我們按照 threshold 降序排列
    sort_idx = np.argsort(all_thresholds)[::-1]
    all_thresholds = all_thresholds[sort_idx]
    precisions = precisions[sort_idx]
    recalls = recalls[sort_idx]
    f1_scores = f1_scores[sort_idx]
    
    # 繪製圖表
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 第一張圖：所有類別的平均 metrics（macro average）
    ax1 = axes[0]
    ax1.plot(all_thresholds, f1_scores, label='F1-score', linewidth=2, color='blue')
    ax1.plot(all_thresholds, precisions, label='Precision', linewidth=2, color='green')
    ax1.plot(all_thresholds, recalls, label='Recall', linewidth=2, color='red')
    
    # 標註最佳閾值
    ax1.axvline(x=best_thresh, color='black', linestyle='--', linewidth=2, 
                label=f'最佳閾值: {best_thresh:.3f}')
    ax1.scatter([best_thresh], [best_f1], color='blue', s=100, zorder=5,
                label=f'最佳 F1: {best_f1:.3f}')
    
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title(f'Threshold vs Metrics (Macro Average){title_suffix}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # 第二張圖：針對 y=1 類別（實際上與第一張相同，因為是二分類）
    # 但我們可以明確標註這是針對正類（y=1）
    ax2 = axes[1]
    ax2.plot(all_thresholds, f1_scores, label='F1-score (y=1)', linewidth=2, color='blue')
    ax2.plot(all_thresholds, precisions, label='Precision (y=1)', linewidth=2, color='green')
    ax2.plot(all_thresholds, recalls, label='Recall (y=1)', linewidth=2, color='red')
    
    # 標註最佳閾值
    ax2.axvline(x=best_thresh, color='black', linestyle='--', linewidth=2,
                label=f'最佳閾值: {best_thresh:.3f}')
    ax2.scatter([best_thresh], [best_f1], color='blue', s=100, zorder=5,
                label=f'最佳 F1: {best_f1:.3f}')
    
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title(f'Threshold vs Metrics (Class y=1){title_suffix}', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_thresh, best_prec, best_rec, best_f1

def save_dataset_artifacts(outdir: str,
                           mode: str,
                           date_cutoff: str,
                           vocab_mode: str,
                           top_n_built: int,
                           X_dense_df: pd.DataFrame,
                           X_tfidf: csr_matrix,
                           y: pd.Series,
                           meta: pd.DataFrame,
                           vocab: list) -> dict:
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
        "meta_columns": list(meta.columns)
    }
    with open(p_dsman, "w", encoding="utf-8-sig") as f:
        json.dump(ds_manifest, f, ensure_ascii=False, indent=2)

    ds_manifest["path"] = p_dsman
    return ds_manifest

def run_one_setting(run_id: str, args, top_n: int, alg_name: str, fs_method: str,
                    X_dense_df, X_tfidf, y, meta, vocab, outdir: str):
    # 組特徵
    X_dense = std_scaler_to_sparse(X_dense_df)
    X_all = hstack([X_dense, X_tfidf], format="csr")

    # 模型
    if alg_name.lower() == "xgboost":
        model = xgb.XGBClassifier(
            tree_method="gpu_hist", predictor="gpu_predictor",
            n_estimators=800, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
            scale_pos_weight=args.suggested_spw if args.oversample == "xgb_scale_pos_weight" else None
        )
        needs_dense = False
    elif alg_name.lower() == "svm":
        if HAS_CUML and cuSVC is not None:
            model = cuSVC(probability=True)
            needs_dense = True
        else:
            model = skSVC(probability=True, kernel="rbf")
            needs_dense = True
    elif alg_name.lower() == "lightgbm" and HAS_LGBM:
        model = LGBMClassifier(device="gpu", n_estimators=800, learning_rate=0.05,
                               num_leaves=127, subsample=0.8, colsample_bytree=0.8, random_state=42)
        needs_dense = False
    else:
        warnings.warn(f"未知或不可用演算法：{alg_name}，略過。")
        return None, None, None, None

    # 10 折
    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
    fold_rows = []

    # 收集所有預測結果（用於最佳閾值分析）
    all_y_true = []
    all_y_score = []
    all_predictions = []
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        Xtr, Xva = X_all[tr_idx], X_all[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        meta_va = meta.iloc[va_idx]

        # FS
        if fs_method == "lgbm_fs":
            Xtr_fs, Xva_fs, idx = select_features_by_lgbm(Xtr, ytr, Xva, top_k=100)
            sel_len = None if idx is None else int(len(idx))
        else:
            Xtr_fs, Xva_fs, sel_len = Xtr, Xva, None

        # Dense 需求
        if needs_dense:
            Xtr_fit = Xtr_fs.toarray()
            Xva_fit = Xva_fs.toarray()
        else:
            Xtr_fit, Xva_fit = Xtr_fs, Xva_fs

        # ====== Oversampling（只對訓練資料做）======
        if args.oversample != "none":
            if not HAS_IMB:
                warnings.warn("未安裝 imbalanced-learn，oversample 選項將被略過。請安裝 imbalanced-learn。")
            else:
                if args.oversample == "random":
                    sampler = RandomOverSampler(random_state=42)
                    Xtr_fit, ytr = sampler.fit_resample(Xtr_fit, ytr)
                elif args.oversample == "smote":
                    Xtr_dense = Xtr_fit.toarray() if hasattr(Xtr_fit, "toarray") else Xtr_fit
                    sampler = SMOTE(random_state=42)
                    Xtr_dense, ytr = sampler.fit_resample(Xtr_dense, ytr)
                    Xtr_fit = Xtr_dense

        model.fit(Xtr_fit, ytr)
        y_pred = model.predict(Xva_fit)
        y_score = get_positive_score(model, Xva_fit)
        m = eval_metrics(yva, y_pred, y_score)
        
        # 收集這一折的預測結果（用於最佳閾值分析）
        all_y_true.extend(yva.values)
        all_y_score.extend(y_score)
        
        # 收集預測結果（用於檔案輸出）
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

    # 摘要
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

    # 寫檔
    os.makedirs(outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"run_{args.date_cutoff.replace('-','')}_{args.vocab_mode}_top{top_n}_{alg_name}_{fs_method}_{stamp}"
    p_folds   = os.path.join(outdir, f"{base}_fold_metrics.csv")
    p_summary = os.path.join(outdir, f"{base}_summary.csv")
    p_predictions = os.path.join(outdir, f"{base}_predictions.csv")
    
    df_folds.to_csv(p_folds, index=False, encoding="utf-8-sig")
    df_summary.to_csv(p_summary, index=False, encoding="utf-8-sig")
    
    # 保存預測結果
    if all_predictions:
        df_predictions = pd.DataFrame(all_predictions)
        df_predictions.to_csv(p_predictions, index=False, encoding="utf-8-sig")
    else:
        p_predictions = None

    # ====== 最佳閾值分析與視覺化 ======
    all_y_true = np.array(all_y_true)
    all_y_score = np.array(all_y_score)
    
    # 計算最佳閾值
    best_thresh, best_prec, best_rec, best_f1 = find_best_threshold(all_y_true, all_y_score)
    
    # 繪製圖表
    plot_path = os.path.join(outdir, f"{base}_threshold_metrics.png")
    title_suffix = f" ({alg_name}, {fs_method})"
    plot_threshold_metrics(all_y_true, all_y_score, plot_path, title_suffix=title_suffix)
    
    # 輸出最佳閾值與分數摘要
    print("\n" + "="*60)
    print(f"最佳閾值分析 - {alg_name} ({fs_method})")
    print("="*60)
    print(f"最佳閾值: {best_thresh:.3f}")
    print(f"Precision: {best_prec:.3f}")
    print(f"Recall: {best_rec:.3f}")
    print(f"F1: {best_f1:.3f}")
    print(f"圖表已儲存至: {plot_path}")
    print("="*60 + "\n")
    
    # 將最佳閾值資訊加入 summary
    threshold_info = {
        "best_threshold": float(best_thresh),
        "best_precision": float(best_prec),
        "best_recall": float(best_rec),
        "best_f1": float(best_f1),
        "threshold_plot": plot_path
    }
    
    # 更新 manifest
    files_dict = {
        "fold_metrics": p_folds,
        "summary": p_summary,
        "threshold_plot": plot_path
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
        "threshold_analysis": threshold_info,
        "shapes": {
            "X_dense": list(X_dense_df.shape),
            "X_tfidf": [int(X_tfidf.shape[0]), int(X_tfidf.shape[1])],
            "y": int(y.shape[0]),
            "vocab": int(len(vocab))
        },
        "dense_features": list(X_dense_df.columns),
        "vocab_preview": vocab[:10],
        "imbalance": {
            "oversample": args.oversample,
            "applied_on": "train_folds"
        },
        "model_params": {
            "algorithm": alg_name,
            "fs_method": fs_method,
            "params": getattr(model, "get_xgb_params", getattr(model, "get_params", lambda: {}))()
                        if hasattr(model, "get_params") else {}
        }
    }
    p_manifest = os.path.join(outdir, f"{base}_manifest.json")
    with open(p_manifest, "w", encoding="utf-8-sig") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return df_folds, df_summary, p_manifest, threshold_info

def main():
    args = parse_args()
    run_id = str(uuid.uuid4())

    # 解析多值參數
    topn_list = [int(x) for x in args.top_n.split(",")]
    alg_list  = [s.strip().lower() for s in args.algorithms.split(",")]
    fs_list   = [s.strip() for s in args.fs_methods.split(",")]
    excluded  = [int(x) for x in args.exclude_products.split(",")] if args.exclude_products else []

    # 取得資料（依 mode）
    if args.mode == "product_level":
        X_dense_df, X_tfidf, y, meta, vocab = load_product_level_training_set(
            date_cutoff=args.date_cutoff,
            top_n=topn_list[0],
            pipeline_version=args.pipeline_version,
            vocab_mode=args.vocab_mode,
            single_keyword=args.keyword,
            exclude_products=excluded
        )
    else:
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
        vocab=vocab
    )

    # === 類別分佈 & 建議權重 ===
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    total = int(len(y))
    pos_ratio = (pos / total) if total else 0.0
    suggested_spw = (neg / pos) if pos > 0 else None

    args.suggested_spw = suggested_spw
    args.pos = pos; args.neg = neg; args.pos_ratio = pos_ratio

    os.makedirs(args.outdir, exist_ok=True)
    all_folds, all_summaries, manifests = [], [], []
    all_threshold_infos = []

    for topn in topn_list:
        for alg in alg_list:
            for fs in fs_list:
                out = run_one_setting(run_id, args, topn, alg, fs, X_dense_df, X_tfidf, y, meta, vocab, args.outdir)
                if out[0] is None:
                    continue
                df_folds, df_summary, p_manifest, threshold_info = out
                all_folds.append(df_folds)
                all_summaries.append(df_summary)
                manifests.append(p_manifest)
                all_threshold_infos.append(threshold_info)

    # 合併輸出便於檢視
    if all_folds:
        folds_cat = pd.concat(all_folds, ignore_index=True)
        folds_cat.to_csv(os.path.join(args.outdir, f"{run_id}_ALL_fold_metrics.csv"),
                         index=False, encoding="utf-8-sig")
    if all_summaries:
        sum_cat = pd.concat(all_summaries, ignore_index=True)
        sum_cat.to_csv(os.path.join(args.outdir, f"{run_id}_ALL_summary.csv"),
                       index=False, encoding="utf-8-sig")

    # 寫入一份 run-level 總 manifest（含多個子 manifest 路徑）
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
        "threshold_analyses": all_threshold_infos,
        "imbalance_config": {
            "oversample": args.oversample,
            "target_distribution": {
                "pos": pos, "neg": neg,
                "total": total, "pos_ratio": pos_ratio
            },
            "suggested_weights": {
                "xgboost": {"scale_pos_weight": suggested_spw if args.oversample == "xgb_scale_pos_weight" else None},
            }
        }
    }

    with open(os.path.join(args.outdir, f"{run_id}_RUN_manifest.json"), "w", encoding="utf-8-sig") as f:
        json.dump(run_manifest, f, ensure_ascii=False, indent=2)

    print("\n[DONE] Outputs saved to:", args.outdir)
    print(f"[DONE] Total runs: {len(manifests)}")
    print(f"[DONE] Threshold analyses completed: {len(all_threshold_infos)}")

if __name__ == "__main__":
    main()

