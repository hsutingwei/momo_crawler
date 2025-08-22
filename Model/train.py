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
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from scipy.sparse import hstack, csr_matrix, save_npz
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, precision_recall_fscore_support
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="product_level",
                    choices=["product_level", "comment_level"])
    ap.add_argument("--date-cutoff", type=str, default="2025-06-25")
    ap.add_argument("--pipeline-version", type=str, default=os.getenv("PIPELINE_VERSION", None))
    ap.add_argument("--vocab-mode", type=str, default="global")          # 先支援 global
    ap.add_argument("--top-n", type=str, default="100")                  # 可給 "100,200"
    ap.add_argument("--algorithms", type=str, default="xgboost,svm")     # 逗號分隔
    ap.add_argument("--fs-methods", type=str, default="no_fs,lgbm_fs")
    ap.add_argument("--cv", type=int, default=10)
    ap.add_argument("--exclude-products", type=str, default="8918452")   # 逗號分隔
    ap.add_argument("--keyword", type=str, default=None)                 # single_keyword 用
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
                    # RandomOverSampler 可處理稀疏矩陣（保持原格式）
                    Xtr_fit, ytr = sampler.fit_resample(Xtr_fit, ytr)
                elif args.oversample == "smote":
                    # SMOTE 需 dense
                    Xtr_dense = Xtr_fit.toarray() if hasattr(Xtr_fit, "toarray") else Xtr_fit
                    sampler = SMOTE(random_state=42, n_jobs=-1)
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

    # manifest（方便之後寫 DB）
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
        "imbalance": {
            "oversample": args.oversample,          # none / random / smote
            "applied_on": "train_folds"
        },
        "model_params": {
            "algorithm": alg_name,
            "fs_method": fs_method,
            # 實際丟進去的參數（像 xgb 有開 scale_pos_weight 就會出現在這裡）
            "params": getattr(model, "get_xgb_params", getattr(model, "get_params", lambda: {}))()
                        if hasattr(model, "get_params") else {}
        }
    }
    p_manifest = os.path.join(outdir, f"{base}_manifest.json")
    with open(p_manifest, "w", encoding="utf-8-sig") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return df_folds, df_summary, p_manifest

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
            top_n=topn_list[0],                 # vocab 以第一個 top_n 建立（若要每個 topN都建 vocab，可外層迭代再呼叫一次 loader）
            pipeline_version=args.pipeline_version,
            vocab_mode=args.vocab_mode,
            single_keyword=args.keyword,
            exclude_products=excluded
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
        vocab=vocab
    )

    # === 類別分佈 & 建議權重 ===
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    total = int(len(y))
    pos_ratio = (pos / total) if total else 0.0
    suggested_spw = (neg / pos) if pos > 0 else None  # XGBoost 推薦 scale_pos_weight

    args.suggested_spw = suggested_spw  # 讓子函式可用
    args.pos = pos; args.neg = neg; args.pos_ratio = pos_ratio

    # 若要對每個 topN 都各跑一次：可在外層重建 vocab+X_tfidf；先給一個實用版本：Dense 同一份，TF 用同一 vocab
    os.makedirs(args.outdir, exist_ok=True)
    all_folds, all_summaries, manifests = [], [], []

    for topn in topn_list:
        # 簡化：當 topn != 建立時的 topN，只是記錄 metadata，不重建（需要重建再擴充）
        for alg in alg_list:
            for fs in fs_list:
                out = run_one_setting(run_id, args, topn, alg, fs, X_dense_df, X_tfidf, y, meta, vocab, args.outdir)
                if out[0] is None:
                    continue
                df_folds, df_summary, p_manifest = out
                all_folds.append(df_folds)
                all_summaries.append(df_summary)
                manifests.append(p_manifest)

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
        "imbalance_config": {
            "oversample": args.oversample,            # none / random / smote
            "target_distribution": {
                "pos": pos, "neg": neg,
                "total": total, "pos_ratio": pos_ratio
            },
            "suggested_weights": {
                "xgboost": {"scale_pos_weight": suggested_spw},
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