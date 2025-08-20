# -*- coding: utf-8-sig -*-
"""
Model/train.py
三模型( SVM / LightGBM / XGBoost ) × 兩種特徵選取(無 / 基於 LGBM 重要度)
- cuML 存在就用 GPU SVM，否則自動退回 sklearn.SVC（CPU）
- LightGBM / XGBoost 預設走 GPU，缺套件會自動略過或退回
- 以 date_cutoff 切訓練/測試；TF-IDF top_n 可切換
- 輸出結果到 model_results.csv / model_results.json
"""

from __future__ import annotations
import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import hstack, csr_matrix
from scipy.sparse import save_npz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_fscore_support

# ----- 嘗試載入 LightGBM（可 GPU），失敗就跳過 -----
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
    LGBMClassifier = None  # type: ignore
    warnings.warn("LightGBM 未安裝，將略過 LightGBM 模型。")

# ----- XGBoost（支援 GPU） -----
import xgboost as xgb

# ----- 嘗試載入 cuML SVM（GPU），失敗退回 sklearn（CPU） -----
try:
    from cuml.svm import SVC as cuSVC  # NVIDIA RAPIDS cuML (GPU)
    HAS_CUML = True
except Exception:
    HAS_CUML = False
    cuSVC = None  # type: ignore
    warnings.warn("cuML 未安裝，SVM 將自動退回 CPU 版本（sklearn.SVC）。")

from sklearn.svm import SVC as skSVC  # CPU fallback

# 讀取 DB + 組特徵（你先把我之前給的 Model/data_loader.py 放好）
from data_loader import load_training_set


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=100,
                    help="TF-IDF Top-N tokens (e.g. 100 or 200)")
    ap.add_argument("--date-cutoff", type=str, default="2025-06-25",
                    help="訓練/測試切割日，<= 為 train，> 為 test (YYYY-MM-DD)")
    ap.add_argument("--pipeline-version", type=str, default=os.getenv("PIPELINE_VERSION", None),
                    help="NLP pipeline version，預設取 .env 的 PIPELINE_VERSION")
    ap.add_argument("--fs-topk", type=int, default=100,
                    help="Feature selection 取前 K 個重要特徵（LGBM 重要度）")
    ap.add_argument("--out", type=str, default="model_results",
                    help="輸出檔名前綴（不含副檔名）")
    return ap.parse_args()


def select_features_by_lgbm(X_train: csr_matrix, y_train: pd.Series,
                            X_test: csr_matrix, top_k: int = 100):
    """
    用 LGBM 做一次 fit，取最重要的 top_k 特徵索引；套用到 train/test。
    注意：importance 對稀疏矩陣在樹模型很好用；top_k 請勿設過大以免顯存不足。
    """
    if not HAS_LGBM:
        warnings.warn("未安裝 LightGBM，無法執行 LGBM 特徵選取；將直接回傳原特徵。")
        return X_train, X_test, np.arange(X_train.shape[1])

    lgb = LGBMClassifier(
        device="gpu",           # 沒有 GPU 也能跑，LightGBM 會自行 fallback
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        random_state=42
    )
    lgb.fit(X_train, y_train)
    importances = lgb.feature_importances_
    order = np.argsort(importances)[::-1]
    # 至少取 top_k；若重要度全 0，避免空集合
    top_idx = order[:max(1, min(top_k, max(1, int((importances > 0).sum()))))]
    
    # 新增：檢查特徵選取是否真的生效
    print(f"[FS] Original features: {X_train.shape[1]}, Selected: {len(top_idx)}")
    print(f"[FS] Top 10 important features: {top_idx[:10]}")
    
    return X_train[:, top_idx], X_test[:, top_idx], top_idx

def get_positive_score(model, X):
    """優先用 predict_proba 的正類機率；不行就用 decision_function 分數；最後退回預測值(0/1)。"""
    try:
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        # multinomial 保底：取最後一欄當正類（不太會用到）
        return proba[:, -1]
    except Exception:
        try:
            score = model.decision_function(X)
            return score  # roc_auc_score 可直接用 margin 分數
        except Exception:
            # 最保底：0/1
            return model.predict(X)

def evaluate(y_true, y_pred, y_score) -> dict:
    # 整體
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    # 整體 ROC-AUC（僅二分類）
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        metrics["auc"] = float("nan")

    # 類別別（labels=[0,1] 的順序）
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    # p[0]/r[0]/f1[0] 對應類別0；p[1]/r[1]/f1[1] 對應類別1
    metrics.update({
        "precision_0": float(p[0]),
        "recall_0": float(r[0]),
        "f1_0": float(f1[0]),
        "support_0": int(support[0]),
        "precision_1": float(p[1]),
        "recall_1": float(r[1]),
        "f1_1": float(f1[1]),
        "support_1": int(support[1]),
    })

    # 修正：類別別 AUC 計算
    try:
        # 對於二分類，auc_1 就是整體 AUC，auc_0 = 1 - auc_1
        auc_1 = roc_auc_score(y_true, y_score)  # 1 vs rest
        auc_0 = 1.0 - auc_1                     # 0 vs rest
    except Exception:
        auc_0 = float("nan")
        auc_1 = float("nan")

    metrics.update({
        "auc_0": float(auc_0),
        "auc_1": float(auc_1),
    })
    return metrics

def output_results(X_dense_df, X_tfidf, y, meta, vocab, args):
        # === 將訓練集產出存檔到 Model/load_training/ ===
    out_root = os.path.join(os.path.dirname(__file__), "load_training")
    os.makedirs(out_root, exist_ok=True)

    # 產生可讀的檔名前綴：例 trainset_dt20250625_top100_pip2608d6_20250815-143012
    def _canon(s: str | None) -> str:
        if not s: return "none"
        return "".join(c for c in s if c.isalnum())[:40]

    dt_key = "".join(args.date_cutoff.split("-")) if args.date_cutoff else "all"
    top_key = f"top{args.top_n}"
    pip_key = f"pip{_canon(args.pipeline_version)}"
    from datetime import datetime
    ts_key = datetime.now().strftime("%Y%m%d-%H%M%S")

    prefix = f"trainset_dt{dt_key}_{top_key}_{pip_key}_{ts_key}"
    # 檔名建議（便於辨識/比對不同參數）：
    #   - {prefix}_Xdense.csv      結構化(密集)特徵
    #   - {prefix}_Xtfidf.npz      稀疏 TF-IDF(1/0) 矩陣 (scipy.sparse)
    #   - {prefix}_y.csv           標籤
    #   - {prefix}_meta.csv        對應 comment/product/batch 等中繼欄位
    #   - {prefix}_vocab.txt       詞彙表（每行一個 token）
    #   - {prefix}_manifest.json   尺寸與欄位摘要

    # 1) X_dense
    X_dense_path = os.path.join(out_root, f"{prefix}_Xdense.csv")
    X_dense_df.to_csv(X_dense_path, index=False, encoding="utf-8-sig")


    # 2) X_tfidf (sparse)
    X_tfidf_path = os.path.join(out_root, f"{prefix}_Xtfidf.npz")
    save_npz(X_tfidf_path, X_tfidf)

    # 3) y
    y_path = os.path.join(out_root, f"{prefix}_y.csv")
    pd.DataFrame({"y": y.astype(int)}).to_csv(y_path, index=False, encoding="utf-8-sig")

    # 4) meta
    meta_path = os.path.join(out_root, f"{prefix}_meta.csv")
    meta.to_csv(meta_path, index=False, encoding="utf-8-sig")

    # 5) vocab
    vocab_path = os.path.join(out_root, f"{prefix}_vocab.txt")
    with open(vocab_path, "w", encoding="utf-8-sig") as f:
        for tok in vocab:
            f.write(f"{tok}\n")

    # 6) manifest（方便之後載入對照）
    manifest = {
        "prefix": prefix,
        "args": {
            "top_n": int(args.top_n),
            "date_cutoff": args.date_cutoff,
            "pipeline_version": args.pipeline_version,
        },
        "shapes": {
            "X_dense": list(X_dense_df.shape),
            "X_tfidf": [int(X_tfidf.shape[0]), int(X_tfidf.shape[1])],
            "y": int(y.shape[0]),
            "meta": list(meta.shape),
            "vocab": int(len(vocab)),
        },
        "files": {
            "Xdense_csv": X_dense_path,
            "Xtfidf_npz": X_tfidf_path,
            "y_csv": y_path,
            "meta_csv": meta_path,
            "vocab_txt": vocab_path,
        },
    }
    manifest_path = os.path.join(out_root, f"{prefix}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8-sig") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[DUMP] training set saved under: {out_root}")
    print(f"[DUMP] prefix = {prefix}")



def main():
    args = parse_args()

    # 1) 從 DB 取得資料 & 特徵
    X_dense_df, X_tfidf, y, meta, vocab = load_training_set(
        top_n=args.top_n,
        pipeline_version=args.pipeline_version,
        date_cutoff=args.date_cutoff,
        vocab_scope="global",
    )

    output_results(X_dense_df, X_tfidf, y, meta, vocab, args)

    # 2) Dense 特徵標準化（SVM/線性模型較受益；樹模型影響小）
    scaler = StandardScaler(with_mean=False)
    X_dense_scaled = scaler.fit_transform(X_dense_df.values)   # ndarray

    # 2.1 轉成稀疏格式，才能跟 TF-IDF 用 hstack
    X_dense_sparse = csr_matrix(X_dense_scaled)

    # 3) 合併稀疏特徵矩陣
    X_all = hstack([X_dense_sparse, X_tfidf], format="csr")

    train_mask = meta["is_train"].values
    X_train, y_train = X_all[train_mask], y[train_mask]
    X_test,  y_test  = X_all[~train_mask], y[~train_mask]

    print(f"[INFO] Train size: {X_train.shape}, Test size: {X_test.shape}")
    print(f"[INFO] Positive rate (train/test): {y_train.mean():.3f} / {y_test.mean():.3f}")
    
    # 新增：診斷資訊
    print(f"[INFO] Feature dimensions - Dense: {X_dense_df.shape[1]}, TF-IDF: {X_tfidf.shape[1]}")
    print(f"[INFO] Total features: {X_all.shape[1]}")
    print(f"[INFO] Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
    
    # 檢查特徵是否有變化
    print(f"[INFO] Dense features range: [{X_dense_df.min().min():.3f}, {X_dense_df.max().max():.3f}]")
    print(f"[INFO] TF-IDF features sparsity: {1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]):.3f}")

    # 4) 兩種特徵選取：無 / LGBM 重要度
    fs_settings = [
        ("no_fs", None),
        ("lgbm_fs", args.fs_topk),
    ]

    # 5) 準備三種模型：SVM（GPU/CPU）、LightGBM（GPU，如無則略過）、XGBoost（GPU）
    models = []

    # 5.1 SVM
    # if HAS_CUML:
    #     models.append(("SVM-GPU(cuML)", cuSVC(probability=True)))
    # else:
    #     # sklearn.SVC 預設 RBF kernel 需要 dense；預測時我會自動 densify
    #     models.append(("SVM-CPU(sklearn)", skSVC(probability=True, kernel="rbf")))

    # # 5.2 LightGBM
    # if HAS_LGBM:
    #     models.append((
    #         "LightGBM",
    #         LGBMClassifier(
    #             device="gpu",            # 無 GPU 會自動 fallback
    #             n_estimators=800,
    #             learning_rate=0.05,
    #             num_leaves=127,
    #             subsample=0.8,
    #             colsample_bytree=0.8,
    #             random_state=42
    #         )
    #     ))
    # else:
    #     warnings.warn("略過 LightGBM。")

    # 5.3 XGBoost
    models.append((
        "XGBoost",
        xgb.XGBClassifier(
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            n_estimators=800,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42
        )
    ))

    results = []

    for fs_name, topk in fs_settings:
        if fs_name == "lgbm_fs":
            print(f"[FS] Using LGBM feature selection top_k={topk}")
            if not HAS_LGBM:
                print("[FS] LightGBM not available, skipping feature selection")
                X_tr_fs, X_te_fs = X_train, X_test
                fs_meta = {"selected_idx_len": None}
            else:
                X_tr_fs, X_te_fs, selected_idx = select_features_by_lgbm(
                    X_train, y_train, X_test, top_k=topk
                )
                fs_meta = {"selected_idx_len": int(len(selected_idx))}
        else:
            X_tr_fs, X_te_fs = X_train, X_test
            fs_meta = {"selected_idx_len": None}

        for model_name, model in tqdm(models, desc="Training models", unit="model"):
            print(f"[TRAIN] {model_name} | {fs_name}")

            # cuML 與 sklearn.SVC（RBF）都偏好 dense
            needs_dense = (
                (HAS_CUML and cuSVC is not None and isinstance(model, cuSVC)) or
                isinstance(model, skSVC)
            )
            if needs_dense:
                Xtr = X_tr_fs.toarray()
                Xte = X_te_fs.toarray()
            else:
                Xtr, Xte = X_tr_fs, X_te_fs

            model.fit(Xtr, y_train)
            y_pred = model.predict(Xte)
            y_score = get_positive_score(model, Xte)
            metrics = evaluate(y_test, y_pred, y_score)

            row = {
                "fs_method": fs_name,
                "model": model_name,
                "top_n_tfidf": int(args.top_n),
                "date_cutoff": args.date_cutoff,
                "pipeline_version": args.pipeline_version,
                **metrics,
                **fs_meta
            }
            print(f"[RESULT] {row}")
            results.append(row)

    df_results = pd.DataFrame(results).sort_values(["fs_method", "model"])
    base = args.out
    df_results.to_csv(f"{base}.csv", index=False, encoding="utf-8-sig")
    with open(f"{base}.json", "w", encoding="utf-8-sig") as f:
        json.dump(df_results.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print(f"[DONE] saved -> {base}.csv / {base}.json")


if __name__ == "__main__":
    main()