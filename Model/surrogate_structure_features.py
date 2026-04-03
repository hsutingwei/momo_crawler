"""
Model/surrogate_structure_features.py
=======================================
v2 Surrogate Structure-Derived Feature Engineering

根據對 v2_error_prod 的 surrogate analysis（SHAP interaction + clustering），
產生 10 個新的商品層級特徵，幫助模型識別「高訊號但高異質性 / 難學樣本」。

此模組供 data_loader.py 的 `load_product_level_training_set()` 尾端呼叫。
直接接受已計算好的 df（含所有 raw features），在上面新增欄位並回傳。

新增的 10 個特徵：
  A. Interaction binary flags（二元旗標）
     1. is_high_arousal_high_volume
     2. is_high_arousal_high_negative
     3. is_high_ncd_high_fit
  B. Interaction continuous scores（連續分數）
     4. interaction_arousal_volume
     5. interaction_arousal_negative
     6. interaction_ncd_fit
  C. Conflict score（綜合衝突分數）
     7. conflict_signal_score
  D. Cluster-derived features（由 surrogate clustering 轉出）
     8. v2_surrogate_cluster_id      (從 KMeans 最近中心分配)
     9. is_cluster_like_high_complexity
    10. is_cluster_like_emotion_ad_mix

分位數計算規則：
  - 所有 P80 均基於傳入 df 的非 null 母體計算（date_cutoff 當下的全體商品）
  - null 欄位的 binary flag 預設為 0，continuous score 預設為 0
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ==========================================================
# 所有新特徵的名稱（供外部引用）
# ==========================================================
SURROGATE_FEATURE_COLS = [
    # A. Binary flags
    "is_high_arousal_high_volume",
    "is_high_arousal_high_negative",
    "is_high_ncd_high_fit",
    # B. Continuous scores
    "interaction_arousal_volume",
    "interaction_arousal_negative",
    "interaction_ncd_fit",
    # C. Conflict score
    "conflict_signal_score",
    # D. Cluster-derived
    "v2_surrogate_cluster_id",
    "is_cluster_like_high_complexity",
    "is_cluster_like_emotion_ad_mix",
]

# 用於 cluster assignment 的 features（若欄位缺失，該欄位補 0）
CLUSTER_FEATURES = [
    "feat_entropy_emb",
    "feat_ncd_spam",
    "bert_advertisement_mean",
    "bert_negative_mean",
    "quality_driven_momentum",
    "clean_arousal_score",
    "comment_count_90d",
    "kin_acc_abs",
    "category_fit_score",
]

# Cluster centers 從 v2_error_prod structure analysis 中得出的近似型態
# 若無法計算，直接用全資料 KMeans 動態學習
_N_CLUSTERS = 4


def _safe_quantile(series: pd.Series, q: float, fallback: float = 0.0) -> float:
    """安全計算分位數（去除 null 後）。若無有效值，回傳 fallback。"""
    valid = series.dropna()
    if len(valid) == 0:
        return fallback
    return float(valid.quantile(q))


def _minmax_norm(series: pd.Series) -> pd.Series:
    """Min-max 正規化，處理常數序列時回傳全 0。"""
    s = series.fillna(0.0)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-10:
        return pd.Series(0.0, index=series.index)
    return (s - lo) / (hi - lo)


def build_surrogate_structure_features(
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    在 df 上新增 10 個 surrogate structure-derived features，並回傳 df。

    Parameters
    ----------
    df : pd.DataFrame
        已含 raw product-level features 的 DataFrame（data_loader 內部的 df）。
    verbose : bool
        若 True，印出每個新特徵的 summary。

    Returns
    -------
    pd.DataFrame
        同一個 df，多了 SURROGATE_FEATURE_COLS 中的所有欄位。
    """
    if verbose:
        print("\n" + "="*60)
        print("🧬 v2 Surrogate Structure-Derived Feature Engineering")
        print("="*60)

    # --------------------------------------------------------
    # 計算所有需要的 P80 門檻
    # --------------------------------------------------------
    def p80(col: str) -> float:
        if col not in df.columns:
            return 0.0
        return _safe_quantile(df[col], 0.80)

    thresholds = {
        "arousal_p80": p80("clean_arousal_score"),
        "volume_p80":  p80("comment_count_90d"),
        "neg_p80":     p80("bert_negative_mean"),
        "ncd_p80":     p80("feat_ncd_spam"),
        "fit_p80":     p80("category_fit_score"),
        "entropy_p80": p80("feat_entropy_emb"),
        "qdm_p80":     p80("quality_driven_momentum"),
        "ad_p80":      p80("bert_advertisement_mean"),
    }

    if verbose:
        print("  Thresholds (P80):")
        for k, v in thresholds.items():
            print(f"    {k} = {v:.4f}")

    def col(name: str) -> pd.Series:
        """取欄位，若不存在回傳全 0 Series。"""
        if name in df.columns:
            return df[name].fillna(0.0)
        return pd.Series(0.0, index=df.index)

    def flag(condition: pd.Series) -> pd.Series:
        """轉成 int 0/1 旗標。"""
        return condition.astype(int)

    # =========================================================
    # A. Interaction Binary Flags
    # =========================================================

    # 1. is_high_arousal_high_volume
    df["is_high_arousal_high_volume"] = flag(
        (col("clean_arousal_score") >= thresholds["arousal_p80"])
        & (col("comment_count_90d") >= thresholds["volume_p80"])
    )

    # 2. is_high_arousal_high_negative
    df["is_high_arousal_high_negative"] = flag(
        (col("clean_arousal_score") >= thresholds["arousal_p80"])
        & (col("bert_negative_mean") >= thresholds["neg_p80"])
    )

    # 3. is_high_ncd_high_fit
    df["is_high_ncd_high_fit"] = flag(
        (col("feat_ncd_spam") >= thresholds["ncd_p80"])
        & (col("category_fit_score") >= thresholds["fit_p80"])
    )

    # =========================================================
    # B. Interaction Continuous Scores
    # =========================================================

    # 4. interaction_arousal_volume
    df["interaction_arousal_volume"] = (
        col("clean_arousal_score") * np.log1p(col("comment_count_90d"))
    )

    # 5. interaction_arousal_negative
    df["interaction_arousal_negative"] = (
        col("clean_arousal_score") * col("bert_negative_mean")
    )

    # 6. interaction_ncd_fit
    df["interaction_ncd_fit"] = (
        col("feat_ncd_spam") * col("category_fit_score")
    )

    # =========================================================
    # C. Conflict Signal Score（綜合衝突分數）
    # 加權公式（依 SHAP interaction 強度排序）
    # =========================================================

    # 7. conflict_signal_score
    #    0.35 × norm(clean_arousal_score)
    #  + 0.25 × norm(log1p(comment_count_90d))
    #  + 0.20 × norm(bert_negative_mean)
    #  + 0.20 × norm(feat_entropy_emb)
    df["conflict_signal_score"] = (
        0.35 * _minmax_norm(col("clean_arousal_score"))
        + 0.25 * _minmax_norm(np.log1p(col("comment_count_90d")))
        + 0.20 * _minmax_norm(col("bert_negative_mean"))
        + 0.20 * _minmax_norm(col("feat_entropy_emb"))
    )

    # =========================================================
    # D. Cluster-Derived Features
    # =========================================================

    # --- 準備 cluster input features ---
    cluster_input_cols = [c for c in CLUSTER_FEATURES if c in df.columns]
    X_cluster = df[cluster_input_cols].fillna(0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # 8. v2_surrogate_cluster_id（動態 KMeans，k=4）
    km = KMeans(n_clusters=_N_CLUSTERS, random_state=42, n_init=10)
    df["v2_surrogate_cluster_id"] = km.fit_predict(X_scaled)

    # 9. is_cluster_like_high_complexity
    #    近似 cluster 中的「高複雜度型」：高熵 + 高 NCD + 高聲量 + 高 QDM
    df["is_cluster_like_high_complexity"] = flag(
        (col("feat_entropy_emb") >= thresholds["entropy_p80"])
        & (col("feat_ncd_spam") >= thresholds["ncd_p80"])
        & (col("comment_count_90d") >= thresholds["volume_p80"])
        & (col("quality_driven_momentum") >= thresholds["qdm_p80"])
    )

    # 10. is_cluster_like_emotion_ad_mix
    #     近似 cluster 中的「情緒業配混合型」：高業配 + 高負面 + 高驚豔
    df["is_cluster_like_emotion_ad_mix"] = flag(
        (col("bert_advertisement_mean") >= thresholds["ad_p80"])
        & (col("bert_negative_mean") >= thresholds["neg_p80"])
        & (col("clean_arousal_score") >= thresholds["arousal_p80"])
    )

    # =========================================================
    # Summary
    # =========================================================
    if verbose:
        print("\n  New Feature Summary:")
        print(f"  {'Feature':<40} {'mean':>8} {'std':>8} {'nonzero%':>10}")
        print("  " + "-"*68)
        for feat in SURROGATE_FEATURE_COLS:
            s = df[feat]
            nonzero_pct = (s != 0).mean() * 100
            print(f"  {feat:<40} {s.mean():>8.4f} {s.std():>8.4f} {nonzero_pct:>9.1f}%")
        print("="*60)

    return df
