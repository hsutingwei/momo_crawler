#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model/discover_v2_error_prod_structure.py
==========================================
v2_error_prod Structure Discovery & Rule Extraction

目的：
  回答 v2_error_prod（z=1）「由哪幾種結構 / 型態」組成，
  並將其轉換為可寫論文的簡單規則。

Steps:
  0. 建立 z 標籤
  1. XGBoost surrogate (max_depth=4, n=100)
  2. SHAP interaction analysis → interaction_summary.csv
  3. 2D heatmap 可視化（Top 3 interaction pairs）
  4. 只對 z=1 做 clustering（PCA + KMeans）
  5. cluster 描述 → cluster_profile.csv
  6. 萃取 2~4 條簡單規則 → extracted_rules.md
  7. 規則評估 → rule_evaluation.csv

Usage:
  python Model/discover_v2_error_prod_structure.py --date-cutoff 2025-06-25
  python Model/discover_v2_error_prod_structure.py --date-cutoff 2025-06-25 --n-clusters 4
"""

import argparse
import os
import sys
import warnings
from datetime import datetime
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.database import DatabaseConfig
from Model.data_loader import load_product_level_training_set

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "analysis_outputs")

FEATURES = [
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


# --------------------------------------------------------------------------
# Step 0: Data + z label
# --------------------------------------------------------------------------

def load_data_with_z(date_cutoff: str) -> pd.DataFrame:
    print(f"\n{'='*70}")
    print(f"📊 Step 0: 載入資料 & 建立 z 標籤 (date_cutoff={date_cutoff})")
    print(f"{'='*70}")

    X_dense_df, _, y, meta, _ = load_product_level_training_set(
        date_cutoff=date_cutoff,
        label_strategy="absolute",
        label_delta_threshold=10,
        skip_tfidf_matrix=True,
    )
    df = X_dense_df.copy()
    df["product_id"] = meta["product_id"].values
    df["y"] = y.values

    # Load v2_error_prod ids
    db = DatabaseConfig()
    conn = db.get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT filter_value FROM ml_data_filters
        WHERE version_tag = 'v2_error_prod' AND filter_level = 'product_id';
    """)
    v2_ids = set(int(r[0]) for r in cur.fetchall())
    cur.close(); conn.close()

    df["z"] = df["product_id"].astype(int).isin(v2_ids).astype(int)
    avail = [f for f in FEATURES if f in df.columns]
    missing = set(FEATURES) - set(avail)

    print(f"  ✅ {len(df)} products  | z=1: {df['z'].sum()}, z=0: {(df['z']==0).sum()}")
    if missing:
        print(f"  ⚠️  Missing features (skipped): {missing}")
    return df, avail


# --------------------------------------------------------------------------
# Step 1: XGBoost surrogate
# --------------------------------------------------------------------------

def train_surrogate(df: pd.DataFrame, features: list):
    print(f"\n{'='*70}")
    print(f"🤖 Step 1: XGBoost surrogate (max_depth=4, n=100)")
    X = df[features].fillna(df[features].median())
    z = df["z"].values
    scale_pos = (z == 0).sum() / max((z == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos,
        random_state=42,
        verbosity=0,
        eval_metric="logloss",
    )
    model.fit(X, z)
    print(f"  ✅ Surrogate trained. scale_pos_weight={scale_pos:.2f}")
    return model, X


# --------------------------------------------------------------------------
# Step 2: SHAP interaction analysis
# --------------------------------------------------------------------------

def shap_interaction_analysis(model, X: pd.DataFrame, features: list, output_dir: str):
    print(f"\n{'='*70}")
    print(f"🔬 Step 2: SHAP interaction analysis")

    explainer = shap.TreeExplainer(model)
    print("  Computing SHAP interaction values (may take a minute)...")
    interaction_vals = explainer.shap_interaction_values(X)  # shape: (n, p, p)

    n, p, _ = interaction_vals.shape

    # Compute mean |interaction| for all pairs
    pair_scores = []
    for i, j in combinations(range(len(features)), 2):
        # interactions are symmetric; off-diagonal terms × 2
        score = np.mean(np.abs(interaction_vals[:, i, j])) * 2
        pair_scores.append({
            "feature_A": features[i],
            "feature_B": features[j],
            "mean_abs_interaction": round(score, 6),
        })

    interaction_df = (
        pd.DataFrame(pair_scores)
        .sort_values("mean_abs_interaction", ascending=False)
        .reset_index(drop=True)
    )

    out_path = os.path.join(output_dir, "interaction_summary.csv")
    interaction_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ Saved: {out_path}")
    print(f"\n  Top 10 interaction pairs:")
    print(interaction_df.head(10).to_string(index=False))

    return interaction_vals, interaction_df


# --------------------------------------------------------------------------
# Step 3: 2D heatmap visualization
# --------------------------------------------------------------------------

def plot_heatmap(df: pd.DataFrame, feat_a: str, feat_b: str, z_col: str,
                 output_dir: str, pair_idx: int):
    """Draw z=1 density heatmap for a pair of features."""
    valid = df[[feat_a, feat_b, z_col]].dropna()
    if len(valid) < 20:
        return

    qa = np.linspace(0, 100, 11)
    bins_a = np.percentile(valid[feat_a], qa)
    bins_b = np.percentile(valid[feat_b], qa)

    # Deduplicate bin edges
    bins_a = np.unique(bins_a)
    bins_b = np.unique(bins_b)
    if len(bins_a) < 3 or len(bins_b) < 3:
        return

    # Bin both features
    valid = valid.copy()
    valid["bin_a"] = pd.cut(valid[feat_a], bins=bins_a, include_lowest=True)
    valid["bin_b"] = pd.cut(valid[feat_b], bins=bins_b, include_lowest=True)

    # Pivot: mean z=1 per cell
    pivot = valid.groupby(["bin_a", "bin_b"])[z_col].mean().unstack(fill_value=np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                   cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_title(f"z=1 density: {feat_a} × {feat_b}", fontsize=11)
    ax.set_xlabel(feat_b)
    ax.set_ylabel(feat_a)

    # Label ticks with bin midpoints
    def mid_labels(bins_arr):
        mids = [(bins_arr[i] + bins_arr[i+1]) / 2 for i in range(len(bins_arr)-1)]
        return [f"{m:.2f}" for m in mids]

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(mid_labels(bins_b), rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(mid_labels(bins_a), fontsize=7)

    plt.colorbar(im, ax=ax, label="z=1 rate")
    plt.tight_layout()

    fname = os.path.join(output_dir, f"heatmap_{pair_idx+1}_{feat_a}_x_{feat_b}.png")
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  📊 Saved heatmap: {fname}")


def run_visualizations(df: pd.DataFrame, interaction_df: pd.DataFrame, output_dir: str):
    print(f"\n{'='*70}")
    print(f"🗺️  Step 3: 2D heatmap visualization (Top 3 interaction pairs)")
    for i, row in interaction_df.head(3).iterrows():
        fa, fb = row["feature_A"], row["feature_B"]
        if fa in df.columns and fb in df.columns:
            plot_heatmap(df, fa, fb, "z", output_dir, i)


# --------------------------------------------------------------------------
# Step 4 & 5: Clustering z=1 samples
# --------------------------------------------------------------------------

def cluster_z1(df: pd.DataFrame, features: list, n_clusters: int, output_dir: str):
    print(f"\n{'='*70}")
    print(f"🔵 Step 4 & 5: Clustering z=1 samples (n_clusters={n_clusters})")

    z1 = df[df["z"] == 1][features + ["product_id"]].copy()
    z1_feats = z1[features].fillna(z1[features].median())

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(z1_feats)

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    z1 = z1.copy()
    z1["cluster"] = labels

    # PCA scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(n_clusters):
        idx = labels == c
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1], s=20, alpha=0.6, label=f"Cluster {c}")
    ax.set_title(f"z=1 Clustering (PCA 2D, k={n_clusters})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"cluster_pca_k{n_clusters}.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  📊 Saved PCA cluster plot: {plot_path}")

    # Cluster profiles
    all_z1_means = z1_feats.mean()
    all_z1_stds  = z1_feats.std()

    profile_rows = []
    total_z1 = len(z1)

    for c in range(n_clusters):
        subset = z1[z1["cluster"] == c][features].fillna(z1[features].median())
        row = {"cluster": c, "size": len(subset),
               "pct_of_z1": round(len(subset) / total_z1 * 100, 1)}
        for feat in features:
            row[f"{feat}_mean"]   = round(subset[feat].mean(), 4)
            row[f"{feat}_median"] = round(subset[feat].median(), 4)
            # z-score vs all z=1
            if all_z1_stds[feat] > 0:
                row[f"{feat}_zscore"] = round(
                    (subset[feat].mean() - all_z1_means[feat]) / all_z1_stds[feat], 2
                )
            else:
                row[f"{feat}_zscore"] = 0.0
        profile_rows.append(row)

    profile_df = pd.DataFrame(profile_rows)
    profile_path = os.path.join(output_dir, "cluster_profile.csv")
    profile_df.to_csv(profile_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ Saved: {profile_path}")

    # Print cluster descriptions
    print(f"\n  {'='*50}")
    print(f"  Cluster Profiles:")
    for c in range(n_clusters):
        subset_z = profile_df[profile_df["cluster"] == c].iloc[0]
        zscore_cols = [f for f in features]
        high = [f for f in zscore_cols if subset_z.get(f"{f}_zscore", 0) >= 0.5]
        low  = [f for f in zscore_cols if subset_z.get(f"{f}_zscore", 0) <= -0.5]
        print(f"\n  Cluster {c} (n={int(subset_z['size'])}, {subset_z['pct_of_z1']}% of z=1):")
        if high:
            print(f"    📈 HIGH: {', '.join(high)}")
        if low:
            print(f"    📉 LOW:  {', '.join(low)}")

    return z1, profile_df


# --------------------------------------------------------------------------
# Step 6: Extract rules
# --------------------------------------------------------------------------

def extract_rules(df: pd.DataFrame, features: list, interaction_df: pd.DataFrame,
                  z1_clusters: pd.DataFrame, profile_df: pd.DataFrame) -> list:
    """
    Generate candidate rules based on:
      - Top SHAP interaction pairs
      - Cluster characteristic features (high z-score)
    Uses actual P-quantile thresholds from full dataset.
    """
    def pct(col, q):
        if col in df.columns:
            return df[col].quantile(q)
        return None

    rules = []

    # --- Rule A: Top interaction pair (usually comment_count × entropy)
    top_pair = interaction_df.iloc[0]
    fa, fb = top_pair["feature_A"], top_pair["feature_B"]
    thresh_fa = pct(fa, 0.80)
    thresh_fb_hi = pct(fb, 0.80)
    thresh_fb_lo = pct(fb, 0.20)

    # Decide direction: check which direction has higher z=1 rate
    if fa in df.columns and fb in df.columns:
        mask_hi_both = (df[fa] >= thresh_fa) & (df[fb] >= thresh_fb_hi)
        mask_hi_lo   = (df[fa] >= thresh_fa) & (df[fb] <= thresh_fb_lo)
        rate_hi = df.loc[mask_hi_both, "z"].mean() if mask_hi_both.sum() > 0 else 0
        rate_lo = df.loc[mask_hi_lo,   "z"].mean() if mask_hi_lo.sum()   > 0 else 0
    else:
        rate_hi, rate_lo = 0, 0

    if rate_hi >= rate_lo:
        rules.append({
            "rule_id": "rule_1",
            "conditions": [
                f"{fa} >= P80  [{thresh_fa:.4f}]",
                f"{fb} >= P80  [{thresh_fb_hi:.4f}]",
            ],
            "description": f"高 {fa} + 高 {fb}（SHAP interaction 最強配對）",
            "capture_type": "高訊號但語意/聲量複雜度同時偏高的樣本",
        })
    else:
        rules.append({
            "rule_id": "rule_1",
            "conditions": [
                f"{fa} >= P80  [{thresh_fa:.4f}]",
                f"{fb} <= P20  [{thresh_fb_lo:.4f}]",
            ],
            "description": f"高 {fa} + 低 {fb}（SHAP interaction 最強配對，反向）",
            "capture_type": f"高 {fa} 但低 {fb} 的衝突型樣本",
        })

    # --- Rule B: 2nd interaction pair
    if len(interaction_df) >= 2:
        row2 = interaction_df.iloc[1]
        fa2, fb2 = row2["feature_A"], row2["feature_B"]
        t_fa2 = pct(fa2, 0.80)
        t_fb2 = pct(fb2, 0.80)
        t_fb2_lo = pct(fb2, 0.25)
        if fa2 in df.columns and fb2 in df.columns:
            m_hi = (df[fa2] >= t_fa2) & (df[fb2] >= t_fb2)
            m_lo = (df[fa2] >= t_fa2) & (df[fb2] <= t_fb2_lo)
            r_hi = df.loc[m_hi, "z"].mean() if m_hi.sum() > 0 else 0
            r_lo = df.loc[m_lo, "z"].mean() if m_lo.sum() > 0 else 0
            if r_hi >= r_lo:
                rules.append({
                    "rule_id": "rule_2",
                    "conditions": [
                        f"{fa2} >= P80  [{t_fa2:.4f}]",
                        f"{fb2} >= P80  [{t_fb2:.4f}]",
                    ],
                    "description": f"高 {fa2} + 高 {fb2}（SHAP 第二強配對）",
                    "capture_type": "情緒衝突型或多重高訊號樣本",
                })
            else:
                rules.append({
                    "rule_id": "rule_2",
                    "conditions": [
                        f"{fa2} >= P80  [{t_fa2:.4f}]",
                        f"{fb2} <= P25  [{t_fb2_lo:.4f}]",
                    ],
                    "description": f"高 {fa2} + 低 {fb2}（SHAP 第二強配對，反向）",
                    "capture_type": "高業配或高複雜度但低品質動量的樣本",
                })

    # --- Rule C: cluster-driven rule (most extreme cluster)
    # Find cluster with highest avg zscore
    z_cols = [f"{f}_zscore" for f in features if f"{f}_zscore" in profile_df.columns]
    if z_cols:
        profile_df["max_abs_z"] = profile_df[z_cols].abs().max(axis=1)
        top_cluster = profile_df.sort_values("max_abs_z", ascending=False).iloc[0]
        c_id = int(top_cluster["cluster"])

        # Get the 2 most extreme features of this cluster
        zscores = {f: top_cluster.get(f"{f}_zscore", 0) for f in features}
        sorted_by_abs = sorted(zscores.items(), key=lambda x: abs(x[1]), reverse=True)
        extreme = sorted_by_abs[:2]

        conds = []
        for feat, z_val in extreme:
            q = 0.80 if z_val > 0 else 0.25
            label = "P80" if z_val > 0 else "P25"
            op    = ">=" if z_val > 0 else "<="
            thresh = pct(feat, q)
            if thresh is not None:
                conds.append(f"{feat} {op} {label}  [{thresh:.4f}]")

        if conds:
            rules.append({
                "rule_id": "rule_3",
                "conditions": conds,
                "description": f"Cluster {c_id} 最具特徵的組合規則（最極端 cluster）",
                "capture_type": "由 cluster 分析萃取，捕捉 z=1 分布最集中的一型",
            })

    # --- Rule D: arousal + negative (emotion conflict) – fixed useful rule
    t_ar = pct("clean_arousal_score", 0.80)
    t_ne = pct("bert_negative_mean", 0.80)
    if t_ar is not None and t_ne is not None:
        rules.append({
            "rule_id": "rule_4",
            "conditions": [
                f"clean_arousal_score >= P80  [{t_ar:.4f}]",
                f"bert_negative_mean >= P80   [{t_ne:.4f}]",
            ],
            "description": "高純淨驚豔 + 高負面情緒（情緒衝突型）",
            "capture_type": "評論同時具有高情緒強度且方向互相矛盾的樣本",
        })

    return rules


# --------------------------------------------------------------------------
# Step 7: Evaluate rules
# --------------------------------------------------------------------------

def evaluate_rules(df: pd.DataFrame, rules: list) -> pd.DataFrame:
    print(f"\n{'='*70}")
    print(f"📏 Step 7: Rule evaluation")
    total = len(df)
    total_z1 = (df["z"] == 1).sum()
    eval_rows = []

    def parse_condition(cond_str: str, df: pd.DataFrame):
        """Parse 'feat >= P80  [0.1234]' style condition into a boolean mask."""
        import re
        # Extract: feature_name, operator, threshold_value
        m = re.match(r"^(\S+)\s*(>=|<=|>|<)\s*\S+\s*\[([^\]]+)\]", cond_str.strip())
        if not m:
            return pd.Series([True] * len(df), index=df.index)
        feat, op, val = m.group(1), m.group(2), float(m.group(3))
        if feat not in df.columns:
            return pd.Series([True] * len(df), index=df.index)
        col = df[feat].fillna(df[feat].median())
        if op == ">=":
            return col >= val
        elif op == "<=":
            return col <= val
        elif op == ">":
            return col > val
        elif op == "<":
            return col < val
        return pd.Series([True] * len(df), index=df.index)

    for rule in rules:
        mask = pd.Series([True] * len(df), index=df.index)
        for cond in rule["conditions"]:
            mask = mask & parse_condition(cond, df)

        n_covered = mask.sum()
        n_z1_in_rule = df.loc[mask, "z"].sum()
        coverage  = n_covered / total if total > 0 else 0
        precision = n_z1_in_rule / n_covered if n_covered > 0 else 0
        recall    = n_z1_in_rule / total_z1 if total_z1 > 0 else 0

        eval_rows.append({
            "rule_id": rule["rule_id"],
            "description": rule["description"],
            "n_covered": int(n_covered),
            "coverage_pct": round(coverage * 100, 2),
            "n_z1_in_rule": int(n_z1_in_rule),
            "precision_z1": round(precision, 4),
            "recall_z1": round(recall, 4),
        })
        print(f"  {rule['rule_id']}: coverage={coverage*100:.1f}%, "
              f"precision={precision:.3f}, recall={recall:.3f}")

    return pd.DataFrame(eval_rows)


# --------------------------------------------------------------------------
# Step 8: Write rules markdown
# --------------------------------------------------------------------------

def write_extracted_rules(rules: list, eval_df: pd.DataFrame, output_dir: str):
    lines = []
    lines.append("# Extracted Proxy Rules from v2_error_prod Structure Analysis")
    lines.append(f"\n> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"\n> **注意**: 這些規則是 surrogate structure discovery 的第一階段輸出，")
    lines.append("> 供研究者判斷是否值得進一步驗證。非最終定案規則。\n")
    lines.append("---\n")

    for rule in rules:
        rid = rule["rule_id"]
        row = eval_df[eval_df["rule_id"] == rid].iloc[0] if rid in eval_df["rule_id"].values else None
        lines.append(f"## {rid}")
        lines.append(f"\n**描述**: {rule['description']}")
        lines.append(f"\n**捕捉型態**: {rule['capture_type']}\n")
        lines.append("**條件（需同時滿足）**:")
        for c in rule["conditions"]:
            lines.append(f"- `{c}`")
        if row is not None:
            lines.append(f"\n**評估指標**:")
            lines.append(f"- Coverage: {row['coverage_pct']}% ({row['n_covered']} samples)")
            lines.append(f"- Precision (z=1 rate in rule): {row['precision_z1']:.4f}")
            lines.append(f"- Recall (% of v2_error_prod covered): {row['recall_z1']:.4f}")
        lines.append("\n---\n")

    out = os.path.join(output_dir, "extracted_rules.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  📝 Saved: {out}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="v2_error_prod Structure Discovery & Rule Extraction"
    )
    ap.add_argument("--date-cutoff", type=str, required=True)
    ap.add_argument("--n-clusters", type=int, default=4,
                    help="KMeans clusters for z=1 (default: 4)")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"🔬 v2_error_prod Structure Discovery")
    print(f"   date_cutoff = {args.date_cutoff}")
    print(f"   n_clusters  = {args.n_clusters}")
    print(f"{'='*70}")

    # Step 0
    df, avail_features = load_data_with_z(args.date_cutoff)

    # Step 1
    model, X_filled = train_surrogate(df, avail_features)

    # Step 2
    interaction_vals, interaction_df = shap_interaction_analysis(
        model, X_filled, avail_features, OUTPUT_DIR
    )

    # Step 3
    run_visualizations(df, interaction_df, OUTPUT_DIR)

    # Step 4 & 5
    z1_df, profile_df = cluster_z1(df, avail_features, args.n_clusters, OUTPUT_DIR)

    # Step 6
    print(f"\n{'='*70}")
    print(f"📋 Step 6: Rule extraction from cluster + interaction analysis")
    rules = extract_rules(df, avail_features, interaction_df, z1_df, profile_df)

    # Step 7
    eval_df = evaluate_rules(df, rules)
    eval_path = os.path.join(OUTPUT_DIR, "rule_evaluation.csv")
    eval_df.to_csv(eval_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ Saved: {eval_path}")

    # Step 8
    print(f"\n{'='*70}")
    print(f"💾 Step 8: Writing output files")
    write_extracted_rules(rules, eval_df, OUTPUT_DIR)

    print(f"\n{'='*70}")
    print("✅ 分析完成！產出檔案（Model/analysis_outputs/）：")
    print("  - interaction_summary.csv      ← SHAP interaction importance")
    print("  - cluster_profile.csv          ← z=1 cluster 特徵統計")
    print("  - extracted_rules.md           ← 可寫論文的萃取規則")
    print("  - rule_evaluation.csv          ← precision / recall / coverage")
    print("  - heatmap_*.png                ← Top 3 interaction 2D 可視化")
    print(f"  - cluster_pca_k{args.n_clusters}.png     ← z=1 clustering PCA 圖")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
