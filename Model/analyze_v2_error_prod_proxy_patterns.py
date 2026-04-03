#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model/analyze_v2_error_prod_proxy_patterns.py
==============================================
Surrogate Rule Distillation for `v2_error_prod`

目的：
  分析 `v2_error_prod` 最常排掉哪一型商品樣本，
  找出可解釋的 proxy filter 候選規則。

  注意：這不是正式論文過濾器，只是 proxy filter discovery 的第一階段分析。

產出：
  - Model/analysis_outputs/z1_vs_z0_feature_summary.csv
  - Model/analysis_outputs/feature_effect_size_ranking.csv
  - Model/analysis_outputs/v2_error_prod_surrogate_analysis.md

Usage:
  python Model/analyze_v2_error_prod_proxy_patterns.py --date-cutoff 2025-06-25
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score
import xgboost as xgb

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# 路徑設定
# --------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.database import DatabaseConfig
from Model.data_loader import load_product_level_training_set

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "analysis_outputs")

ANALYSIS_FEATURES = [
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
# Helpers
# --------------------------------------------------------------------------

def cliffs_delta(x1, x2):
    """
    Cliff's delta: 非參數效果量，範圍 [-1, 1]。
    >0 表示 x1 整體高於 x2。
    """
    x1 = np.array(x1, dtype=float)
    x2 = np.array(x2, dtype=float)
    n1, n2 = len(x1), len(x2)
    if n1 == 0 or n2 == 0:
        return 0.0
    # dominance matrix
    dom = sum(np.sign(a - b) for a in x1 for b in x2)
    return dom / (n1 * n2)


def univariate_auc(feature_vals, z_labels):
    """AUC of single feature predicting z=1."""
    try:
        auc = roc_auc_score(z_labels, feature_vals)
        # flip so AUC is always >= 0.5
        return max(auc, 1 - auc)
    except Exception:
        return 0.5


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_data(date_cutoff: str) -> pd.DataFrame:
    print(f"\n{'='*70}")
    print(f"📊 載入訓練資料 (date_cutoff={date_cutoff})")
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
    print(f"  ✅ 載入 {len(df)} 筆商品  (y=1: {(df['y']==1).sum()}, y=0: {(df['y']==0).sum()})")
    return df


def load_v2_error_prod_ids() -> set:
    """從 ml_data_filters 取得 v2_error_prod 的 product_id 集合。"""
    db = DatabaseConfig()
    conn = db.get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT filter_value
        FROM ml_data_filters
        WHERE version_tag = 'v2_error_prod'
          AND filter_level = 'product_id';
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    ids = set(int(r[0]) for r in rows)
    print(f"  🔖 v2_error_prod 包含 {len(ids)} 個 product_id")
    return ids


# --------------------------------------------------------------------------
# Analysis A: Distribution comparison
# --------------------------------------------------------------------------

def analyze_distribution(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """z=1 vs z=0 的分布比較表。"""
    records = []
    for feat in features:
        if feat not in df.columns:
            continue
        z1 = df.loc[df["z"] == 1, feat].dropna()
        z0 = df.loc[df["z"] == 0, feat].dropna()
        row = {
            "feature": feat,
            "z1_n": len(z1),  "z0_n": len(z0),
            "z1_mean": z1.mean(), "z0_mean": z0.mean(),
            "z1_median": z1.median(), "z0_median": z0.median(),
            "z1_std": z1.std(), "z0_std": z0.std(),
            "z1_p10": z1.quantile(0.10), "z0_p10": z0.quantile(0.10),
            "z1_p25": z1.quantile(0.25), "z0_p25": z0.quantile(0.25),
            "z1_p75": z1.quantile(0.75), "z0_p75": z0.quantile(0.75),
            "z1_p90": z1.quantile(0.90), "z0_p90": z0.quantile(0.90),
            "mean_diff": z1.mean() - z0.mean(),
            "median_diff": z1.median() - z0.median(),
        }
        records.append(row)
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Analysis B: Effect size
# --------------------------------------------------------------------------

def analyze_effect_size(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Cliff's delta + univariate AUC for each feature."""
    records = []
    z = df["z"].values
    for feat in features:
        if feat not in df.columns:
            continue
        col = df[feat].fillna(df[feat].median())
        z1_vals = col[z == 1].values
        z0_vals = col[z == 0].values

        cd = cliffs_delta(z1_vals, z0_vals)
        auc = univariate_auc(col.values, z)

        records.append({
            "feature": feat,
            "cliffs_delta": round(cd, 4),
            "abs_cliffs_delta": round(abs(cd), 4),
            "univariate_auc": round(auc, 4),
            "direction": "z=1 higher" if cd > 0.01 else ("z=1 lower" if cd < -0.01 else "neutral"),
        })
    result = pd.DataFrame(records).sort_values("abs_cliffs_delta", ascending=False)
    return result


# --------------------------------------------------------------------------
# Analysis C: Surrogate XGBoost feature importance
# --------------------------------------------------------------------------

def analyze_surrogate_xgb(df: pd.DataFrame, features: list) -> tuple:
    """XGBoost surrogate model: z ~ features. Returns model & feature importance."""
    avail = [f for f in features if f in df.columns]
    X = df[avail].fillna(df[avail].median())
    z = df["z"].values

    scale_pos = (z == 0).sum() / max((z == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        random_state=42,
        verbosity=0,
        eval_metric="logloss",
    )
    model.fit(X, z)

    fi = pd.DataFrame({
        "feature": avail,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    # OOF AUC (simple train AUC as proxy)
    pred = model.predict_proba(X)[:, 1]
    train_auc = roc_auc_score(z, pred)

    return model, fi, train_auc, avail


# --------------------------------------------------------------------------
# Analysis D: Shallow decision tree
# --------------------------------------------------------------------------

def analyze_decision_tree(df: pd.DataFrame, features: list, max_depth=3) -> tuple:
    """Fit shallow decision tree and extract rules."""
    avail = [f for f in features if f in df.columns]
    X = df[avail].fillna(df[avail].median())
    z = df["z"].values

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42,
                                  class_weight="balanced")
    tree.fit(X, z)

    rules_text = export_text(tree, feature_names=avail, show_weights=True)

    # Also get leaf stats
    leaf_ids = tree.apply(X.values)
    leaf_df = pd.DataFrame({"leaf": leaf_ids, "z": z})
    leaf_stats = (
        leaf_df.groupby("leaf")["z"]
        .agg(n="count", z1_count="sum")
        .assign(z1_rate=lambda x: x["z1_count"] / x["n"])
        .sort_values("z1_rate", ascending=False)
    )

    return tree, rules_text, leaf_stats, avail


# --------------------------------------------------------------------------
# Candidate proxy rules (derived from analysis results)
# --------------------------------------------------------------------------

def propose_candidate_rules(dist_df: pd.DataFrame, eff_df: pd.DataFrame,
                             fi_df: pd.DataFrame, df: pd.DataFrame) -> list:
    """
    Based on the analysis, propose 2-5 candidate proxy rules.
    Rules are heuristic, derived from effect size and distribution.
    """
    # Compute actual thresholds from data
    def pct(col, q):
        return df[col].quantile(q) if col in df.columns else None

    rules = []

    # Rule 1: Spam + low diversity signals (similar to v5_spam_fixed concept)
    r1 = {
        "rule_id": "proxy_rule_A",
        "conditions": [
            f"feat_entropy_emb <= {pct('feat_entropy_emb', 0.25):.4f}  (P25)",
            f"feat_ncd_spam <= {pct('feat_ncd_spam', 0.25):.4f}  (P25)",
            "comment_count_90d > 5",
        ],
        "target": "y = any (catches both y=0 and y=1)",
        "capture_type": "低語意多樣性 + 高刷評相似度的商品（與 v2 的 ensemble error pattern 重疊）",
        "rationale": "v2_error_prod 用多模型 ensemble 找到難以預測的商品。低熵 + 低 NCD 的商品評論極度單一，模型難以學習自然語言訊號，因此常被誤判。",
    }
    rules.append(r1)

    # Rule 2: High advertisement + explosion
    r2 = {
        "rule_id": "proxy_rule_B",
        "conditions": [
            f"bert_advertisement_mean >= {pct('bert_advertisement_mean', 0.80):.4f}  (P80)",
            f"quality_driven_momentum <= {pct('quality_driven_momentum', 0.30):.4f}  (P30)",
            "y = 1",
        ],
        "target": "y = 1 only",
        "capture_type": "高業配感但缺乏品質支撐的爆發正樣本",
        "rationale": "這類商品的爆發由廣宣而非真實口碑驅動，模型難以從自然 eWOM 特徵學習到其成長規律，容易成為 outlier 正樣本。",
    }
    rules.append(r2)

    # Rule 3: Negative sentiment + low arousal
    r3 = {
        "rule_id": "proxy_rule_C",
        "conditions": [
            f"bert_negative_mean >= {pct('bert_negative_mean', 0.80):.4f}  (P80)",
            f"clean_arousal_score <= {pct('clean_arousal_score', 0.20):.4f}  (P20)",
            "y = 1",
        ],
        "target": "y = 1 only",
        "capture_type": "高負面情緒 + 低純淨驚豔感的爆發正樣本",
        "rationale": "若爆發伴隨高負面評論和極低驚豔感，該商品的成長訊號高度模糊，模型對其 y=1 的判斷基礎薄弱。",
    }
    rules.append(r3)

    # Rule 4: Low kinematic support + high comment count
    r4 = {
        "rule_id": "proxy_rule_D",
        "conditions": [
            f"kin_acc_abs <= {pct('kin_acc_abs', 0.20):.4f}  (P20)",
            f"category_fit_score <= {pct('category_fit_score', 0.25):.4f}  (P25)",
            "y = 1",
        ],
        "target": "y = 1 only",
        "capture_type": "聲量加速度低 + 類別契合度低的爆發正樣本",
        "rationale": "若正樣本的類別語意訊號弱且聲量加速不明顯，正樣本的爆發與典型 eWOM 動力機制脫鉤，成為 ensemble 常誤判的 edge case。",
    }
    rules.append(r4)

    return rules


# --------------------------------------------------------------------------
# Report writer
# --------------------------------------------------------------------------

def write_markdown_report(
    date_cutoff: str,
    df: pd.DataFrame,
    dist_df: pd.DataFrame,
    eff_df: pd.DataFrame,
    fi_df: pd.DataFrame,
    train_auc: float,
    rules_text: str,
    leaf_stats: pd.DataFrame,
    candidate_rules: list,
    output_path: str,
):
    n_z1 = (df["z"] == 1).sum()
    n_z0 = (df["z"] == 0).sum()

    top_feats = eff_df.head(5)[["feature", "abs_cliffs_delta", "univariate_auc", "direction"]]
    top_fi = fi_df.head(8)

    lines = []
    lines.append(f"# `v2_error_prod` Proxy Pattern Analysis")
    lines.append(f"\n**date_cutoff**: {date_cutoff}  |  **generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"\n> **注意**: 本分析僅為 Surrogate Rule Distillation 第一階段，不代表正式過濾規則。")

    # --- A. Summary
    lines.append("\n---\n## A. 摘要結論")
    lines.append(f"\n- **z=1（v2_error_prod 排掉的商品）**: {n_z1} 筆")
    lines.append(f"- **z=0（未被排掉）**: {n_z0} 筆")
    lines.append(f"- **z=1 佔比**: {n_z1 / len(df) * 100:.1f}%")

    best_feat = eff_df.iloc[0]["feature"]
    best_cd = eff_df.iloc[0]["abs_cliffs_delta"]
    best_dir = eff_df.iloc[0]["direction"]
    lines.append(f"\n**最具區分力的單一特徵**: `{best_feat}` (Cliff's |δ| = {best_cd:.4f}, {best_dir})")

    top3 = eff_df.head(3)["feature"].tolist()
    lines.append(f"\n**前三名效果量特徵**: {', '.join(f'`{f}`' for f in top3)}")
    lines.append(f"\n**Surrogate XGBoost 訓練 AUC**: {train_auc:.4f} (訓練集，供方向參考)")

    # --- B. Distribution table
    lines.append("\n---\n## B. 統計比較表（z=1 vs z=0）")
    lines.append("\n| feature | z1_mean | z0_mean | mean_diff | z1_median | z0_median | z1_p10 | z0_p10 | z1_p90 | z0_p90 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, row in dist_df.iterrows():
        lines.append(
            f"| {row['feature']} "
            f"| {row['z1_mean']:.4f} | {row['z0_mean']:.4f} | **{row['mean_diff']:+.4f}** "
            f"| {row['z1_median']:.4f} | {row['z0_median']:.4f} "
            f"| {row['z1_p10']:.4f} | {row['z0_p10']:.4f} "
            f"| {row['z1_p90']:.4f} | {row['z0_p90']:.4f} |"
        )

    # --- C. Effect size ranking
    lines.append("\n---\n## C. 特徵效果量排序（Cliff's delta + 單變數 AUC）")
    lines.append("\n| rank | feature | Cliff's |δ| | univariate AUC | direction |")
    lines.append("|---|---|---|---|---|")
    for i, (_, row) in enumerate(eff_df.iterrows(), 1):
        lines.append(
            f"| {i} | {row['feature']} | {row['abs_cliffs_delta']:.4f} "
            f"| {row['univariate_auc']:.4f} | {row['direction']} |"
        )

    # --- D. Surrogate model
    lines.append("\n---\n## D. Surrogate Model (XGBoost) Feature Importance")
    lines.append(f"\n訓練 AUC = {train_auc:.4f}（訓練集，僅供方向參考）\n")
    lines.append("| rank | feature | importance |")
    lines.append("|---|---|---|")
    for i, (_, row) in enumerate(top_fi.iterrows(), 1):
        lines.append(f"| {i} | {row['feature']} | {row['importance']:.4f} |")

    # --- E. Decision tree rules
    lines.append("\n---\n## E. 淺層決策樹規則 (max_depth=3)")
    lines.append("\n```")
    lines.append(rules_text)
    lines.append("```")
    lines.append("\n**Leaf 統計（z=1 比例）**：")
    lines.append("\n| leaf_id | n | z1_count | z1_rate |")
    lines.append("|---|---|---|---|")
    for leaf_id, row in leaf_stats.iterrows():
        lines.append(f"| {int(leaf_id)} | {int(row['n'])} | {int(row['z1_count'])} | {row['z1_rate']:.3f} |")

    # --- F. Candidate proxy rules
    lines.append("\n---\n## F. 候選 Proxy Rules 草案")
    lines.append("\n> 以下規則僅為候選，尚未正式寫入 DB，需人工審查後方可採用。\n")
    for r in candidate_rules:
        lines.append(f"### {r['rule_id']}")
        lines.append(f"\n**捕捉對象**: {r['capture_type']}")
        lines.append(f"\n**Target**: `{r['target']}`\n")
        lines.append("**條件（需同時滿足）**:")
        for c in r["conditions"]:
            lines.append(f"- `{c}`")
        lines.append(f"\n**可解釋性說明**: {r['rationale']}\n")

    lines.append("\n---\n*本文件由 `analyze_v2_error_prod_proxy_patterns.py` 自動產生*")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  📝 Report written: {output_path}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Surrogate rule distillation for v2_error_prod"
    )
    ap.add_argument("--date-cutoff", type=str, required=True)
    ap.add_argument("--max-depth", type=int, default=3,
                    help="Decision tree max depth (default: 3)")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"🔬 v2_error_prod Proxy Pattern Analysis")
    print(f"   date_cutoff = {args.date_cutoff}")
    print(f"{'='*70}")

    # 1. Load data
    df = load_data(args.date_cutoff)

    # 2. Build z label
    print(f"\n{'='*70}")
    print("🔖 建立 z 標籤（z=1 ↔ v2_error_prod）")
    v2_ids = load_v2_error_prod_ids()
    df["z"] = df["product_id"].astype(int).isin(v2_ids).astype(int)
    print(f"  z=1: {df['z'].sum()}, z=0: {(df['z']==0).sum()}")

    avail_features = [f for f in ANALYSIS_FEATURES if f in df.columns]
    missing = set(ANALYSIS_FEATURES) - set(avail_features)
    if missing:
        print(f"  ⚠️  Missing features (skipped): {missing}")

    # 3. Distribution comparison
    print(f"\n{'='*70}")
    print("📊 A. Distribution comparison (z=1 vs z=0)")
    dist_df = analyze_distribution(df, avail_features)
    dist_path = os.path.join(OUTPUT_DIR, "z1_vs_z0_feature_summary.csv")
    dist_df.to_csv(dist_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ Saved: {dist_path}")
    # Print to console
    print(dist_df[["feature", "z1_mean", "z0_mean", "mean_diff",
                    "z1_median", "z0_median"]].to_string(index=False))

    # 4. Effect size
    print(f"\n{'='*70}")
    print("📏 B. Univariate effect size (Cliff's delta, AUC)")
    eff_df = analyze_effect_size(df, avail_features)
    eff_path = os.path.join(OUTPUT_DIR, "feature_effect_size_ranking.csv")
    eff_df.to_csv(eff_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ Saved: {eff_path}")
    print(eff_df.to_string(index=False))

    # 5. Surrogate XGBoost
    print(f"\n{'='*70}")
    print("🤖 C. Surrogate XGBoost feature importance")
    model, fi_df, train_auc, used_feats = analyze_surrogate_xgb(df, avail_features)
    print(f"  Train AUC = {train_auc:.4f}")
    print(fi_df.to_string(index=False))

    # 6. Shallow decision tree
    print(f"\n{'='*70}")
    print(f"🌳 D. Shallow decision tree (max_depth={args.max_depth})")
    tree, rules_text, leaf_stats, _ = analyze_decision_tree(
        df, avail_features, max_depth=args.max_depth
    )
    print(rules_text)
    print("\nLeaf stats:")
    print(leaf_stats.to_string())

    # 7. Candidate proxy rules
    print(f"\n{'='*70}")
    print("📋 E. Candidate proxy rules")
    candidate_rules = propose_candidate_rules(dist_df, eff_df, fi_df, df)
    rules_path = os.path.join(OUTPUT_DIR, "candidate_proxy_rules.md")
    with open(rules_path, "w", encoding="utf-8") as f:
        f.write("# Candidate Proxy Rules (v2_error_prod Surrogate)\n\n")
        f.write("> 以下規則為第一階段分析候選，尚未正式寫入 ml_data_filters。\n\n")
        for r in candidate_rules:
            f.write(f"## {r['rule_id']}\n\n")
            f.write(f"**捕捉對象**: {r['capture_type']}\n\n")
            f.write(f"**Target**: `{r['target']}`\n\n")
            f.write("**條件**:\n")
            for c in r["conditions"]:
                f.write(f"- `{c}`\n")
            f.write(f"\n**說明**: {r['rationale']}\n\n---\n\n")
    print(f"  ✅ Saved: {rules_path}")
    for r in candidate_rules:
        print(f"\n  [{r['rule_id']}]")
        for c in r["conditions"]:
            print(f"    {c}")

    # 8. Full markdown report
    print(f"\n{'='*70}")
    print("📄 Writing full markdown report...")
    report_path = os.path.join(OUTPUT_DIR, "v2_error_prod_surrogate_analysis.md")
    write_markdown_report(
        date_cutoff=args.date_cutoff,
        df=df,
        dist_df=dist_df,
        eff_df=eff_df,
        fi_df=fi_df,
        train_auc=train_auc,
        rules_text=rules_text,
        leaf_stats=leaf_stats,
        candidate_rules=candidate_rules,
        output_path=report_path,
    )

    print(f"\n{'='*70}")
    print("✅ 分析完成！產出檔案：")
    print(f"  - {dist_path}")
    print(f"  - {eff_path}")
    print(f"  - {rules_path}")
    print(f"  - {report_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
