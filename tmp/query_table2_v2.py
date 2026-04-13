# -*- coding: utf-8 -*-
"""
論文表 2 正確統計腳本 v2
修正項目：
1. 評論筆數 → 限定在觀測期間內 (comment_date 在 min～max 之間，且 <= cutoff T)
   注意：根據 各關鍵字時間範圍.md，評論觀測期間 = min(comment_date)～max(comment_date)
         但論文表 2 的「評論觀測期間」直接取自 DB 全部資料的 min/max comment_date
         此腳本的「評論筆數」即所有 comment_date 的總筆數（不限 cutoff 前），
         因為表 2 呈現的是「資料集整體概況」；模型特徵只用 cutoff 前資料另見說明。
2. 額外產出兩版：(A) 未過濾版  (B) 套用 v2_error_prod 過濾後版
3. 有效樣本數：有評論 AND 有截止日後銷售快照可計算標籤的商品數

定義釐清：
- 「商品數量」= products 表中各 keyword 的不重複商品數（爬蟲所有抓到的）
- 「評論筆數」= product_comments 中各 keyword 的評論筆數（不限 cutoff）
- 「評論觀測期間」= min(comment_date) ~ max(comment_date)（整體資料集範圍）
- 「爆品/非爆品」= 以 cutoff T=2025-06-25, absolute, delta>=10 計算的標籤
- 「有效樣本數」= 有截止後銷售快照能計算標籤的商品數（= 爆品數 + 非爆品數）
  ★ 套用 v2_error_prod 後：再從有效樣本中扣除被 v2_error_prod 排除的商品
"""

import sys
import os
sys.path.append(r'C:\YvesProject\中央\線上評論\momo_crawler-main')

from dotenv import load_dotenv
load_dotenv(r'C:\YvesProject\中央\線上評論\momo_crawler-main\.env')

from config.database import DatabaseConfig
import pandas as pd

def get_conn():
    db_config = DatabaseConfig()
    return db_config.get_connection()

DATE_CUTOFF = '2025-06-25'
DELTA_THRESHOLD = 10.0
FILTER_TAG = 'v2_error_prod'

# ── A. 各 keyword 商品數量 ──────────────────────────────────────────────
SQL_PRODUCT_COUNT = """
SELECT keyword, COUNT(DISTINCT id) AS product_count
FROM products
GROUP BY keyword
ORDER BY keyword;
"""

# ── B. 評論筆數 & 觀測期間（不限 cutoff，呈現全資料集範圍）──────────────
SQL_COMMENT_STATS = """
SELECT
    p.keyword,
    COUNT(DISTINCT pc.comment_id)               AS comment_count,
    MIN(pc.comment_date)::date                  AS min_comment_date,
    MAX(pc.comment_date)::date                  AS max_comment_date,
    ROUND(AVG(pc.score::float)::numeric, 2)     AS avg_score,
    ROUND(STDDEV(pc.score::float)::numeric, 2)  AS std_score
FROM product_comments pc
JOIN products p ON p.id = pc.product_id
GROUP BY p.keyword
ORDER BY p.keyword;
"""

# ── C. 爆品 / 非爆品（無過濾）──────────────────────────────────────────
SQL_LABEL = """
WITH comment_batches AS (
  SELECT p.keyword, pc.product_id, pc.capture_time AS batch_time
  FROM product_comments pc
  JOIN products p ON p.id = pc.product_id
  GROUP BY p.keyword, pc.product_id, pc.capture_time
),
snap_mapped AS (
  SELECT
    s.product_id, p.keyword,
    s.capture_time AS snapshot_time, s.sales_count,
    cb_near.batch_time
  FROM sales_snapshots s
  JOIN products p ON p.id = s.product_id
  LEFT JOIN LATERAL (
    SELECT cb.batch_time
    FROM comment_batches cb
    WHERE cb.keyword = p.keyword
      AND cb.product_id = s.product_id
      AND cb.batch_time <= s.capture_time
    ORDER BY cb.batch_time DESC LIMIT 1
  ) AS cb_near ON TRUE
),
batch_repr AS (
  SELECT * FROM (
    SELECT product_id, keyword, batch_time, sales_count, snapshot_time,
           ROW_NUMBER() OVER (PARTITION BY product_id, batch_time ORDER BY snapshot_time DESC) AS rn
    FROM snap_mapped WHERE batch_time IS NOT NULL
  ) t WHERE rn = 1
),
seq AS (
  SELECT product_id, keyword, batch_time, sales_count,
         LAG(sales_count) OVER (PARTITION BY product_id ORDER BY batch_time) AS prev_sales
  FROM batch_repr
),
y_post AS (
  SELECT product_id, keyword,
    MAX(CASE WHEN batch_time > %(cutoff)s::timestamp
                  AND prev_sales IS NOT NULL
                  AND (sales_count - prev_sales) >= %(delta)s
             THEN 1 ELSE 0 END) AS y
  FROM seq
  GROUP BY product_id, keyword
)
SELECT
    keyword,
    COUNT(*)                                              AS valid_sample_count,
    SUM(CASE WHEN y = 1 THEN 1 ELSE 0 END)               AS explosive_count,
    SUM(CASE WHEN y = 0 THEN 1 ELSE 0 END)               AS non_explosive_count,
    ROUND(100.0 * SUM(CASE WHEN y=1 THEN 1 ELSE 0 END) / COUNT(*), 1) AS explosive_pct
FROM y_post
GROUP BY keyword
ORDER BY keyword;
"""

# ── D. 爆品 / 非爆品（套用 v2_error_prod 過濾）───────────────────────────
SQL_LABEL_FILTERED = """
WITH excluded AS (
  SELECT filter_value::int AS product_id
  FROM ml_data_filters
  WHERE version_tag = %(filter_tag)s
    AND filter_level = 'product_id'
),
comment_batches AS (
  SELECT p.keyword, pc.product_id, pc.capture_time AS batch_time
  FROM product_comments pc
  JOIN products p ON p.id = pc.product_id
  WHERE pc.product_id NOT IN (SELECT product_id FROM excluded)
  GROUP BY p.keyword, pc.product_id, pc.capture_time
),
snap_mapped AS (
  SELECT
    s.product_id, p.keyword,
    s.capture_time AS snapshot_time, s.sales_count,
    cb_near.batch_time
  FROM sales_snapshots s
  JOIN products p ON p.id = s.product_id
  LEFT JOIN LATERAL (
    SELECT cb.batch_time
    FROM comment_batches cb
    WHERE cb.keyword = p.keyword
      AND cb.product_id = s.product_id
      AND cb.batch_time <= s.capture_time
    ORDER BY cb.batch_time DESC LIMIT 1
  ) AS cb_near ON TRUE
  WHERE s.product_id NOT IN (SELECT product_id FROM excluded)
),
batch_repr AS (
  SELECT * FROM (
    SELECT product_id, keyword, batch_time, sales_count, snapshot_time,
           ROW_NUMBER() OVER (PARTITION BY product_id, batch_time ORDER BY snapshot_time DESC) AS rn
    FROM snap_mapped WHERE batch_time IS NOT NULL
  ) t WHERE rn = 1
),
seq AS (
  SELECT product_id, keyword, batch_time, sales_count,
         LAG(sales_count) OVER (PARTITION BY product_id ORDER BY batch_time) AS prev_sales
  FROM batch_repr
),
y_post AS (
  SELECT product_id, keyword,
    MAX(CASE WHEN batch_time > %(cutoff)s::timestamp
                  AND prev_sales IS NOT NULL
                  AND (sales_count - prev_sales) >= %(delta)s
             THEN 1 ELSE 0 END) AS y
  FROM seq
  GROUP BY product_id, keyword
)
SELECT
    keyword,
    COUNT(*)                                              AS valid_sample_count,
    SUM(CASE WHEN y = 1 THEN 1 ELSE 0 END)               AS explosive_count,
    SUM(CASE WHEN y = 0 THEN 1 ELSE 0 END)               AS non_explosive_count,
    ROUND(100.0 * SUM(CASE WHEN y=1 THEN 1 ELSE 0 END) / COUNT(*), 1) AS explosive_pct
FROM y_post
GROUP BY keyword
ORDER BY keyword;
"""

# ── E. v2_error_prod 過濾後的商品數量（各 keyword 剩多少） ─────────────────
SQL_PRODUCT_COUNT_FILTERED = """
WITH excluded AS (
  SELECT filter_value::int AS product_id
  FROM ml_data_filters
  WHERE version_tag = %(filter_tag)s
    AND filter_level = 'product_id'
)
SELECT p.keyword, COUNT(DISTINCT p.id) AS product_count_filtered
FROM products p
WHERE p.id NOT IN (SELECT product_id FROM excluded)
GROUP BY p.keyword
ORDER BY p.keyword;
"""

# ── F. v2_error_prod 被排除的商品數 ──────────────────────────────────────
SQL_EXCLUDED_COUNT = """
SELECT
  COUNT(*) AS total_excluded,
  COUNT(*) FILTER (WHERE filter_level='product_id') AS by_product_id
FROM ml_data_filters
WHERE version_tag = %(filter_tag)s;
"""

# ── G. 每商品評論分佈 ─────────────────────────────────────────────────────
SQL_AVG_COMMENTS = """
SELECT
    p.keyword,
    ROUND(AVG(cnt)::numeric, 1)    AS avg_comments_per_product,
    MIN(cnt)                       AS min_per_product,
    MAX(cnt)                       AS max_per_product,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cnt) AS median_per_product
FROM (
    SELECT pc.product_id, COUNT(*) AS cnt
    FROM product_comments pc
    GROUP BY pc.product_id
) sub
JOIN products p ON p.id = sub.product_id
GROUP BY p.keyword
ORDER BY p.keyword;
"""

def fmt(n):
    """Format number with comma."""
    return f"{int(n):,}"

def main():
    conn = get_conn()
    params = {'cutoff': DATE_CUTOFF, 'delta': DELTA_THRESHOLD}
    params_f = {'cutoff': DATE_CUTOFF, 'delta': DELTA_THRESHOLD, 'filter_tag': FILTER_TAG}

    import warnings
    warnings.filterwarnings('ignore')

    df_prod    = pd.read_sql(SQL_PRODUCT_COUNT, conn)
    df_comment = pd.read_sql(SQL_COMMENT_STATS, conn)
    df_label   = pd.read_sql(SQL_LABEL, conn, params=params)
    df_label_f = pd.read_sql(SQL_LABEL_FILTERED, conn, params=params_f)
    df_prod_f  = pd.read_sql(SQL_PRODUCT_COUNT_FILTERED, conn, params={'filter_tag': FILTER_TAG})
    df_excl    = pd.read_sql(SQL_EXCLUDED_COUNT, conn, params={'filter_tag': FILTER_TAG})
    df_avg     = pd.read_sql(SQL_AVG_COMMENTS, conn)

    print("=" * 80)
    print("【v2_error_prod 排除筆數】")
    print(df_excl.to_string(index=False))

    print("\n" + "=" * 80)
    print("【表 A：未過濾版 — 各 keyword 彙整】")
    m = df_prod \
        .merge(df_comment[['keyword','comment_count','min_comment_date','max_comment_date','avg_score']], on='keyword') \
        .merge(df_label[['keyword','valid_sample_count','explosive_count','non_explosive_count','explosive_pct']], on='keyword') \
        .merge(df_avg[['keyword','avg_comments_per_product','median_per_product']], on='keyword')
    print(m.to_string(index=False))

    print("\n" + "=" * 80)
    print("【表 B：套用 v2_error_prod 過濾後 — 各 keyword 彙整】")
    m_f = df_prod_f \
        .merge(df_comment[['keyword','comment_count','min_comment_date','max_comment_date','avg_score']], on='keyword') \
        .merge(df_label_f[['keyword','valid_sample_count','explosive_count','non_explosive_count','explosive_pct']], on='keyword') \
        .merge(df_avg[['keyword','avg_comments_per_product','median_per_product']], on='keyword')
    print(m_f.to_string(index=False))

    # 存 CSV
    m.to_csv(r'C:\YvesProject\中央\線上評論\momo_crawler-main\tmp\table2_unfiltered.csv', index=False, encoding='utf-8-sig')
    m_f.to_csv(r'C:\YvesProject\中央\線上評論\momo_crawler-main\tmp\table2_v2error_filtered.csv', index=False, encoding='utf-8-sig')
    print("\n✅ 已存 CSV 至 tmp/ 資料夾")

    conn.close()

if __name__ == '__main__':
    main()
