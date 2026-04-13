# -*- coding: utf-8 -*-
"""
查詢論文表 2 所需的各關鍵字統計資料
- 商品數量（各 keyword 的 products count）
- 爆品數 / 非爆品數（label y=1 / y=0，使用 absolute 策略, delta_threshold=10, cutoff=2025-06-25）
- 評論筆數、觀測期間（已知，從 各關鍵字時間範圍.md 確認）
- 平均每商品評論數
- 平均星數分佈
- 正負類比例
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
DELTA_THRESHOLD = 10.0  # absolute strategy, same as production

SQL_PRODUCT_COUNT = """
SELECT
    keyword,
    COUNT(DISTINCT id) AS product_count
FROM products
GROUP BY keyword
ORDER BY keyword;
"""

SQL_COMMENT_STATS = """
SELECT
    p.keyword,
    COUNT(DISTINCT pc.comment_id)      AS comment_count,
    MIN(pc.comment_date)               AS min_comment_date,
    MAX(pc.comment_date)               AS max_comment_date,
    COUNT(DISTINCT pc.product_id)      AS products_with_comments,
    ROUND(AVG(pc.score::float)::numeric, 2) AS avg_score,
    ROUND(STDDEV(pc.score::float)::numeric, 2) AS std_score
FROM product_comments pc
JOIN products p ON p.id = pc.product_id
GROUP BY p.keyword
ORDER BY p.keyword;
"""

SQL_LABEL = """
WITH comment_batches AS (
  SELECT p.keyword, pc.product_id, pc.capture_time AS batch_time
  FROM product_comments pc
  JOIN products p ON p.id = pc.product_id
  GROUP BY p.keyword, pc.product_id, pc.capture_time
),
snap_mapped AS (
  SELECT
    s.product_id,
    p.keyword,
    s.capture_time AS snapshot_time,
    s.sales_count,
    cb_near.batch_time
  FROM sales_snapshots s
  JOIN products p ON p.id = s.product_id
  LEFT JOIN LATERAL (
    SELECT cb.batch_time
    FROM comment_batches cb
    WHERE cb.keyword = p.keyword
      AND cb.product_id = s.product_id
      AND cb.batch_time <= s.capture_time
    ORDER BY cb.batch_time DESC
    LIMIT 1
  ) AS cb_near ON TRUE
),
batch_repr AS (
  SELECT *
  FROM (
    SELECT
      product_id, keyword, batch_time, sales_count, snapshot_time,
      ROW_NUMBER() OVER (PARTITION BY product_id, batch_time ORDER BY snapshot_time DESC) AS rn
    FROM snap_mapped
    WHERE batch_time IS NOT NULL
  ) t
  WHERE rn = 1
),
seq AS (
  SELECT
    product_id, keyword, batch_time, snapshot_time, sales_count,
    LAG(sales_count) OVER (PARTITION BY product_id ORDER BY batch_time) AS prev_sales
  FROM batch_repr
),
y_post AS (
  SELECT
    product_id,
    keyword,
    MAX(
      CASE
        WHEN batch_time > %(cutoff)s::timestamp
             AND prev_sales IS NOT NULL
             AND (sales_count - prev_sales) >= %(delta)s
        THEN 1 ELSE 0
      END
    ) AS y
  FROM seq
  GROUP BY product_id, keyword
)
SELECT
    keyword,
    COUNT(*) AS total_products_labeled,
    SUM(CASE WHEN y = 1 THEN 1 ELSE 0 END) AS explosive_count,
    SUM(CASE WHEN y = 0 THEN 1 ELSE 0 END) AS non_explosive_count,
    ROUND(100.0 * SUM(CASE WHEN y = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) AS explosive_pct
FROM y_post
GROUP BY keyword
ORDER BY keyword;
"""

SQL_AVG_COMMENTS_PER_PRODUCT = """
SELECT
    p.keyword,
    ROUND(AVG(cnt)::numeric, 1) AS avg_comments_per_product,
    MIN(cnt) AS min_comments_per_product,
    MAX(cnt) AS max_comments_per_product,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cnt) AS median_comments_per_product
FROM (
    SELECT pc.product_id, p.keyword, COUNT(*) AS cnt
    FROM product_comments pc
    JOIN products p ON p.id = pc.product_id
    GROUP BY pc.product_id, p.keyword
) sub
JOIN products p ON p.id = sub.product_id
GROUP BY p.keyword
ORDER BY p.keyword;
"""

def main():
    conn = get_conn()
    try:
        print("=" * 70)
        print("【各關鍵字商品數量】")
        df_prod = pd.read_sql(SQL_PRODUCT_COUNT, conn)
        print(df_prod.to_string(index=False))

        print("\n" + "=" * 70)
        print("【各關鍵字評論統計】")
        df_comment = pd.read_sql(SQL_COMMENT_STATS, conn)
        print(df_comment.to_string(index=False))

        print("\n" + "=" * 70)
        print(f"【各關鍵字爆品/非爆品數量】(cutoff={DATE_CUTOFF}, delta={DELTA_THRESHOLD})")
        df_label = pd.read_sql(SQL_LABEL, conn, params={'cutoff': DATE_CUTOFF, 'delta': DELTA_THRESHOLD})
        print(df_label.to_string(index=False))

        print("\n" + "=" * 70)
        print("【各關鍵字每商品評論數分佈】")
        df_avg = pd.read_sql(SQL_AVG_COMMENTS_PER_PRODUCT, conn)
        print(df_avg.to_string(index=False))

        # 合併輸出完整表格
        print("\n" + "=" * 70)
        print("【彙整表】")
        m = df_prod.merge(df_comment[['keyword','comment_count','min_comment_date','max_comment_date','avg_score']], on='keyword') \
                   .merge(df_label[['keyword','total_products_labeled','explosive_count','non_explosive_count','explosive_pct']], on='keyword') \
                   .merge(df_avg[['keyword','avg_comments_per_product','median_comments_per_product']], on='keyword')
        print(m.to_string(index=False))

        # 存成 CSV
        out_path = r'C:\YvesProject\中央\線上評論\momo_crawler-main\tmp\table2_stats.csv'
        m.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"\n已存至: {out_path}")

    finally:
        conn.close()

if __name__ == '__main__':
    main()
