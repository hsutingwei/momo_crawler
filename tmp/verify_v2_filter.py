# -*- coding: utf-8 -*-
"""
驗證腳本：確認兩種過濾順序的結果是否一致
方法A（train_v2.py 的方法）：先算 label（4939），再 drop v2_error_prod → 理論上 3729
方法B（我的 SQL）：先排除 v2_error_prod，再算 label → 查詢結果 3729
"""
import sys, os
sys.path.append(r'C:\YvesProject\中央\線上評論\momo_crawler-main')
from dotenv import load_dotenv
load_dotenv(r'C:\YvesProject\中央\線上評論\momo_crawler-main\.env')
from config.database import DatabaseConfig
import pandas as pd, warnings
warnings.filterwarnings('ignore')

def get_conn():
    return DatabaseConfig().get_connection()

DATE_CUTOFF = '2025-06-25'
DELTA = 10.0
FILTER_TAG = 'v2_error_prod'

# 方法A：模擬 train_v2.py 的行為
# 先算完整 label（不過濾），再 drop v2_error_prod 的商品
SQL_ALL_LABEL = """
WITH comment_batches AS (
  SELECT p.keyword, pc.product_id, pc.capture_time AS batch_time
  FROM product_comments pc JOIN products p ON p.id = pc.product_id
  GROUP BY p.keyword, pc.product_id, pc.capture_time
),
snap_mapped AS (
  SELECT s.product_id, p.keyword,
    s.capture_time AS snapshot_time, s.sales_count, cb_near.batch_time
  FROM sales_snapshots s
  JOIN products p ON p.id = s.product_id
  LEFT JOIN LATERAL (
    SELECT cb.batch_time FROM comment_batches cb
    WHERE cb.keyword = p.keyword AND cb.product_id = s.product_id
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
    MAX(CASE WHEN batch_time > %(cutoff)s::timestamp AND prev_sales IS NOT NULL
             AND (sales_count - prev_sales) >= %(delta)s
        THEN 1 ELSE 0 END) AS y
  FROM seq GROUP BY product_id, keyword
)
SELECT product_id, keyword, y FROM y_post;
"""

SQL_EXCLUDED = """
SELECT filter_value::int AS product_id
FROM ml_data_filters
WHERE version_tag = 'v2_error_prod' AND filter_level = 'product_id';
"""

conn = get_conn()
df_all = pd.read_sql(SQL_ALL_LABEL, conn, params={'cutoff': DATE_CUTOFF, 'delta': DELTA})
df_excl = pd.read_sql(SQL_EXCLUDED, conn)
conn.close()

# 方法 A：先全算，再 drop
excluded_ids = set(df_excl['product_id'])
df_a = df_all[~df_all['product_id'].isin(excluded_ids)]

print("=" * 60)
print("【方法 A：模擬 train_v2.py（先算 label，再 drop v2_error_prod）】")
print(f"  全量 labeled 商品: {len(df_all)}")
print(f"  v2_error_prod 排除: {len(excluded_ids)}")
print(f"  過濾後 n_samples: {len(df_a)}")
print(f"  爆品 (y=1): {(df_a['y']==1).sum()}")
print(f"  非爆品 (y=0): {(df_a['y']==0).sum()}")
print(f"  正類比例: {(df_a['y']==1).mean():.1%}")
print()
print("【各 keyword 明細】")
g = df_a.groupby('keyword').agg(
    valid_sample=('y','count'),
    explosive=('y','sum'),
).reset_index()
g['non_explosive'] = g['valid_sample'] - g['explosive']
g['pct'] = (g['explosive']/g['valid_sample']*100).round(1)
print(g.to_string(index=False))
