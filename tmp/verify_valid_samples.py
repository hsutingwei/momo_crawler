# -*- coding: utf-8 -*-
"""
精準版：逐層驗證有效樣本的形成過程
重點：y_post 的條件 — 商品只要「存在於 batch_repr（有對齊紀錄）」就會出現在 y_post
(y=0 或 y=1)，不需要有 post-cutoff 批次
但 y 的意義是：有 post-cutoff AND prev_sales NOT NULL 的 batch 才能賦予 y=1
沒有 post-cutoff 的商品 → y=MAX(0)=0（標記為非爆品，即使沒有後期觀測）

實際上：爬蟲時間是 2025-04-28 ~ 2025-08-23，全部商品都有 post-cutoff 快照
所以「可評估」的商品幾乎等於 batch_repr 裡的商品 = 4939

讓我確認各層數字邏輯
"""
import sys, os
sys.path.append(r'C:\YvesProject\中央\線上評論\momo_crawler-main')
from dotenv import load_dotenv
load_dotenv(r'C:\YvesProject\中央\線上評論\momo_crawler-main\.env')
from config.database import DatabaseConfig
import pandas as pd, warnings
warnings.filterwarnings('ignore')

conn = DatabaseConfig().get_connection()
DATE_CUTOFF = '2025-06-25'

# ===== 逐層查詢 =====

# 1. 全部商品
q1 = "SELECT COUNT(DISTINCT id) AS n FROM products"
n1 = int(pd.read_sql(q1, conn).iloc[0,0])

# 2. 有評論（進入 comment_batches）
q2 = "SELECT COUNT(DISTINCT product_id) AS n FROM product_comments"
n2 = int(pd.read_sql(q2, conn).iloc[0,0])

# 2b. 沒評論的商品（= n1 - n2）
n1_no_comment = n1 - n2

# 3. batch_repr 裡出現（有評論 + 有銷售快照可對齊）= 最終 y_post 的商品
q3 = f"""
WITH comment_batches AS (
  SELECT p.keyword, pc.product_id, pc.capture_time AS batch_time
  FROM product_comments pc JOIN products p ON p.id = pc.product_id
  GROUP BY p.keyword, pc.product_id, pc.capture_time
),
snap_mapped AS (
  SELECT s.product_id, p.keyword, s.capture_time AS snapshot_time, s.sales_count, cb_near.batch_time
  FROM sales_snapshots s
  JOIN products p ON p.id = s.product_id
  LEFT JOIN LATERAL (
    SELECT cb.batch_time FROM comment_batches cb
    WHERE cb.keyword = p.keyword AND cb.product_id = s.product_id AND cb.batch_time <= s.capture_time
    ORDER BY cb.batch_time DESC LIMIT 1
  ) AS cb_near ON TRUE
),
batch_repr AS (
  SELECT product_id, batch_time, sales_count, snapshot_time
  FROM (
    SELECT product_id, batch_time, sales_count, snapshot_time,
      ROW_NUMBER() OVER (PARTITION BY product_id, batch_time ORDER BY snapshot_time DESC) AS rn
    FROM snap_mapped WHERE batch_time IS NOT NULL
  ) t WHERE rn = 1
)
SELECT COUNT(DISTINCT product_id) AS n FROM batch_repr
"""
n3 = int(pd.read_sql(q3, conn).iloc[0,0])

# 3b. 有評論但銷售快照無法對齊 = n2 - n3
n2_no_snap = n2 - n3

# 4. seq 中 prev_sales IS NOT NULL 的商品（≥2 批次紀錄）
q4 = f"""
WITH comment_batches AS (
  SELECT p.keyword, pc.product_id, pc.capture_time AS batch_time
  FROM product_comments pc JOIN products p ON p.id = pc.product_id
  GROUP BY p.keyword, pc.product_id, pc.capture_time
),
snap_mapped AS (
  SELECT s.product_id, p.keyword, s.capture_time AS snapshot_time, s.sales_count, cb_near.batch_time
  FROM sales_snapshots s JOIN products p ON p.id = s.product_id
  LEFT JOIN LATERAL (
    SELECT cb.batch_time FROM comment_batches cb
    WHERE cb.keyword = p.keyword AND cb.product_id = s.product_id AND cb.batch_time <= s.capture_time
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
  SELECT product_id, batch_time,
    LAG(sales_count) OVER (PARTITION BY product_id ORDER BY batch_time) AS prev_sales
  FROM batch_repr
)
SELECT COUNT(DISTINCT product_id) AS n FROM seq WHERE prev_sales IS NOT NULL
"""
n4 = int(pd.read_sql(q4, conn).iloc[0,0])

# 5. 截止日後有 batch_time > cutoff AND prev_sales NOT NULL 的商品
q5 = f"""
WITH comment_batches AS (
  SELECT p.keyword, pc.product_id, pc.capture_time AS batch_time
  FROM product_comments pc JOIN products p ON p.id = pc.product_id
  GROUP BY p.keyword, pc.product_id, pc.capture_time
),
snap_mapped AS (
  SELECT s.product_id, p.keyword, s.capture_time AS snapshot_time, s.sales_count, cb_near.batch_time
  FROM sales_snapshots s JOIN products p ON p.id = s.product_id
  LEFT JOIN LATERAL (
    SELECT cb.batch_time FROM comment_batches cb
    WHERE cb.keyword = p.keyword AND cb.product_id = s.product_id AND cb.batch_time <= s.capture_time
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
  SELECT product_id, batch_time,
    LAG(sales_count) OVER (PARTITION BY product_id ORDER BY batch_time) AS prev_sales
  FROM batch_repr
)
SELECT COUNT(DISTINCT product_id) AS n FROM seq
WHERE batch_time > '{DATE_CUTOFF}'::timestamp AND prev_sales IS NOT NULL
"""
n5 = int(pd.read_sql(q5, conn).iloc[0,0])

# 6. capture_time 的分佈（確認是否橫跨 cutoff）
q6 = """
SELECT
  MIN(capture_time)::date AS min_capture,
  MAX(capture_time)::date AS max_capture,
  COUNT(DISTINCT CASE WHEN capture_time <= '2025-06-25' THEN product_id END) AS products_before_cutoff,
  COUNT(DISTINCT CASE WHEN capture_time >  '2025-06-25' THEN product_id END) AS products_after_cutoff
FROM product_comments
"""
df6 = pd.read_sql(q6, conn)

# 7. batch_repr 中各商品的批次數分佈
q7 = f"""
WITH comment_batches AS (
  SELECT p.keyword, pc.product_id, pc.capture_time AS batch_time
  FROM product_comments pc JOIN products p ON p.id = pc.product_id
  GROUP BY p.keyword, pc.product_id, pc.capture_time
),
snap_mapped AS (
  SELECT s.product_id, p.keyword, s.capture_time AS snapshot_time, s.sales_count, cb_near.batch_time
  FROM sales_snapshots s JOIN products p ON p.id = s.product_id
  LEFT JOIN LATERAL (
    SELECT cb.batch_time FROM comment_batches cb
    WHERE cb.keyword = p.keyword AND cb.product_id = s.product_id AND cb.batch_time <= s.capture_time
    ORDER BY cb.batch_time DESC LIMIT 1
  ) AS cb_near ON TRUE
),
batch_repr AS (
  SELECT * FROM (
    SELECT product_id, batch_time,
      ROW_NUMBER() OVER (PARTITION BY product_id, batch_time ORDER BY snapshot_time DESC) AS rn
    FROM snap_mapped WHERE batch_time IS NOT NULL
  ) t WHERE rn = 1
),
counts AS (
  SELECT product_id, COUNT(*) AS n_batches FROM batch_repr GROUP BY product_id
)
SELECT
  n_batches,
  COUNT(*) AS product_count
FROM counts
GROUP BY n_batches
ORDER BY n_batches
"""
df7 = pd.read_sql(q7, conn)
conn.close()

# ===== 輸出報告 =====
print("=" * 60)
print("【各層過濾的商品數追蹤】")
print("=" * 60)
print(f"① 全部商品（products 表）           : {n1:,}")
print(f"  └─ 沒有任何評論 → 無法建立特徵    : {n1_no_comment:,}  (剔除)")
print(f"② 有評論的商品                      : {n2:,}")
print(f"  └─ 評論批次無對應銷售快照 → 無標籤 : {n2_no_snap:,}  (剔除)")
print(f"③ 進入 batch_repr（可計算標籤母群） : {n3:,}  ← 即本研究「有效樣本」上限")
print()
print(f"【批次數分佈（n_batches = 有幾個爬蟲時間點）】")
print(df7.to_string(index=False))
print()
print(f"【評論 capture_time 範圍 & 跨 cutoff 分佈】")
print(df6.to_string(index=False))
print()
print(f"④ 有 ≥2 批次紀錄（prev_sales 存在）  : {n4:,}")
print(f"   └ 只有1批次（計算 Δ 需要前後對比） : {n3-n4:,}  (y 仍為 0，算入有效樣本)")
print(f"⑤ 截止日後有可評估批次               : {n5:,}")
print(f"   └ 只有截止前批次                  : {n3-n5:,}  (y 仍為 0，算入有效樣本)")
print()
print(f"⑥ 最終 y_post（全量有效樣本）         : {n3:,}  ← 這是因為所有")
print(f"   進入 batch_repr 的商品都會在 y_post 得到 y（0 或 1）")
