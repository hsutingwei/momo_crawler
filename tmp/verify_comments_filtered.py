# -*- coding: utf-8 -*-
"""
驗證：過濾後的評論筆數與觀測期間
"""
import sys
sys.path.append(r'C:\YvesProject\中央\線上評論\momo_crawler-main')
from dotenv import load_dotenv
load_dotenv(r'C:\YvesProject\中央\線上評論\momo_crawler-main\.env')
from config.database import DatabaseConfig
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

conn = DatabaseConfig().get_connection()
FILTER_TAG = 'v2_error_prod'

SQL_COMMENT_FILTERED = """
WITH excluded AS (
  SELECT filter_value::int AS product_id
  FROM ml_data_filters
  WHERE version_tag = %(filter_tag)s
    AND filter_level = 'product_id'
)
SELECT
    p.keyword,
    COUNT(DISTINCT pc.comment_id)               AS comment_count_filtered,
    MIN(pc.comment_date)::date                  AS min_comment_date_filtered,
    MAX(pc.comment_date)::date                  AS max_comment_date_filtered
FROM product_comments pc
JOIN products p ON p.id = pc.product_id
WHERE pc.product_id NOT IN (SELECT product_id FROM excluded)
GROUP BY p.keyword
ORDER BY p.keyword;
"""

df_f = pd.read_sql(SQL_COMMENT_FILTERED, conn, params={'filter_tag': FILTER_TAG})
print(df_f.to_string(index=False))
print(f"\n總評論數(套用過濾後): {df_f['comment_count_filtered'].sum()}")
conn.close()
