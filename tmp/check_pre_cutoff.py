import sys
sys.path.insert(0, '.')
from config.database import DatabaseConfig

db_config = DatabaseConfig()
conn = db_config.get_connection()
cur = conn.cursor()

# 取得 v2_error_prod 排除清單
cur.execute("SELECT filter_value::int FROM ml_data_filters WHERE version_tag=%s AND filter_level=%s",
            ('v2_error_prod', 'product_id'))
excluded_ids = [r[0] for r in cur.fetchall()]
print(f"v2_error_prod 排除: {len(excluded_ids)} 件")

# 最終有效商品數
cur.execute("""
    SELECT COUNT(DISTINCT p.id)
    FROM products p
    WHERE p.id <> ALL(%s)
      AND EXISTS (SELECT 1 FROM product_comments pc WHERE pc.product_id = p.id)
""", (excluded_ids,))
n_final = cur.fetchone()[0]
print(f"最終商品數: {n_final}")

# 精確「截止日前評論數」= sql_dense pre_comments 的筆數
cur.execute("""
    SELECT COUNT(*)
    FROM product_comments pc
    JOIN products p ON p.id = pc.product_id
    WHERE pc.comment_date <= %s::date
      AND p.id <> ALL(%s)
      AND EXISTS (SELECT 1 FROM product_comments pc2 WHERE pc2.product_id = p.id)
""", ('2025-06-25', excluded_ids))
pre_cutoff_total = cur.fetchone()[0]
print(f"截止日 T 前評論總數 (comment_date 版): {pre_cutoff_total}")

# TF-IDF doc_text_tokenized 用的 capture_time 版
cur.execute("""
    SELECT COUNT(DISTINCT pc.comment_id)
    FROM product_comments pc
    JOIN products p ON p.id = pc.product_id
    WHERE pc.capture_time <= %s::timestamp
      AND p.id <> ALL(%s)
      AND EXISTS (SELECT 1 FROM product_comments pc2 WHERE pc2.product_id = p.id)
      AND EXISTS (SELECT 1 FROM comment_tokens ct WHERE ct.comment_id = pc.comment_id)
""", ('2025-06-25', excluded_ids))
tfidf_comment_count = cur.fetchone()[0]
print(f"doc_text_tokenized 用的評論數 (capture_time, 有 token): {tfidf_comment_count}")

# 分關鍵字明細
cur.execute("""
    SELECT p.keyword,
           COUNT(DISTINCT p.id) AS n_products,
           SUM(CASE WHEN pc.comment_date <= '2025-06-25'::date THEN 1 ELSE 0 END) AS n_pre_cutoff,
           COUNT(pc.comment_id) AS n_all_comments
    FROM products p
    LEFT JOIN product_comments pc ON pc.product_id = p.id
    WHERE p.id <> ALL(%s)
      AND EXISTS (SELECT 1 FROM product_comments pc2 WHERE pc2.product_id = p.id)
    GROUP BY p.keyword
    ORDER BY p.keyword
""", (excluded_ids,))
rows = cur.fetchall()
print()
print("keyword|n_products|n_pre_cutoff_comments|n_all_comments")
total_prod = 0
total_pre = 0
total_all = 0
for r in rows:
    keyword = r[0]
    n_p = r[1]
    n_pre = r[2]
    n_all = r[3]
    print(f"  {keyword}|{n_p}|{n_pre}|{n_all}")
    total_prod += n_p
    total_pre += n_pre
    total_all += n_all
print(f"  合計|{total_prod}|{total_pre}|{total_all}")

cur.close()
conn.close()
