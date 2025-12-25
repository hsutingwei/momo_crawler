import os
import sys
import pandas as pd
import psycopg2
import psycopg2.extras

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.database import DatabaseConfig

def get_db_connection():
    db = DatabaseConfig()
    return db.get_connection()

def main():
    sql = """
    WITH product_stats AS (
        SELECT 
            p.keyword,
            COUNT(p.id) AS total_products,
            COUNT(f.filter_value) AS filtered_count
        FROM products p
        LEFT JOIN ml_data_filters f ON 
            CAST(p.id AS VARCHAR) = f.filter_value 
            AND f.version_tag = 'v2_error_prod'
            AND f.filter_level = 'product_id'
            AND f.reason = 'ensemble_error_count'
        GROUP BY p.keyword
    )
    SELECT 
        keyword,
        total_products AS original_count,
        filtered_count AS removed_count,
        ROUND((filtered_count::NUMERIC / NULLIF(total_products, 0)) * 100, 2) AS removed_percentage,
        (total_products - filtered_count) AS remaining_count
    FROM product_stats
    WHERE filtered_count > 0
    ORDER BY removed_percentage DESC, total_products DESC
    LIMIT 20;
    """
    
    conn = get_db_connection()
    df = pd.read_sql(sql, conn)
    conn.close()
    
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
