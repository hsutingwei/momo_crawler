import psycopg2
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.database import DatabaseConfig

def run_ddl():
    db = DatabaseConfig()
    conn = db.get_connection()
    try:
        sql_path = os.path.join(os.path.dirname(__file__), 'create_ml_filters_table.sql')
        with open(sql_path, 'r', encoding='utf-8') as f:
            ddl = f.read()
        
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
        print("Successfully created table: ml_data_filters")
    except Exception as e:
        conn.rollback()
        print(f"Error creating table: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    run_ddl()
