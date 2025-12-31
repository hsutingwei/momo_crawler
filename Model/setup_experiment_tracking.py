"""
建立實驗追蹤資料表
使用 Python + psycopg2 執行 DDL
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import DatabaseConfig

def create_experiment_tracking_tables():
    # 使用 DatabaseConfig 的正確方式
    db_config = DatabaseConfig()
    conn = db_config.get_connection()
    
    # 讀取 DDL
    ddl_path = os.path.join(os.path.dirname(__file__), 'create_experiment_tracking_tables.sql')
    with open(ddl_path, 'r', encoding='utf-8') as f:
        ddl_sql = f.read()
    
    try:
        cursor = conn.cursor()
        cursor.execute(ddl_sql)
        conn.commit()
        print("✅ 成功建立實驗追蹤資料表！")
        
        # 驗證表是否建立成功
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
              AND table_name LIKE 'experiment_%'
            ORDER BY table_name
        """)
        
        tables = cursor.fetchall()
        print(f"\n已建立 {len(tables)} 張表：")
        for (table_name,) in tables:
            print(f"  - {table_name}")
            
    except Exception as e:
        conn.rollback()
        print(f"❌ 建立表失敗：{e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    create_experiment_tracking_tables()
