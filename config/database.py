import psycopg2
import os
import logging
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConfig:
    def __init__(self):
        # 資料庫連接設定 - 請根據你的環境修改
        self.config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', ''),
            'dbname': os.getenv('DB_NAME', 'momo_crawler'),
        }
    
    def get_connection(self):
        """取得資料庫連接"""
        try:
            conn = psycopg2.connect(**self.config)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            logger.info("成功連接到 PostgreSQL 資料庫")
            return conn
        except Exception as e:
            logger.error(f"資料庫連接錯誤: {e}")
            raise
    
    def init_database(self):
        """初始化資料庫表結構"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        try:
            # 建立商品表
            cur.execute('''
                CREATE TABLE IF NOT EXISTS products (
                    id BIGINT PRIMARY KEY,
                    name VARCHAR(500) NOT NULL,
                    price NUMERIC(10,2),
                    product_link TEXT,
                    keyword VARCHAR(100),
                    is_complete BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_keyword ON products(keyword);
                CREATE INDEX IF NOT EXISTS idx_created_at ON products(created_at);
            ''')
            
            # 建立銷售快照表
            cur.execute('''
                CREATE TABLE IF NOT EXISTS sales_snapshots (
                    id BIGSERIAL PRIMARY KEY,
                    product_id BIGINT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
                    sales_count INTEGER,
                    sales_unit VARCHAR(10),
                    capture_time TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(product_id, capture_time)
                );
                CREATE INDEX IF NOT EXISTS idx_product_id ON sales_snapshots(product_id);
                CREATE INDEX IF NOT EXISTS idx_capture_time ON sales_snapshots(capture_time);
            ''')
            
            # 建立商品評論表
            cur.execute('''
                CREATE TABLE IF NOT EXISTS product_comments (
                    id BIGSERIAL PRIMARY KEY,
                    comment_id VARCHAR(100) UNIQUE NOT NULL,
                    product_id BIGINT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
                    comment_text TEXT,
                    customer_name VARCHAR(200),
                    comment_date TIMESTAMP NULL,
                    goods_type VARCHAR(100),
                    image_urls JSONB,
                    is_like BOOLEAN,
                    is_show_like BOOLEAN,
                    like_count INTEGER,
                    reply_content TEXT,
                    reply_date TIMESTAMP NULL,
                    score NUMERIC(2,1),
                    video_thumbnail_img TEXT,
                    video_url TEXT,
                    capture_time TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_product_id_comment ON product_comments(product_id);
                CREATE INDEX IF NOT EXISTS idx_comment_id ON product_comments(comment_id);
                CREATE INDEX IF NOT EXISTS idx_comment_date ON product_comments(comment_date);
                CREATE INDEX IF NOT EXISTS idx_capture_time_comment ON product_comments(capture_time);
            ''')
            
            # 建立資料驗證錯誤記錄表
            cur.execute('''
                CREATE TABLE IF NOT EXISTS data_validation_errors (
                    id BIGSERIAL PRIMARY KEY,
                    file_path VARCHAR(500) NOT NULL,
                    keyword VARCHAR(100) NOT NULL,
                    data_type VARCHAR(20) NOT NULL,
                    row_number INTEGER,
                    field_name VARCHAR(100),
                    original_value TEXT,
                    error_type VARCHAR(20) NOT NULL,
                    error_message TEXT,
                    suggested_value TEXT,
                    is_fixed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_file_path ON data_validation_errors(file_path);
                CREATE INDEX IF NOT EXISTS idx_keyword ON data_validation_errors(keyword);
                CREATE INDEX IF NOT EXISTS idx_error_type ON data_validation_errors(error_type);
                CREATE INDEX IF NOT EXISTS idx_is_fixed ON data_validation_errors(is_fixed);
            ''')
            
            # 建立檔案同步記錄表
            cur.execute('''
                CREATE TABLE IF NOT EXISTS file_sync_logs (
                    id BIGSERIAL PRIMARY KEY,
                    keyword VARCHAR(100) NOT NULL,
                    file_type VARCHAR(20) NOT NULL,
                    file_path VARCHAR(500) NOT NULL,
                    file_size BIGINT,
                    total_records INTEGER DEFAULT 0,
                    valid_records INTEGER DEFAULT 0,
                    error_records INTEGER DEFAULT 0,
                    sync_status VARCHAR(20) DEFAULT 'pending',
                    sync_start_time TIMESTAMP NULL,
                    sync_end_time TIMESTAMP NULL,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(keyword, file_type, file_path)
                );
                CREATE INDEX IF NOT EXISTS idx_keyword_sync ON file_sync_logs(keyword);
                CREATE INDEX IF NOT EXISTS idx_sync_status ON file_sync_logs(sync_status);
            ''')
            
            conn.commit()
            logger.info("資料庫表結構初始化完成 (PostgreSQL)")
            
        except Exception as e:
            logger.error(f"初始化資料庫錯誤: {e}")
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close() 