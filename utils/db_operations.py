import psycopg2
from typing import List, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseOperations:
    """資料庫操作類別 (PostgreSQL 版)"""
    def __init__(self, db_connection):
        self.db = db_connection

    def insert_product(self, product_id: int, name: str, price: float, 
                      product_link: str, keyword: str) -> bool:
        try:
            with self.db.cursor() as cursor:
                query = '''
                    INSERT INTO products (id, name, price, product_link, keyword, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        price = EXCLUDED.price,
                        product_link = EXCLUDED.product_link,
                        keyword = EXCLUDED.keyword,
                        updated_at = NOW()
                '''
                cursor.execute(query, (product_id, name, price, product_link, keyword))
                self.db.commit()
                logger.debug(f"成功插入/更新商品: {product_id}")
                return True
        except Exception as e:
            logger.error(f"插入商品資料時發生錯誤: {e}")
            self.db.rollback()
            return False

    def insert_sales_snapshot(self, product_id: int, sales_count: int, 
                            sales_unit: str, capture_time: datetime) -> bool:
        try:
            with self.db.cursor() as cursor:
                query = '''
                    INSERT INTO sales_snapshots (product_id, sales_count, sales_unit, capture_time, created_at)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON CONFLICT (product_id, capture_time) DO UPDATE SET
                        sales_count = EXCLUDED.sales_count,
                        sales_unit = EXCLUDED.sales_unit,
                        created_at = NOW()
                '''
                cursor.execute(query, (product_id, sales_count, sales_unit, capture_time))
                self.db.commit()
                logger.debug(f"成功插入銷售快照: {product_id} - {capture_time}")
                return True
        except Exception as e:
            logger.error(f"插入銷售快照時發生錯誤: {e}")
            self.db.rollback()
            return False

    def insert_product_comment(self, comment_id: str, product_id: int, comment_text: str,
                             customer_name: str, comment_date: datetime, goods_type: str,
                             image_urls: str, is_like: bool, is_show_like: bool,
                             like_count: int, reply_content: str, reply_date: datetime,
                             score: float, video_thumbnail_img: str, video_url: str,
                             capture_time: datetime) -> bool:
        try:
            with self.db.cursor() as cursor:
                query = '''
                    INSERT INTO product_comments 
                    (comment_id, product_id, comment_text, customer_name, comment_date,
                     goods_type, image_urls, is_like, is_show_like, like_count,
                     reply_content, reply_date, score, video_thumbnail_img, video_url,
                     capture_time, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (comment_id) DO UPDATE SET
                        comment_text = EXCLUDED.comment_text,
                        customer_name = EXCLUDED.customer_name,
                        comment_date = EXCLUDED.comment_date,
                        goods_type = EXCLUDED.goods_type,
                        image_urls = EXCLUDED.image_urls,
                        is_like = EXCLUDED.is_like,
                        is_show_like = EXCLUDED.is_show_like,
                        like_count = EXCLUDED.like_count,
                        reply_content = EXCLUDED.reply_content,
                        reply_date = EXCLUDED.reply_date,
                        score = EXCLUDED.score,
                        video_thumbnail_img = EXCLUDED.video_thumbnail_img,
                        video_url = EXCLUDED.video_url,
                        capture_time = EXCLUDED.capture_time,
                        created_at = NOW()
                '''
                cursor.execute(query, (
                    comment_id, product_id, comment_text, customer_name, comment_date,
                    goods_type, image_urls, is_like, is_show_like, like_count,
                    reply_content, reply_date, score, video_thumbnail_img, video_url,
                    capture_time
                ))
                self.db.commit()
                logger.debug(f"成功插入評論: {comment_id}")
                return True
        except Exception as e:
            logger.error(f"插入評論資料時發生錯誤: {e}")
            self.db.rollback()
            return False

    def check_product_exists(self, product_id: int) -> bool:
        try:
            with self.db.cursor() as cursor:
                query = "SELECT COUNT(*) FROM products WHERE id = %s"
                cursor.execute(query, (product_id,))
                count = cursor.fetchone()[0]
                return count > 0
        except Exception as e:
            logger.error(f"檢查商品存在時發生錯誤: {e}")
            return False

    def check_comment_exists(self, comment_id: str) -> bool:
        try:
            with self.db.cursor() as cursor:
                query = "SELECT COUNT(*) FROM product_comments WHERE comment_id = %s"
                cursor.execute(query, (comment_id,))
                count = cursor.fetchone()[0]
                return count > 0
        except Exception as e:
            logger.error(f"檢查評論存在時發生錯誤: {e}")
            return False

    def check_snapshot_exists(self, product_id: int, capture_time: datetime) -> bool:
        try:
            with self.db.cursor() as cursor:
                query = "SELECT COUNT(*) FROM sales_snapshots WHERE product_id = %s AND capture_time = %s"
                cursor.execute(query, (product_id, capture_time))
                count = cursor.fetchone()[0]
                return count > 0
        except Exception as e:
            logger.error(f"檢查銷售快照存在時發生錯誤: {e}")
            return False

    def get_existing_comment_ids(self, keyword: str = None) -> set:
        try:
            with self.db.cursor() as cursor:
                if keyword:
                    query = '''
                        SELECT DISTINCT pc.comment_id 
                        FROM product_comments pc
                        JOIN products p ON pc.product_id = p.id
                        WHERE p.keyword = %s
                    '''
                    cursor.execute(query, (keyword,))
                else:
                    query = "SELECT DISTINCT comment_id FROM product_comments"
                    cursor.execute(query)
                comment_ids = {row[0] for row in cursor.fetchall()}
                logger.info(f"取得 {len(comment_ids)} 個已存在的評論ID")
                return comment_ids
        except Exception as e:
            logger.error(f"取得已存在評論ID時發生錯誤: {e}")
            return set()

    def get_existing_snapshot_keys(self, keyword: str = None) -> set:
        try:
            with self.db.cursor() as cursor:
                if keyword:
                    query = '''
                        SELECT DISTINCT ss.product_id, ss.capture_time
                        FROM sales_snapshots ss
                        JOIN products p ON ss.product_id = p.id
                        WHERE p.keyword = %s
                    '''
                    cursor.execute(query, (keyword,))
                else:
                    query = "SELECT DISTINCT product_id, capture_time FROM sales_snapshots"
                    cursor.execute(query)
                snapshot_keys = {(row[0], row[1]) for row in cursor.fetchall()}
                logger.info(f"取得 {len(snapshot_keys)} 個已存在的銷售快照")
                return snapshot_keys
        except Exception as e:
            logger.error(f"取得已存在銷售快照時發生錯誤: {e}")
            return set()

    def batch_insert_products(self, products_data: List[Tuple]) -> int:
        if not products_data:
            return 0
        try:
            with self.db.cursor() as cursor:
                query = '''
                    INSERT INTO products (id, name, price, product_link, keyword, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        price = EXCLUDED.price,
                        product_link = EXCLUDED.product_link,
                        keyword = EXCLUDED.keyword,
                        updated_at = NOW()
                '''
                cursor.executemany(query, products_data)
                self.db.commit()
                inserted_count = cursor.rowcount
                logger.info(f"批次插入商品資料: {inserted_count} 筆")
                return inserted_count
        except Exception as e:
            logger.error(f"批次插入商品資料時發生錯誤: {e}")
            self.db.rollback()
            return 0

    def batch_insert_snapshots(self, snapshots_data: List[Tuple]) -> int:
        if not snapshots_data:
            return 0
        try:
            with self.db.cursor() as cursor:
                query = '''
                    INSERT INTO sales_snapshots (product_id, sales_count, sales_unit, capture_time, created_at)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON CONFLICT (product_id, capture_time) DO UPDATE SET
                        sales_count = EXCLUDED.sales_count,
                        sales_unit = EXCLUDED.sales_unit,
                        created_at = NOW()
                '''
                cursor.executemany(query, snapshots_data)
                self.db.commit()
                inserted_count = cursor.rowcount
                logger.info(f"批次插入銷售快照: {inserted_count} 筆")
                return inserted_count
        except Exception as e:
            logger.error(f"批次插入銷售快照時發生錯誤: {e}")
            self.db.rollback()
            return 0

    def batch_insert_comments(self, comments_data: List[Tuple]) -> int:
        if not comments_data:
            return 0
        try:
            with self.db.cursor() as cursor:
                query = '''
                    INSERT INTO product_comments 
                    (comment_id, product_id, comment_text, customer_name, comment_date,
                     goods_type, image_urls, is_like, is_show_like, like_count,
                     reply_content, reply_date, score, video_thumbnail_img, video_url,
                     capture_time, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (comment_id) DO UPDATE SET
                        comment_text = EXCLUDED.comment_text,
                        customer_name = EXCLUDED.customer_name,
                        comment_date = EXCLUDED.comment_date,
                        goods_type = EXCLUDED.goods_type,
                        image_urls = EXCLUDED.image_urls,
                        is_like = EXCLUDED.is_like,
                        is_show_like = EXCLUDED.is_show_like,
                        like_count = EXCLUDED.like_count,
                        reply_content = EXCLUDED.reply_content,
                        reply_date = EXCLUDED.reply_date,
                        score = EXCLUDED.score,
                        video_thumbnail_img = EXCLUDED.video_thumbnail_img,
                        video_url = EXCLUDED.video_url,
                        capture_time = EXCLUDED.capture_time,
                        created_at = NOW()
                '''
                cursor.executemany(query, comments_data)
                self.db.commit()
                inserted_count = cursor.rowcount
                logger.info(f"批次插入評論資料: {inserted_count} 筆")
                return inserted_count
        except Exception as e:
            logger.error(f"批次插入評論資料時發生錯誤: {e}")
            self.db.rollback()
            return 0 