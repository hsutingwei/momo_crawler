import mysql.connector
from mysql.connector import Error
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseOperations:
    """資料庫操作類別"""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def insert_product(self, product_id: int, name: str, price: float, 
                      product_link: str, keyword: str) -> bool:
        """
        插入商品資料
        
        Args:
            product_id: 商品ID
            name: 商品名稱
            price: 價格
            product_link: 商品連結
            keyword: 關鍵字
            
        Returns:
            是否成功插入
        """
        try:
            cursor = self.db.cursor()
            
            query = """
                INSERT INTO products (id, name, price, product_link, keyword, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                ON DUPLICATE KEY UPDATE
                name = VALUES(name),
                price = VALUES(price),
                product_link = VALUES(product_link),
                keyword = VALUES(keyword),
                updated_at = NOW()
            """
            
            cursor.execute(query, (product_id, name, price, product_link, keyword))
            self.db.commit()
            
            logger.debug(f"成功插入/更新商品: {product_id}")
            return True
            
        except Error as e:
            logger.error(f"插入商品資料時發生錯誤: {e}")
            self.db.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
    
    def insert_sales_snapshot(self, product_id: int, sales_count: int, 
                            sales_unit: str, capture_time: datetime) -> bool:
        """
        插入銷售快照資料
        
        Args:
            product_id: 商品ID
            sales_count: 銷售數量
            sales_unit: 銷售單位
            capture_time: 擷取時間
            
        Returns:
            是否成功插入
        """
        try:
            cursor = self.db.cursor()
            
            query = """
                INSERT INTO sales_snapshots (product_id, sales_count, sales_unit, capture_time, created_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON DUPLICATE KEY UPDATE
                sales_count = VALUES(sales_count),
                sales_unit = VALUES(sales_unit),
                created_at = NOW()
            """
            
            cursor.execute(query, (product_id, sales_count, sales_unit, capture_time))
            self.db.commit()
            
            logger.debug(f"成功插入銷售快照: {product_id} - {capture_time}")
            return True
            
        except Error as e:
            logger.error(f"插入銷售快照時發生錯誤: {e}")
            self.db.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
    
    def insert_product_comment(self, comment_id: str, product_id: int, comment_text: str,
                             customer_name: str, comment_date: datetime, goods_type: str,
                             image_urls: str, is_like: bool, is_show_like: bool,
                             like_count: int, reply_content: str, reply_date: datetime,
                             score: float, video_thumbnail_img: str, video_url: str,
                             capture_time: datetime) -> bool:
        """
        插入商品評論資料
        
        Args:
            comment_id: 評論ID
            product_id: 商品ID
            comment_text: 評論內容
            customer_name: 客戶名稱
            comment_date: 評論日期
            goods_type: 商品類型
            image_urls: 圖片URLs (JSON格式)
            is_like: 是否按讚
            is_show_like: 是否顯示按讚
            like_count: 按讚數
            reply_content: 回覆內容
            reply_date: 回覆日期
            score: 評分
            video_thumbnail_img: 影片縮圖
            video_url: 影片URL
            capture_time: 擷取時間
            
        Returns:
            是否成功插入
        """
        try:
            cursor = self.db.cursor()
            
            query = """
                INSERT INTO product_comments 
                (comment_id, product_id, comment_text, customer_name, comment_date,
                 goods_type, image_urls, is_like, is_show_like, like_count,
                 reply_content, reply_date, score, video_thumbnail_img, video_url,
                 capture_time, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON DUPLICATE KEY UPDATE
                comment_text = VALUES(comment_text),
                customer_name = VALUES(customer_name),
                comment_date = VALUES(comment_date),
                goods_type = VALUES(goods_type),
                image_urls = VALUES(image_urls),
                is_like = VALUES(is_like),
                is_show_like = VALUES(is_show_like),
                like_count = VALUES(like_count),
                reply_content = VALUES(reply_content),
                reply_date = VALUES(reply_date),
                score = VALUES(score),
                video_thumbnail_img = VALUES(video_thumbnail_img),
                video_url = VALUES(video_url),
                capture_time = VALUES(capture_time),
                created_at = NOW()
            """
            
            cursor.execute(query, (
                comment_id, product_id, comment_text, customer_name, comment_date,
                goods_type, image_urls, is_like, is_show_like, like_count,
                reply_content, reply_date, score, video_thumbnail_img, video_url,
                capture_time
            ))
            
            self.db.commit()
            logger.debug(f"成功插入評論: {comment_id}")
            return True
            
        except Error as e:
            logger.error(f"插入評論資料時發生錯誤: {e}")
            self.db.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
    
    def check_product_exists(self, product_id: int) -> bool:
        """
        檢查商品是否存在
        
        Args:
            product_id: 商品ID
            
        Returns:
            是否存在
        """
        try:
            cursor = self.db.cursor()
            
            query = "SELECT COUNT(*) FROM products WHERE id = %s"
            cursor.execute(query, (product_id,))
            
            count = cursor.fetchone()[0]
            return count > 0
            
        except Error as e:
            logger.error(f"檢查商品存在時發生錯誤: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
    
    def check_comment_exists(self, comment_id: str) -> bool:
        """
        檢查評論是否存在
        
        Args:
            comment_id: 評論ID
            
        Returns:
            是否存在
        """
        try:
            cursor = self.db.cursor()
            
            query = "SELECT COUNT(*) FROM product_comments WHERE comment_id = %s"
            cursor.execute(query, (comment_id,))
            
            count = cursor.fetchone()[0]
            return count > 0
            
        except Error as e:
            logger.error(f"檢查評論存在時發生錯誤: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
    
    def check_snapshot_exists(self, product_id: int, capture_time: datetime) -> bool:
        """
        檢查銷售快照是否存在
        
        Args:
            product_id: 商品ID
            capture_time: 擷取時間
            
        Returns:
            是否存在
        """
        try:
            cursor = self.db.cursor()
            
            query = "SELECT COUNT(*) FROM sales_snapshots WHERE product_id = %s AND capture_time = %s"
            cursor.execute(query, (product_id, capture_time))
            
            count = cursor.fetchone()[0]
            return count > 0
            
        except Error as e:
            logger.error(f"檢查銷售快照存在時發生錯誤: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
    
    def get_existing_comment_ids(self, keyword: str = None) -> set:
        """
        取得已存在的評論ID集合
        
        Args:
            keyword: 關鍵字篩選
            
        Returns:
            評論ID集合
        """
        try:
            cursor = self.db.cursor()
            
            if keyword:
                query = """
                    SELECT DISTINCT pc.comment_id 
                    FROM product_comments pc
                    JOIN products p ON pc.product_id = p.id
                    WHERE p.keyword = %s
                """
                cursor.execute(query, (keyword,))
            else:
                query = "SELECT DISTINCT comment_id FROM product_comments"
                cursor.execute(query)
            
            comment_ids = {row[0] for row in cursor.fetchall()}
            logger.info(f"取得 {len(comment_ids)} 個已存在的評論ID")
            return comment_ids
            
        except Error as e:
            logger.error(f"取得已存在評論ID時發生錯誤: {e}")
            return set()
        finally:
            if cursor:
                cursor.close()
    
    def get_existing_snapshot_keys(self, keyword: str = None) -> set:
        """
        取得已存在的銷售快照鍵值集合 (product_id + capture_time)
        
        Args:
            keyword: 關鍵字篩選
            
        Returns:
            快照鍵值集合
        """
        try:
            cursor = self.db.cursor()
            
            if keyword:
                query = """
                    SELECT DISTINCT ss.product_id, ss.capture_time
                    FROM sales_snapshots ss
                    JOIN products p ON ss.product_id = p.id
                    WHERE p.keyword = %s
                """
                cursor.execute(query, (keyword,))
            else:
                query = "SELECT DISTINCT product_id, capture_time FROM sales_snapshots"
                cursor.execute(query)
            
            snapshot_keys = {(row[0], row[1]) for row in cursor.fetchall()}
            logger.info(f"取得 {len(snapshot_keys)} 個已存在的銷售快照")
            return snapshot_keys
            
        except Error as e:
            logger.error(f"取得已存在銷售快照時發生錯誤: {e}")
            return set()
        finally:
            if cursor:
                cursor.close()
    
    def batch_insert_products(self, products_data: List[Tuple]) -> int:
        """
        批次插入商品資料
        
        Args:
            products_data: 商品資料列表 [(product_id, name, price, product_link, keyword), ...]
            
        Returns:
            成功插入的數量
        """
        if not products_data:
            return 0
        
        try:
            cursor = self.db.cursor()
            
            query = """
                INSERT INTO products (id, name, price, product_link, keyword, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                ON DUPLICATE KEY UPDATE
                name = VALUES(name),
                price = VALUES(price),
                product_link = VALUES(product_link),
                keyword = VALUES(keyword),
                updated_at = NOW()
            """
            
            cursor.executemany(query, products_data)
            self.db.commit()
            
            inserted_count = cursor.rowcount
            logger.info(f"批次插入商品資料: {inserted_count} 筆")
            return inserted_count
            
        except Error as e:
            logger.error(f"批次插入商品資料時發生錯誤: {e}")
            self.db.rollback()
            return 0
        finally:
            if cursor:
                cursor.close()
    
    def batch_insert_snapshots(self, snapshots_data: List[Tuple]) -> int:
        """
        批次插入銷售快照資料
        
        Args:
            snapshots_data: 快照資料列表 [(product_id, sales_count, sales_unit, capture_time), ...]
            
        Returns:
            成功插入的數量
        """
        if not snapshots_data:
            return 0
        
        try:
            cursor = self.db.cursor()
            
            query = """
                INSERT INTO sales_snapshots (product_id, sales_count, sales_unit, capture_time, created_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON DUPLICATE KEY UPDATE
                sales_count = VALUES(sales_count),
                sales_unit = VALUES(sales_unit),
                created_at = NOW()
            """
            
            cursor.executemany(query, snapshots_data)
            self.db.commit()
            
            inserted_count = cursor.rowcount
            logger.info(f"批次插入銷售快照: {inserted_count} 筆")
            return inserted_count
            
        except Error as e:
            logger.error(f"批次插入銷售快照時發生錯誤: {e}")
            self.db.rollback()
            return 0
        finally:
            if cursor:
                cursor.close()
    
    def batch_insert_comments(self, comments_data: List[Tuple]) -> int:
        """
        批次插入評論資料
        
        Args:
            comments_data: 評論資料列表
            
        Returns:
            成功插入的數量
        """
        if not comments_data:
            return 0
        
        try:
            cursor = self.db.cursor()
            
            query = """
                INSERT INTO product_comments 
                (comment_id, product_id, comment_text, customer_name, comment_date,
                 goods_type, image_urls, is_like, is_show_like, like_count,
                 reply_content, reply_date, score, video_thumbnail_img, video_url,
                 capture_time, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON DUPLICATE KEY UPDATE
                comment_text = VALUES(comment_text),
                customer_name = VALUES(customer_name),
                comment_date = VALUES(comment_date),
                goods_type = VALUES(goods_type),
                image_urls = VALUES(image_urls),
                is_like = VALUES(is_like),
                is_show_like = VALUES(is_show_like),
                like_count = VALUES(like_count),
                reply_content = VALUES(reply_content),
                reply_date = VALUES(reply_date),
                score = VALUES(score),
                video_thumbnail_img = VALUES(video_thumbnail_img),
                video_url = VALUES(video_url),
                capture_time = VALUES(capture_time),
                created_at = NOW()
            """
            
            cursor.executemany(query, comments_data)
            self.db.commit()
            
            inserted_count = cursor.rowcount
            logger.info(f"批次插入評論資料: {inserted_count} 筆")
            return inserted_count
            
        except Error as e:
            logger.error(f"批次插入評論資料時發生錯誤: {e}")
            self.db.rollback()
            return 0
        finally:
            if cursor:
                cursor.close() 