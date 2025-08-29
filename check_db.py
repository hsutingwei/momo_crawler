#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Checker
檢查資料庫中的數據
"""

from dotenv import load_dotenv
load_dotenv()

import os
import argparse
import logging
from datetime import datetime
from typing import Optional

# 添加專案路徑
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database import DatabaseConfig

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseChecker:
    """資料庫檢查器"""
    
    def __init__(self):
        self.db_config = DatabaseConfig()
        self.connection = None
        
    def connect(self):
        """連接資料庫"""
        try:
            self.connection = self.db_config.get_connection()
            logger.info("資料庫連接成功")
        except Exception as e:
            logger.error(f"資料庫連接失敗: {e}")
            raise
    
    def close(self):
        """關閉資料庫連接"""
        if self.connection:
            self.connection.close()
            logger.info("資料庫連接已關閉")
    
    def get_table_count(self, table_name: str) -> int:
        """獲取表的記錄數"""
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            return count
        finally:
            cursor.close()
    
    def get_table_info(self):
        """獲取所有表的基本資訊"""
        cursor = self.connection.cursor()
        try:
            # 獲取所有表名
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            tables = cursor.fetchall()
            
            print("\n=== 資料庫表資訊 ===")
            print(f"{'表名':<20} {'記錄數':<10} {'說明'}")
            print("-" * 50)
            
            for table in tables:
                table_name = table[0]
                count = self.get_table_count(table_name)
                
                # 根據表名提供說明
                description = {
                    'products': '商品資料',
                    'sales_snapshots': '銷售快照',
                    'product_comments': '商品評論',
                    'data_validation_errors': '資料驗證錯誤',
                    'file_sync_logs': '檔案同步記錄'
                }.get(table_name, '未知表')
                
                print(f"{table_name:<20} {count:<10} {description}")
                
        finally:
            cursor.close()
    
    def show_products(self, limit: int = 10, keyword: Optional[str] = None):
        """顯示商品資料"""
        cursor = self.connection.cursor()
        try:
            if keyword:
                cursor.execute("""
                    SELECT id, name, price, keyword, created_at 
                    FROM products 
                    WHERE keyword = %s 
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (keyword, limit))
            else:
                cursor.execute("""
                    SELECT id, name, price, keyword, created_at 
                    FROM products 
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (limit,))
            
            products = cursor.fetchall()
            
            print(f"\n=== 商品資料 (顯示前 {len(products)} 筆) ===")
            if keyword:
                print(f"關鍵字篩選: {keyword}")
            print(f"{'商品ID':<15} {'價格':<10} {'關鍵字':<15} {'建立時間':<20}")
            print("-" * 70)
            
            for product in products:
                product_id, name, price, kw, created_at = product
                # 截斷商品名稱以適應顯示
                display_name = name[:30] + "..." if len(name) > 30 else name
                print(f"{product_id:<15} {price:<10} {kw:<15} {created_at}")
                
        finally:
            cursor.close()
    
    def show_snapshots(self, limit: int = 10, keyword: Optional[str] = None):
        """顯示銷售快照資料"""
        cursor = self.connection.cursor()
        try:
            if keyword:
                cursor.execute("""
                    SELECT s.id, s.product_id, p.name, s.sales_count, s.sales_unit, s.capture_time
                    FROM sales_snapshots s
                    JOIN products p ON s.product_id = p.id
                    WHERE p.keyword = %s
                    ORDER BY s.capture_time DESC 
                    LIMIT %s
                """, (keyword, limit))
            else:
                cursor.execute("""
                    SELECT s.id, s.product_id, p.name, s.sales_count, s.sales_unit, s.capture_time
                    FROM sales_snapshots s
                    JOIN products p ON s.product_id = p.id
                    ORDER BY s.capture_time DESC 
                    LIMIT %s
                """, (limit,))
            
            snapshots = cursor.fetchall()
            
            print(f"\n=== 銷售快照資料 (顯示前 {len(snapshots)} 筆) ===")
            if keyword:
                print(f"關鍵字篩選: {keyword}")
            print(f"{'快照ID':<10} {'商品ID':<15} {'銷售數量':<10} {'單位':<8} {'擷取時間':<20}")
            print("-" * 70)
            
            for snapshot in snapshots:
                snapshot_id, product_id, name, sales_count, sales_unit, capture_time = snapshot
                display_name = name[:20] + "..." if len(name) > 20 else name
                print(f"{snapshot_id:<10} {product_id:<15} {sales_count:<10} {sales_unit:<8} {capture_time}")
                
        finally:
            cursor.close()
    
    def show_comments(self, limit: int = 10, keyword: Optional[str] = None):
        """顯示評論資料"""
        cursor = self.connection.cursor()
        try:
            if keyword:
                cursor.execute("""
                    SELECT c.id, c.comment_id, c.product_id, p.name, c.customer_name, c.score, c.comment_date
                    FROM product_comments c
                    JOIN products p ON c.product_id = p.id
                    WHERE p.keyword = %s
                    ORDER BY c.comment_date DESC 
                    LIMIT %s
                """, (keyword, limit))
            else:
                cursor.execute("""
                    SELECT c.id, c.comment_id, c.product_id, p.name, c.customer_name, c.score, c.comment_date
                    FROM product_comments c
                    JOIN products p ON c.product_id = p.id
                    ORDER BY c.comment_date DESC 
                    LIMIT %s
                """, (limit,))
            
            comments = cursor.fetchall()
            
            print(f"\n=== 評論資料 (顯示前 {len(comments)} 筆) ===")
            if keyword:
                print(f"關鍵字篩選: {keyword}")
            print(f"{'評論ID':<10} {'商品ID':<15} {'客戶名稱':<15} {'評分':<5} {'評論時間':<20}")
            print("-" * 70)
            
            for comment in comments:
                comment_id, comment_id_str, product_id, name, customer_name, score, comment_date = comment
                display_name = name[:20] + "..." if len(name) > 20 else name
                print(f"{comment_id:<10} {product_id:<15} {customer_name:<15} {score:<5} {comment_date}")
                
        finally:
            cursor.close()
    
    def show_keywords(self):
        """顯示所有關鍵字及其統計"""
        cursor = self.connection.cursor()
        try:
            cursor.execute("""
                SELECT 
                    keyword,
                    COUNT(*) as product_count,
                    MIN(created_at) as first_created,
                    MAX(created_at) as last_created
                FROM products 
                GROUP BY keyword 
                ORDER BY product_count DESC
            """)
            
            keywords = cursor.fetchall()
            
            print("\n=== 關鍵字統計 ===")
            print(f"{'關鍵字':<20} {'商品數':<10} {'首次建立':<20} {'最後更新':<20}")
            print("-" * 80)
            
            for keyword in keywords:
                kw, count, first_created, last_created = keyword
                print(f"{kw:<20} {count:<10} {first_created:<20} {last_created:<20}")
                
        finally:
            cursor.close()
    
    def show_sync_status(self, keyword: Optional[str] = None):
        """顯示檔案同步狀態"""
        cursor = self.connection.cursor()
        try:
            if keyword:
                cursor.execute("""
                    SELECT keyword, file_type, file_path, sync_status, 
                           total_records, valid_records, error_records, 
                           sync_start_time, sync_end_time
                    FROM file_sync_logs 
                    WHERE keyword = %s
                    ORDER BY created_at DESC
                """, (keyword,))
            else:
                cursor.execute("""
                    SELECT keyword, file_type, file_path, sync_status, 
                           total_records, valid_records, error_records, 
                           sync_start_time, sync_end_time
                    FROM file_sync_logs 
                    ORDER BY created_at DESC
                    LIMIT 20
                """)
            
            logs = cursor.fetchall()
            
            print(f"\n=== 檔案同步狀態 ===")
            if keyword:
                print(f"關鍵字篩選: {keyword}")
            print(f"{'關鍵字':<15} {'類型':<10} {'狀態':<10} {'總數':<8} {'有效':<8} {'錯誤':<8}")
            print("-" * 70)
            
            for log in logs:
                kw, file_type, file_path, status, total, valid, error, start_time, end_time = log
                # 只顯示檔案名，不顯示完整路徑
                filename = file_path.split('/')[-1] if '/' in file_path else file_path
                filename = filename[:20] + "..." if len(filename) > 20 else filename
                print(f"{kw:<15} {file_type:<10} {status:<10} {total:<8} {valid:<8} {error:<8}")
                
        finally:
            cursor.close()
    
    def show_errors(self, limit: int = 10, keyword: Optional[str] = None):
        """顯示資料驗證錯誤"""
        cursor = self.connection.cursor()
        try:
            if keyword:
                cursor.execute("""
                    SELECT file_path, data_type, field_name, error_type, error_message, created_at
                    FROM data_validation_errors 
                    WHERE keyword = %s
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (keyword, limit))
            else:
                cursor.execute("""
                    SELECT file_path, data_type, field_name, error_type, error_message, created_at
                    FROM data_validation_errors 
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (limit,))
            
            errors = cursor.fetchall()
            
            print(f"\n=== 資料驗證錯誤 (顯示前 {len(errors)} 筆) ===")
            if keyword:
                print(f"關鍵字篩選: {keyword}")
            print(f"{'檔案':<30} {'類型':<10} {'欄位':<15} {'錯誤類型':<15} {'建立時間':<20}")
            print("-" * 100)
            
            for error in errors:
                file_path, data_type, field_name, error_type, error_message, created_at = error
                # 只顯示檔案名
                filename = file_path.split('/')[-1] if '/' in file_path else file_path
                filename = filename[:25] + "..." if len(filename) > 25 else filename
                print(f"{filename:<30} {data_type:<10} {field_name:<15} {error_type:<15} {created_at}")
                
        finally:
            cursor.close()

def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='Database Checker - 檢查資料庫中的數據')
    parser.add_argument('--table-info', action='store_true', help='顯示所有表的基本資訊')
    parser.add_argument('--products', action='store_true', help='顯示商品資料')
    parser.add_argument('--snapshots', action='store_true', help='顯示銷售快照資料')
    parser.add_argument('--comments', action='store_true', help='顯示評論資料')
    parser.add_argument('--keywords', action='store_true', help='顯示關鍵字統計')
    parser.add_argument('--sync-status', action='store_true', help='顯示檔案同步狀態')
    parser.add_argument('--errors', action='store_true', help='顯示資料驗證錯誤')
    parser.add_argument('--keyword', type=str, help='指定關鍵字篩選')
    parser.add_argument('--limit', type=int, default=10, help='顯示記錄數限制 (預設: 10)')
    parser.add_argument('--all', action='store_true', help='顯示所有資訊')
    
    args = parser.parse_args()
    
    try:
        checker = DatabaseChecker()
        checker.connect()
        
        # 如果沒有指定任何選項，顯示表資訊
        if not any([args.table_info, args.products, args.snapshots, args.comments, 
                   args.keywords, args.sync_status, args.errors, args.all]):
            args.table_info = True
        
        if args.all:
            args.table_info = True
            args.products = True
            args.snapshots = True
            args.comments = True
            args.keywords = True
            args.sync_status = True
            args.errors = True
        
        if args.table_info:
            checker.get_table_info()
        
        if args.keywords:
            checker.show_keywords()
        
        if args.products:
            checker.show_products(args.limit, args.keyword)
        
        if args.snapshots:
            checker.show_snapshots(args.limit, args.keyword)
        
        if args.comments:
            checker.show_comments(args.limit, args.keyword)
        
        if args.sync_status:
            checker.show_sync_status(args.keyword)
        
        if args.errors:
            checker.show_errors(args.limit, args.keyword)
        
    except Exception as e:
        logger.error(f"程式執行失敗: {e}")
        sys.exit(1)
    finally:
        checker.close()

if __name__ == "__main__":
    main()