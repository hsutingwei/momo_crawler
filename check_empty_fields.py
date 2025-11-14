#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empty Fields Checker
檢查資料庫中各表的空欄位
"""

from dotenv import load_dotenv
load_dotenv()

import os
import argparse
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple

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

class EmptyFieldsChecker:
    """空欄位檢查器"""
    
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
    
    def get_table_columns(self, table_name: str) -> List[str]:
        """獲取表的所有欄位名稱"""
        cursor = self.connection.cursor()
        try:
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s AND table_schema = 'public'
                ORDER BY ordinal_position
            """, (table_name,))
            
            columns = [row[0] for row in cursor.fetchall()]
            return columns
        finally:
            cursor.close()
    
    def check_empty_fields_in_table(self, table_name: str, keyword: Optional[str] = None) -> Dict:
        """檢查指定表中的空欄位"""
        cursor = self.connection.cursor()
        try:
            # 獲取表的所有欄位
            columns = self.get_table_columns(table_name)
            
            # 排除主鍵和自動生成的欄位
            exclude_columns = ['id', 'created_at', 'updated_at']
            check_columns = [col for col in columns if col not in exclude_columns]
            
            results = {}
            
            for column in check_columns:
                # 構建查詢條件
                where_clause = ""
                params = []
                
                if keyword and 'keyword' in columns:
                    where_clause = "WHERE keyword = %s"
                    params.append(keyword)
                
                # 檢查空值
                if where_clause:
                    null_query = f"""
                        SELECT COUNT(*) 
                        FROM {table_name} 
                        {where_clause} AND {column} IS NULL
                    """
                else:
                    null_query = f"""
                        SELECT COUNT(*) 
                        FROM {table_name} 
                        WHERE {column} IS NULL
                    """
                cursor.execute(null_query, params)
                null_count = cursor.fetchone()[0]
                
                # 檢查空字串
                if where_clause:
                    empty_query = f"""
                        SELECT COUNT(*) 
                        FROM {table_name} 
                        {where_clause} AND {column} = ''
                    """
                else:
                    empty_query = f"""
                        SELECT COUNT(*) 
                        FROM {table_name} 
                        WHERE {column} = ''
                    """
                cursor.execute(empty_query, params)
                empty_count = cursor.fetchone()[0]
                
                # 檢查只包含空白字元的字串
                if where_clause:
                    whitespace_query = f"""
                        SELECT COUNT(*) 
                        FROM {table_name} 
                        {where_clause} AND {column} ~ '^\\s*$'
                    """
                else:
                    whitespace_query = f"""
                        SELECT COUNT(*) 
                        FROM {table_name} 
                        WHERE {column} ~ '^\\s*$'
                    """
                cursor.execute(whitespace_query, params)
                whitespace_count = cursor.fetchone()[0]
                
                # 總計空欄位數
                total_empty = null_count + empty_count + whitespace_count
                
                if total_empty > 0:
                    results[column] = {
                        'null_count': null_count,
                        'empty_count': empty_count,
                        'whitespace_count': whitespace_count,
                        'total_empty': total_empty
                    }
            
            return results
            
        finally:
            cursor.close()
    
    def check_products_empty_fields(self, keyword: Optional[str] = None):
        """檢查商品表的空欄位"""
        print("\n=== 商品表 (products) 空欄位檢查 ===")
        if keyword:
            print(f"關鍵字篩選: {keyword}")
        
        results = self.check_empty_fields_in_table('products', keyword)
        
        if not results:
            print("✅ 沒有發現空欄位")
            return
        
        print(f"{'欄位名稱':<20} {'NULL值':<10} {'空字串':<10} {'空白字串':<10} {'總計':<10}")
        print("-" * 70)
        
        for column, counts in results.items():
            print(f"{column:<20} {counts['null_count']:<10} {counts['empty_count']:<10} "
                  f"{counts['whitespace_count']:<10} {counts['total_empty']:<10}")
    
    def check_snapshots_empty_fields(self, keyword: Optional[str] = None):
        """檢查銷售快照表的空欄位"""
        print("\n=== 銷售快照表 (sales_snapshots) 空欄位檢查 ===")
        if keyword:
            print(f"關鍵字篩選: {keyword}")
        
        cursor = self.connection.cursor()
        try:
            # 構建查詢條件
            where_clause = ""
            params = []
            
            if keyword:
                where_clause = """
                    WHERE p.keyword = %s
                """
                params.append(keyword)
            
            # 檢查各欄位的空值
            columns_to_check = ['sales_count', 'sales_unit', 'capture_time']
            
            for column in columns_to_check:
                query = f"""
                    SELECT COUNT(*) 
                    FROM sales_snapshots s
                    JOIN products p ON s.product_id = p.id
                    {where_clause} AND s.{column} IS NULL
                """
                cursor.execute(query, params)
                null_count = cursor.fetchone()[0]
                
                if column in ['sales_count']:
                    # 檢查數值欄位
                    query = f"""
                        SELECT COUNT(*) 
                        FROM sales_snapshots s
                        JOIN products p ON s.product_id = p.id
                        {where_clause} AND (s.{column} = 0 OR s.{column} < 0)
                    """
                    cursor.execute(query, params)
                    zero_count = cursor.fetchone()[0]
                    
                    if null_count > 0 or zero_count > 0:
                        print(f"⚠️  {column}: NULL值={null_count}, 零值={zero_count}")
                else:
                    # 檢查字串欄位
                    query = f"""
                        SELECT COUNT(*) 
                        FROM sales_snapshots s
                        JOIN products p ON s.product_id = p.id
                        {where_clause} AND (s.{column} = '' OR s.{column} ~ '^\\s*$')
                    """
                    cursor.execute(query, params)
                    empty_count = cursor.fetchone()[0]
                    
                    if null_count > 0 or empty_count > 0:
                        print(f"⚠️  {column}: NULL值={null_count}, 空值={empty_count}")
            
            if not any([null_count > 0, zero_count > 0, empty_count > 0]):
                print("✅ 沒有發現空欄位")
                
        finally:
            cursor.close()
    
    def check_comments_empty_fields(self, keyword: Optional[str] = None):
        """檢查評論表的空欄位"""
        print("\n=== 評論表 (product_comments) 空欄位檢查 ===")
        if keyword:
            print(f"關鍵字篩選: {keyword}")
        
        cursor = self.connection.cursor()
        try:
            # 構建查詢條件
            where_clause = ""
            params = []
            
            if keyword:
                where_clause = """
                    WHERE p.keyword = %s
                """
                params.append(keyword)
            
            # 檢查重要欄位的空值
            important_columns = [
                'comment_id', 'product_id', 'comment_text', 'customer_name', 
                'comment_date', 'score', 'capture_time'
            ]
            
            has_empty = False
            
            for column in important_columns:
                # 檢查NULL值
                query = f"""
                    SELECT COUNT(*) 
                    FROM product_comments c
                    JOIN products p ON c.product_id = p.id
                    {where_clause} AND c.{column} IS NULL
                """
                cursor.execute(query, params)
                null_count = cursor.fetchone()[0]
                
                # 檢查空字串
                query = f"""
                    SELECT COUNT(*) 
                    FROM product_comments c
                    JOIN products p ON c.product_id = p.id
                    {where_clause} AND (c.{column} = '' OR c.{column} ~ '^\\s*$')
                """
                cursor.execute(query, params)
                empty_count = cursor.fetchone()[0]
                
                if null_count > 0 or empty_count > 0:
                    print(f"⚠️  {column}: NULL值={null_count}, 空值={empty_count}")
                    has_empty = True
            
            if not has_empty:
                print("✅ 沒有發現重要欄位為空")
                
        finally:
            cursor.close()
    
    def show_sample_empty_records(self, table_name: str, column: str, keyword: Optional[str] = None, limit: int = 5):
        """顯示空欄位的範例記錄"""
        cursor = self.connection.cursor()
        try:
            # 構建查詢條件
            where_clause = f"WHERE {column} IS NULL OR {column} = '' OR {column} ~ '^\\s*$'"
            params = []
            
            if keyword and table_name == 'products':
                where_clause += " AND keyword = %s"
                params.append(keyword)
            elif keyword and table_name in ['sales_snapshots', 'product_comments']:
                where_clause = f"""
                    WHERE ({column} IS NULL OR {column} = '' OR {column} ~ '^\\s*$')
                    AND p.keyword = %s
                """
                params.append(keyword)
            
            # 根據表名選擇要顯示的欄位
            if table_name == 'products':
                select_fields = 'id, name, price, keyword'
                query = f"SELECT {select_fields} FROM {table_name} {where_clause} LIMIT %s"
            elif table_name == 'sales_snapshots':
                select_fields = 's.id, s.product_id, p.name, s.sales_count, s.sales_unit'
                query = f"""
                    SELECT {select_fields} 
                    FROM {table_name} s
                    JOIN products p ON s.product_id = p.id
                    {where_clause} 
                    LIMIT %s
                """
            elif table_name == 'product_comments':
                select_fields = 'c.id, c.comment_id, c.product_id, c.customer_name, c.score'
                query = f"""
                    SELECT {select_fields} 
                    FROM {table_name} c
                    JOIN products p ON c.product_id = p.id
                    {where_clause} 
                    LIMIT %s
                """
            
            params.append(limit)
            cursor.execute(query, params)
            records = cursor.fetchall()
            
            if records:
                print(f"\n📋 {table_name}.{column} 空欄位範例記錄:")
                for record in records:
                    print(f"   {record}")
                    
        finally:
            cursor.close()
    
    def check_all_tables(self, keyword: Optional[str] = None, show_samples: bool = False):
        """檢查所有表的空欄位"""
        print("🔍 開始檢查所有表的空欄位...")
        
        # 檢查商品表
        self.check_products_empty_fields(keyword)
        
        # 檢查銷售快照表
        self.check_snapshots_empty_fields(keyword)
        
        # 檢查評論表
        self.check_comments_empty_fields(keyword)
        
        if show_samples:
            print("\n" + "="*50)
            print("📋 顯示空欄位範例記錄")
            print("="*50)
            
            # 這裡可以添加顯示範例記錄的邏輯
            # 例如：self.show_sample_empty_records('products', 'name', keyword)

def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='Empty Fields Checker - 檢查資料庫中的空欄位')
    parser.add_argument('--products', action='store_true', help='檢查商品表空欄位')
    parser.add_argument('--snapshots', action='store_true', help='檢查銷售快照表空欄位')
    parser.add_argument('--comments', action='store_true', help='檢查評論表空欄位')
    parser.add_argument('--keyword', type=str, help='指定關鍵字篩選')
    parser.add_argument('--samples', action='store_true', help='顯示空欄位範例記錄')
    parser.add_argument('--all', action='store_true', help='檢查所有表')
    
    args = parser.parse_args()
    
    try:
        checker = EmptyFieldsChecker()
        checker.connect()
        
        # 如果沒有指定任何選項，檢查所有表
        if not any([args.products, args.snapshots, args.comments, args.all]):
            args.all = True
        
        if args.all:
            checker.check_all_tables(args.keyword, args.samples)
        else:
            if args.products:
                checker.check_products_empty_fields(args.keyword)
            if args.snapshots:
                checker.check_snapshots_empty_fields(args.keyword)
            if args.comments:
                checker.check_comments_empty_fields(args.keyword)
        
    except Exception as e:
        logger.error(f"程式執行失敗: {e}")
        sys.exit(1)
    finally:
        checker.close()

if __name__ == "__main__":
    main() 