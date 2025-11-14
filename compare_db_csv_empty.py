#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare DB Empty Fields with CSV
自動比對所有表有空欄位的資料，並與對應CSV做比對，輸出比對報告。
"""

from dotenv import load_dotenv
load_dotenv()

import os
import argparse
import logging
import pandas as pd
from typing import Optional, List

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database import DatabaseConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_available_keywords() -> List[str]:
    """自動檢測所有可用的關鍵字"""
    keywords = set()
    
    # 檢查根目錄的商品資料檔案
    for file in os.listdir('.'):
        if file.endswith('_商品資料.csv'):
            keyword = file.replace('_商品資料.csv', '')
            keywords.add(keyword)
    
    # 檢查 crawler 目錄的檔案
    if os.path.exists('crawler'):
        for file in os.listdir('crawler'):
            if file.endswith('_商品銷售快照.csv'):
                keyword = file.replace('_商品銷售快照.csv', '')
                keywords.add(keyword)
            elif file.endswith('_商品留言資料_') and file.endswith('.csv'):
                # 提取關鍵字（去掉時間戳）
                parts = file.replace('_商品留言資料_', '_').replace('.csv', '').split('_')
                if len(parts) >= 2:
                    keyword = parts[0]
                    keywords.add(keyword)
    
    return sorted(list(keywords))

def select_keyword(keywords: List[str]) -> str:
    """讓使用者選擇關鍵字"""
    print("\n=== 可用的關鍵字 ===")
    for i, keyword in enumerate(keywords, 1):
        print(f"{i:2d}. {keyword}")
    
    while True:
        try:
            choice = input(f"\n請選擇關鍵字 (1-{len(keywords)}) 或直接輸入關鍵字: ").strip()
            
            # 如果輸入的是數字
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(keywords):
                    return keywords[idx]
                else:
                    print(f"請輸入 1-{len(keywords)} 之間的數字")
            else:
                # 如果直接輸入關鍵字
                if choice in keywords:
                    return choice
                else:
                    print(f"找不到關鍵字 '{choice}'，請重新選擇")
        except (ValueError, KeyboardInterrupt):
            print("請重新選擇")

def get_db_connection():
    db_config = DatabaseConfig()
    return db_config.get_connection()

def fetch_empty_products(conn, keyword: str):
    # 查詢 products 表有空欄位的記錄
    query = '''
        SELECT id, name, price, product_link, keyword
        FROM products
        WHERE keyword = %s AND (
            name IS NULL OR name = '' OR name ~ '^\\s*$' OR
            price IS NULL OR product_link IS NULL OR product_link = '' OR product_link ~ '^\\s*$'
        )
    '''
    df = pd.read_sql_query(query, conn, params=[keyword])
    return df

def fetch_empty_snapshots(conn, keyword: str):
    # 查詢 sales_snapshots 表有空欄位的記錄
    query = '''
        SELECT s.product_id, s.sales_count, s.sales_unit, s.capture_time, p.keyword
        FROM sales_snapshots s
        JOIN products p ON s.product_id = p.id
        WHERE p.keyword = %s AND (
            s.sales_count IS NULL OR s.sales_unit IS NULL OR s.sales_unit = '' OR s.sales_unit ~ '^\\s*$' OR
            s.capture_time IS NULL
        )
    '''
    df = pd.read_sql_query(query, conn, params=[keyword])
    return df

def fetch_empty_comments(conn, keyword: str):
    # 查詢 product_comments 表有空欄位的記錄
    query = '''
        SELECT c.comment_id, c.product_id, c.comment_text, c.customer_name, c.comment_date, c.score, c.capture_time, p.keyword
        FROM product_comments c
        JOIN products p ON c.product_id = p.id
        WHERE p.keyword = %s AND (
            c.comment_id IS NULL OR c.comment_id = '' OR c.comment_id ~ '^\\s*$' OR
            c.product_id IS NULL OR
            c.comment_text IS NULL OR c.comment_text = '' OR c.comment_text ~ '^\\s*$' OR
            c.customer_name IS NULL OR c.customer_name = '' OR c.customer_name ~ '^\\s*$' OR
            c.comment_date IS NULL OR
            c.score IS NULL OR
            c.capture_time IS NULL
        )
    '''
    df = pd.read_sql_query(query, conn, params=[keyword])
    return df

def find_csv_file(keyword: str, table: str) -> Optional[str]:
    # 根據 crawler.py 的命名規則尋找對應CSV
    if table == 'products':
        fname = f'{keyword}_商品資料.csv'
        if os.path.exists(fname):
            return fname
    elif table == 'sales_snapshots':
        fname = f'crawler/{keyword}_商品銷售快照.csv'
        if os.path.exists(fname):
            return fname
    elif table == 'product_comments':
        # 找最新的留言檔案
        files = [f for f in os.listdir('crawler') if f.startswith(f'{keyword}_商品留言資料_') and f.endswith('.csv')]
        if files:
            files.sort(reverse=True)
            return os.path.join('crawler', files[0])
    return None

def compare_and_report(df_db, df_csv, table: str, keyword: str, output_dir: str):
    # 根據主鍵合併，標註來源行數
    if table == 'products':
        merge_col = '商品ID'
        db_col = 'id'
    elif table == 'sales_snapshots':
        merge_col = ['商品ID', '擷取時間']
        db_col = ['product_id', 'capture_time']
    elif table == 'product_comments':
        merge_col = '留言ID'
        db_col = 'comment_id'
    else:
        return

    # 轉型以便比對
    if isinstance(merge_col, list):
        for c, d in zip(merge_col, db_col):
            df_db[d] = df_db[d].astype(str)
            df_csv[c] = df_csv[c].astype(str)
    else:
        df_db[db_col] = df_db[db_col].astype(str)
        df_csv[merge_col] = df_csv[merge_col].astype(str)

    # 加入來源行數
    df_csv['_csv_row'] = df_csv.index + 2  # +2: header+1-based
    # 合併
    df_merged = pd.merge(df_db, df_csv, left_on=db_col, right_on=merge_col, how='left', suffixes=('_db', '_csv'))
    # 輸出
    outname = os.path.join(output_dir, f'{table}_empty_compare_{keyword}.csv')
    df_merged.to_csv(outname, encoding='utf-8-sig', index=False)
    print(f'✅ {table} 比對報告已輸出: {outname}')

def main():
    parser = argparse.ArgumentParser(description='比對DB空欄位與CSV原始資料')
    parser.add_argument('--keyword', type=str, help='指定關鍵字（如益生菌）')
    parser.add_argument('--output-dir', type=str, default='compare_report', help='報告輸出資料夾')
    args = parser.parse_args()

    # 獲取可用關鍵字
    keywords = get_available_keywords()
    
    if not keywords:
        print("❌ 找不到任何關鍵字的CSV檔案")
        return
    
    # 選擇關鍵字
    if args.keyword:
        if args.keyword not in keywords:
            print(f"❌ 找不到關鍵字 '{args.keyword}' 的CSV檔案")
            print(f"可用的關鍵字: {', '.join(keywords)}")
            return
        keyword = args.keyword
    else:
        keyword = select_keyword(keywords)
    
    print(f"\n🔍 開始比對關鍵字: {keyword}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    conn = get_db_connection()

    # 1. products
    print('🔍 查詢 products 空欄位...')
    df_db = fetch_empty_products(conn, keyword)
    csv_path = find_csv_file(keyword, 'products')
    if csv_path and not df_db.empty:
        df_csv = pd.read_csv(csv_path, dtype=str)
        compare_and_report(df_db, df_csv, 'products', keyword, args.output_dir)
    else:
        print('products 無空欄位或找不到對應CSV')

    # 2. sales_snapshots
    print('🔍 查詢 sales_snapshots 空欄位...')
    df_db = fetch_empty_snapshots(conn, keyword)
    csv_path = find_csv_file(keyword, 'sales_snapshots')
    if csv_path and not df_db.empty:
        df_csv = pd.read_csv(csv_path, dtype=str)
        compare_and_report(df_db, df_csv, 'sales_snapshots', keyword, args.output_dir)
    else:
        print('sales_snapshots 無空欄位或找不到對應CSV')

    # 3. product_comments
    print('🔍 查詢 product_comments 空欄位...')
    df_db = fetch_empty_comments(conn, keyword)
    csv_path = find_csv_file(keyword, 'product_comments')
    if csv_path and not df_db.empty:
        df_csv = pd.read_csv(csv_path, dtype=str)
        compare_and_report(df_db, df_csv, 'product_comments', keyword, args.output_dir)
    else:
        print('product_comments 無空欄位或找不到對應CSV')

    conn.close()
    print('🎉 比對完成！')

if __name__ == '__main__':
    main() 