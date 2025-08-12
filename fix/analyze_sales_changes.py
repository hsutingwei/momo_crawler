#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sales Changes Analyzer
分析所有快照CSV檔案，找出同商品ID下銷售數量有變化的記錄
"""

import os
import pandas as pd
import argparse
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SalesChangesAnalyzer:
    """銷售變化分析器"""
    
    def __init__(self):
        self.changes_data = []
        
    def extract_number(self, sales_str) -> int:
        """從銷售字串中提取數字"""
        if pd.isna(sales_str) or sales_str == '':
            return 0
        
        import re
        # 轉換為字串
        sales_str = str(sales_str)
        
        # 移除萬字並轉換為數字
        if '萬' in sales_str:
            # 提取數字部分
            match = re.search(r'(\d+(?:\.\d+)?)', sales_str)
            if match:
                return int(float(match.group(1)) * 10000)  # 萬轉為實際數字
        else:
            # 直接提取數字
            match = re.search(r'(\d+)', sales_str)
            if match:
                return int(match.group(1))
        return 0
    
    def find_snapshot_files(self) -> List[str]:
        """尋找所有快照CSV檔案"""
        snapshot_files = []
        
        # 檢查多個可能的路徑
        possible_paths = ['crawler', '../crawler', './crawler']
        
        for path in possible_paths:
            if os.path.exists(path):
                for file in os.listdir(path):
                    if file.endswith('_商品銷售快照.csv') and not file.endswith('.bak'):
                        snapshot_files.append(os.path.join(path, file))
                break
        
        return sorted(snapshot_files)
    
    def read_csv_with_encoding(self, file_path: str) -> pd.DataFrame:
        """嘗試多種編碼讀取CSV檔案"""
        encodings = ['utf-8', 'big5', 'gbk', 'gb2312', 'utf-8-sig']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, dtype={'商品ID': str}, encoding=encoding)
                logger.info(f"成功使用 {encoding} 編碼讀取檔案: {file_path}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"使用 {encoding} 編碼讀取檔案時發生錯誤: {e}")
                continue
        
        # 如果所有編碼都失敗，嘗試不指定編碼
        try:
            df = pd.read_csv(file_path, dtype={'商品ID': str})
            logger.info(f"使用預設編碼成功讀取檔案: {file_path}")
            return df
        except Exception as e:
            logger.error(f"無法讀取檔案 {file_path}: {e}")
            raise
    
    def analyze_single_file(self, file_path: str) -> List[Dict]:
        """分析單一快照檔案"""
        logger.info(f"分析檔案: {file_path}")
        
        try:
            # 讀取CSV檔案，嘗試多種編碼
            df = self.read_csv_with_encoding(file_path)
            
            # 提取關鍵字
            filename = os.path.basename(file_path)
            keyword = filename.replace('_商品銷售快照.csv', '')
            
            changes = []
            
            # 按商品ID分組
            grouped = df.groupby('商品ID')
            
            for product_id, group in grouped:
                if len(group) > 1:  # 只有多筆記錄才需要比較
                    # 按擷取時間排序
                    group_sorted = group.sort_values('擷取時間')
                    
                    # 提取銷售數量數字
                    sales_numbers = []
                    for _, row in group_sorted.iterrows():
                        sales_num = self.extract_number(row['銷售數量'])
                        sales_numbers.append({
                            'capture_time': row['擷取時間'],
                            'sales_count': sales_num,
                            'original_sales': row['銷售數量']
                        })
                    
                    # 檢查是否有變化
                    for i in range(1, len(sales_numbers)):
                        prev_sales = sales_numbers[i-1]['sales_count']
                        curr_sales = sales_numbers[i]['sales_count']
                        
                        if prev_sales != curr_sales:
                            changes.append({
                                'keyword': keyword,
                                'product_id': product_id,
                                'product_name': group_sorted.iloc[0]['商品名稱'],
                                'previous_capture_time': sales_numbers[i-1]['capture_time'],
                                'previous_sales': sales_numbers[i-1]['original_sales'],
                                'previous_sales_num': prev_sales,
                                'current_capture_time': sales_numbers[i]['capture_time'],
                                'current_sales': sales_numbers[i]['original_sales'],
                                'current_sales_num': curr_sales,
                                'change_amount': curr_sales - prev_sales,
                                'change_percentage': round(((curr_sales - prev_sales) / prev_sales * 100) if prev_sales > 0 else 0, 2),
                                'file_source': file_path
                            })
            
            logger.info(f"在 {keyword} 中找到 {len(changes)} 筆銷售變化")
            return changes
            
        except Exception as e:
            logger.error(f"分析檔案 {file_path} 時發生錯誤: {e}")
            return []
    
    def analyze_all_files(self) -> pd.DataFrame:
        """分析所有快照檔案"""
        snapshot_files = self.find_snapshot_files()
        
        if not snapshot_files:
            logger.warning("找不到任何快照CSV檔案")
            return pd.DataFrame()
        
        logger.info(f"找到 {len(snapshot_files)} 個快照檔案")
        
        all_changes = []
        
        for file_path in snapshot_files:
            changes = self.analyze_single_file(file_path)
            all_changes.extend(changes)
        
        if not all_changes:
            logger.info("沒有找到任何銷售變化")
            return pd.DataFrame()
        
        # 轉換為DataFrame
        df_changes = pd.DataFrame(all_changes)
        
        # 按商品ID和時間排序
        df_changes = df_changes.sort_values(['product_id', 'current_capture_time'])
        
        logger.info(f"總共找到 {len(df_changes)} 筆銷售變化記錄")
        return df_changes
    
    def save_results(self, df: pd.DataFrame, output_file: Optional[str] = None):
        """儲存結果"""
        if df.empty:
            logger.warning("沒有資料可儲存")
            return
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'sales_changes_analysis_{timestamp}.csv'
        
        # 確保輸出目錄存在
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
        os.makedirs(output_dir, exist_ok=True)
        
        # 儲存CSV
        df.to_csv(output_file, encoding='utf-8-sig', index=False)
        logger.info(f"結果已儲存至: {output_file}")
        
        # 顯示統計資訊
        self.show_statistics(df)
    
    def show_statistics(self, df: pd.DataFrame):
        """顯示統計資訊"""
        if df.empty:
            return
        
        print("\n=== 銷售變化統計 ===")
        print(f"總變化記錄數: {len(df)}")
        print(f"涉及商品數: {df['product_id'].nunique()}")
        print(f"涉及關鍵字: {', '.join(df['keyword'].unique())}")
        
        # 變化幅度統計
        print(f"\n變化幅度統計:")
        print(f"  增加記錄: {len(df[df['change_amount'] > 0])} 筆")
        print(f"  減少記錄: {len(df[df['change_amount'] < 0])} 筆")
        
        if len(df[df['change_amount'] > 0]) > 0:
            max_increase = df[df['change_amount'] > 0]['change_amount'].max()
            print(f"  最大增加: +{max_increase:,}")
        
        if len(df[df['change_amount'] < 0]) > 0:
            max_decrease = df[df['change_amount'] < 0]['change_amount'].min()
            print(f"  最大減少: {max_decrease:,}")
        
        # 按關鍵字統計
        print(f"\n按關鍵字統計:")
        keyword_stats = df.groupby('keyword').agg({
            'product_id': 'nunique',
            'change_amount': ['count', 'sum']
        }).round(2)
        keyword_stats.columns = ['商品數', '變化次數', '總變化量']
        print(keyword_stats)

def main():
    parser = argparse.ArgumentParser(description='分析快照CSV檔案中的銷售變化')
    parser.add_argument('--output', type=str, help='輸出檔案名稱')
    parser.add_argument('--keyword', type=str, help='只分析特定關鍵字')
    args = parser.parse_args()
    
    analyzer = SalesChangesAnalyzer()
    
    # 分析所有檔案
    df_changes = analyzer.analyze_all_files()
    
    if not df_changes.empty:
        # 如果指定了關鍵字，進行篩選
        if args.keyword:
            df_changes = df_changes[df_changes['keyword'] == args.keyword].copy()
            if df_changes.empty:
                logger.info(f"關鍵字 '{args.keyword}' 沒有找到銷售變化")
                return
            logger.info(f"篩選關鍵字 '{args.keyword}' 後，找到 {len(df_changes)} 筆變化")
        
        # 儲存結果
        analyzer.save_results(df_changes, args.output if args.output else None)
    else:
        logger.info("沒有找到任何銷售變化")

if __name__ == '__main__':
    main() 