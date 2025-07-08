#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean Sales Decimal
清理快照檔案中銷售數量的小數點，確保都是整數格式
"""

import os
import pandas as pd
import argparse
import logging
import shutil
from typing import List, Optional
from datetime import datetime

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SalesDecimalCleaner:
    """銷售數量小數點清理器"""
    
    def __init__(self):
        self.fixed_count = 0
        self.backup_files = []
        
    def backup_file(self, file_path: str) -> str:
        """備份檔案"""
        backup_path = file_path.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        shutil.copy2(file_path, backup_path)
        self.backup_files.append(backup_path)
        logger.info(f"已備份: {file_path} -> {backup_path}")
        return backup_path
    
    def find_snapshot_files(self) -> List[str]:
        """尋找所有快照檔案"""
        snapshot_files = []
        crawler_dir = '../crawler'
        
        if os.path.exists(crawler_dir):
            for file in os.listdir(crawler_dir):
                if file.endswith('_商品銷售快照.csv') and not file.endswith('_backup_'):
                    snapshot_files.append(os.path.join(crawler_dir, file))
        
        return snapshot_files
    
    def clean_sales_number(self, sales_str: str) -> str:
        """清理銷售數量，移除小數點"""
        if pd.isna(sales_str) or sales_str == '':
            return ''
        
        sales_str = str(sales_str).strip()
        
        # 處理萬字
        if '萬' in sales_str:
            # 移除萬字，轉換為數字，再轉回整數萬字格式
            number_part = sales_str.replace('萬', '').strip()
            try:
                # 轉換為數字
                number = float(number_part)
                # 轉換為整數萬字格式
                if number >= 1:
                    return f"{int(number)}萬"
                else:
                    return f"{int(number * 10000)}"
            except ValueError:
                return sales_str
        
        # 處理純數字
        try:
            number = float(sales_str)
            return str(int(number))
        except ValueError:
            return sales_str
    
    def clean_single_snapshot(self, snapshot_path: str) -> bool:
        """清理單一快照檔案"""
        logger.info(f"開始清理: {snapshot_path}")
        
        try:
            # 讀取快照檔案
            df = pd.read_csv(snapshot_path, dtype={'商品ID': str})
            original_count = len(df)
            
            # 記錄修正的記錄
            fixed_records = []
            
            # 清理銷售數量
            for index, row in df.iterrows():
                original_sales = row['銷售數量']
                cleaned_sales = self.clean_sales_number(original_sales)
                
                if original_sales != cleaned_sales:
                    fixed_records.append({
                        'product_id': row['商品ID'],
                        'capture_time': row['擷取時間'],
                        'original_sales': original_sales,
                        'cleaned_sales': cleaned_sales
                    })
                    
                    # 更新資料
                    df.at[index, '銷售數量'] = cleaned_sales
            
            if not fixed_records:
                logger.info(f"檔案 {snapshot_path} 沒有需要清理的小數點")
                return True
            
            logger.info(f"發現 {len(fixed_records)} 筆需要清理的記錄")
            
            # 備份原始檔案
            backup_path = self.backup_file(snapshot_path)
            
            # 儲存清理後的檔案
            df.to_csv(snapshot_path, encoding='utf-8-sig', index=False)
            
            self.fixed_count += len(fixed_records)
            logger.info(f"清理完成: {snapshot_path}, 清理了 {len(fixed_records)} 筆記錄")
            
            # 輸出清理報告
            self.save_clean_report(snapshot_path, fixed_records, backup_path)
            
            return True
            
        except Exception as e:
            logger.error(f"清理檔案 {snapshot_path} 時發生錯誤: {e}")
            return False
    
    def save_clean_report(self, snapshot_path: str, fixed_records: List[dict], backup_path: str):
        """儲存清理報告"""
        if not fixed_records:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        keyword = os.path.basename(snapshot_path).replace('_商品銷售快照.csv', '')
        report_path = f'clean_decimal_report_{keyword}_{timestamp}.csv'
        
        # 創建詳細報告
        df_report = pd.DataFrame(fixed_records)
        df_report.to_csv(report_path, encoding='utf-8-sig', index=False)
        
        logger.info(f"清理報告已儲存: {report_path}")
        
        # 顯示清理摘要
        print(f"\n=== {keyword} 小數點清理摘要 ===")
        print(f"清理記錄數: {len(fixed_records)}")
        print(f"備份檔案: {backup_path}")
        print(f"清理報告: {report_path}")
        
        # 顯示前幾筆清理記錄
        print(f"\n清理記錄範例:")
        for i, record in enumerate(fixed_records[:5]):
            print(f"  {i+1}. 商品ID: {record['product_id']}")
            print(f"     時間: {record['capture_time']}")
            print(f"     清理: {record['original_sales']} -> {record['cleaned_sales']}")
    
    def preview_cleaning(self, snapshot_path: str):
        """預覽清理內容（不實際修改檔案）"""
        logger.info(f"預覽清理: {snapshot_path}")
        
        try:
            df = pd.read_csv(snapshot_path, dtype={'商品ID': str})
            
            # 記錄需要清理的記錄
            records_to_clean = []
            
            for index, row in df.iterrows():
                original_sales = row['銷售數量']
                cleaned_sales = self.clean_sales_number(original_sales)
                
                if original_sales != cleaned_sales:
                    records_to_clean.append({
                        'product_id': row['商品ID'],
                        'capture_time': row['擷取時間'],
                        'original_sales': original_sales,
                        'cleaned_sales': cleaned_sales
                    })
            
            if not records_to_clean:
                print(f"檔案 {snapshot_path} 沒有需要清理的小數點")
                return
            
            keyword = os.path.basename(snapshot_path).replace('_商品銷售快照.csv', '')
            print(f"\n=== {keyword} 小數點清理預覽 ===")
            print(f"發現 {len(records_to_clean)} 筆需要清理的記錄")
            
            # 顯示前幾筆清理內容
            for i, record in enumerate(records_to_clean[:5]):
                print(f"  {i+1}. 商品ID: {record['product_id']}")
                print(f"     時間: {record['capture_time']}")
                print(f"     清理: {record['original_sales']} -> {record['cleaned_sales']}")
            
            if len(records_to_clean) > 5:
                print(f"     ... 還有 {len(records_to_clean) - 5} 筆")
            
        except Exception as e:
            logger.error(f"預覽檔案 {snapshot_path} 時發生錯誤: {e}")
    
    def clean_all_snapshots(self, snapshot_files: Optional[List[str]] = None):
        """清理所有快照檔案"""
        if snapshot_files is None:
            snapshot_files = self.find_snapshot_files()
        
        if not snapshot_files:
            logger.warning("沒有找到快照檔案")
            return
        
        logger.info(f"準備清理 {len(snapshot_files)} 個快照檔案")
        
        success_count = 0
        for snapshot_path in snapshot_files:
            if self.clean_single_snapshot(snapshot_path):
                success_count += 1
        
        # 顯示總體結果
        print(f"\n=== 小數點清理完成 ===")
        print(f"成功處理: {success_count}/{len(snapshot_files)} 個檔案")
        print(f"總清理記錄數: {self.fixed_count}")
        print(f"備份檔案數: {len(self.backup_files)}")
        
        if self.backup_files:
            print(f"\n備份檔案列表:")
            for backup_file in self.backup_files:
                print(f"  {backup_file}")
    
    def preview_all_cleaning(self, snapshot_files: Optional[List[str]] = None):
        """預覽所有檔案的清理內容"""
        if snapshot_files is None:
            snapshot_files = self.find_snapshot_files()
        
        if not snapshot_files:
            logger.warning("沒有找到快照檔案")
            return
        
        logger.info(f"預覽 {len(snapshot_files)} 個快照檔案的小數點清理")
        
        total_records = 0
        for snapshot_path in snapshot_files:
            try:
                df = pd.read_csv(snapshot_path, dtype={'商品ID': str})
                
                # 計算需要清理的記錄數
                records_to_clean = 0
                for _, row in df.iterrows():
                    original_sales = row['銷售數量']
                    cleaned_sales = self.clean_sales_number(original_sales)
                    if original_sales != cleaned_sales:
                        records_to_clean += 1
                
                total_records += records_to_clean
                
                if records_to_clean > 0:
                    self.preview_cleaning(snapshot_path)
            except Exception as e:
                logger.error(f"預覽檔案 {snapshot_path} 時發生錯誤: {e}")
        
        print(f"\n=== 總體預覽結果 ===")
        print(f"處理檔案數: {len(snapshot_files)}")
        print(f"總清理記錄數: {total_records}")

def main():
    parser = argparse.ArgumentParser(description='清理快照檔案中銷售數量的小數點')
    parser.add_argument('--files', type=str, nargs='+', help='指定要處理的快照檔案（不指定則處理所有）')
    parser.add_argument('--preview', action='store_true', help='只預覽清理內容，不實際修改檔案')
    args = parser.parse_args()
    
    cleaner = SalesDecimalCleaner()
    
    if args.preview:
        # 只預覽清理內容
        cleaner.preview_all_cleaning(args.files)
    else:
        # 實際執行清理
        cleaner.clean_all_snapshots(args.files)

if __name__ == '__main__':
    main() 