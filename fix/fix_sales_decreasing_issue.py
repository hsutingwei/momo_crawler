#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix Sales Decreasing Issue
修正快照檔案中銷售數量不合理的遞減問題
由於爬蟲錯誤導致後期銷售數量比前期少，這在momo的銷售紀錄中是不可能的
"""

import os
import pandas as pd
import argparse
import logging
import shutil
from typing import List, Dict, Tuple
from datetime import datetime
import numpy as np

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SalesDecreasingFixer:
    """銷售數量遞減問題修正器"""
    
    def __init__(self):
        self.fixed_count = 0
        self.backup_files = []
        self.issues_found = []
        
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
        crawler_dir = 'crawler'
        
        if os.path.exists(crawler_dir):
            for file in os.listdir(crawler_dir):
                if file.endswith('_商品銷售快照.csv') and not file.endswith('_backup_'):
                    snapshot_files.append(os.path.join(crawler_dir, file))
        
        return snapshot_files
    
    def parse_sales_number(self, sales_str: str) -> float:
        """解析銷售數量字串為數字"""
        if pd.isna(sales_str) or sales_str == '':
            return 0.0
        
        sales_str = str(sales_str).strip()
        
        # 處理萬字
        if '萬' in sales_str:
            # 移除萬字並轉換為數字
            number_part = sales_str.replace('萬', '').strip()
            try:
                return float(number_part) * 10000
            except ValueError:
                return 0.0
        
        # 處理純數字
        try:
            return float(sales_str)
        except ValueError:
            return 0.0
    
    def format_sales_number(self, sales_num: float) -> str:
        """將數字格式化為萬字格式"""
        if sales_num >= 10000:
            return f"{sales_num / 10000:.1f}萬"
        else:
            return str(int(sales_num))
    
    def detect_decreasing_issues(self, df: pd.DataFrame) -> List[Dict]:
        """檢測銷售數量遞減問題"""
        issues = []
        
        # 按商品ID分組
        for product_id, group in df.groupby('商品ID'):
            # 按擷取時間排序
            group_sorted = group.sort_values('擷取時間')
            
            if len(group_sorted) < 2:
                continue
            
            # 轉換銷售數量為數字進行比較
            sales_numbers = []
            for _, row in group_sorted.iterrows():
                sales_num = self.parse_sales_number(row['銷售數量'])
                sales_numbers.append(sales_num)
            
            # 檢查是否有遞減問題
            max_sales = sales_numbers[0]
            for i, sales in enumerate(sales_numbers[1:], 1):
                if sales < max_sales:
                    # 找到遞減問題
                    issues.append({
                        'product_id': product_id,
                        'capture_time': group_sorted.iloc[i]['擷取時間'],
                        'current_sales': group_sorted.iloc[i]['銷售數量'],
                        'current_sales_num': sales,
                        'max_sales': self.format_sales_number(max_sales),
                        'max_sales_num': max_sales,
                        'suggested_sales': self.format_sales_number(max_sales)
                    })
                else:
                    max_sales = sales
        
        return issues
    
    def fix_single_snapshot(self, snapshot_path: str) -> bool:
        """修正單一快照檔案"""
        logger.info(f"開始處理: {snapshot_path}")
        
        try:
            # 讀取快照檔案
            df = pd.read_csv(snapshot_path, dtype={'商品ID': str})
            original_count = len(df)
            
            # 檢測問題
            issues = self.detect_decreasing_issues(df)
            
            if not issues:
                logger.info(f"檔案 {snapshot_path} 沒有發現遞減問題")
                return True
            
            logger.info(f"發現 {len(issues)} 筆遞減問題")
            
            # 備份原始檔案
            backup_path = self.backup_file(snapshot_path)
            
            # 記錄修正的記錄
            fixed_records = []
            
            # 修正問題
            for issue in issues:
                product_id = issue['product_id']
                capture_time = issue['capture_time']
                suggested_sales = issue['suggested_sales']
                
                # 找到對應的記錄
                mask = (df['商品ID'] == product_id) & (df['擷取時間'] == capture_time)
                if mask.any():
                    # 記錄修正前後的值
                    original_sales = df.loc[mask, '銷售數量'].iloc[0]
                    df.loc[mask, '銷售數量'] = suggested_sales
                    
                    fixed_records.append({
                        'product_id': product_id,
                        'capture_time': capture_time,
                        'original_sales': original_sales,
                        'fixed_sales': suggested_sales,
                        'max_sales': issue['max_sales']
                    })
                    
                    logger.info(f"修正: 商品ID {product_id}, 時間 {capture_time}, {original_sales} -> {suggested_sales}")
            
            # 儲存修正後的檔案
            df.to_csv(snapshot_path, encoding='utf-8-sig', index=False)
            
            self.fixed_count += len(fixed_records)
            logger.info(f"修正完成: {snapshot_path}, 修正了 {len(fixed_records)} 筆記錄")
            
            # 輸出修正報告
            self.save_fix_report(snapshot_path, fixed_records, backup_path, issues)
            
            return True
            
        except Exception as e:
            logger.error(f"處理檔案 {snapshot_path} 時發生錯誤: {e}")
            return False
    
    def save_fix_report(self, snapshot_path: str, fixed_records: List[Dict], backup_path: str, all_issues: List[Dict]):
        """儲存修正報告"""
        if not fixed_records:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        keyword = os.path.basename(snapshot_path).replace('_商品銷售快照.csv', '')
        report_path = f'fix_decreasing_report_{keyword}_{timestamp}.csv'
        
        # 創建詳細報告
        report_data = []
        for record in fixed_records:
            report_data.append({
                'product_id': record['product_id'],
                'capture_time': record['capture_time'],
                'original_sales': record['original_sales'],
                'fixed_sales': record['fixed_sales'],
                'max_sales': record['max_sales']
            })
        
        df_report = pd.DataFrame(report_data)
        df_report.to_csv(report_path, encoding='utf-8-sig', index=False)
        
        logger.info(f"修正報告已儲存: {report_path}")
        
        # 顯示修正摘要
        print(f"\n=== {keyword} 遞減問題修正摘要 ===")
        print(f"發現問題數: {len(all_issues)}")
        print(f"修正記錄數: {len(fixed_records)}")
        print(f"備份檔案: {backup_path}")
        print(f"修正報告: {report_path}")
        
        # 顯示前幾筆修正記錄
        print(f"\n修正記錄範例:")
        for i, record in enumerate(fixed_records[:5]):
            print(f"  {i+1}. 商品ID: {record['product_id']}")
            print(f"     時間: {record['capture_time']}")
            print(f"     修正: {record['original_sales']} -> {record['fixed_sales']}")
            print(f"     最大銷售: {record['max_sales']}")
    
    def preview_issues(self, snapshot_path: str):
        """預覽問題（不實際修改檔案）"""
        logger.info(f"預覽問題: {snapshot_path}")
        
        try:
            df = pd.read_csv(snapshot_path, dtype={'商品ID': str})
            issues = self.detect_decreasing_issues(df)
            
            if not issues:
                print(f"檔案 {snapshot_path} 沒有發現遞減問題")
                return
            
            keyword = os.path.basename(snapshot_path).replace('_商品銷售快照.csv', '')
            print(f"\n=== {keyword} 遞減問題預覽 ===")
            print(f"發現 {len(issues)} 筆遞減問題")
            
            # 顯示前幾筆問題
            for i, issue in enumerate(issues[:5]):
                print(f"  {i+1}. 商品ID: {issue['product_id']}")
                print(f"     時間: {issue['capture_time']}")
                print(f"     當前銷售: {issue['current_sales']}")
                print(f"     建議修正: {issue['suggested_sales']}")
                print(f"     最大銷售: {issue['max_sales']}")
            
            if len(issues) > 5:
                print(f"     ... 還有 {len(issues) - 5} 筆問題")
            
        except Exception as e:
            logger.error(f"預覽檔案 {snapshot_path} 時發生錯誤: {e}")
    
    def fix_all_snapshots(self, snapshot_files: List[str] = None):
        """修正所有快照檔案"""
        if snapshot_files is None:
            snapshot_files = self.find_snapshot_files()
        
        if not snapshot_files:
            logger.warning("沒有找到快照檔案")
            return
        
        logger.info(f"準備處理 {len(snapshot_files)} 個快照檔案")
        
        success_count = 0
        for snapshot_path in snapshot_files:
            if self.fix_single_snapshot(snapshot_path):
                success_count += 1
        
        # 顯示總體結果
        print(f"\n=== 遞減問題修正完成 ===")
        print(f"成功處理: {success_count}/{len(snapshot_files)} 個檔案")
        print(f"總修正記錄數: {self.fixed_count}")
        print(f"備份檔案數: {len(self.backup_files)}")
        
        if self.backup_files:
            print(f"\n備份檔案列表:")
            for backup_file in self.backup_files:
                print(f"  {backup_file}")
    
    def preview_all_issues(self, snapshot_files: List[str] = None):
        """預覽所有檔案的遞減問題"""
        if snapshot_files is None:
            snapshot_files = self.find_snapshot_files()
        
        if not snapshot_files:
            logger.warning("沒有找到快照檔案")
            return
        
        logger.info(f"預覽 {len(snapshot_files)} 個快照檔案的遞減問題")
        
        total_issues = 0
        for snapshot_path in snapshot_files:
            try:
                df = pd.read_csv(snapshot_path, dtype={'商品ID': str})
                issues = self.detect_decreasing_issues(df)
                total_issues += len(issues)
                
                if issues:
                    self.preview_issues(snapshot_path)
            except Exception as e:
                logger.error(f"預覽檔案 {snapshot_path} 時發生錯誤: {e}")
        
        print(f"\n=== 總體預覽結果 ===")
        print(f"處理檔案數: {len(snapshot_files)}")
        print(f"總問題數: {total_issues}")

def main():
    parser = argparse.ArgumentParser(description='修正快照檔案中的銷售數量遞減問題')
    parser.add_argument('--files', type=str, nargs='+', help='指定要處理的快照檔案（不指定則處理所有）')
    parser.add_argument('--preview', action='store_true', help='只預覽問題，不實際修改檔案')
    args = parser.parse_args()
    
    fixer = SalesDecreasingFixer()
    
    if args.preview:
        # 只預覽問題
        fixer.preview_all_issues(args.files)
    else:
        # 實際執行修正
        fixer.fix_all_snapshots(args.files)

if __name__ == '__main__':
    main() 