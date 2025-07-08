#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix Sales Snapshots
根據格式檢查結果自動修正快照檔案，保持萬字格式
"""

import os
import pandas as pd
import argparse
import logging
import shutil
from typing import List, Dict, Optional
from datetime import datetime

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SalesSnapshotFixer:
    """銷售快照修正器"""
    
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
    
    def load_format_issues(self, issues_file: str) -> pd.DataFrame:
        """載入格式問題檢查結果"""
        try:
            df = pd.read_csv(issues_file)
            logger.info(f"載入格式問題檔案: {issues_file}, 共 {len(df)} 筆問題")
            return df
        except Exception as e:
            logger.error(f"載入格式問題檔案失敗: {e}")
            return pd.DataFrame()
    
    def find_snapshot_file(self, keyword: str) -> str:
        """尋找對應的快照檔案"""
        snapshot_path = f'crawler/{keyword}_商品銷售快照.csv'
        if os.path.exists(snapshot_path):
            return snapshot_path
        return ""
    
    def fix_single_snapshot(self, keyword: str, issues_df: pd.DataFrame) -> bool:
        """修正單一快照檔案"""
        snapshot_path = self.find_snapshot_file(keyword)
        if not snapshot_path:
            logger.warning(f"找不到快照檔案: {keyword}")
            return False
        
        # 篩選該關鍵字的問題
        keyword_issues = issues_df[issues_df['keyword'] == keyword]
        if keyword_issues.empty:
            logger.info(f"關鍵字 {keyword} 沒有需要修正的問題")
            return False
        
        logger.info(f"開始修正 {keyword} 快照檔案，共 {len(keyword_issues)} 筆需要修正")
        
        # 備份原始檔案
        backup_path = self.backup_file(snapshot_path)
        
        try:
            # 讀取快照檔案
            df = pd.read_csv(snapshot_path, dtype={'商品ID': str})
            original_count = len(df)
            
            # 記錄修正的記錄
            fixed_records = []
            
            # 對每個問題進行修正
            for _, issue in keyword_issues.iterrows():
                product_id = issue['product_id']
                problematic_time = issue['problematic_time']
                suggested_sales = issue['suggested_sales']
                
                # 找到對應的記錄
                mask = (df['商品ID'] == product_id) & (df['擷取時間'] == problematic_time)
                if mask.any():
                    # 記錄修正前後的值
                    original_sales = df.loc[mask, '銷售數量'].iloc[0]
                    df.loc[mask, '銷售數量'] = suggested_sales
                    
                    fixed_records.append({
                        'product_id': product_id,
                        'capture_time': problematic_time,
                        'original_sales': original_sales,
                        'fixed_sales': suggested_sales
                    })
                    
                    logger.info(f"修正: 商品ID {product_id}, 時間 {problematic_time}, {original_sales} -> {suggested_sales}")
            
            # 儲存修正後的檔案
            df.to_csv(snapshot_path, encoding='utf-8-sig', index=False)
            
            self.fixed_count += len(fixed_records)
            logger.info(f"修正完成: {keyword}, 修正了 {len(fixed_records)} 筆記錄")
            
            # 輸出修正報告
            self.save_fix_report(keyword, fixed_records, backup_path)
            
            return True
            
        except Exception as e:
            logger.error(f"修正檔案 {snapshot_path} 時發生錯誤: {e}")
            # 如果修正失敗，恢復備份
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, snapshot_path)
                logger.info(f"已恢復備份檔案: {backup_path}")
            return False
    
    def save_fix_report(self, keyword: str, fixed_records: List[Dict], backup_path: str):
        """儲存修正報告"""
        if not fixed_records:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f'fix_report_{keyword}_{timestamp}.csv'
        
        df_report = pd.DataFrame(fixed_records)
        df_report.to_csv(report_path, encoding='utf-8-sig', index=False)
        
        logger.info(f"修正報告已儲存: {report_path}")
        
        # 顯示修正摘要
        print(f"\n=== {keyword} 修正摘要 ===")
        print(f"修正記錄數: {len(fixed_records)}")
        print(f"備份檔案: {backup_path}")
        print(f"修正報告: {report_path}")
        
        # 顯示前幾筆修正記錄
        print(f"\n修正記錄範例:")
        for i, record in enumerate(fixed_records[:5]):
            print(f"  {i+1}. 商品ID: {record['product_id']}")
            print(f"     時間: {record['capture_time']}")
            print(f"     修正: {record['original_sales']} -> {record['fixed_sales']}")
    
    def fix_all_snapshots(self, issues_file: str, keywords: Optional[List[str]] = None):
        """修正所有快照檔案"""
        # 載入格式問題
        issues_df = self.load_format_issues(issues_file)
        if issues_df.empty:
            logger.error("沒有載入到格式問題資料")
            return
        
        # 取得所有需要修正的關鍵字
        if keywords is None:
            keywords = issues_df['keyword'].unique().tolist()
        else:
            keywords = keywords
        
        logger.info(f"準備修正 {len(keywords)} 個關鍵字的快照檔案")
        
        success_count = 0
        for keyword in keywords:
            if self.fix_single_snapshot(keyword, issues_df):
                success_count += 1
        
        # 顯示總體結果
        print(f"\n=== 修正完成 ===")
        print(f"成功修正: {success_count}/{len(keywords)} 個檔案")
        print(f"總修正記錄數: {self.fixed_count}")
        print(f"備份檔案數: {len(self.backup_files)}")
        
        if self.backup_files:
            print(f"\n備份檔案列表:")
            for backup_file in self.backup_files:
                print(f"  {backup_file}")
    
    def preview_fixes(self, issues_file: str, keywords: Optional[List[str]] = None):
        """預覽修正內容（不實際修改檔案）"""
        issues_df = self.load_format_issues(issues_file)
        if issues_df.empty:
            logger.error("沒有載入到格式問題資料")
            return
        
        if keywords is None:
            keywords = issues_df['keyword'].unique().tolist()
        else:
            keywords = keywords
        
        print(f"\n=== 修正預覽 ===")
        print(f"將修正 {len(keywords)} 個關鍵字的快照檔案")
        
        for keyword in keywords:
            keyword_issues = issues_df[issues_df['keyword'] == keyword]
            if not keyword_issues.empty:
                print(f"\n{keyword}: {len(keyword_issues)} 筆需要修正")
                
                # 顯示前幾筆修正內容
                for i, (_, issue) in enumerate(keyword_issues.head(3).iterrows()):
                    print(f"  {i+1}. 商品ID: {issue['product_id']}")
                    print(f"     時間: {issue['problematic_time']}")
                    print(f"     修正: {issue['problematic_sales']} -> {issue['suggested_sales']}")
                
                if len(keyword_issues) > 3:
                    print(f"     ... 還有 {len(keyword_issues) - 3} 筆")

def main():
    parser = argparse.ArgumentParser(description='修正快照檔案中的銷售數量格式問題')
    parser.add_argument('--issues-file', type=str, required=True, help='格式問題檢查結果檔案')
    parser.add_argument('--keywords', type=str, nargs='+', help='指定要修正的關鍵字（不指定則修正所有）')
    parser.add_argument('--preview', action='store_true', help='只預覽修正內容，不實際修改檔案')
    args = parser.parse_args()
    
    fixer = SalesSnapshotFixer()
    
    if args.preview:
        # 只預覽修正內容
        fixer.preview_fixes(args.issues_file, args.keywords)
    else:
        # 實際執行修正
        fixer.fix_all_snapshots(args.issues_file, args.keywords)

if __name__ == '__main__':
    main() 