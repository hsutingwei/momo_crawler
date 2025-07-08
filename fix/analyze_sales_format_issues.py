#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sales Format Issues Analyzer
分析快照檔案中銷售數量格式不一致的問題，特別是萬字被過濾的情況
"""

import os
import pandas as pd
import argparse
import logging
from typing import List, Dict, Tuple
from datetime import datetime
import re

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SalesFormatAnalyzer:
    """銷售格式分析器"""
    
    def __init__(self):
        self.issues_data = []
        
    def extract_number_and_unit(self, sales_str: str) -> Tuple[int, str]:
        """從銷售字串中提取數字和單位"""
        if pd.isna(sales_str) or sales_str == '':
            return 0, ''
        
        if isinstance(sales_str, str):
            # 檢查是否包含萬字
            if '萬' in sales_str:
                # 提取數字部分
                match = re.search(r'(\d+(?:\.\d+)?)', sales_str)
                if match:
                    return int(float(match.group(1)) * 10000), '萬'
            else:
                # 直接提取數字
                match = re.search(r'(\d+)', sales_str)
                if match:
                    return int(match.group(1)), ''
        
        return 0, ''
    
    def find_snapshot_files(self) -> List[str]:
        """尋找所有快照CSV檔案"""
        snapshot_files = []
        
        # 檢查 crawler 目錄
        if os.path.exists('crawler'):
            for file in os.listdir('crawler'):
                if file.endswith('_商品銷售快照.csv') and not file.endswith('.bak'):
                    snapshot_files.append(os.path.join('crawler', file))
        
        return sorted(snapshot_files)
    
    def analyze_single_file(self, file_path: str) -> List[Dict]:
        """分析單一快照檔案的格式問題"""
        logger.info(f"分析檔案: {file_path}")
        
        try:
            # 讀取CSV檔案
            df = pd.read_csv(file_path, dtype={'商品ID': str})
            
            # 提取關鍵字
            filename = os.path.basename(file_path)
            keyword = filename.replace('_商品銷售快照.csv', '')
            
            issues = []
            
            # 按商品ID分組
            grouped = df.groupby('商品ID')
            
            for product_id, group in grouped:
                if len(group) > 1:  # 只有多筆記錄才需要比較
                    # 按擷取時間排序
                    group_sorted = group.sort_values('擷取時間')
                    
                    # 分析每筆記錄的格式
                    sales_records = []
                    for _, row in group_sorted.iterrows():
                        sales_num, unit = self.extract_number_and_unit(row['銷售數量'])
                        sales_records.append({
                            'capture_time': row['擷取時間'],
                            'original_sales': row['銷售數量'],
                            'sales_num': sales_num,
                            'unit': unit,
                            'has_wan': '萬' in str(row['銷售數量'])
                        })
                    
                    # 檢查格式不一致的問題
                    issues.extend(self.check_format_issues(product_id, group_sorted, sales_records, keyword, file_path))
            
            logger.info(f"在 {keyword} 中找到 {len(issues)} 筆格式問題")
            return issues
            
        except Exception as e:
            logger.error(f"分析檔案 {file_path} 時發生錯誤: {e}")
            return []
    
    def check_format_issues(self, product_id: str, group: pd.DataFrame, sales_records: List[Dict], keyword: str, file_path: str) -> List[Dict]:
        """檢查格式問題"""
        issues = []
        
        # 檢查是否有萬字格式不一致
        has_wan_records = [r for r in sales_records if r['has_wan']]
        no_wan_records = [r for r in sales_records if not r['has_wan']]
        
        if has_wan_records and no_wan_records:
            # 有萬字和無萬字混合的情況
            for no_wan_record in no_wan_records:
                # 檢查這個無萬字的記錄是否可能是萬字被過濾了
                potential_wan_value = no_wan_record['sales_num'] * 10000
                
                # 檢查是否有其他記錄的數值接近這個潛在的萬字值
                for wan_record in has_wan_records:
                    if abs(wan_record['sales_num'] - potential_wan_value) <= 1000:  # 允許1000的誤差
                        issues.append({
                            'keyword': keyword,
                            'product_id': product_id,
                            'product_name': group.iloc[0]['商品名稱'],
                            'issue_type': '萬字可能被過濾',
                            'problematic_time': no_wan_record['capture_time'],
                            'problematic_sales': no_wan_record['original_sales'],
                            'problematic_sales_num': no_wan_record['sales_num'],
                            'suggested_sales': f"{no_wan_record['sales_num']}萬",
                            'suggested_sales_num': potential_wan_value,
                            'reference_time': wan_record['capture_time'],
                            'reference_sales': wan_record['original_sales'],
                            'reference_sales_num': wan_record['sales_num'],
                            'file_source': file_path
                        })
                        break
        
        # 檢查銷售數量異常減少的情況（可能是萬字被過濾）
        for i in range(1, len(sales_records)):
            prev_record = sales_records[i-1]
            curr_record = sales_records[i]
            
            # 如果後面的記錄有萬字，前面的沒有，且數值相差很大
            if (curr_record['has_wan'] and not prev_record['has_wan'] and 
                curr_record['sales_num'] > prev_record['sales_num'] * 100):
                
                # 檢查前面的記錄是否可能是萬字被過濾
                potential_prev_wan = prev_record['sales_num'] * 10000
                if abs(curr_record['sales_num'] - potential_prev_wan) <= 10000:  # 允許1萬的誤差
                    issues.append({
                        'keyword': keyword,
                        'product_id': product_id,
                        'product_name': group.iloc[0]['商品名稱'],
                        'issue_type': '萬字可能被過濾（異常減少）',
                        'problematic_time': prev_record['capture_time'],
                        'problematic_sales': prev_record['original_sales'],
                        'problematic_sales_num': prev_record['sales_num'],
                        'suggested_sales': f"{prev_record['sales_num']}萬",
                        'suggested_sales_num': potential_prev_wan,
                        'reference_time': curr_record['capture_time'],
                        'reference_sales': curr_record['original_sales'],
                        'reference_sales_num': curr_record['sales_num'],
                        'file_source': file_path
                    })
        
        return issues
    
    def analyze_all_files(self) -> pd.DataFrame:
        """分析所有快照檔案"""
        snapshot_files = self.find_snapshot_files()
        
        if not snapshot_files:
            logger.warning("找不到任何快照CSV檔案")
            return pd.DataFrame()
        
        logger.info(f"找到 {len(snapshot_files)} 個快照檔案")
        
        all_issues = []
        
        for file_path in snapshot_files:
            issues = self.analyze_single_file(file_path)
            all_issues.extend(issues)
        
        if not all_issues:
            logger.info("沒有找到任何格式問題")
            return pd.DataFrame()
        
        # 轉換為DataFrame
        df_issues = pd.DataFrame(all_issues)
        
        # 按商品ID和時間排序
        df_issues = df_issues.sort_values(['product_id', 'problematic_time'])
        
        logger.info(f"總共找到 {len(df_issues)} 筆格式問題記錄")
        return df_issues
    
    def save_results(self, df: pd.DataFrame, output_file: str = None):
        """儲存結果"""
        if df.empty:
            logger.warning("沒有資料可儲存")
            return
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'sales_format_issues_{timestamp}.csv'
        
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
        
        print("\n=== 銷售格式問題統計 ===")
        print(f"總問題記錄數: {len(df)}")
        print(f"涉及商品數: {df['product_id'].nunique()}")
        print(f"涉及關鍵字: {', '.join(df['keyword'].unique())}")
        
        # 按問題類型統計
        print(f"\n按問題類型統計:")
        issue_type_stats = df.groupby('issue_type').size()
        for issue_type, count in issue_type_stats.items():
            print(f"  {issue_type}: {count} 筆")
        
        # 按關鍵字統計
        print(f"\n按關鍵字統計:")
        keyword_stats = df.groupby('keyword').agg({
            'product_id': 'nunique',
            'issue_type': 'count'
        })
        keyword_stats.columns = ['商品數', '問題數']
        print(keyword_stats)
        
        # 顯示一些範例
        print(f"\n=== 範例問題記錄 ===")
        for i, (_, row) in enumerate(df.head(5).iterrows()):
            print(f"\n{i+1}. 商品ID: {row['product_id']}")
            print(f"   商品名稱: {row['product_name']}")
            print(f"   問題類型: {row['issue_type']}")
            print(f"   問題時間: {row['problematic_time']}")
            print(f"   問題銷售: {row['problematic_sales']} (數值: {row['problematic_sales_num']})")
            print(f"   建議修正: {row['suggested_sales']} (數值: {row['suggested_sales_num']})")
            print(f"   參考時間: {row['reference_time']}")
            print(f"   參考銷售: {row['reference_sales']} (數值: {row['reference_sales_num']})")

def main():
    parser = argparse.ArgumentParser(description='分析快照CSV檔案中銷售數量格式問題')
    parser.add_argument('--output', type=str, help='輸出檔案名稱')
    parser.add_argument('--keyword', type=str, help='只分析特定關鍵字')
    args = parser.parse_args()
    
    analyzer = SalesFormatAnalyzer()
    
    # 分析所有檔案
    df_issues = analyzer.analyze_all_files()
    
    if not df_issues.empty:
        # 如果指定了關鍵字，進行篩選
        if args.keyword:
            df_issues = df_issues[df_issues['keyword'] == args.keyword]
            if df_issues.empty:
                logger.info(f"關鍵字 '{args.keyword}' 沒有找到格式問題")
                return
            logger.info(f"篩選關鍵字 '{args.keyword}' 後，找到 {len(df_issues)} 筆問題")
        
        # 儲存結果
        analyzer.save_results(df_issues, args.output)
    else:
        logger.info("沒有找到任何格式問題")

if __name__ == '__main__':
    main() 