#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
from datetime import datetime

# 載入環境變數
from dotenv import load_dotenv
load_dotenv()

# 添加當前目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database import DatabaseConfig
from utils.data_validator import DataValidator
from utils.error_logger import ErrorLogger
from utils.db_operations import DatabaseOperations
from utils.file_utils import FileUtils

def test_snapshot_processing():
    """測試銷售快照資料處理"""
    
    print("=== 測試銷售快照資料處理 ===")
    
    # 建立資料庫設定
    db_config = DatabaseConfig()
    
    try:
        # 連接資料庫
        db_connection = db_config.get_connection()
        db_ops = DatabaseOperations(db_connection)
        error_logger = ErrorLogger(db_connection)
        validator = DataValidator()
        
        print("✓ 資料庫連接成功")
        
        # 初始化資料庫
        db_config.init_database()
        print("✓ 資料庫初始化完成")
        
        # 讀取CSV檔案
        csv_file = "crawler/雞精_商品銷售快照.csv"
        
        if not os.path.exists(csv_file):
            print(f"✗ 檔案不存在: {csv_file}")
            return
        
        df = FileUtils.read_csv_safely(csv_file)
        if df is None:
            print("✗ 無法讀取CSV檔案")
            return
        
        print(f"✓ 成功讀取CSV檔案: {csv_file}")
        print(f"  資料行數: {len(df)}")
        print(f"  欄位名稱: {list(df.columns)}")
        
        # 驗證CSV結構
        expected_columns = FileUtils.get_expected_columns("snapshot")
        is_valid, errors = FileUtils.validate_csv_structure(df, expected_columns, "snapshot")
        
        if not is_valid:
            print(f"✗ CSV結構驗證失敗:")
            for error in errors:
                print(f"  - {error}")
            return
        
        print("✓ CSV結構驗證通過")
        
        # 取得已存在的快照鍵值
        existing_snapshots = db_ops.get_existing_snapshot_keys("雞精")
        print(f"  已存在快照數量: {len(existing_snapshots)}")
        
        # 處理前幾筆資料作為測試
        test_rows = min(5, len(df))
        print(f"\n處理前 {test_rows} 筆資料:")
        
        for i in range(test_rows):
            row = df.iloc[i]
            print(f"\n第 {i+1} 筆資料:")
            print(f"  商品ID: {row['商品ID']}")
            print(f"  商品名稱: {row['商品名稱'][:50]}...")
            print(f"  價格: {row['價格']}")
            print(f"  銷售數量: {row['銷售數量']}")
            print(f"  擷取時間: {row['擷取時間']}")
            
            # 驗證資料
            product_id, product_id_error = validator.validate_product_id(str(row['商品ID']))
            if product_id_error:
                print(f"  ✗ 商品ID驗證失敗: {product_id_error}")
                continue
            
            sales_count, sales_unit, sales_error = validator.validate_sales_count(str(row['銷售數量']))
            if sales_error:
                print(f"  ✗ 銷售數量驗證失敗: {sales_error}")
                continue
            
            capture_time, time_error = validator.validate_timestamp(str(row['擷取時間']))
            if time_error:
                print(f"  ✗ 擷取時間驗證失敗: {time_error}")
                continue
            
            # 檢查是否已存在
            snapshot_key = (product_id, capture_time)
            if snapshot_key in existing_snapshots:
                print(f"  ⚠ 快照已存在，跳過")
                continue
            
            print(f"  ✓ 所有欄位驗證通過")
            print(f"    商品ID: {product_id}")
            print(f"    銷售數量: {sales_count} {sales_unit}")
            print(f"    擷取時間: {capture_time}")
            
            # 準備資料
            snapshot_data = (product_id, sales_count, sales_unit, capture_time)
            
            # 插入資料庫
            success = db_ops.insert_sales_snapshot(*snapshot_data)
            if success:
                print("  ✓ 資料插入成功")
            else:
                print("  ✗ 資料插入失敗")
        
        db_connection.close()
        print("\n✓ 測試完成")
        
    except Exception as e:
        print(f"✗ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_snapshot_processing() 