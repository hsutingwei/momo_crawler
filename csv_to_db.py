#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV to Database Converter
將爬蟲產生的CSV檔案轉換並儲存到資料庫
"""

from dotenv import load_dotenv
load_dotenv()

import os
print('DB_HOST:', os.getenv('DB_HOST'))

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd

# 添加專案路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database import DatabaseConfig
from utils.data_validator import DataValidator
from utils.error_logger import ErrorLogger
from utils.db_operations import DatabaseOperations
from utils.file_utils import FileUtils

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('csv_to_db.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CSVToDBConverter:
    """CSV到資料庫轉換器"""
    
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
        self.db_connection = None
        self.db_ops = None
        self.error_logger = None
        self.validator = DataValidator()
        
        # 立即連接資料庫並初始化相關物件
        self.connect_database()
        
    def connect_database(self):
        """連接資料庫"""
        try:
            self.db_connection = self.db_config.get_connection()
            self.db_ops = DatabaseOperations(self.db_connection)
            self.error_logger = ErrorLogger(self.db_connection)
            logger.info("資料庫連接成功")
        except Exception as e:
            logger.error(f"資料庫連接失敗: {e}")
            # 確保物件不為 None
            self.db_connection = None
            self.db_ops = None
            self.error_logger = None
            raise
    
    def close_database(self):
        """關閉資料庫連接"""
        if self.db_connection:
            self.db_connection.close()
            logger.info("資料庫連接已關閉")
    
    def process_products_csv(self, file_path: str, keyword: str) -> Tuple[int, int, int]:
        """
        處理商品CSV檔案
        
        Args:
            file_path: 檔案路徑
            keyword: 關鍵字
            
        Returns:
            (總記錄數, 有效記錄數, 錯誤記錄數)
        """
        logger.info(f"開始處理商品檔案: {file_path}")
        
        # 讀取CSV檔案
        df = FileUtils.read_csv_safely(file_path)
        if df is None:
            return 0, 0, 0
        
        # 驗證CSV結構
        expected_columns = FileUtils.get_expected_columns("product")
        is_valid, errors = FileUtils.validate_csv_structure(df, expected_columns, "product")
        
        if not is_valid:
            for error in errors:
                self.error_logger.log_validation_error(
                    file_path, keyword, "product", 0, "structure", "",
                    "format_error", error
                )
            return len(df), 0, len(df)
        
        total_records = len(df)
        valid_records = 0
        error_records = 0
        
        # 批次處理資料
        products_data = []
        
        for index, row in df.iterrows():
            try:
                row_index = int(str(index)) + 2
                # 驗證商品ID
                product_id, product_id_error = self.validator.validate_product_id(str(row['商品ID']))
                if product_id_error:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "product", row_index, "商品ID",
                        str(row['商品ID']), "format_error", product_id_error
                    )
                    error_records += 1
                    continue
                
                # 驗證商品名稱
                name, name_error = self.validator.validate_text_field(str(row['商品名稱']), 500)
                if name_error:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "product", row_index, "商品名稱",
                        str(row['商品名稱']), "format_error", name_error
                    )
                    error_records += 1
                    continue
                
                # 驗證價格
                price, price_error = self.validator.validate_price(str(row['價格']))
                if price_error:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "product", row_index, "價格",
                        str(row['價格']), "format_error", price_error
                    )
                    error_records += 1
                    continue
                
                # 驗證商品連結
                product_link, link_error = self.validator.validate_url(str(row['商品連結']))
                if link_error:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "product", row_index, "商品連結",
                        str(row['商品連結']), "format_error", link_error
                    )
                    error_records += 1
                    continue
                
                # 準備資料
                products_data.append((product_id, name, price, product_link, keyword))
                valid_records += 1
                
            except Exception as e:
                self.error_logger.log_validation_error(
                    file_path, keyword, "product", row_index, "general",
                    str(row), "format_error", str(e)
                )
                error_records += 1
        
        # 批次插入資料庫
        if products_data:
            inserted_count = self.db_ops.batch_insert_products(products_data)
            logger.info(f"商品資料插入完成: {inserted_count} 筆")
        
        return total_records, valid_records, error_records
    
    def process_snapshots_csv(self, file_path: str, keyword: str) -> Tuple[int, int, int]:
        """
        處理銷售快照CSV檔案
        
        Args:
            file_path: 檔案路徑
            keyword: 關鍵字
            
        Returns:
            (總記錄數, 有效記錄數, 錯誤記錄數)
        """
        logger.info(f"開始處理銷售快照檔案: {file_path}")
        
        # 讀取CSV檔案
        df = FileUtils.read_csv_safely(file_path)
        if df is None:
            return 0, 0, 0
        
        # 驗證CSV結構
        expected_columns = FileUtils.get_expected_columns("snapshot")
        is_valid, errors = FileUtils.validate_csv_structure(df, expected_columns, "snapshot")
        
        if not is_valid:
            for error in errors:
                self.error_logger.log_validation_error(
                    file_path, keyword, "snapshot", 0, "structure", "",
                    "format_error", error
                )
            return len(df), 0, len(df)
        
        # 取得已存在的快照鍵值
        existing_snapshots = self.db_ops.get_existing_snapshot_keys(keyword)
        
        total_records = len(df)
        valid_records = 0
        error_records = 0
        
        # 批次處理資料
        snapshots_data = []
        
        for index, row in df.iterrows():
            try:
                row_index = int(str(index)) + 2
                # 驗證商品ID
                product_id, product_id_error = self.validator.validate_product_id(str(row['商品ID']))
                if product_id_error:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "snapshot", row_index, "商品ID",
                        str(row['商品ID']), "format_error", product_id_error
                    )
                    error_records += 1
                    continue
                
                # 驗證銷售數量
                sales_count, sales_unit, sales_error = self.validator.validate_sales_count(str(row['銷售數量']))
                if sales_error:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "snapshot", row_index, "銷售數量",
                        str(row['銷售數量']), "format_error", sales_error
                    )
                    error_records += 1
                    continue
                
                # 驗證擷取時間
                capture_time, time_error = self.validator.validate_timestamp(str(row['擷取時間']))
                if time_error:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "snapshot", row_index, "擷取時間",
                        str(row['擷取時間']), "format_error", time_error
                    )
                    error_records += 1
                    continue
                
                # 檢查是否已存在
                snapshot_key = (product_id, capture_time)
                if snapshot_key in existing_snapshots:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "snapshot", row_index, "duplicate",
                        f"{product_id}_{capture_time}", "duplicate", "銷售快照已存在"
                    )
                    error_records += 1
                    continue
                
                # 準備資料
                snapshots_data.append((product_id, sales_count, sales_unit, capture_time))
                valid_records += 1
                
            except Exception as e:
                self.error_logger.log_validation_error(
                    file_path, keyword, "snapshot", row_index, "general",
                    str(row), "format_error", str(e)
                )
                error_records += 1
        
        # 批次插入資料庫
        if snapshots_data:
            inserted_count = self.db_ops.batch_insert_snapshots(snapshots_data)
            logger.info(f"銷售快照插入完成: {inserted_count} 筆")
        
        return total_records, valid_records, error_records
    
    def process_comments_csv(self, file_path: str, keyword: str) -> Tuple[int, int, int]:
        """
        處理評論CSV檔案
        
        Args:
            file_path: 檔案路徑
            keyword: 關鍵字
            
        Returns:
            (總記錄數, 有效記錄數, 錯誤記錄數)
        """
        logger.info(f"開始處理評論檔案: {file_path}")
        
        # 讀取CSV檔案
        df = FileUtils.read_csv_safely(file_path)
        if df is None:
            return 0, 0, 0
        
        # 驗證CSV結構
        expected_columns = FileUtils.get_expected_columns("comment")
        is_valid, errors = FileUtils.validate_csv_structure(df, expected_columns, "comment")
        
        if not is_valid:
            for error in errors:
                self.error_logger.log_validation_error(
                    file_path, keyword, "comment", 0, "structure", "",
                    "format_error", error
                )
            return len(df), 0, len(df)
        
        # 取得已存在的評論ID
        existing_comment_ids = self.db_ops.get_existing_comment_ids(keyword)
        
        total_records = len(df)
        valid_records = 0
        error_records = 0
        
        # 批次處理資料
        comments_data = []
        
        for index, row in df.iterrows():
            try:
                row_index = int(str(index)) + 2
                # 驗證評論ID
                comment_id, comment_id_error = self.validator.validate_comment_id(str(row['留言ID']))
                if comment_id_error:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "comment", row_index, "留言ID",
                        str(row['留言ID']), "format_error", comment_id_error
                    )
                    error_records += 1
                    continue
                
                # 檢查是否已存在
                if comment_id and comment_id in existing_comment_ids:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "comment", row_index, "留言ID",
                        str(comment_id), "duplicate", "評論已存在"
                    )
                    error_records += 1
                    continue
                
                # 驗證商品ID
                product_id, product_id_error = self.validator.validate_product_id(str(row['商品ID']))
                if product_id_error:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "comment", row_index, "商品ID",
                        str(row['商品ID']), "format_error", product_id_error
                    )
                    error_records += 1
                    continue
                
                # 驗證評論內容
                comment_text = self.validator.clean_comment_text(str(row['留言']))
                comment_text, text_error = self.validator.validate_text_field(comment_text)
                if text_error:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "comment", row_index, "留言",
                        str(row['留言']), "format_error", text_error
                    )
                    error_records += 1
                    continue
                
                # 驗證客戶名稱
                customer_name, name_error = self.validator.validate_text_field(str(row['留言者名稱']), 200)
                if name_error:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "comment", row_index, "留言者名稱",
                        str(row['留言者名稱']), "format_error", name_error
                    )
                    error_records += 1
                    continue
                
                # 驗證評論時間
                comment_date, date_error = self.validator.validate_timestamp(str(row['留言時間']))
                if date_error:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "comment", row_index, "留言時間",
                        str(row['留言時間']), "format_error", date_error
                    )
                    error_records += 1
                    continue
                
                # 驗證商品類型
                goods_type, type_error = self.validator.validate_text_field(str(row['商品 Type']), 100)
                if type_error:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "comment", row_index, "商品 Type",
                        str(row['商品 Type']), "format_error", type_error
                    )
                    error_records += 1
                    continue
                
                # 驗證圖片URLs
                image_urls, urls_error = self.validator.validate_json_field(str(row['留言 圖']))
                if urls_error:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "comment", row_index, "留言 圖",
                        str(row['留言 圖']), "format_error", urls_error
                    )
                    error_records += 1
                    continue
                
                # 驗證布林值欄位
                is_like, like_error = self.validator.validate_boolean_field(str(row['isLike']))
                if like_error:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "comment", row_index, "isLike",
                        str(row['isLike']), "format_error", like_error
                    )
                    error_records += 1
                    continue
                
                is_show_like, show_like_error = self.validator.validate_boolean_field(str(row['isShowLike']))
                if show_like_error:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "comment", row_index, "isShowLike",
                        str(row['isShowLike']), "format_error", show_like_error
                    )
                    error_records += 1
                    continue
                
                # 驗證整數欄位
                like_count, like_count_error = self.validator.validate_integer_field(str(row['留言 likeCount']))
                if like_count_error:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "comment", row_index, "留言 likeCount",
                        str(row['留言 likeCount']), "format_error", like_count_error
                    )
                    error_records += 1
                    continue
                
                # 驗證回覆內容
                reply_content, reply_error = self.validator.validate_text_field(str(row['留言 replyContent']))
                if reply_error:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "comment", row_index, "留言 replyContent",
                        str(row['留言 replyContent']), "format_error", reply_error
                    )
                    error_records += 1
                    continue
                
                # 驗證回覆時間
                reply_date, reply_date_error = self.validator.validate_timestamp(str(row['留言 replyDate']))
                if reply_date_error:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "comment", row_index, "留言 replyDate",
                        str(row['留言 replyDate']), "format_error", reply_date_error
                    )
                    error_records += 1
                    continue
                
                # 驗證評分
                score, score_error = self.validator.validate_score(str(row['留言星數']))
                if score_error:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "comment", row_index, "留言星數",
                        str(row['留言星數']), "format_error", score_error
                    )
                    error_records += 1
                    continue
                
                # 驗證影片相關欄位
                video_thumbnail_img, thumb_error = self.validator.validate_url(str(row['videoThumbnailImg']))
                if thumb_error:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "comment", row_index, "videoThumbnailImg",
                        str(row['videoThumbnailImg']), "format_error", thumb_error
                    )
                    error_records += 1
                    continue
                
                video_url, video_error = self.validator.validate_url(str(row['videoUrl']))
                if video_error:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "comment", row_index, "videoUrl",
                        str(row['videoUrl']), "format_error", video_error
                    )
                    error_records += 1
                    continue
                
                # 驗證擷取時間
                capture_time, capture_error = self.validator.validate_timestamp(str(row['資料擷取時間']))
                if capture_error:
                    self.error_logger.log_validation_error(
                        file_path, keyword, "comment", row_index, "資料擷取時間",
                        str(row['資料擷取時間']), "format_error", capture_error
                    )
                    error_records += 1
                    continue
                
                # 準備資料
                comments_data.append((
                    comment_id, product_id, comment_text, customer_name, comment_date,
                    goods_type, image_urls, is_like, is_show_like, like_count,
                    reply_content, reply_date, score, video_thumbnail_img, video_url,
                    capture_time
                ))
                valid_records += 1
                
            except Exception as e:
                self.error_logger.log_validation_error(
                    file_path, keyword, "comment", row_index, "general",
                    str(row), "format_error", str(e)
                )
                error_records += 1
        
        # 批次插入資料庫
        if comments_data:
            inserted_count = self.db_ops.batch_insert_comments(comments_data)
            logger.info(f"評論資料插入完成: {inserted_count} 筆")
        
        return total_records, valid_records, error_records
    
    def process_file(self, file_info: Dict) -> bool:
        """
        處理單一檔案
        
        Args:
            file_info: 檔案資訊
            
        Returns:
            是否成功處理
        """
        file_path = file_info['file_path']
        file_type = file_info['file_type']
        keyword = file_info['keyword']
        file_size = file_info['file_size']
        
        logger.info(f"開始處理檔案: {file_path} (類型: {file_type}, 關鍵字: {keyword})")
        
        try:
            # 更新同步狀態為處理中
            self.error_logger.update_sync_status(
                file_path, keyword, file_type, "processing", file_size=file_size
            )
            
            # 根據檔案類型處理
            if file_type == "product":
                total, valid, error = self.process_products_csv(file_path, keyword)
            elif file_type == "snapshot":
                total, valid, error = self.process_snapshots_csv(file_path, keyword)
            elif file_type == "comment":
                total, valid, error = self.process_comments_csv(file_path, keyword)
            else:
                logger.error(f"不支援的檔案類型: {file_type}")
                return False
            
            # 更新同步狀態為完成
            self.error_logger.update_sync_status(
                file_path, keyword, file_type, "completed",
                total, valid, error, file_size
            )
            
            logger.info(f"檔案處理完成: {file_path} - 總計: {total}, 有效: {valid}, 錯誤: {error}")
            return True
            
        except Exception as e:
            logger.error(f"處理檔案時發生錯誤: {file_path} - {e}")
            self.error_logger.log_sync_error(file_path, keyword, file_type, str(e), file_size)
            return False
    
    def run(self, base_dir: str = ".", keyword: Optional[str] = None, 
            move_processed: bool = False, generate_report: bool = True):
        """
        執行轉換程序
        
        Args:
            base_dir: 基礎目錄
            keyword: 關鍵字篩選
            move_processed: 是否移動已處理的檔案
            generate_report: 是否生成錯誤報告
        """
        logger.info("開始CSV到資料庫轉換程序")
        
        try:
            # 檢查資料庫連接狀態
            if self.db_connection is None or self.db_ops is None or self.error_logger is None:
                self.connect_database()
            
            # 初始化資料庫
            self.db_config.init_database()
            
            # 掃描CSV檔案
            csv_files = FileUtils.scan_csv_files(base_dir, keyword)
            
            if not csv_files:
                logger.warning("沒有找到需要處理的CSV檔案")
                return
            
            # 處理每個檔案
            processed_count = 0
            failed_count = 0
            
            for file_info in csv_files:
                if self.process_file(file_info):
                    processed_count += 1
                    
                    # 移動已處理的檔案
                    if move_processed:
                        FileUtils.move_processed_file(file_info['file_path'])
                else:
                    failed_count += 1
            
            # 生成錯誤報告
            if generate_report:
                report_content = self.error_logger.generate_error_report(
                    keyword, f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                )
                logger.info("錯誤報告已生成")
            
            logger.info(f"轉換程序完成 - 成功: {processed_count}, 失敗: {failed_count}")
            
        except Exception as e:
            logger.error(f"轉換程序發生錯誤: {e}")
            raise
        finally:
            self.close_database()

def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='CSV to Database Converter')
    parser.add_argument('--base-dir', type=str, default='.', 
                       help='基礎目錄 (預設: 當前目錄)')
    parser.add_argument('--keyword', type=str, 
                       help='關鍵字篩選 (預設: 處理所有關鍵字)')
    parser.add_argument('--move-processed', action='store_true',
                       help='移動已處理的檔案到 processed 目錄')
    parser.add_argument('--no-report', action='store_true',
                       help='不生成錯誤報告')
    parser.add_argument('--init-db', action='store_true',
                       help='只初始化資料庫表結構')
    
    args = parser.parse_args()
    
    try:
        # 建立資料庫設定
        db_config = DatabaseConfig()
        
        if args.init_db:
            # 只初始化資料庫
            db_config.init_database()
            logger.info("資料庫初始化完成")
            return
        
        # 建立轉換器並執行
        converter = CSVToDBConverter(db_config)
        converter.run(
            base_dir=args.base_dir,
            keyword=args.keyword,
            move_processed=args.move_processed,
            generate_report=not args.no_report
        )
        
    except KeyboardInterrupt:
        logger.info("程式被使用者中斷")
    except Exception as e:
        logger.error(f"程式執行失敗: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 