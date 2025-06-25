import mysql.connector
from mysql.connector import Error
from typing import Dict, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ErrorLogger:
    """錯誤記錄器"""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def log_validation_error(self, file_path: str, keyword: str, data_type: str, 
                           row_number: int, field_name: str, original_value: str,
                           error_type: str, error_message: str, suggested_value: str = None):
        """
        記錄資料驗證錯誤
        
        Args:
            file_path: 檔案路徑
            keyword: 關鍵字
            data_type: 資料類型 (product, snapshot, comment)
            row_number: CSV行號
            field_name: 欄位名稱
            original_value: 原始值
            error_type: 錯誤類型 (format_error, null_value, invalid_range, duplicate)
            error_message: 錯誤訊息
            suggested_value: 建議修正值
        """
        try:
            cursor = self.db.cursor()
            
            # 截斷過長的值以避免資料庫錯誤
            original_value = str(original_value)[:1000] if original_value else None
            error_message = str(error_message)[:1000] if error_message else None
            suggested_value = str(suggested_value)[:1000] if suggested_value else None
            
            query = """
                INSERT INTO data_validation_errors 
                (file_path, keyword, data_type, row_number, field_name, original_value, 
                 error_type, error_message, suggested_value, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """
            
            cursor.execute(query, (
                file_path, keyword, data_type, row_number, field_name, original_value,
                error_type, error_message, suggested_value
            ))
            
            self.db.commit()
            logger.debug(f"記錄驗證錯誤: {field_name} - {error_message}")
            
        except Error as e:
            logger.error(f"記錄驗證錯誤時發生資料庫錯誤: {e}")
            self.db.rollback()
        finally:
            if cursor:
                cursor.close()
    
    def log_sync_error(self, file_path: str, keyword: str, file_type: str, 
                      error_message: str, file_size: int = 0):
        """
        記錄同步錯誤
        
        Args:
            file_path: 檔案路徑
            keyword: 關鍵字
            file_type: 檔案類型
            error_message: 錯誤訊息
            file_size: 檔案大小
        """
        try:
            cursor = self.db.cursor()
            
            # 更新檔案同步記錄
            query = """
                INSERT INTO file_sync_logs 
                (keyword, file_type, file_path, file_size, sync_status, 
                 sync_start_time, sync_end_time, error_message, created_at)
                VALUES (%s, %s, %s, %s, 'failed', NOW(), NOW(), %s, NOW())
                ON DUPLICATE KEY UPDATE
                sync_status = 'failed',
                sync_end_time = NOW(),
                error_message = VALUES(error_message),
                updated_at = NOW()
            """
            
            cursor.execute(query, (
                keyword, file_type, file_path, file_size, error_message
            ))
            
            self.db.commit()
            logger.error(f"記錄同步錯誤: {file_path} - {error_message}")
            
        except Error as e:
            logger.error(f"記錄同步錯誤時發生資料庫錯誤: {e}")
            self.db.rollback()
        finally:
            if cursor:
                cursor.close()
    
    def update_sync_status(self, file_path: str, keyword: str, file_type: str,
                          status: str, total_records: int = 0, valid_records: int = 0, 
                          error_records: int = 0, file_size: int = 0):
        """
        更新同步狀態
        
        Args:
            file_path: 檔案路徑
            keyword: 關鍵字
            file_type: 檔案類型
            status: 狀態 (pending, processing, completed, failed)
            total_records: 總記錄數
            valid_records: 有效記錄數
            error_records: 錯誤記錄數
            file_size: 檔案大小
        """
        try:
            cursor = self.db.cursor()
            
            if status == 'processing':
                query = """
                    INSERT INTO file_sync_logs 
                    (keyword, file_type, file_path, file_size, sync_status, 
                     sync_start_time, created_at)
                    VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                    ON DUPLICATE KEY UPDATE
                    sync_status = VALUES(sync_status),
                    sync_start_time = NOW(),
                    updated_at = NOW()
                """
                cursor.execute(query, (keyword, file_type, file_path, file_size, status))
            else:
                query = """
                    INSERT INTO file_sync_logs 
                    (keyword, file_type, file_path, file_size, total_records, valid_records, error_records,
                     sync_status, sync_start_time, sync_end_time, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW(), NOW())
                    ON DUPLICATE KEY UPDATE
                    total_records = VALUES(total_records),
                    valid_records = VALUES(valid_records),
                    error_records = VALUES(error_records),
                    sync_status = VALUES(sync_status),
                    sync_end_time = NOW(),
                    updated_at = NOW()
                """
                cursor.execute(query, (
                    keyword, file_type, file_path, file_size, total_records, 
                    valid_records, error_records, status
                ))
            
            self.db.commit()
            logger.info(f"更新同步狀態: {file_path} - {status}")
            
        except Error as e:
            logger.error(f"更新同步狀態時發生資料庫錯誤: {e}")
            self.db.rollback()
        finally:
            if cursor:
                cursor.close()
    
    def get_error_summary(self, keyword: str = None, file_path: str = None) -> Dict:
        """
        取得錯誤摘要統計
        
        Args:
            keyword: 關鍵字篩選
            file_path: 檔案路徑篩選
            
        Returns:
            錯誤統計字典
        """
        try:
            cursor = self.db.cursor(dictionary=True)
            
            where_clause = "WHERE 1=1"
            params = []
            
            if keyword:
                where_clause += " AND keyword = %s"
                params.append(keyword)
            
            if file_path:
                where_clause += " AND file_path = %s"
                params.append(file_path)
            
            # 統計各類型錯誤
            query = f"""
                SELECT 
                    error_type,
                    COUNT(*) as error_count,
                    COUNT(DISTINCT file_path) as affected_files,
                    COUNT(DISTINCT keyword) as affected_keywords
                FROM data_validation_errors
                {where_clause}
                GROUP BY error_type
                ORDER BY error_count DESC
            """
            
            cursor.execute(query, params)
            error_types = cursor.fetchall()
            
            # 統計總錯誤數
            query = f"""
                SELECT 
                    COUNT(*) as total_errors,
                    COUNT(DISTINCT file_path) as total_files,
                    COUNT(DISTINCT keyword) as total_keywords
                FROM data_validation_errors
                {where_clause}
            """
            
            cursor.execute(query, params)
            total_stats = cursor.fetchone()
            
            # 統計最近錯誤
            query = f"""
                SELECT 
                    field_name,
                    COUNT(*) as error_count
                FROM data_validation_errors
                {where_clause}
                GROUP BY field_name
                ORDER BY error_count DESC
                LIMIT 10
            """
            
            cursor.execute(query, params)
            field_errors = cursor.fetchall()
            
            return {
                'error_types': error_types,
                'total_stats': total_stats,
                'field_errors': field_errors
            }
            
        except Error as e:
            logger.error(f"取得錯誤摘要時發生資料庫錯誤: {e}")
            return {}
        finally:
            if cursor:
                cursor.close()
    
    def get_sync_summary(self, keyword: str = None) -> Dict:
        """
        取得同步摘要統計
        
        Args:
            keyword: 關鍵字篩選
            
        Returns:
            同步統計字典
        """
        try:
            cursor = self.db.cursor(dictionary=True)
            
            where_clause = "WHERE 1=1"
            params = []
            
            if keyword:
                where_clause += " AND keyword = %s"
                params.append(keyword)
            
            # 統計各狀態檔案
            query = f"""
                SELECT 
                    sync_status,
                    COUNT(*) as file_count,
                    SUM(total_records) as total_records,
                    SUM(valid_records) as valid_records,
                    SUM(error_records) as error_records
                FROM file_sync_logs
                {where_clause}
                GROUP BY sync_status
                ORDER BY file_count DESC
            """
            
            cursor.execute(query, params)
            status_stats = cursor.fetchall()
            
            # 統計總體情況
            query = f"""
                SELECT 
                    COUNT(*) as total_files,
                    SUM(CASE WHEN sync_status = 'completed' THEN 1 ELSE 0 END) as completed_files,
                    SUM(CASE WHEN sync_status = 'failed' THEN 1 ELSE 0 END) as failed_files,
                    SUM(CASE WHEN sync_status = 'processing' THEN 1 ELSE 0 END) as processing_files,
                    SUM(total_records) as total_records,
                    SUM(valid_records) as valid_records,
                    SUM(error_records) as error_records
                FROM file_sync_logs
                {where_clause}
            """
            
            cursor.execute(query, params)
            overall_stats = cursor.fetchone()
            
            return {
                'status_stats': status_stats,
                'overall_stats': overall_stats
            }
            
        except Error as e:
            logger.error(f"取得同步摘要時發生資料庫錯誤: {e}")
            return {}
        finally:
            if cursor:
                cursor.close()
    
    def generate_error_report(self, keyword: str = None, output_file: str = None) -> str:
        """
        生成錯誤報告
        
        Args:
            keyword: 關鍵字篩選
            output_file: 輸出檔案路徑
            
        Returns:
            報告內容
        """
        error_summary = self.get_error_summary(keyword)
        sync_summary = self.get_sync_summary(keyword)
        
        report = []
        report.append("=" * 60)
        report.append("資料驗證錯誤報告")
        report.append("=" * 60)
        report.append(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if keyword:
            report.append(f"關鍵字: {keyword}")
        report.append("")
        
        # 總體統計
        if error_summary.get('total_stats'):
            stats = error_summary['total_stats']
            report.append("總體統計:")
            report.append(f"  總錯誤數: {stats['total_errors']}")
            report.append(f"  受影響檔案數: {stats['total_files']}")
            report.append(f"  受影響關鍵字數: {stats['total_keywords']}")
            report.append("")
        
        # 錯誤類型統計
        if error_summary.get('error_types'):
            report.append("錯誤類型統計:")
            for error_type in error_summary['error_types']:
                report.append(f"  {error_type['error_type']}: {error_type['error_count']} 個錯誤")
            report.append("")
        
        # 欄位錯誤統計
        if error_summary.get('field_errors'):
            report.append("欄位錯誤統計 (前10名):")
            for field_error in error_summary['field_errors']:
                report.append(f"  {field_error['field_name']}: {field_error['error_count']} 個錯誤")
            report.append("")
        
        # 同步統計
        if sync_summary.get('overall_stats'):
            stats = sync_summary['overall_stats']
            report.append("同步統計:")
            report.append(f"  總檔案數: {stats['total_files']}")
            report.append(f"  成功檔案數: {stats['completed_files']}")
            report.append(f"  失敗檔案數: {stats['failed_files']}")
            report.append(f"  處理中檔案數: {stats['processing_files']}")
            report.append(f"  總記錄數: {stats['total_records']}")
            report.append(f"  有效記錄數: {stats['valid_records']}")
            report.append(f"  錯誤記錄數: {stats['error_records']}")
        
        report_content = "\n".join(report)
        
        # 寫入檔案
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.info(f"錯誤報告已寫入: {output_file}")
            except Exception as e:
                logger.error(f"寫入錯誤報告檔案時發生錯誤: {e}")
        
        return report_content 