import os
import glob
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class FileUtils:
    """檔案處理工具類別"""
    
    @staticmethod
    def scan_csv_files(base_dir: str = ".", keyword: str = None) -> List[Dict]:
        """
        掃描CSV檔案
        
        Args:
            base_dir: 基礎目錄
            keyword: 關鍵字篩選
            
        Returns:
            CSV檔案資訊列表
        """
        csv_files = []
        
        # 掃描商品資料檔案
        product_pattern = f"{base_dir}/*_商品資料.csv"
        if keyword:
            product_pattern = f"{base_dir}/{keyword}_商品資料.csv"
        
        for file_path in glob.glob(product_pattern):
            file_info = FileUtils._get_file_info(file_path, "product", keyword)
            if file_info:
                csv_files.append(file_info)
        
        # 掃描銷售快照檔案
        snapshot_pattern = f"{base_dir}/crawler/*_商品銷售快照.csv"
        if keyword:
            snapshot_pattern = f"{base_dir}/crawler/{keyword}_商品銷售快照.csv"
        
        for file_path in glob.glob(snapshot_pattern):
            file_info = FileUtils._get_file_info(file_path, "snapshot", keyword)
            if file_info:
                csv_files.append(file_info)
        
        # 掃描評論資料檔案
        comment_pattern = f"{base_dir}/crawler/*_商品留言資料_*.csv"
        if keyword:
            comment_pattern = f"{base_dir}/crawler/{keyword}_商品留言資料_*.csv"
        
        for file_path in glob.glob(comment_pattern):
            file_info = FileUtils._get_file_info(file_path, "comment", keyword)
            if file_info:
                csv_files.append(file_info)
        
        logger.info(f"掃描到 {len(csv_files)} 個CSV檔案")
        return csv_files
    
    @staticmethod
    def _get_file_info(file_path: str, file_type: str, keyword: str = None) -> Optional[Dict]:
        """
        取得檔案資訊
        
        Args:
            file_path: 檔案路徑
            file_type: 檔案類型
            keyword: 關鍵字
            
        Returns:
            檔案資訊字典
        """
        try:
            if not os.path.exists(file_path):
                return None
            
            file_stat = os.stat(file_path)
            
            # 從檔案名提取關鍵字
            if not keyword:
                if file_type == "product":
                    match = re.search(r'([^_]+)_商品資料\.csv', os.path.basename(file_path))
                elif file_type == "snapshot":
                    match = re.search(r'([^_]+)_商品銷售快照\.csv', os.path.basename(file_path))
                elif file_type == "comment":
                    match = re.search(r'([^_]+)_商品留言資料_', os.path.basename(file_path))
                else:
                    match = None
                
                if match:
                    keyword = match.group(1)
            
            return {
                'file_path': file_path,
                'file_type': file_type,
                'keyword': keyword,
                'file_size': file_stat.st_size,
                'modified_time': datetime.fromtimestamp(file_stat.st_mtime),
                'created_time': datetime.fromtimestamp(file_stat.st_ctime)
            }
            
        except Exception as e:
            logger.error(f"取得檔案資訊時發生錯誤: {file_path} - {e}")
            return None
    
    @staticmethod
    def read_csv_safely(file_path: str, encoding: str = 'utf-8-sig') -> Optional[pd.DataFrame]:
        """
        安全讀取CSV檔案
        
        Args:
            file_path: 檔案路徑
            encoding: 編碼格式
            
        Returns:
            DataFrame 或 None
        """
        try:
            # 檢查檔案是否存在
            if not os.path.exists(file_path):
                logger.error(f"檔案不存在: {file_path}")
                return None
            
            # 檢查檔案大小
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logger.warning(f"檔案為空: {file_path}")
                return None
            
            # 嘗試讀取CSV
            df = pd.read_csv(file_path, encoding=encoding)
            
            if df.empty:
                logger.warning(f"CSV檔案為空: {file_path}")
                return None
            
            # 清理欄位名稱（移除前後空格）
            df.columns = df.columns.str.strip()
            
            logger.info(f"成功讀取CSV檔案: {file_path}, 共 {len(df)} 筆記錄")
            return df
            
        except UnicodeDecodeError as e:
            logger.error(f"編碼錯誤: {file_path} - {e}")
            # 嘗試其他編碼
            for enc in ['utf-8', 'big5', 'gbk', 'gb2312']:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    logger.info(f"使用編碼 {enc} 成功讀取: {file_path}")
                    return df
                except UnicodeDecodeError:
                    continue
            return None
            
        except Exception as e:
            logger.error(f"讀取CSV檔案時發生錯誤: {file_path} - {e}")
            return None
    
    @staticmethod
    def extract_keyword_from_filename(file_path: str) -> Optional[str]:
        """
        從檔案名提取關鍵字
        
        Args:
            file_path: 檔案路徑
            
        Returns:
            關鍵字或None
        """
        filename = os.path.basename(file_path)
        
        # 商品資料檔案
        match = re.search(r'([^_]+)_商品資料\.csv', filename)
        if match:
            return match.group(1)
        
        # 銷售快照檔案
        match = re.search(r'([^_]+)_商品銷售快照\.csv', filename)
        if match:
            return match.group(1)
        
        # 評論資料檔案
        match = re.search(r'([^_]+)_商品留言資料_', filename)
        if match:
            return match.group(1)
        
        return None
    
    @staticmethod
    def extract_timestamp_from_filename(file_path: str) -> Optional[str]:
        """
        從檔案名提取時間戳記
        
        Args:
            file_path: 檔案路徑
            
        Returns:
            時間戳記或None
        """
        filename = os.path.basename(file_path)
        
        # 評論資料檔案中的時間戳記
        match = re.search(r'_商品留言資料_(\d{14})\.csv', filename)
        if match:
            timestamp = match.group(1)
            # 轉換為標準格式
            try:
                dt = datetime.strptime(timestamp, '%Y%m%d%H%M%S')
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                return timestamp
        
        return None
    
    @staticmethod
    def validate_csv_structure(df: pd.DataFrame, expected_columns: List[str], 
                             file_type: str) -> Tuple[bool, List[str]]:
        """
        驗證CSV結構
        
        Args:
            df: DataFrame
            expected_columns: 預期欄位列表
            file_type: 檔案類型
            
        Returns:
            (是否有效, 錯誤訊息列表)
        """
        errors = []
        
        if df is None or df.empty:
            errors.append("DataFrame為空或None")
            return False, errors
        
        # 檢查必要欄位
        missing_columns = []
        for col in expected_columns:
            if col not in df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            errors.append(f"缺少必要欄位: {', '.join(missing_columns)}")
        
        # 檢查資料型別
        if file_type == "product":
            if '商品ID' in df.columns:
                # 檢查商品ID是否為數字
                non_numeric_ids = df[~df['商品ID'].astype(str).str.match(r'^\d+$')]
                if not non_numeric_ids.empty:
                    errors.append(f"發現 {len(non_numeric_ids)} 筆非數字商品ID")
        
        elif file_type == "snapshot":
            if '商品ID' in df.columns and '擷取時間' in df.columns:
                # 銷售快照允許重複的商品ID+擷取時間組合，因為可能有多筆記錄
                # 重複檢查將在資料庫插入時處理
                pass
        
        elif file_type == "comment":
            if '留言ID' in df.columns:
                # 檢查留言ID是否重複
                duplicates = df.duplicated(subset=['留言ID'])
                if duplicates.any():
                    errors.append(f"發現 {duplicates.sum()} 筆重複的留言ID")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def get_expected_columns(file_type: str) -> List[str]:
        """
        取得預期欄位列表
        
        Args:
            file_type: 檔案類型
            
        Returns:
            預期欄位列表
        """
        if file_type == "product":
            return ['商品ID', '商品名稱', '商品連結', '價格', '銷售數量']
        elif file_type == "snapshot":
            return ['商品ID', '商品名稱', '價格', '銷售數量', '商品連結', '擷取時間']
        elif file_type == "comment":
            return [
                '商品ID', '商品名稱', '商品連結', '價格', '銷售數量', '留言', '留言ID',
                '留言者名稱', '留言時間', '商品 Type', '留言 圖', 'isLike', 'isShowLike',
                '留言 likeCount', '留言 replyContent', '留言 replyDate', '留言星數',
                'videoThumbnailImg', 'videoUrl', '資料擷取時間'
            ]
        else:
            return []
    
    @staticmethod
    def backup_file(file_path: str, backup_dir: str = "backup") -> Optional[str]:
        """
        備份檔案
        
        Args:
            file_path: 原始檔案路徑
            backup_dir: 備份目錄
            
        Returns:
            備份檔案路徑或None
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"原始檔案不存在: {file_path}")
                return None
            
            # 建立備份目錄
            os.makedirs(backup_dir, exist_ok=True)
            
            # 生成備份檔案名
            filename = os.path.basename(file_path)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"{timestamp}_{filename}"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            # 複製檔案
            import shutil
            shutil.copy2(file_path, backup_path)
            
            logger.info(f"檔案已備份: {file_path} -> {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"備份檔案時發生錯誤: {file_path} - {e}")
            return None
    
    @staticmethod
    def move_processed_file(file_path: str, processed_dir: str = "processed") -> bool:
        """
        移動已處理的檔案
        
        Args:
            file_path: 原始檔案路徑
            processed_dir: 已處理檔案目錄
            
        Returns:
            是否成功移動
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"原始檔案不存在: {file_path}")
                return False
            
            # 建立已處理目錄
            os.makedirs(processed_dir, exist_ok=True)
            
            # 生成目標檔案路徑
            filename = os.path.basename(file_path)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            processed_filename = f"{timestamp}_{filename}"
            processed_path = os.path.join(processed_dir, processed_filename)
            
            # 移動檔案
            import shutil
            shutil.move(file_path, processed_path)
            
            logger.info(f"檔案已移動: {file_path} -> {processed_path}")
            return True
            
        except Exception as e:
            logger.error(f"移動檔案時發生錯誤: {file_path} - {e}")
            return False 