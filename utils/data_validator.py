import re
import json
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """資料驗證器"""
    
    @staticmethod
    def validate_price(price_str: str) -> Tuple[Optional[float], Optional[str]]:
        """
        驗證價格格式
        支援格式: "1,234", "1,234.56", "$1,234", "1234"
        """
        if not price_str or str(price_str).strip() == '':
            return None, "價格為空值"
        
        try:
            # 移除 $ 符號、千分位逗號和空白
            clean_price = re.sub(r'[$,]', '', str(price_str).strip())
            price = float(clean_price)
            
            if price < 0:
                return None, f"價格不能為負數: {price_str}"
            
            return price, None
        except (ValueError, AttributeError) as e:
            return None, f"無法解析價格格式: {price_str} - {str(e)}"
    
    @staticmethod
    def validate_sales_count(sales_str: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """
        驗證銷售數量格式
        支援格式: "1,234", "1,234萬", "1234", "1.5萬"
        """
        # 空值或常見缺值
        if sales_str is None:
            return 0, '個', "銷售數量為空值"

        s = str(sales_str).strip()
        if s == "" or s in {"-", "--", "N/A", "n/a", "null", "None"}:
            return 0, '個', "銷售數量為空值"
        
        try:
            sales_str = str(sales_str).strip()
            # 全形→半形；統一標點與空白
            trans = str.maketrans("０１２３４５６７８９，．。　 ", "0123456789,. . ")
            sales_str = sales_str.translate(trans)
            sales_str = sales_str.replace(" ", "")
            
            # 處理 "萬" 單位
            if '萬' in sales_str:
                number_str = re.sub(r'[萬,]', '', sales_str)
                count = int(float(number_str) * 10000)
                return count, '萬', None
            else:
                # 處理一般數字
                number_str = re.sub(r'[,]', '', sales_str)
                count = int(float(number_str))
                return count, '個', None
                
        except (ValueError, AttributeError) as e:
            return None, None, f"無法解析銷售數量格式: {sales_str} - {str(e)}"
    
    @staticmethod
    def validate_timestamp(date_str: str) -> Tuple[Optional[datetime], Optional[str]]:
        """
        驗證時間格式
        支援多種常見的日期格式
        """
        if not date_str or str(date_str).strip() == '' or str(date_str).lower() == 'nan':
            return None, None  # 空值不算錯誤
        
        date_str = str(date_str).strip()
        
        # 常見的日期格式
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y年%m月%d日',
            '%Y年%m月%d日 %H:%M:%S',
            '%Y%m%d%H%M%S'  # 新增：20250802215524 格式
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt), None
            except ValueError:
                continue
        
        return None, f"無法解析日期格式: {date_str}"
    
    @staticmethod
    def validate_score(score_str: str) -> Tuple[Optional[float], Optional[str]]:
        """
        驗證評分格式
        支援 0-5 分的評分系統
        """
        if not score_str or str(score_str).strip() == '':
            return None, "評分為空值"
        
        try:
            score = float(str(score_str).strip())
            if 0 <= score <= 5:
                return round(score, 1), None
            else:
                return None, f"評分超出範圍 (0-5): {score_str}"
        except (ValueError, AttributeError) as e:
            return None, f"無法解析評分格式: {score_str} - {str(e)}"
    
    @staticmethod
    def validate_product_id(product_id: Any) -> Tuple[Optional[int], Optional[str]]:
        """
        驗證商品ID格式
        """
        if not product_id:
            return None, "商品ID為空值"
        
        try:
            pid = int(str(product_id).strip())
            if pid > 0:
                return pid, None
            else:
                return None, f"商品ID必須為正整數: {product_id}"
        except (ValueError, AttributeError) as e:
            return None, f"無法解析商品ID格式: {product_id} - {str(e)}"
    
    @staticmethod
    def validate_comment_id(comment_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        驗證評論ID格式
        """
        if not comment_id or str(comment_id).strip() == '':
            return None, "評論ID為空值"
        
        comment_id = str(comment_id).strip()
        if len(comment_id) > 100:
            return None, f"評論ID過長 (最大100字元): {comment_id}"
        
        return comment_id, None
    
    @staticmethod
    def validate_json_field(json_str: str) -> Tuple[Optional[str], Optional[str]]:
        """
        驗證JSON格式欄位
        """
        if not json_str or str(json_str).strip() == '' or str(json_str).lower() == 'nan':
            return None, None  # 空值不算錯誤
        
        json_str = str(json_str).strip()
        
        try:
            # 嘗試標準JSON解析
            json.loads(json_str)
            return json_str, None
        except (json.JSONDecodeError, TypeError):
            # 如果是Python列表格式（單引號），轉換為標準JSON格式
            if json_str.startswith('[') and json_str.endswith(']') and "'" in json_str:
                try:
                    # 替換單引號為雙引號
                    json_str_converted = json_str.replace("'", '"')
                    json.loads(json_str_converted)  # 驗證轉換後的格式
                    return json_str_converted, None
                except (json.JSONDecodeError, TypeError) as e:
                    return None, f"轉換後JSON格式錯誤: {json_str} - {str(e)}"
            else:
                return None, f"JSON格式錯誤: {json_str}"
    
    @staticmethod
    def validate_boolean_field(bool_str: str) -> Tuple[Optional[bool], Optional[str]]:
        """
        驗證布林值格式
        """
        if not bool_str or str(bool_str).strip() == '' or str(bool_str).lower() == 'nan':
            return None, None  # 空值不算錯誤
        
        bool_str = str(bool_str).strip().lower()
        
        if bool_str in ['true', '1', 'yes', '是']:
            return True, None
        elif bool_str in ['false', '0', 'no', '否']:
            return False, None
        else:
            return None, f"無法解析布林值: {bool_str}"
    
    @staticmethod
    def validate_integer_field(int_str: str) -> Tuple[Optional[int], Optional[str]]:
        """
        驗證整數格式
        """
        if not int_str or str(int_str).strip() == '' or str(int_str).lower() == 'nan':
            return None, None  # 空值不算錯誤
        
        try:
            value = int(float(str(int_str).strip()))
            return value, None
        except (ValueError, AttributeError) as e:
            return None, f"無法解析整數格式: {int_str} - {str(e)}"
    
    @staticmethod
    def validate_text_field(text: str, max_length: int = None) -> Tuple[Optional[str], Optional[str]]:
        """
        驗證文字欄位
        """
        if not text or str(text).strip() == '' or str(text).lower() == 'nan':
            return None, None  # 空值不算錯誤
        
        text = str(text).strip()
        
        if max_length and len(text) > max_length:
            return None, f"文字長度超過限制 ({max_length}字元): {text[:50]}..."
        
        return text, None
    
    @staticmethod
    def clean_comment_text(text: str) -> str:
        """
        清理評論文字
        """
        if not text:
            return ""
        
        # 移除多餘的換行和逗號
        text = str(text).replace(',', '，')
        text = '，'.join(text.splitlines())
        text = re.sub(r'[，]+', '，', text)
        
        return text.strip()
    
    @staticmethod
    def validate_url(url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        驗證URL格式
        """
        if not url or str(url).strip() == '' or str(url).lower() == 'nan':
            return None, None  # 空值不算錯誤
        
        url = str(url).strip()
        
        # 簡單的URL驗證
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        if url_pattern.match(url):
            return url, None
        else:
            return None, f"URL格式不正確: {url}" 