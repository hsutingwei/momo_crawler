# -*- coding: utf-8 -*-
import requests
import json
import pandas as pd
import time
from seleniumwire import webdriver # 需安裝：pip install selenium-wire
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import re
import random
import zlib
from tqdm import tqdm
from bs4 import BeautifulSoup
import os
from datetime import datetime
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

keyword = '維他命'
page = 60
ecode = 'utf-8-sig'
product_csv_path = f'{keyword}_商品資料.csv'
snapshot_path = f'crawler/{keyword}_商品銷售快照.csv'
current_time_str = datetime.now().strftime('%Y%m%d%H%M%S')
# 是否包含第一次「舊資料」中的銷售量當成一筆快照記錄
include_original_snapshot = True

def is_valid_row(line):
    # 判斷行是否為合法的開頭（例如：商品ID 開頭是數字）
    return re.match(r'^\d+,', line.strip()) is not None

def clean_broken_csv(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8-sig') as infile, open(output_path, 'w', encoding='utf-8-sig') as outfile:
        buffer = ''
        for line in infile:
            line = line.rstrip('\n')
            if is_valid_row(line):
                # 若上一行有累積，寫入
                if buffer:
                    outfile.write(buffer + '\n')
                buffer = line
            else:
                buffer += ' ' + line  # 合併為同一行（中間加空格避免直接黏住）
        
        # 最後一行補寫
        if buffer:
            outfile.write(buffer + '\n')

def extract_number(input_string):
    """
    從字串中提取數字，支持千分位符號，並將結果轉換為 int 型別。

    :param input_string: 含有數字的字串
    :return: 整數型別的數字，若無數字則回傳 None
    """
    # 定義正規表達式，匹配含有千分位的數字
    match = re.search(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?', input_string)
    if match:
        # 去掉千分位符號，並轉換為 int
        number_str = match.group(0).replace(',', '')
        return int(float(number_str))  # 支援整數和小數
    return None
    
   
def get_goods_comments(goods_code, cur_page=1, cust_no="", filter_type="total", host="web", multi_filter_type=None):
    """
    呼叫 momoshop API 獲取商品評論列表

    :param goods_code: 商品代碼
    :param cur_page: 當前頁面 (預設為 1)
    :param cust_no: 使用者代碼 (預設為空字串)
    :param filter_type: 篩選類型 (預設為 'total')
    :param host: 請求來源 (預設為 'web')
    :param multi_filter_type: 多重篩選類型 (預設為 ['hasComment'])
    :return: 回傳 JSON 格式的評論資料
    """
    if multi_filter_type is None:
        multi_filter_type = ["hasComment"]

    url = "https://eccapi.momoshop.com.tw/user/getGoodsCommentList"
    headers = {
        "Content-Type": "application/json",  # 設定為 JSON 格式
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    payload = {
        "curPage": cur_page,
        "custNo": cust_no,
        "filterType": filter_type,
        "goodsCode": str(goods_code),
        "host": host,
        "multiFilterType": multi_filter_type
    }

    try:
        # 發送 POST 請求
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # 若 HTTP 狀態碼為 4xx 或 5xx，則拋出異常

        # 解析回應 JSON 資料
        data = response.json()
        return data
    except requests.RequestException as e:
        print(f"HTTP 請求錯誤: {e}")
        return None
    except json.JSONDecodeError:
        print("無法解析回應的 JSON 資料")
        return None

def get_current_sales(goods_code, host="momoshop"):
    url = "https://eccapi.momoshop.com.tw/user/getGoodsComment"
    headers = {
        "Content-Type": "application/json",  # 設定為 JSON 格式
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    payload = {
        "goodsCode": str(goods_code),
        "host": host,
    }

    try:
        # 發送 POST 請求
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # 若 HTTP 狀態碼為 4xx 或 5xx，則拋出異常

        # 解析回應 JSON 資料
        data = response.json()
        if data is None or data.get("saleCount") is None:
            print(f"【警告】無法取得 {goods_code} 的銷售數")
            return 0
        sales_text = data.get("saleCount")
        count = extract_number(sales_text)
        if count is None:
            return 0
        return str(count) + ('萬' if sales_text.endswith('萬') else '')
    except requests.RequestException as e:
        print(f"HTTP 請求錯誤: {e}")
        return None
    except json.JSONDecodeError:
        print("無法解析回應的 JSON 資料")
        return None

def save_sales_snapshot_long_format():
    if not os.path.exists(product_csv_path):
        print(f"找不到商品資料檔案：{product_csv_path}")
        return

    df = pd.read_csv(product_csv_path, encoding=ecode)
    all_rows = []

    # ✅ 加入第一次原始資料的銷售快照
    if include_original_snapshot:
        for _, row in df.iterrows():
            all_rows.append([
                row['商品ID'],
                row['商品名稱'],
                row['價格'],
                row['銷售數量'],
                row['商品連結'],
                current_time_str
            ])
        print(f'已從原始商品資料加入 {len(df)} 筆初始快照')

    # ✅ 爬取當下銷售量
    service = ChromeService(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_experimental_option("prefs", {"profile.default_content_setting_values.notifications": 2})
    driver = webdriver.Chrome(service=service, options=options)

    for _, row in df.iterrows():
        product_id = row['商品ID']
        product_name = row['商品名稱']
        price = row['價格']
        link = row['商品連結']

        latest_sales = get_current_sales(driver, link)
        if latest_sales is None:
            print(f"【警告】無法取得 {product_name} 的銷售數")
            latest_sales = 0

        all_rows.append([
            product_id,
            product_name,
            price,
            latest_sales,
            link,
            current_time_str
        ])
        time.sleep(random.uniform(1, 2.2))

    driver.quit()

    if not all_rows:
        print("⚠ 無可寫入的快照資料")
        return

    # 準備新快照資料表
    df_snapshot = pd.DataFrame(all_rows, columns=[
        '商品ID', '商品名稱', '價格', '銷售數量', '商品連結', '擷取時間'
    ])

    os.makedirs('crawler', exist_ok=True)

    # ✅ 若已有快照檔案，先讀出比對，避免重複寫入
    if os.path.exists(snapshot_path):
        try:
            df_existing = pd.read_csv(snapshot_path, encoding=ecode, dtype={'商品ID': str})
            # 轉型後進行去重（根據商品ID+擷取時間）
            before_len = len(df_snapshot)
            df_combined = pd.concat([df_existing, df_snapshot], ignore_index=True)
            df_combined.drop_duplicates(subset=['商品ID', '擷取時間'], keep='first', inplace=True)
            new_records = df_combined[~df_combined.duplicated(subset=['商品ID', '擷取時間'], keep='last')]

            df_snapshot = new_records[df_snapshot.columns]
            actual_new = len(df_snapshot)
            if actual_new == 0:
                print("🚫 沒有新增的快照資料，跳過寫入")
                return
            else:
                print(f"✅ 實際寫入 {actual_new} 筆去重後的新快照資料")
                df_snapshot.to_csv(snapshot_path, mode='a', encoding=ecode, index=False, header=False)
        except Exception as e:
            print(f"❌ 讀取或處理現有快照檔案時出錯：{e}")
    else:
        # 第一次建立快照檔
        df_snapshot.to_csv(snapshot_path, encoding=ecode, index=False)
        print(f"✅ 首次建立快照檔，寫入 {len(df_snapshot)} 筆")

# 自動下載ChromeDriver
service = ChromeService(executable_path=ChromeDriverManager().install())

# 關閉通知提醒
options = webdriver.ChromeOptions()
prefs = {"profile.default_content_setting_values.notifications" : 2}
options.add_experimental_option("prefs",prefs)
# 不載入圖片，提升爬蟲速度
# options.add_argument('blink-settings=imagesEnabled=false') 

# 開啟瀏覽器
driver = webdriver.Chrome(service=service, chrome_options=options)
time.sleep(random.randint(5,10))

# 開啟網頁，進到首頁
driver.get('https://www.momoshop.com.tw' )
time.sleep(random.randint(5,10))

#---------- Part 1. 主要先抓下商品名稱與連結，之後再慢慢補上詳細資料 ----------
print('---------- 開始進行商品爬蟲 ----------')
tStart = time.time()#計時開始
# 準備用來存放資料的陣列
itemid = []
shopid =[]
name = []
link = []
price = []
sales = []
if (False):
    for i in tqdm(range(int(page))):
    #for i in tqdm(range(1)):
        driver.get('https://www.momoshop.com.tw/search/searchShop.jsp?keyword=' + keyword + '&cateLevel=0&_isFuzzy=0&searchType=1&curPage=' + str(i))
        time.sleep(random.randint(2,4))
        # 滾動頁面
        for scroll in range(6):
            driver.execute_script('window.scrollBy(0,1000)')
            time.sleep(random.randint(2,3))
        
        # 取得商品內容
        for block in driver.find_elements(by=By.XPATH, value='//div[contains(@class, "goodsUrl")]'):
            # 將整個網站的Html進行解析
            soup = BeautifulSoup(block.get_attribute('innerHTML'), "html.parser")

            tname = soup.select_one('.prdName').text.strip()
            if len(tname) <= 0:
                print('抓不到資料，直接是空的')
                continue # 沒抓到這個商品就別爬了  
            tmpSales = soup.select_one('.totalSales').text.strip()
            salseCount = str(extract_number(tmpSales)) + ("萬" if tmpSales.endswith("萬") else "")
            #if (salseCount is None or salseCount < 1000):
                #continue # 銷售量小於1000就跳過  
            tprice = soup.select_one('.price > b').text.strip()
            tlink = soup.select_one('.goods-img-url').get('href')
            # 使用正規表達式匹配 i_code 的值
            match = re.search(r'i_code=([^&]+)', tlink)

            i_code = ''
            # 獲取匹配到的值
            if match:
                i_code = match.group(1)

            # 到這邊確認都有抓到資料，才將它塞入陣列，否則可能會有缺漏
            link.append(tlink)
            itemid.append(i_code)
            #shopid.append(theshopid)
            name.append(tname)
            price.append(tprice)
            sales.append(salseCount)


        time.sleep(random.randint(2,4)) # 休息久一點

    # 先將每頁抓到的商品儲存下來，方便後續追蹤並爬蟲
    dic = {
        '商品ID':itemid,
        #'賣家ID':shopid,
        '商品名稱':name,
        '商品連結':link,
        '價格':price,
        '銷售數量': sales,
        '資料已完整爬取':[ False for x in range(len(itemid)) ] ,
    }
    pd.DataFrame(dic).to_csv(product_csv_path, encoding = ecode, index=False)

    tEnd = time.time()#計時結束
    totalTime = int(tEnd - tStart)
    minute = totalTime // 60
    second = totalTime % 60
    print('資料儲存完成，花費時間（約）： ' + str(minute) + ' 分 ' + str(second) + '秒')

#print('---------- 開始進行銷售數量快照爬蟲 ----------')
#save_sales_snapshot_long_format()

#---------- Part 2. 補上商品的詳細資料，由於多設了爬取的標記，因此爬過的就不會再爬了 ----------
print('---------- 開始進行留言爬蟲（只抓取新留言） ----------')
tStart = time.time()

# 當前爬蟲時間作為檔名用
comments_file_path = f'crawler/{keyword}_商品留言資料_{current_time_str}.csv'

# 嘗試讀取過去已爬留言ID，避免重複爬取
existing_comment_ids = set()
all_existing_files = [f for f in os.listdir('crawler') if f.startswith(keyword + '_商品留言資料_') and f.endswith('.csv')]

for file in all_existing_files:
    original_path = os.path.join('crawler', file)
    fixed_path = original_path.replace('.csv', '_fixed_temp.csv')

    # 修復該檔案的換行錯誤到暫存檔
    clean_broken_csv(original_path, fixed_path)

    try:
        df = pd.read_csv(fixed_path, encoding=ecode)
        existing_comment_ids.update(df['留言ID'].astype(str).tolist())
    except Exception as e:
        print(f"讀取修復後檔案 {file} 時發生錯誤: {e}")
    
    # 用完就刪除暫存檔
    os.remove(fixed_path)

# 欄位容器（新增 資料擷取時間）
data2 = [[] for _ in range(20)]  # 原本19個欄位 + 資料擷取時間

getData = pd.read_csv(product_csv_path, encoding=ecode)
all_rows = []

# ✅ 加入第一次原始資料的銷售快照
if include_original_snapshot:
    for _, row in getData.iterrows():
        all_rows.append([
            row['商品ID'],
            row['商品名稱'],
            row['價格'],
            row['銷售數量'],
            row['商品連結'],
            current_time_str
        ])
    print(f'已從原始商品資料加入 {len(df)} 筆初始快照')

for i in tqdm(range(len(getData))):
    if getData.iloc[i]['資料已完整爬取'] == True:
        continue

    product_id = int(getData.iloc[i]['商品ID'])
    basic_info = [
        product_id,
        getData.iloc[i]['商品名稱'],
        getData.iloc[i]['商品連結'],
        getData.iloc[i]['價格'],
        getData.iloc[i]['銷售數量']
    ]

    tmpDetail = get_goods_comments(product_id)
    if tmpDetail is None or tmpDetail.get("goodsCommentList") is None:
        continue

    # 有流言才會重新爬銷售數量的快照 (不然每次都爬會很慢)
    product_id = getData.iloc[i]['商品ID']
    product_name = getData.iloc[i]['商品名稱']
    price = getData.iloc[i]['價格']
    link = getData.iloc[i]['商品連結']

    latest_sales = get_current_sales(product_id)

    all_rows.append([
        product_id,
        product_name,
        price,
        latest_sales,
        link,
        current_time_str
    ])

    all_pages = int(tmpDetail.get("filterList")[0]['count'])
    loopCount = all_pages // 10 + (1 if all_pages % 10 > 0 else 0)

    for page in range(1, loopCount + 1):
        if page > 1:
            tmpDetail = get_goods_comments(product_id, cur_page=page)
            if tmpDetail is None:
                continue
        itemDetail = tmpDetail.get("goodsCommentList")
        if not itemDetail:
            continue

        for obj in itemDetail:
            comment_id = str(obj["commentId"])
            if comment_id in existing_comment_ids:
                continue  # Skip duplicates
            existing_comment_ids.add(comment_id)

            tmp_str = obj["comment"].replace(',', '，')
            tmp_str = '，'.join(tmp_str.splitlines())
            tmp_str = re.sub(r'[，]+', '，', tmp_str)

            data2[0].append(basic_info[0])
            data2[1].append(basic_info[1])
            data2[2].append(basic_info[2])
            data2[3].append(basic_info[3])
            data2[4].append(basic_info[4])
            data2[5].append(tmp_str)
            data2[6].append(comment_id)
            data2[7].append(obj["customName"].replace(',', '，'))
            data2[8].append(obj["date"])
            data2[9].append(obj["goodsType"].replace(',', '，'))
            data2[10].append(obj["imageUrlList"])
            data2[11].append(obj["isLike"])
            data2[12].append(obj["isShowLike"])
            data2[13].append(obj["likeCount"])
            data2[14].append(obj["replyContent"].replace(',', '，'))
            data2[15].append(obj["replyDate"])
            data2[16].append(obj["score"])
            data2[17].append(obj["videoThumbnailImg"])
            data2[18].append(obj["videoUrl"])
            data2[19].append(current_time_str)  # 資料擷取時間

        time.sleep(random.randint(2, 4))

# 儲存新留言資料（以新檔案儲存）
dic = {
    '商品ID': data2[0],
    '商品名稱': data2[1],
    '商品連結': data2[2],
    '價格': data2[3],
    '銷售數量': data2[4],
    '留言': data2[5],
    '留言ID': data2[6],
    '留言者名稱': data2[7],
    '留言時間': data2[8],
    '商品 Type': data2[9],
    '留言 圖': data2[10],
    'isLike': data2[11],
    'isShowLike': data2[12],
    '留言 likeCount': data2[13],
    '留言 replyContent': data2[14],
    '留言 replyDate': data2[15],
    '留言星數': data2[16],
    'videoThumbnailImg': data2[17],
    'videoUrl': data2[18],
    '資料擷取時間': data2[19]
}

# 儲存原始留言檔
pd.DataFrame(dic).to_csv(comments_file_path, encoding=ecode, index=False)

# 對剛儲存的留言檔案進行換行修復處理
fixed_file_path = comments_file_path.replace('.csv', '_fixed.csv')
clean_broken_csv(comments_file_path, fixed_file_path)

# 若要用修正後版本直接覆蓋原始檔案（避免雙份），取消下面註解
os.replace(fixed_file_path, comments_file_path)

# 結束計時
tEnd = time.time()
totalTime = int(tEnd - tStart)
minute = totalTime // 60
second = totalTime % 60
print(f'新留言資料儲存完成，檔名：{comments_file_path}，耗時：約 {minute} 分 {second} 秒')


if not all_rows:
    print("⚠ 無可寫入的快照資料")
else:
    # 準備新快照資料表
    df_snapshot = pd.DataFrame(all_rows, columns=[
        '商品ID', '商品名稱', '價格', '銷售數量', '商品連結', '擷取時間'
    ])

    os.makedirs('crawler', exist_ok=True)

    # ✅ 若已有快照檔案，先讀出比對，避免重複寫入
    if os.path.exists(snapshot_path):
        try:
            df_existing = pd.read_csv(snapshot_path, encoding=ecode, dtype={'商品ID': str})
            # 轉型後進行去重（根據商品ID+擷取時間）
            before_len = len(df_snapshot)
            df_combined = pd.concat([df_existing, df_snapshot], ignore_index=True)
            df_combined.drop_duplicates(subset=['商品ID', '擷取時間'], keep='first', inplace=True)
            new_records = df_combined[~df_combined.duplicated(subset=['商品ID', '擷取時間'], keep='last')]

            df_snapshot = new_records[df_snapshot.columns]
            actual_new = len(df_snapshot)
            if actual_new == 0:
                print("🚫 沒有新增的快照資料，跳過寫入")
            else:
                print(f"✅ 實際寫入 {actual_new} 筆去重後的新快照資料")
                df_snapshot.to_csv(snapshot_path, mode='a', encoding=ecode, index=False, header=False)
        except Exception as e:
            print(f"❌ 讀取或處理現有快照檔案時出錯：{e}")
    else:
        # 第一次建立快照檔
        df_snapshot.to_csv(snapshot_path, encoding=ecode, index=False)
        print(f"✅ 首次建立快照檔，寫入 {len(df_snapshot)} 筆")