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

keyword = '耳機'
page = 60
ecode = 'utf-8-sig'

# 2022/11/21 由於蝦皮API更新，商品細節資料無法再單純使用request爬取，因此header只留給爬留言使用，而流言的API沒什麼檢查
my_headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
    'if-none-match-': '55b03-6d83b58414a54cb5ffbe81099940f836'
    }     

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
if (True):
    for i in tqdm(range(int(page))):
    #for i in tqdm(range(1)):
        driver.get('https://www.momoshop.com.tw/search/searchShop.jsp?keyword=' + keyword + '&cateLevel=0&_isFuzzy=0&searchType=1&curPage=' + str(i))
        time.sleep(random.randint(5,10))
        # 滾動頁面
        for scroll in range(6):
            driver.execute_script('window.scrollBy(0,1000)')
            time.sleep(random.randint(3,10))
        
        # 2023/04/20 由於使用selenium取得商品有些不穩定，因此以下換成全部使用bs4去解析
        # 取得商品內容
        for block in driver.find_elements(by=By.XPATH, value='//div[contains(@class, "goodsUrl")]'):
            # 將整個網站的Html進行解析
            soup = BeautifulSoup(block.get_attribute('innerHTML'), "html.parser")

            tname = soup.select_one('.prdName').text.strip()
            if len(tname) <= 0:
                print('抓不到資料，直接是空的')
                continue # 沒抓到這個商品就別爬了  
            salseCount = extract_number(soup.select_one('.totalSales').text.strip())
            if (salseCount is None or salseCount < 1000):
                continue # 銷售量小於1000就跳過  
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


        time.sleep(random.randint(20,30)) # 休息久一點

    # 2023/04/20 先將每頁抓到的商品儲存下來，方便後續追蹤並爬蟲
    dic = {
        '商品ID':itemid,
        #'賣家ID':shopid,
        '商品名稱':name,
        '商品連結':link,
        '價格':price,
        '資料已完整爬取':[ False for x in range(len(itemid)) ] ,
    }
    pd.DataFrame(dic).to_csv(keyword +'_商品資料.csv', encoding = ecode, index=False)

    tEnd = time.time()#計時結束
    totalTime = int(tEnd - tStart)
    minute = totalTime // 60
    second = totalTime % 60
    print('資料儲存完成，花費時間（約）： ' + str(minute) + ' 分 ' + str(second) + '秒')



#---------- Part 2. 補上商品的詳細資料，由於多設了爬取的標記，因此爬過的就不會再爬了 ----------
print('---------- 開始進行留言爬蟲 ----------')
tStart = time.time()#計時開始

# 2023/04/20 先取得之前爬下來的紀錄
getData = pd.read_csv(keyword +'_商品資料.csv')
data2 = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
for i in tqdm(range(len(getData))):
    data = []
    # 2023/04/20 備標註已經抓過的，就不用再抓了，這樣就算之前爬到一半被中斷，也不會努力付諸東流
    if getData.iloc[i]['資料已完整爬取']==True:
        continue
    data.append(int(getData.iloc[i]['商品ID']))
    data.append(getData.iloc[i]['商品名稱'])
    data.append(getData.iloc[i]['商品連結'])
    data.append(getData.iloc[i]['價格'])
    #請求商品詳細資料
    #itemDetail = goods_detail(url = data[2], item_id = data[0])
    tmpDetail = get_goods_comments(int(data[0]))
    if (tmpDetail.get("goodsCommentList") is not None):
        itemDetail = tmpDetail.get("goodsCommentList")
        responseCount = int(tmpDetail.get("filterList")[0]['count']) # 共有幾筆留言
        loopCount = responseCount // 10 + (1 if responseCount % 10 > 0 else 0) # 根據留言數，判斷需要call 幾次 web API

        # 抓不到資料就先跳過
        if itemDetail is None:
            print('抓不到商品詳細資料...\n')
            continue

        data.append(True)# 資料已完整爬取

        getData.iloc[i] = data #塞入所有資料
        getData.to_csv(keyword +'_商品資料.csv', encoding = ecode, index=False)

        for obj in itemDetail:
            data2[0].append(int(getData.iloc[i]['商品ID']))
            data2[1].append(getData.iloc[i]['商品名稱'])
            data2[2].append(getData.iloc[i]['商品連結'])
            data2[3].append(getData.iloc[i]['價格'])
            data2[4].append(obj["comment"].replace(',', '，')) #留言
            data2[5].append(obj["commentId"].replace(',', '，')) #留言ID
            data2[6].append(obj["customName"].replace(',', '，')) #留言者名稱
            data2[7].append(obj["date"]) #留言時間
            data2[8].append(obj["goodsType"].replace(',', '，')) #商品 Type
            data2[9].append(obj["imageUrlList"]) #留言 圖
            data2[10].append(obj["isLike"]) #isLike
            data2[11].append(obj["isShowLike"]) #isShowLike
            data2[12].append(obj["likeCount"]) #留言 likeCount
            data2[13].append(obj["replyContent"].replace(',', '，')) #留言 replyContent
            data2[14].append(obj["replyDate"]) #留言 replyDate
            data2[15].append(obj["score"]) #留言星數
            data2[16].append(obj["videoThumbnailImg"]) #videoThumbnailImg
            data2[17].append(obj["videoUrl"]) #videoUrl

        for j in range(2, loopCount + 1):
            tmpDetail = get_goods_comments(int(data[0]), cur_page=j)
            itemDetail = tmpDetail.get("goodsCommentList")
            
            for obj in itemDetail:
                data2[0].append(int(getData.iloc[i]['商品ID']))
                data2[1].append(getData.iloc[i]['商品名稱'])
                data2[2].append(getData.iloc[i]['商品連結'])
                data2[3].append(getData.iloc[i]['價格'])
                data2[4].append(obj["comment"].replace(',', '，')) #留言
                data2[5].append(obj["commentId"].replace(',', '，')) #留言ID
                data2[6].append(obj["customName"].replace(',', '，')) #留言者名稱
                data2[7].append(obj["date"]) #留言時間
                data2[8].append(obj["goodsType"].replace(',', '，')) #商品 Type
                data2[9].append(obj["imageUrlList"]) #留言 圖
                data2[10].append(obj["isLike"]) #isLike
                data2[11].append(obj["isShowLike"]) #isShowLike
                data2[12].append(obj["likeCount"]) #留言 likeCount
                data2[13].append(obj["replyContent"].replace(',', '，')) #留言 replyContent
                data2[14].append(obj["replyDate"]) #留言 replyDate
                data2[15].append(obj["score"]) #留言星數
                data2[16].append(obj["videoThumbnailImg"]) #videoThumbnailImg
                data2[17].append(obj["videoUrl"]) #videoUrl
            time.sleep(random.randint(5,10))
        
        time.sleep(random.randint(45,70)) # 休息久一點
        #time.sleep(random.randint(5,10)) # 休息久一點

        # 每爬5個商品，會再有一次更長的休息
        if i%5 == 0 :
            time.sleep(random.randint(30,150)) 

dic = {
    '商品ID':data2[0],
    #'賣家ID':shopid,
    '商品名稱':data2[1],
    '商品連結':data2[2],
    '價格':data2[3],
    '留言': data2[4] ,
    '留言ID': data2[5],
    '留言者名稱': data2[6],
    '留言時間': data2[7],
    '商品 Type': data2[8],
    '留言 圖': data2[9],
    'isLike': data2[10],
    'isShowLike': data2[11],
    '留言 likeCount': data2[12],
    '留言 replyContent': data2[13],
    '留言 replyDate': data2[14],
    '留言星數': data2[15],
    'videoThumbnailImg': data2[16],
    'videoUrl': data2[17],
}
pd.DataFrame(dic).to_csv(keyword +'_商品留言資料.csv', encoding = ecode, index=False)

tEnd = time.time()#計時結束
totalTime = int(tEnd - tStart)
minute = totalTime // 60
second = totalTime % 60
print('資料儲存完成，花費時間（約）： ' + str(minute) + ' 分 ' + str(second) + '秒')

driver.close() 
