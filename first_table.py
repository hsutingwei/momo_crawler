# -*- coding: utf-8 -*-
"""
初步研究成果：
1. 從 preprocess 資料夾讀取所有 pre_<keyword>_<timestamp>.csv，組成留言 DataFrame
2. 從 crawler 快照檔計算每週銷售增幅
3. 從 seg 資料夾讀取關鍵詞頻特徵 (fin_*.csv)
4. 聚合到 (商品ID, 週) 層級：留言量、平均星數、平均情緒、關鍵詞頻
5. 探索性分析：相關矩陣 & 線性迴歸示例
"""
import os
import glob
import pandas as pd
from snownlp import SnowNLP
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 參數
ROOT = r"C:\YvesProject\中央\線上評論\momo_crawler-main"
keyword = '益生菌'
PRE_DIR = os.path.join(ROOT, 'preprocess')
SEG_DIR = os.path.join(ROOT, 'seg', keyword)
SALES_PATH = os.path.join(ROOT, 'crawler', f'{keyword}_商品銷售快照.csv')

# 1. 載入所有留言並標記週期
msg_list = []
expected_cols = [
    '商品ID','商品名稱','商品連結','價格','銷售數量','留言','留言ID','留言者名稱',
    '留言時間','商品 Type','留言 圖','isLike','isShowLike','留言 likeCount',
    '留言 replyContent','留言 replyDate','留言星數','videoThumbnailImg','videoUrl'
]
for path in glob.glob(os.path.join(PRE_DIR, f'pre_{keyword}_*.csv')):
    df = pd.read_csv(path, encoding='utf-8-sig')
    # 若欄位數多於預期，截斷多餘的尾欄；少於預期視為錯誤
    if df.shape[1] < len(expected_cols):
        raise ValueError(f"讀取 {path} 時欄位數(={df.shape[1]}) 少於預期({len(expected_cols)})")
    if df.shape[1] > len(expected_cols):
        # 截斷多餘列
        df = df.iloc[:, :len(expected_cols)]
        # 設置欄位名稱
    df.columns = expected_cols
    # 轉換時間與週期
    df['留言時間'] = pd.to_datetime(df['留言時間'], errors='coerce')
    df['週'] = df['留言時間'].dt.to_period('W')
    msg_list.append(df)

if not msg_list:
    raise RuntimeError('找不到任何預處理後的留言檔案')
df_msg = pd.concat(msg_list, ignore_index=True)
df_msg = df_msg.drop_duplicates('留言ID')

# 2. 計算基礎特徵：留言量、平均星數、平均情緒
# 星等 & 情緒

df_msg['情緒'] = df_msg['留言'].fillna('').map(lambda x: SnowNLP(str(x)).sentiments)
feat = (
    df_msg.groupby(['商品ID','週'])
          .agg(
              留言量=('留言ID','size'),
              平均星數=('留言星數','mean'),
              平均情緒=('情緒','mean')
          )
          .reset_index()
)

# 3. 關鍵詞頻 (取 fin_* 檔，對預設關鍵詞計數)
kw_cols = ['效果','包裝','價格','客服','味道']  # 可依需求調整
kw_list = []
import csv

for path in glob.glob(os.path.join(SEG_DIR, f'fin_{keyword}_*.csv')):
    # 用 csv.reader 讀取每一行 tokens
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    # 建立 DataFrame: 每行即是一則清洗後詞列表
    dfkw = pd.DataFrame({'kw_list': rows})
    # 對應商品ID、週 (假設順序與 df_msg 對齊)
    dfkw['商品ID'] = df_msg['商品ID'].values[:len(rows)]
    dfkw['週'] = df_msg['週'].values[:len(rows)]
    # 計算詞頻
    for w in kw_cols:
        dfkw[w] = dfkw['kw_list'].map(lambda lst: lst.count(w))
    kw_list.append(dfkw[['商品ID','週'] + kw_cols])

df_kw = pd.concat(kw_list, ignore_index=True)
kw_feat = df_kw.groupby(['商品ID','週'])[kw_cols].sum().reset_index()

# 合併留言特徵與關鍵詞特徵
df_all = feat.merge(kw_feat, on=['商品ID','週'], how='left').fillna(0)

# 4. 載入銷售快照並計算週增幅

df_sales = pd.read_csv(SALES_PATH, encoding='utf-8-sig')
df_sales['週'] = pd.to_datetime(df_sales['擷取時間'].astype(str).str[:8]).dt.to_period('W')
# 有時候數值帶 ".0" 或千分位逗號，先移除再轉型
# 例如 '8000.0' -> '8000'
sales_num = (
    df_sales['銷售數量']
      .astype(str)
      .str.replace(',','')
      .str.replace(r'\.0+$','', regex=True)
)
df_sales['銷售數量'] = pd.to_numeric(sales_num, errors='coerce').fillna(0).astype(int)
sales_week = (
    df_sales.sort_values('擷取時間')
            .groupby(['商品ID','週'])['銷售數量']
            .last()
            .reset_index()
)
sales_week['銷售增幅'] = sales_week.groupby('商品ID')['銷售數量'].pct_change().fillna(0)

# 合併銷售增幅
df_final = df_all.merge(sales_week, on=['商品ID','週'], how='left').fillna(0)

# 5. 探索與迴歸分析
print(df_final[['銷售增幅','留言量','平均星數','平均情緒'] + kw_cols]
      .corr(method='spearman')['銷售增幅'])

# 線性迴歸示例
target = '銷售增幅'
features = ['留言量','平均星數','平均情緒'] + kw_cols
X = df_final[features]
y = df_final[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression().fit(X_train, y_train)
print('Coefficients:', dict(zip(features, model.coef_)))
print('R2:', model.score(X_test, y_test))