# -*- coding: utf-8 -*-
"""
初步研究成果：
1. 從 preprocess 資料夾讀取所有 pre_<keyword>_<timestamp>.csv，組成留言 DataFrame，並加上 snapshot 欄
2. 從 crawler 快照檔計算每次爬蟲時間點的銷售增幅
3. 從 seg 資料夾讀取關鍵詞頻特徵 (fin_*.csv)
4. 聚合到 (商品ID, snapshot) 層級：留言量、平均星數、平均情緒、關鍵詞頻
5. 探索性分析：相關矩陣 & 線性迴歸示例
"""
import os
import glob
import pandas as pd
from snownlp import SnowNLP
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import csv

# 在第 2 步計算「情緒」前，先定義一個安全的 wrapper：
def safe_sentiment(text):
    """
    回傳該文字的情緒分數，若計算過程有除以零等錯誤，回傳 0.0
    """
    try:
        # 只有在非空文字下才呼叫 SnowNLP
        if not text or text.strip()=='':
            return 0.0
        return SnowNLP(text).sentiments
    except ZeroDivisionError:
        return 0.0
    except Exception:
        # 其他意外也回 0
        return 0.0

# 參數
ROOT = r"C:\YvesProject\中央\線上評論\momo_crawler-main"
keyword = '益生菌'
PRE_DIR = os.path.join(ROOT, 'preprocess')
SEG_DIR = os.path.join(ROOT, 'seg', keyword)
SALES_PATH = os.path.join(ROOT, 'crawler', f'{keyword}_商品銷售快照.csv')

# 載入銷售快照，準備 mapping
df_sales = pd.read_csv(SALES_PATH, encoding='utf-8-sig')
# 確保擷取時間為純字串格式
df_sales['擷取時間'] = df_sales['擷取時間'].astype(str)
# 處理銷售數量格式
sales_num = (
    df_sales['銷售數量'].astype(str)
               .str.replace(',','')
               .str.replace(r'\.0+$','', regex=True)
)
df_sales['銷售數量'] = pd.to_numeric(sales_num, errors='coerce').fillna(0).astype(int)
# 建立 mapping: (商品ID, 擷取時間) -> 銷售數量
sales_map = df_sales.set_index(['商品ID','擷取時間'])['銷售數量'].to_dict()

# 1. 載入所有留言並標記 snapshot 時間
expected_cols = [
    '商品ID','商品名稱','商品連結','價格','銷售數量','留言','留言ID','留言者名稱',
    '留言時間','商品 Type','留言 圖','isLike','isShowLike','留言 likeCount',
    '留言 replyContent','留言 replyDate','留言星數','videoThumbnailImg','videoUrl'
]
msg_list = []
for path in glob.glob(os.path.join(PRE_DIR, f'pre_{keyword}_*.csv')):
    # 取檔名 timestamp
    ts = os.path.basename(path).split('_')[-1].replace('.csv','')
    snap_dt = pd.to_datetime(ts, format='%Y%m%d%H%M%S')
    # 讀取留言檔
    df = pd.read_csv(path, encoding='utf-8-sig')
    # 欄位截斷或驗證
    if df.shape[1] < len(expected_cols):
        raise ValueError(f"讀取 {path} 時欄位數(={df.shape[1]}) 少於預期({len(expected_cols)})")
    if df.shape[1] > len(expected_cols):
        df = df.iloc[:, :len(expected_cols)]
    df.columns = expected_cols
    # 標記這次 snapshot
    df['snapshot'] = snap_dt
    # 填入對應銷售數量
    # 以檔名 timestamp 做為 key
    df['銷售數量'] = df.apply(
        lambda row: sales_map.get(
            (str(row['商品ID']), ts),
            0
        ), axis=1
    )
    # 處理留言時間
    df['留言時間'] = pd.to_datetime(df['留言時間'], errors='coerce')
    msg_list.append(df)
if not msg_list:
    raise RuntimeError('找不到任何預處理後的留言檔案')
# 合併並去重留言 ID
all_msg = pd.concat(msg_list, ignore_index=True).drop_duplicates('留言ID')

# 2. 計算基礎特徵，以 (商品ID, snapshot) 聚合
# 用 safe_sentiment 取代直接呼叫，避免 ZeroDivisionError
all_msg['情緒'] = all_msg['留言'].fillna('').map(lambda x: safe_sentiment(str(x)))
feat_snap = (
    all_msg.groupby(['商品ID','snapshot'])
           .agg(
               留言量=('留言ID','size'),
               平均星數=('留言星數','mean'),
               平均情緒=('情緒','mean')
           )
           .reset_index()
)

# 3. 關鍵詞頻 (fin_* 檔，以 snapshot 對齊)
kw_cols = ['效果','包裝','價格','客服','味道']
kw_list = []
for path in glob.glob(os.path.join(SEG_DIR, f'fin_{keyword}_*.csv')):
    # 檔名 timestamp
    filename = os.path.basename(path)
    timestamp = filename.rsplit('_', 1)[1].replace('.csv','')
    snap_dt = pd.to_datetime(timestamp, format='%Y%m%d%H%M%S')

    # 讀取對應的 pre_<keyword>_<timestamp>.csv
    pre_file = os.path.join(PRE_DIR, f'pre_{keyword}_{timestamp}.csv')
    df_pre = pd.read_csv(pre_file, encoding='utf-8-sig')
    # 截斷多餘欄位並設欄名
    if df_pre.shape[1] > len(expected_cols):
        df_pre = df_pre.iloc[:, :len(expected_cols)]
    df_pre.columns = expected_cols
    df_pre['snapshot'] = snap_dt

        # 讀 segmentation tokens
    with open(path, 'r', encoding='utf-8-sig', newline='') as f_seg:
        rows = list(csv.reader(f_seg))
    # 去除空白列（可能最末行為空）
    rows = [r for r in rows if any(cell.strip() for cell in r)]
    dfkw = pd.DataFrame({'kw_list': rows})
    # 若 rows 數量與 df_pre 不符，截取對齊最小長度
    n = min(len(rows), len(df_pre))
    dfkw = dfkw.iloc[:n]
    df_pre_ids = df_pre['商品ID'].astype(str).values[:n]
    dfkw['商品ID'] = df_pre_ids
    dfkw['snapshot'] = snap_dt
    for w in kw_cols:
        dfkw[w] = dfkw['kw_list'].map(lambda lst: lst.count(w))
        dfkw[w] = dfkw['kw_list'].map(lambda lst: lst.count(w))
    kw_list.append(dfkw[['商品ID','snapshot'] + kw_cols])
if not kw_list:
    raise RuntimeError('找不到任何分詞檔案')
kw_feat_snap = pd.concat(kw_list, ignore_index=True)
kw_feat_snap = kw_feat_snap.groupby(['商品ID','snapshot'])[kw_cols].sum().reset_index()
# 4. 統一類型並合併留言與關鍵詞特徵
# 確保 商品ID 同為字串
feat_snap['商品ID'] = feat_snap['商品ID'].astype(str)
kw_feat_snap['商品ID'] = kw_feat_snap['商品ID'].astype(str)
# snapshot 同為 datetime (已設置為 pandas Timestamp 格式)

df_all = feat_snap.merge(
    kw_feat_snap, on=['商品ID','snapshot'], how='left'
).fillna(0)
df_all = feat_snap.merge(
    kw_feat_snap, on=['商品ID','snapshot'], how='left'
).fillna(0)

# 5. 載入銷售快照並計算快照增幅
sales = pd.read_csv(SALES_PATH, encoding='utf-8-sig')
sales['擷取時間'] = pd.to_datetime(sales['擷取時間'], format='%Y%m%d%H%M%S')
sales['商品ID'] = sales['商品ID'].astype(str)
# 按商品ID與時間排序，取當日銷售數
sales = sales.sort_values(['商品ID','擷取時間'])
# 前向填充缺失
sales['銷售數量'] = sales['銷售數量'].astype(str).str.replace(',','')
sales['銷售數量'] = pd.to_numeric(sales['銷售數量'], errors='coerce').fillna(method='ffill').astype(int)
# 計算增幅
sales['銷售增幅'] = sales.groupby('商品ID')['銷售數量'].pct_change().fillna(0)


# --- 校验 & 改用 merge_asof 对齐 ---------------

# 1. 抽样校验：随机选 5 组 (商品ID, snapshot)，看它们前后的销量记录
sample = df_all[['商品ID','snapshot']].drop_duplicates().sample(5, random_state=42)
print("\n— 抽样校验销量走势 —")
for pid, snap in sample.itertuples(index=False):
    window = (sales['商品ID']==pid) & \
             (sales['擷取時間'] >= snap - pd.Timedelta(days=7)) & \
             (sales['擷取時間'] <= snap + pd.Timedelta(days=7))
    sel = sales.loc[window, ['擷取時間','銷售數量']].sort_values('擷取時間')
    print(f"\n商品 {pid} 在 {snap.date()} ±7天 的销量：")
    print(sel.to_string(index=False))

# 2. 按商品用 merge_asof 重新对齐销量到 snapshot
# 先把 df_all 按 (商品ID, snapshot) 排序
df_all = df_all.sort_values(['商品ID','snapshot']).reset_index(drop=True)

# 右表也同理
sales2 = sales.sort_values(['商品ID','擷取時間']).reset_index(drop=True)

# 然后再做 merge_asof
df_all = pd.merge_asof(
    df_all,
    sales2[['商品ID','擷取時間','銷售數量']],
    left_on='snapshot',      # df_all 里要对齐的时间列
    right_on='擷取時間',     # sales2 里对应的时间列
    by='商品ID',             # 按商品ID 分组
    direction='backward'     # 向后找最近一次小于等于 snapshot 的记录
)

# 最后再基于新对齐的 “銷售數量” 计算增幅
df_all['銷售增幅'] = (
    df_all.groupby('商品ID')['銷售數量']
          .pct_change()
          .fillna(0)
)
print("\n— merge_asof 对齐后销量增幅描述 —")
print(df_all['銷售增幅'].describe())



# 6. 合併 df_all 與 sales（按商品ID 和 snapshot 對齊擷取時間）
""" df_all['商品ID'] = df_all['商品ID'].astype(str)
df_all['擷取時間'] = df_all['snapshot']
merge_check = df_all.merge(
    sales[['商品ID','擷取時間','銷售增幅']],
    on=['商品ID','擷取時間'], how='left', indicator=True
) """
# 6. 合併 df_all 與 sales（按商品ID 和 snapshot 對齊擷取時間）
merge_check = df_all.copy()
print("\nMerge status via snapshot:\n", merge_check.shape)

# 7. 最终结果
df_final = merge_check.copy()
# 填補无对应增幅的为 0
df_final['銷售增幅'] = df_final['銷售增幅'].fillna(0)
print("銷售增幅 描述 via snapshot:\n", df_final['銷售增幅'].describe())

# 8. 探索与迴歸分析
corr = df_final[['銷售增幅','留言量','平均星數','平均情緒'] + kw_cols] \
          .corr(method='spearman')['銷售增幅']
print("相關矩陣 (Spearman)\n", corr)

X = df_final[['留言量','平均星數','平均情緒'] + kw_cols]
y = df_final['銷售增幅']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression().fit(X_train, y_train)
print('Coefficients:', dict(zip(X.columns, model.coef_)))
print('R2:', model.score(X_test, y_test))