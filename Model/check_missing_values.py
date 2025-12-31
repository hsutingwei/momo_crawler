"""
檢查數據集特徵的缺失值情況
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.data_loader import load_product_level_training_set
import pandas as pd

def check_missing_values():
    print("載入數據集...")
    
    # 使用基本參數載入
    X_dense, X_tfidf, y, meta, vocab = load_product_level_training_set(
        date_cutoff="2025-06-25",
        top_n=100,
        vocab_mode="global",
        label_delta_threshold=10,
        label_strategy="hybrid",
        label_params={"ratio_threshold": 1.0, "delta_threshold": 10},
        min_comments=0
    )
    
    print(f"\n數據集大小：{len(y)} 筆")
    print(f"Dense 特徵：{X_dense.shape[1]} 維")
    print(f"TF-IDF 特徵：{X_tfidf.shape[1]} 維")
    
    # 檢查 Dense 特徵的缺失值
    print("\n" + "="*60)
    print("Dense 特徵缺失值檢查")
    print("="*60)
    
    missing_info = []
    for col in X_dense.columns:
        null_count = X_dense[col].isnull().sum()
        if null_count > 0:
            null_pct = (null_count / len(X_dense)) * 100
            missing_info.append({
                'feature': col,
                'null_count': null_count,
                'null_pct': f"{null_pct:.2f}%"
            })
    
    if missing_info:
        print(f"\n發現 {len(missing_info)} 個特徵有缺失值：\n")
        for info in sorted(missing_info, key=lambda x: x['null_count'], reverse=True):
            print(f"  {info['feature']:<30} {info['null_count']:>6} ({info['null_pct']:>6})")
    else:
        print("\n✅ 所有 Dense 特徵都沒有缺失值！")
    
    # 檢查數值統計
    print("\n" + "="*60)
    print("特徵數值範圍檢查（前 10 個特徵）")
    print("="*60)
    print(X_dense.iloc[:, :10].describe().T[['min', 'max', 'mean', 'std']])
    
    # 檢查是否有 inf 或極端值
    print("\n" + "="*60)
    print("極端值檢查")
    print("="*60)
    
    inf_features = []
    for col in X_dense.columns:
        if X_dense[col].dtype in ['float64', 'float32']:
            inf_count = (X_dense[col] == float('inf')).sum() + (X_dense[col] == float('-inf')).sum()
            if inf_count > 0:
                inf_features.append({'feature': col, 'inf_count': inf_count})
    
    if inf_features:
        print(f"\n⚠️  發現 {len(inf_features)} 個特徵有 inf 值：\n")
        for info in inf_features:
            print(f"  {info['feature']:<30} {info['inf_count']:>6}")
    else:
        print("\n✅ 沒有發現 inf 值！")

if __name__ == "__main__":
    check_missing_values()
