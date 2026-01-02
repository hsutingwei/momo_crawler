# -*- coding: utf-8 -*-
"""
feature_transformer.py
Feature Engineering Pipeline with Leakage Prevention

實現規格參考：implementation_plan.md
- Phase 2: Feature Transform Profiles (phaseA/phaseB)
- Phase 3: Data Leakage Prevention (train_fold_only fit scope)
"""

import os
import json
import pickle
import warnings
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler


# =============================================================================
# Feature Transform Profiles
# =============================================================================

TRANSFORM_PROFILES = {
    'none': {
        'impute_strategy': 'none',
        'clip_profile': 'none',
        'log_transform_features': [],
        'scaler': 'none'
    },
    'phaseA': {
        'impute_strategy': 'none',  # PhaseA: 不做 imputation (當前數據無缺失值)
        'clip_profile': 'none',     # PhaseA: 不做 clipping，只產 outlier_report
        'log_transform_features': ['price'],  # 價格做 log
        'scaler': 'none'            # PhaseA: 不做 scaling
    },
    'phaseB': {
        'impute_strategy': 'median',  # PhaseB: 可選用 median imputation
        'clip_profile': 'p99',        # PhaseB: 依 outlier_report 決定 (p99 或 p995)
        'log_transform_features': ['price'],
        'scaler': 'standard'          # PhaseB: StandardScaler
    }
}


# =============================================================================
# FeatureTransformer Class
# =============================================================================

class FeatureTransformer:
    """
    Feature Engineering Pipeline with Leakage Prevention
    
    規格：implementation_plan.md #2.1-2.5
    
    支援功能：
    - Imputation (median/zero/none)
    - Clipping (p99/p995/none)
    - Log Transform
    - Scaling (standard/robust/none)
    - Outlier Reporting
    
    Leakage Prevention:
    - 所有 fit 操作只在 train fold 上執行
    - transform 可應用到 train/val/test
    """
    
    def __init__(
        self,
        profile: str = 'phaseA',
        feature_whitelist: Optional[List[str]] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化 FeatureTransformer
        
        Args:
            profile: 'none' | 'phaseA' | 'phaseB'
            feature_whitelist: 允許的特徵列表（來自 get_feature_whitelist）
            custom_config: 自定義配置（覆蓋 profile 預設值）
        """
        self.profile = profile
        self.feature_whitelist = feature_whitelist
        
        # 載入 profile 配置
        if profile not in TRANSFORM_PROFILES:
            raise ValueError(f"Unknown profile: {profile}")
        
        self.config = TRANSFORM_PROFILES[profile].copy()
        
        # 應用自定義配置
        if custom_config:
            self.config.update(custom_config)
        
        # 初始化學習到的參數（fit 時填充）
        self.impute_values_ = {}
        self.clip_bounds_ = {}
        self.log_transform_features_ = []
        self.scaler_ = None
        
        # Metadata
        self.fitted_ = False
        self.feature_names_ = []
        
    def fit(self, X: pd.DataFrame, feature_names: Optional[List[str]] = None) -> 'FeatureTransformer':
        """
        在 train fold 上學習 transformation 參數
        
        Args:
            X: 訓練集特徵（僅 train fold，不含 val/test）
            feature_names: 特徵名稱列表（若 None 則用 X.columns）
        
        Returns:
            self
        """
        if feature_names is None:
            feature_names = list(X.columns)
        
        self.feature_names_ = feature_names
        
        # 過濾 whitelist
        if self.feature_whitelist:
            feature_names = [f for f in feature_names if f in self.feature_whitelist]
        
        X_work = X[feature_names].copy()
        
        print(f"[FeatureTransformer] Fitting on {len(X_work)} samples, {len(feature_names)} features")
        print(f"  Profile: {self.profile}")
        print(f"  Config: {self.config}")
        
        # Step 1: Imputation
        self._fit_imputation(X_work)
        
        # Step 2: Clipping (學習 bounds，但 phaseA 不實際 clip)
        self._fit_clipping(X_work)
        
        # Step 3: Log Transform (確定要轉換的特徵)
        self._fit_log_transform(X_work)
        
        # Apply log transform for scaler fitting
        X_work = self._transform_log(X_work)
        
        # Step 4: Scaler (在 log transform 之後 fit)
        self._fit_scaler(X_work)
        
        self.fitted_ = True
        print(f"[FeatureTransformer] Fit completed ✅")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        應用 transformation
        
        Args:
            X: 輸入特徵
        
        Returns:
            Transformed 特徵
        """
        if not self.fitted_:
            raise ValueError("FeatureTransformer must be fitted before transform")
        
        # 過濾 whitelist
        feature_names = self.feature_names_
        if self.feature_whitelist:
            feature_names = [f for f in feature_names if f in self.feature_whitelist]
        
        X_work = X[feature_names].copy()
        
        # Step 1: Imputation
        X_work = self._transform_imputation(X_work)
        
        # Step 2: Clipping
        X_work = self._transform_clipping(X_work)
        
        # Step 3: Log Transform
        X_work = self._transform_log(X_work)
        
        # Step 4: Scaling
        X_work = self._transform_scaler(X_work)
        
        return X_work
    
    def fit_transform(self, X: pd.DataFrame, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X, feature_names).transform(X)
    
    # =========================================================================
    # Imputation
    # =========================================================================
    
    def _fit_imputation(self, X: pd.DataFrame) -> None:
        """學習 imputation 值"""
        strategy = self.config['impute_strategy']
        
        if strategy == 'none':
            return
        
        print(f"  [Imputation] Strategy: {strategy}")
        
        for col in X.columns:
            if X[col].isna().any():
                if strategy == 'median':
                    self.impute_values_[col] = X[col].median()
                elif strategy == 'zero':
                    self.impute_values_[col] = 0
                else:
                    raise ValueError(f"Unknown impute strategy: {strategy}")
                
                print(f"    {col}: {self.impute_values_[col]:.4f}")
    
    def _transform_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        """應用 imputation"""
        if not self.impute_values_:
            return X
        
        X = X.copy()
        for col, value in self.impute_values_.items():
            if col in X.columns:
                X[col] = X[col].fillna(value)
        
        return X
    
    # =========================================================================
    # Clipping
    # =========================================================================
    
    def _fit_clipping(self, X: pd.DataFrame) -> None:
        """學習 clipping bounds（並生成 outlier report）"""
        clip_profile = self.config['clip_profile']
        
        # 永遠生成 outlier report（即使不 clip）
        outlier_stats = []
        
        for col in X.columns:
            if X[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                stats = {
                    'feature': col,
                    'p95': X[col].quantile(0.95),
                    'p99': X[col].quantile(0.99),
                    'p995': X[col].quantile(0.995),
                    'max': X[col].max(),
                    'min': X[col].min()
                }
                outlier_stats.append(stats)
                
                # 學習 clip bounds
                if clip_profile == 'p99':
                    self.clip_bounds_[col] = (X[col].quantile(0.01), X[col].quantile(0.99))
                elif clip_profile == 'p995':
                    self.clip_bounds_[col] = (X[col].quantile(0.005), X[col].quantile(0.995))
        
        self.outlier_report_ = pd.DataFrame(outlier_stats)
        
        if clip_profile != 'none':
            print(f"  [Clipping] Profile: {clip_profile}")
            print(f"    Learned bounds for {len(self.clip_bounds_)} features")
    
    def _transform_clipping(self, X: pd.DataFrame) -> pd.DataFrame:
        """應用 clipping"""
        if not self.clip_bounds_:
            return X
        
        X = X.copy()
        for col, (lower, upper) in self.clip_bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower, upper)
        
        return X
    
    # =========================================================================
    # Log Transform
    # =========================================================================
    
    def _fit_log_transform(self, X: pd.DataFrame) -> None:
        """確定要 log transform 的特徵"""
        log_features = self.config['log_transform_features']
        
        # 驗證特徵存在
        self.log_transform_features_ = [f for f in log_features if f in X.columns]
        
        if self.log_transform_features_:
            print(f"  [Log Transform] Features: {self.log_transform_features_}")
    
    def _transform_log(self, X: pd.DataFrame) -> pd.DataFrame:
        """應用 log transform"""
        if not self.log_transform_features_:
            return X
        
        X = X.copy()
        for col in self.log_transform_features_:
            if col in X.columns:
                # log(x + 1) to avoid log(0)
                X[f'{col}_log'] = np.log1p(X[col])
        
        return X
    
    # =========================================================================
    # Scaling
    # =========================================================================
    
    def _fit_scaler(self, X: pd.DataFrame) -> None:
        """訓練 scaler"""
        scaler_type = self.config['scaler']
        
        if scaler_type == 'none':
            return
        
        # 選取數值特徵（排除 log 生成的特徵，因為還沒 transform）
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if scaler_type == 'standard':
            self.scaler_ = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler_ = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler: {scaler_type}")
        
        self.scaler_.fit(X[numeric_cols])
        print(f"  [Scaler] Type: {scaler_type}, Features: {len(numeric_cols)}")
    
    def _transform_scaler(self, X: pd.DataFrame) -> pd.DataFrame:
        """應用 scaler"""
        if self.scaler_ is None:
            return X
        
        X = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        X[numeric_cols] = self.scaler_.transform(X[numeric_cols])
        
        return X
    
    # =========================================================================
    # Save/Load
    # =========================================================================
    
    def save(self, output_dir: str) -> None:
        """保存 transformer 及其參數"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save config
        with open(f'{output_dir}/transform_config.json', 'w') as f:
            json.dump({
                'profile': self.profile,
                'config': self.config,
                'feature_names': self.feature_names_
            }, f, indent=2)
        
        # Save learned parameters
        params = {
            'impute_values': self.impute_values_,
            'clip_bounds': self.clip_bounds_,
            'log_transform_features': self.log_transform_features_
        }
        with open(f'{output_dir}/transform_params.json', 'w') as f:
            json.dump(params, f, indent=2)
        
        # Save scaler
        if self.scaler_ is not None:
            with open(f'{output_dir}/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler_, f)
        
        # Save outlier report
        if hasattr(self, 'outlier_report_'):
            self.outlier_report_.to_csv(f'{output_dir}/outlier_report.csv', index=False)
        
        print(f"[FeatureTransformer] Saved to {output_dir} ✅")
    
    @classmethod
    def load(cls, input_dir: str) -> 'FeatureTransformer':
        """載入已保存的 transformer"""
        # Load config
        with open(f'{input_dir}/transform_config.json', 'r') as f:
            config_data = json.load(f)
        
        transformer = cls(
            profile=config_data['profile'],
            custom_config=config_data['config']
        )
        transformer.feature_names_ = config_data['feature_names']
        
        # Load parameters
        with open(f'{input_dir}/transform_params.json', 'r') as f:
            params = json.load(f)
        
        transformer.impute_values_ = params['impute_values']
        transformer.clip_bounds_ = {k: tuple(v) for k, v in params['clip_bounds'].items()}
        transformer.log_transform_features_ = params['log_transform_features']
        
        # Load scaler
        scaler_path = f'{input_dir}/scaler.pkl'
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                transformer.scaler_ = pickle.load(f)
        
        # Load outlier report
        outlier_path = f'{input_dir}/outlier_report.csv'
        if os.path.exists(outlier_path):
            transformer.outlier_report_ = pd.read_csv(outlier_path)
        
        transformer.fitted_ = True
        
        print(f"[FeatureTransformer] Loaded from {input_dir} ✅")
        return transformer


if __name__ == "__main__":
    # 簡單測試
    print("feature_transformer.py loaded successfully")
    
    # 創建模擬數據測試
    np.random.seed(42)
    X_train = pd.DataFrame({
        'price': np.random.uniform(100, 10000, 100),
        'comment_count': np.random.randint(0, 1000, 100),
        'score_mean': np.random.uniform(3, 5, 100)
    })
    
    print("\n測試 phaseA profile:")
    transformer = FeatureTransformer(profile='phaseA')
    X_transformed = transformer.fit_transform(X_train)
    print(f"Transformed shape: {X_transformed.shape}")
    print(f"Transformed columns: {list(X_transformed.columns)}")
