# -*- coding: utf-8 -*-
"""
artifact_manager.py
實驗產物生成與管理 (Artifact Generation & Management)

實現規格參考：implementation_plan.md
- Phase 5: Artifacts & Reproducibility (每次 run 生成 15+ artifacts)
- Specification #5: Mandatory Artifacts, OOF/Test Separation (強制產物, OOF/Test 分離)
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
import numpy as np


# =============================================================================
# Artifact Manager
# =============================================================================

class ArtifactManager:
    """
    管理實驗 artifacts 的生成與保存
    
    規格：implementation_plan.md #5
    """
    
    def __init__(self, run_id: str, output_dir: str):
        """
        初始化 ArtifactManager
        
        Args:
            run_id: 實驗 run ID
            output_dir: 輸出目錄（如 runs/{run_id}）
        """
        self.run_id = run_id
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.artifacts = {}  # 記錄已生成的 artifacts
    
    def save_config(self, config: Dict[str, Any]) -> str:
        """
        保存 run_config.json
        
        規格：#5.1 Mandatory Artifacts
        """
        filepath = os.path.join(self.output_dir, 'run_config.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        self.artifacts['run_config'] = filepath
        return filepath
    
    def save_splits(self, splits_df: pd.DataFrame) -> str:
        """保存 splits.parquet"""
        filepath = os.path.join(self.output_dir, 'splits.parquet')
        splits_df.to_parquet(filepath, index=False)
        
        self.artifacts['splits'] = filepath
        return filepath
    
    def save_labels(self, labels_df: pd.DataFrame) -> str:
        """保存 label.parquet"""
        filepath = os.path.join(self.output_dir, 'label.parquet')
        labels_df.to_parquet(filepath, index=False)
        
        self.artifacts['label'] = filepath
        return filepath
    
    def save_hashes(self, hashes: Dict[str, str]) -> str:
        """
        保存 hashes.json
        
        包含: dataset_hash, split_hash, feature_hash
        """
        filepath = os.path.join(self.output_dir, 'hashes.json')
        with open(filepath, 'w') as f:
            json.dump(hashes, f, indent=2)
        
        self.artifacts['hashes'] = filepath
        return filepath
    
    def save_samples_metadata(self, metadata: Dict[str, Any]) -> str:
        """保存 samples_metadata.json"""
        filepath = os.path.join(self.output_dir, 'samples_metadata.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        self.artifacts['samples_metadata'] = filepath
        return filepath
    
    def save_predictions_oof(self, predictions_df: pd.DataFrame) -> str:
        """
        保存 predictions_oof.parquet
        
        規格：#5.2 OOF Predictions
        必須包含: product_id, fold_id, y_true, y_prob, is_oof=1
        """
        # 確保有 is_oof 欄位
        if 'is_oof' not in predictions_df.columns:
            predictions_df = predictions_df.copy()
            predictions_df['is_oof'] = 1
        
        filepath = os.path.join(self.output_dir, 'predictions_oof.parquet')
        predictions_df.to_parquet(filepath, index=False)
        
        self.artifacts['predictions_oof'] = filepath
        return filepath
    
    def save_predictions_test(self, predictions_df: pd.DataFrame) -> str:
        """
        保存 predictions_test.parquet
        
        規格：#5.2 Test Predictions
        必須包含: product_id, y_true, y_prob, y_pred, threshold, is_oof=0
        """
        # 確保有 is_oof 欄位
        if 'is_oof' not in predictions_df.columns:
            predictions_df = predictions_df.copy()
            predictions_df['is_oof'] = 0
        
        filepath = os.path.join(self.output_dir, 'predictions_test.parquet')
        predictions_df.to_parquet(filepath, index=False)
        
        self.artifacts['predictions_test'] = filepath
        return filepath
    
    def save_metrics(self, metrics: Dict[str, Any]) -> str:
        """
        保存 metrics.json
        
        規格：#5.2, #17 Fixed Schema
        必須包含: oof_by_fold, oof_aggregate, oof_global, test, threshold
        """
        filepath = os.path.join(self.output_dir, 'metrics.json')
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.artifacts['metrics'] = filepath
        return filepath
    
    def save_chosen_threshold(self, threshold_info: Dict[str, Any]) -> str:
        """保存 chosen_threshold.json"""
        filepath = os.path.join(self.output_dir, 'chosen_threshold.json')
        with open(filepath, 'w') as f:
            json.dump(threshold_info, f, indent=2)
        
        self.artifacts['chosen_threshold'] = filepath
        return filepath
    
    def save_feature_list(self, features: List[str]) -> str:
        """保存 feature_list.json"""
        filepath = os.path.join(self.output_dir, 'feature_list.json')
        with open(filepath, 'w') as f:
            json.dump({'features': sorted(features)}, f, indent=2)
        
        self.artifacts['feature_list'] = filepath
        return filepath
    
    def save_feature_hash_context(self, context: Dict[str, Any]) -> str:
        """
        保存 feature_hash_context.json
        
        規格：#10, #5.3
        """
        filepath = os.path.join(self.output_dir, 'feature_hash_context.json')
        with open(filepath, 'w') as f:
            json.dump(context, f, indent=2)
        
        self.artifacts['feature_hash_context'] = filepath
        return filepath
    
    def save_imbalance_report(self, report: Dict[str, Any]) -> str:
        """
        保存 imbalance_report.json
        
        規格：#21
        """
        filepath = os.path.join(self.output_dir, 'imbalance_report.json')
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.artifacts['imbalance_report'] = filepath
        return filepath
    
    def save_leakage_checks(self, checks: Dict[str, Any]) -> str:
        """保存 leakage_checks.json"""
        filepath = os.path.join(self.output_dir, 'leakage_checks.json')
        with open(filepath, 'w') as f:
            json.dump(checks, f, indent=2)
        
        self.artifacts['leakage_checks'] = filepath
        return filepath
    
    def save_feature_importance(self, importance_df: pd.DataFrame) -> str:
        """保存 feature_importance.csv"""
        filepath = os.path.join(self.output_dir, 'feature_importance.csv')
        importance_df.to_csv(filepath, index=False)
        
        self.artifacts['feature_importance'] = filepath
        return filepath
    
    def get_artifact_summary(self) -> Dict[str, Any]:
        """獲取已生成 artifacts 的摘要"""
        return {
            'run_id': self.run_id,
            'output_dir': self.output_dir,
            'total_artifacts': len(self.artifacts),
            'artifacts': self.artifacts,
            'generated_at': datetime.now().isoformat()
        }
    
    def save_artifact_summary(self) -> str:
        """保存 artifact summary"""
        summary = self.get_artifact_summary()
        filepath = os.path.join(self.output_dir, 'artifact_summary.json')
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return filepath
    
    def verify_mandatory_artifacts(self) -> Dict[str, Any]:
        """
        驗證所有必須的 artifacts 是否已生成
        
        規格：#5.1-5.5
        """
        mandatory = [
            'run_config',
            'splits',
            'label',
            'hashes',
            'samples_metadata',
            'predictions_oof',
            'predictions_test',
            'metrics',
            'chosen_threshold',
            'feature_list',
            'feature_hash_context'
        ]
        
        missing = [a for a in mandatory if a not in self.artifacts]
        
        return {
            'all_present': len(missing) == 0,
            'mandatory_count': len(mandatory),
            'generated_count': len([a for a in mandatory if a in self.artifacts]),
            'missing': missing,
            'total_artifacts': len(self.artifacts)
        }


# =============================================================================
# Baseline Best Params Management (for Ablation)
# =============================================================================

def save_baseline_best_params(
    output_path: str,
    best_params: Dict[str, Any],
    split_hash: str,
    dataset_hash: str,
    feature_transform_profile: str,
    imbalance_mode: str,
    scale_pos_weight_scope: str,
    threshold_mode: str,
    cv_metric: str = 'f1',
    tuning_trials: int = 50
) -> None:
    """
    保存 baseline_best_params.json
    
    規格：#12, #25
    """
    params = {
        'split_hash': split_hash,
        'dataset_hash': dataset_hash,
        'feature_transform_profile': feature_transform_profile,
        'imbalance_mode': imbalance_mode,
        'scale_pos_weight_scope': scale_pos_weight_scope,
        'threshold_mode': threshold_mode,
        'cv_metric': cv_metric,
        'tuning_trials': tuning_trials,
        'best_params': best_params,
        'created_at': datetime.now().isoformat()
    }
    
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"✅ Baseline 參數已保存至 {output_path}")


def load_and_verify_baseline_params(
    params_path: str,
    current_split_hash: str,
    current_dataset_hash: str,
    fail_on_mismatch: bool = True
) -> Dict[str, Any]:
    """
    載入並驗證 baseline_best_params.json
    
    規格：#12 Guardrail
    """
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # 驗證 hashes
    mismatches = []
    
    if params['split_hash'] != current_split_hash:
        mismatches.append(f"split_hash: {params['split_hash']} vs {current_split_hash}")
    
    if params['dataset_hash'] != current_dataset_hash:
        mismatches.append(f"dataset_hash: {params['dataset_hash']} vs {current_dataset_hash}")
    
    if mismatches and fail_on_mismatch:
        raise ValueError(
            f"❌ Hash 不匹配 (Hash Mismatch)! 無法使用鎖定的參數:\n" +
            "\n".join(mismatches)
        )
    
    if mismatches:
        print(f"⚠️  警告: 檢測到 Hash 不匹配:\n" + "\n".join(mismatches))
    
    return params


if __name__ == "__main__":
    print("artifact_manager.py 加載成功")
    
    # 測試
    import tempfile
    import shutil
    
    test_dir = tempfile.mkdtemp()
    print(f"\n測試目錄: {test_dir}")
    
    try:
        # 創建 ArtifactManager
        manager = ArtifactManager('test_run_001', test_dir)
        
        # 保存各種 artifacts
        manager.save_config({'test': True, 'mode': 'paper'})
        manager.save_hashes({
            'dataset_hash': 'abc123',
            'split_hash': 'def456',
            'feature_hash': 'ghi789'
        })
        manager.save_samples_metadata({
            'total_products': 1000,
            'positives': 50,
            'positive_rate': 0.05
        })
        
        # 驗證
        verification = manager.verify_mandatory_artifacts()
        print(f"\n驗證結果:")
        print(f"  Generated: {verification['generated_count']}/{verification['mandatory_count']}")
        print(f"  Missing: {verification['missing']}")
        
        # Save summary
        summary_path = manager.save_artifact_summary()
        print(f"\n✅ Summary 已保存至 {summary_path}")
        
    finally:
        shutil.rmtree(test_dir)
        print(f"\n清理測試目錄")
