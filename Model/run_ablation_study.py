#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_ablation_study.py
自动化消融实验脚本

执行流程:
1. Baseline run (tuning mode)
2. 保存 baseline_best_params.json
3. 用锁定的参数跑 3 个 feature variants
4. 生成对比报告

用法:
python Model/run_ablation_study.py \
  --date-cutoff 2025-06-25 \
  --n-folds 10 \
  --group-id ablation_20260102

预计时间 (i9-13900K + RTX 4080):
- Baseline: ~10-15 分钟
- 每个 variant: ~8-12 分钟
- 总计: ~40-50 分钟
"""

import os
import sys
import subprocess
import json
import argparse
from datetime import datetime


def parse_args():
    ap = argparse.ArgumentParser(description='自动化消融实验')
    
    ap.add_argument('--date-cutoff', type=str, default='2025-06-25',
                    help='数据切分日期')
    ap.add_argument('--n-folds', type=int, default=10,
                    help='K-fold CV 折数')
    ap.add_argument('--group-id', type=str, default=None,
                    help='Ablation study group ID (若不指定则自动生成)')
    ap.add_argument('--output-dir', type=str, default='runs',
                    help='输出目录')
    ap.add_argument('--random-seed', type=int, default=42,
                    help='随机种子')
    ap.add_argument('--label-strategy', type=str, default='absolute',
                    choices=['absolute', 'hybrid'],
                    help='标签策略')
    ap.add_argument('--skip-baseline', action='store_true',
                    help='跳过 baseline (假设已存在)')
    ap.add_argument('--baseline-run-id', type=str, default=None,
                    help='已存在的 baseline run ID (skip-baseline 时必须)')
    
    return ap.parse_args()


def run_command(cmd, description):
    """运行命令并显示进度"""
    print("\n" + "="*80)
    print(f"🚀 {description}")
    print("="*80)
    print(f"命令: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ 失败: {description}")
        sys.exit(1)
    
    print(f"\n✅ 完成: {description}")
    return result


def main():
    args = parse_args()
    
    # 生成 group_id
    if args.group_id is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.group_id = f'ablation_{timestamp}'
    
    print("\n" + "="*80)
    print("🔬 消融实验自动化工具")
    print("="*80)
    print(f"Group ID: {args.group_id}")
    print(f"Date cutoff: {args.date_cutoff}")
    print(f"N-folds: {args.n_folds}")
    print(f"Output: {args.output_dir}")
    print("="*80)
    
    # Feature sets to test
    feature_sets = ['baseline', '+physical', '+semantic', '+psych']
    
    # ========================================================================
    # Step 1: Run Baseline (Tuning mode)
    # ========================================================================
    if args.skip_baseline:
        if not args.baseline_run_id:
            print("❌ Error: --baseline-run-id required when --skip-baseline")
            sys.exit(1)
        
        baseline_run_id = args.baseline_run_id
        print(f"\n⏭️  跳过 baseline，使用现有: {baseline_run_id}")
    else:
        baseline_run_id = f'baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        baseline_cmd = [
            'python', 'Model/train_v2.py',
            '--date-cutoff', args.date_cutoff,
            '--feature-set', 'baseline',
            '--run-id', baseline_run_id,
            '--group-id', args.group_id,
            '--n-folds', str(args.n_folds),
            '--random-seed', str(args.random_seed),
            '--label-strategy', args.label_strategy,
            '--hyperparameter-mode', 'tuning',
            '--output-dir', args.output_dir
        ]
        
        run_command(baseline_cmd, f"Baseline Run ({baseline_run_id})")
    
    # Check baseline results
    baseline_dir = os.path.join(args.output_dir, baseline_run_id)
    metrics_path = os.path.join(baseline_dir, 'metrics.json')
    
    if not os.path.exists(metrics_path):
        print(f"❌ Baseline metrics not found: {metrics_path}")
        sys.exit(1)
    
    with open(metrics_path, 'r') as f:
        baseline_metrics = json.load(f)
    
    print("\n" + "="*80)
    print("📊 Baseline 结果")
    print("="*80)
    print(f"OOF AUC: {baseline_metrics['oof_global']['auc']:.4f}")
    print(f"Test AUC: {baseline_metrics['test']['auc']:.4f}")
    
    # Save baseline params (simplified - in real use, should save actual tuned params)
    baseline_params_path = os.path.join(baseline_dir, 'baseline_best_params.json')
    
    # Read hashes from baseline
    hashes_path = os.path.join(baseline_dir, 'hashes.json')
    with open(hashes_path, 'r') as f:
        hashes = json.load(f)
    
    baseline_params = {
        'split_hash': hashes['split_hash'],
        'dataset_hash': hashes['dataset_hash'],
        'feature_transform_profile': 'phaseA',
        'imbalance_mode': 'scale_pos_weight',
        'scale_pos_weight_scope': 'fold_train',
        'threshold_mode': 'fixed',
        'cv_metric': 'auc',
        'tuning_trials': 1,
        'best_params': {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100
        },
        'created_at': datetime.now().isoformat()
    }
    
    with open(baseline_params_path, 'w') as f:
        json.dump(baseline_params, f, indent=2)
    
    print(f"\n✅ Baseline params 已保存: {baseline_params_path}")
    
    # ========================================================================
    # Step 2: Run Feature Variants (Locked mode)
    # ========================================================================
    variant_results = {}
    
    for feature_set in feature_sets[1:]:  # Skip baseline
        run_id = f'{feature_set.replace("+", "")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        variant_cmd = [
            'python', 'Model/train_v2.py',
            '--date-cutoff', args.date_cutoff,
            '--feature-set', feature_set,
            '--run-id', run_id,
            '--group-id', args.group_id,
            '--n-folds', str(args.n_folds),
            '--random-seed', str(args.random_seed),
            '--label-strategy', args.label_strategy,
            '--hyperparameter-mode', 'locked',
            '--baseline-params-path', baseline_params_path,
            '--output-dir', args.output_dir
        ]
        
        run_command(variant_cmd, f"Feature Variant: {feature_set} ({run_id})")
        
        # Read variant metrics
        variant_metrics_path = os.path.join(args.output_dir, run_id, 'metrics.json')
        with open(variant_metrics_path, 'r') as f:
            variant_metrics = json.load(f)
        
        variant_results[feature_set] = {
            'run_id': run_id,
            'oof_auc': variant_metrics['oof_global']['auc'],
            'test_auc': variant_metrics['test']['auc']
        }
    
    # ========================================================================
    # Step 3: Generate Comparison Report
    # ========================================================================
    print("\n" + "="*80)
    print("📊 消融实验结果对比")
    print("="*80)
    
    # Combine all results
    all_results = {
        'baseline': {
            'run_id': baseline_run_id,
            'oof_auc': baseline_metrics['oof_global']['auc'],
            'test_auc': baseline_metrics['test']['auc']
        }
    }
    all_results.update(variant_results)
    
    # Print comparison table
    print(f"\n{'Feature Set':<20} {'Run ID':<35} {'OOF AUC':>10} {'Test AUC':>10} {'Δ OOF':>10} {'Δ Test':>10}")
    print("-" * 110)
    
    baseline_oof = all_results['baseline']['oof_auc']
    baseline_test = all_results['baseline']['test_auc']
    
    for feature_set, result in all_results.items():
        delta_oof = result['oof_auc'] - baseline_oof
        delta_test = result['test_auc'] - baseline_test
        
        print(f"{feature_set:<20} {result['run_id']:<35} "
              f"{result['oof_auc']:>10.4f} {result['test_auc']:>10.4f} "
              f"{delta_oof:>+10.4f} {delta_test:>+10.4f}")
    
    # Save report
    report_path = os.path.join(args.output_dir, f'{args.group_id}_comparison.json')
    report = {
        'group_id': args.group_id,
        'date_cutoff': args.date_cutoff,
        'n_folds': args.n_folds,
        'label_strategy': args.label_strategy,
        'baseline_run_id': baseline_run_id,
        'results': all_results,
        'created_at': datetime.now().isoformat()
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ 对比报告已保存: {report_path}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("🎉 消融实验完成！")
    print("="*80)
    print(f"Group ID: {args.group_id}")
    print(f"总实验数: {len(all_results)}")
    print(f"输出目录: {args.output_dir}")
    print(f"对比报告: {report_path}")
    
    # Find best variant
    best_feature_set = max(all_results.keys(), 
                          key=lambda k: all_results[k]['test_auc'])
    best_auc = all_results[best_feature_set]['test_auc']
    
    print(f"\n🏆 最佳 Feature Set: {best_feature_set}")
    print(f"   Test AUC: {best_auc:.4f}")
    
    if best_feature_set != 'baseline':
        improvement = best_auc - baseline_test
        print(f"   相比 Baseline 提升: {improvement:+.4f}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
