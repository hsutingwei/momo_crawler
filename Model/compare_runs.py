#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_runs.py - Compare Multiple Experiment Runs

Usage:
  python Model/compare_runs.py --run-ids baseline_top500_5fold_20260105,baseline_top500_5fold_v2_filter,baseline_min5
  python Model/compare_runs.py --output-dir comparison_results
"""

import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def load_run_metrics(run_id: str, runs_dir: str = 'runs'):
    """
    Load metrics from a single run
    
    Returns:
        dict with run_id, metrics, config, etc.
    """
    run_dir = os.path.join(runs_dir, run_id)
    
    if not os.path.exists(run_dir):
        print(f"⚠️  Run directory not found: {run_dir}")
        return None
    
    metrics_path = os.path.join(run_dir, 'metrics.json')
    config_path = os.path.join(run_dir, 'run_config.json')
    
    if not os.path.exists(metrics_path):
        print(f"⚠️  Metrics file not found: {metrics_path}")
        return None
    
    # Load metrics
    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    # Load config if exists
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    return {
        'run_id': run_id,
        'metrics': metrics,
        'config': config
    }


def create_comparison_dataframe(run_data_list):
    """
    Create comparison DataFrame from list of run data
    
    Returns:
        DataFrame with columns: run_id, test_auc, test_f1, oof_auc_mean, oof_auc_std, etc.
    """
    data = []
    
    for run_data in run_data_list:
        if run_data is None:
            continue
        
        run_id = run_data['run_id']
        metrics = run_data['metrics']
        config = run_data['config']
        
        record = {
            'run_id': run_id,
            'feature_set': config.get('feature_set', 'unknown'),
            'date_cutoff': config.get('date_cutoff', 'unknown'),
            'n_folds': config.get('n_folds', 'unknown'),
        }
        
        # Extract test metrics
        if 'test' in metrics:
            record['test_auc'] = metrics['test'].get('auc')
            record['test_f1'] = metrics['test'].get('f1')
            record['test_precision'] = metrics['test'].get('precision')
            record['test_recall'] = metrics['test'].get('recall')
        else:
            record['test_auc'] = None
            record['test_f1'] = None
            record['test_precision'] = None
            record['test_recall'] = None
        
        # Extract OOF aggregate metrics
        if 'oof_aggregate' in metrics:
            oof_agg = metrics['oof_aggregate']
            record['oof_auc_mean'] = oof_agg.get('auc', {}).get('mean')
            record['oof_auc_std'] = oof_agg.get('auc', {}).get('std')
            record['oof_f1_mean'] = oof_agg.get('f1', {}).get('mean')
            record['oof_f1_std'] = oof_agg.get('f1', {}).get('std')
        else:
            record['oof_auc_mean'] = None
            record['oof_auc_std'] = None
            record['oof_f1_mean'] = None
            record['oof_f1_std'] = None
        
        # Sample counts
        record['n_samples'] = metrics.get('n_samples')
        record['n_train'] = metrics.get('n_train')
        record['n_test'] = metrics.get('n_test')
        
        data.append(record)
    
    return pd.DataFrame(data)


def print_comparison_table(df: pd.DataFrame):
    """Print formatted comparison table"""
    print("\n" + "="*120)
    print("📊 Runs Comparison")
    print("="*120)
    
    # Select key columns to display
    display_cols = [
        'run_id', 'feature_set', 'n_samples', 'test_auc', 'test_f1', 
        'oof_auc_mean', 'oof_auc_std'
    ]
    
    display_df = df[display_cols].copy()
    
    # Format numeric columns
    for col in ['test_auc', 'test_f1', 'oof_auc_mean', 'oof_auc_std']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    
    print(display_df.to_string(index=False))
    print()


def save_comparison_csv(df: pd.DataFrame, output_dir: str):
    """Save comparison table to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"comparison_{timestamp}.csv")
    
    df.to_csv(csv_path, index=False)
    print(f"💾 Saved comparison table to: {csv_path}")
    return csv_path


def plot_test_auc_comparison(df: pd.DataFrame, output_dir: str):
    """Plot Test AUC comparison bar chart"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter completed runs with test_auc
    plot_df = df[df['test_auc'].notna()].copy()
    
    if len(plot_df) == 0:
        print("⚠️  No completed runs with test_auc to plot")
        return
    
    plt.figure(figsize=(14, 7))
    
    # Create bar plot
    bars = plt.bar(range(len(plot_df)), plot_df['test_auc'].values, color='steelblue', alpha=0.8, edgecolor='navy')
    
    # Customize
    plt.xlabel('Run ID', fontsize=13, fontweight='bold')
    plt.ylabel('Test AUC', fontsize=13, fontweight='bold')
    plt.title(f'Test AUC Comparison ({len(plot_df)} Runs)', fontsize=15, fontweight='bold', pad=20)
    plt.xticks(range(len(plot_df)), plot_df['run_id'].values, rotation=45, ha='right', fontsize=10)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, plot_df['test_auc'].values)):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.015, f'{val:.4f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"test_auc_comparison_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved Test AUC comparison plot to: {plot_path}")
    return plot_path


def plot_oof_auc_comparison(df: pd.DataFrame, output_dir: str):
    """Plot OOF AUC mean comparison with error bars"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter completed runs with oof_auc_mean
    plot_df = df[df['oof_auc_mean'].notna()].copy()
    
    if len(plot_df) == 0:
        print("⚠️  No completed runs with oof_auc_mean to plot")
        return
    
    plt.figure(figsize=(14, 7))
    
    # Create bar plot with error bars
    x_pos = range(len(plot_df))
    bars = plt.bar(x_pos, plot_df['oof_auc_mean'].values, 
                   yerr=plot_df['oof_auc_std'].fillna(0).values,
                   color='coral', alpha=0.8, capsize=6, error_kw={'linewidth': 2.5}, edgecolor='darkred')
    
    # Customize
    plt.xlabel('Run ID', fontsize=13, fontweight='bold')
    plt.ylabel('OOF AUC (Mean ± Std)', fontsize=13, fontweight='bold')
    plt.title(f'OOF AUC Comparison ({len(plot_df)} Runs)', fontsize=15, fontweight='bold', pad=20)
    plt.xticks(x_pos, plot_df['run_id'].values, rotation=45, ha='right', fontsize=10)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, plot_df['oof_auc_mean'].values, plot_df['oof_auc_std'].fillna(0).values)):
        label_y = mean_val + std_val + 0.015
        plt.text(bar.get_x() + bar.get_width()/2, label_y, 
                f'{mean_val:.4f}±{std_val:.4f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"oof_auc_comparison_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved OOF AUC comparison plot to: {plot_path}")
    return plot_path


def main():
    parser = argparse.ArgumentParser(description='Compare experiment results from multiple runs')
    parser.add_argument('--run-ids', type=str, required=False,
                       help='Comma-separated list of run IDs to compare. If omitted, will compare all valid runs in --runs-dir')
    parser.add_argument('--runs-dir', type=str, default='runs',
                       help='Directory containing run folders')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    # Parse run IDs
    if args.run_ids:
        run_ids = [rid.strip() for rid in args.run_ids.split(',')]
    else:
        import glob
        run_ids = [os.path.basename(d) for d in glob.glob(os.path.join(args.runs_dir, '*')) if os.path.isdir(d)]
        print(f"📌 No --run-ids provided. Auto-detected {len(run_ids)} valid folders in {args.runs_dir}.")
    
    print(f"\n🔍 Loading metrics for {len(run_ids)} runs...")
    
    # Load all run data
    run_data_list = []
    for run_id in run_ids:
        print(f"  Loading: {run_id}...")
        run_data = load_run_metrics(run_id, args.runs_dir)
        if run_data:
            run_data_list.append(run_data)
    
    if not run_data_list:
        print("❌ No valid runs found!")
        return
    
    print(f"✅ Loaded {len(run_data_list)} runs successfully")
    
    # Create comparison DataFrame
    df = create_comparison_dataframe(run_data_list)
    
    # Print comparison table
    print_comparison_table(df)
    
    # Save CSV
    csv_path = save_comparison_csv(df, args.output_dir)
    
    # Plot Test AUC
    test_plot = plot_test_auc_comparison(df, args.output_dir)
    
    # Plot OOF AUC
    oof_plot = plot_oof_auc_comparison(df, args.output_dir)
    
    print(f"\n✅ Comparison complete! Results saved to: {args.output_dir}")
    print(f"   CSV: {csv_path}")
    if test_plot:
        print(f"   Test AUC Plot: {test_plot}")
    if oof_plot:
        print(f"   OOF AUC Plot: {oof_plot}")


if __name__ == '__main__':
    main()
