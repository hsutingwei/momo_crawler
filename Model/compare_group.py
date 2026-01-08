#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_group.py - Compare Experiment Results within Same Group

Usage:
  python Model/compare_group.py --group-id ablation_top500_5fold
  python Model/compare_group.py --group-id ablation_top500_5fold --output-dir comparison_results
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import DatabaseConfig


def get_group_runs(group_id: str):
    """
    Query all runs in the same group
    
    Returns:
        DataFrame with columns: run_id, status, created_at, test_auc, oof_auc_mean, oof_auc_std, etc.
    """
    db_config = DatabaseConfig()
    conn = db_config.get_connection()
    cur = conn.cursor()
    
    sql = """
    SELECT 
        run_id,
        status,
        group_id,
        feature_set,
        date_cutoff,
        created_at,
        finished_at,
        metrics_json
    FROM experiment_runs
    WHERE group_id = %s
    ORDER BY created_at ASC;
    """
    
    cur.execute(sql, (group_id,))
    rows = cur.fetchall()
    
    if not rows:
        cur.close()
        conn.close()
        return None
    
    # Parse rows and extract metrics from JSONB
    data = []
    for row in rows:
        run_id, status, group_id, feature_set, date_cutoff, created_at, finished_at, metrics_json = row
        
        record = {
            'run_id': run_id,
            'status': status,
            'group_id': group_id,
            'feature_set': feature_set,
            'date_cutoff': date_cutoff,
            'created_at': created_at,
            'finished_at': finished_at,
        }
        
        # Extract metrics from JSONB
        if metrics_json:
            # Test metrics
            record['test_auc'] = metrics_json.get('test', {}).get('auc')
            record['test_f1'] = metrics_json.get('test', {}).get('f1')
            record['test_precision'] = metrics_json.get('test', {}).get('precision')
            record['test_recall'] = metrics_json.get('test', {}).get('recall')
            
            # OOF aggregate metrics
            oof_agg = metrics_json.get('oof_aggregate', {})
            if isinstance(oof_agg, dict):
                record['oof_auc_mean'] = oof_agg.get('auc', {}).get('mean') if isinstance(oof_agg.get('auc'), dict) else None
                record['oof_auc_std'] = oof_agg.get('auc', {}).get('std') if isinstance(oof_agg.get('auc'), dict) else None
                record['oof_f1_mean'] = oof_agg.get('f1', {}).get('mean') if isinstance(oof_agg.get('f1'), dict) else None
                record['oof_f1_std'] = oof_agg.get('f1', {}).get('std') if isinstance(oof_agg.get('f1'), dict) else None
            else:
                record['oof_auc_mean'] = None
                record['oof_auc_std'] = None
                record['oof_f1_mean'] = None
                record['oof_f1_std'] = None
            
            # Get sample count
            record['n_samples'] = metrics_json.get('n_samples')
        else:
            record['test_auc'] = None
            record['test_f1'] = None
            record['test_precision'] = None
            record['test_recall'] = None
            record['oof_auc_mean'] = None
            record['oof_auc_std'] = None
            record['oof_f1_mean'] = None
            record['oof_f1_std'] = None
            record['n_samples'] = None
        
        data.append(record)
    
    df = pd.DataFrame(data)

    
    cur.close()
    conn.close()
    
    return df


def print_comparison_table(df: pd.DataFrame):
    """Print formatted comparison table"""
    print("\n" + "="*120)
    print(f"📊 Group Comparison: {df['group_id'].iloc[0]}")
    print("="*120)
    
    # Select key columns to display
    display_cols = [
        'run_id', 'n_samples', 'test_auc', 'test_f1', 
        'oof_auc_mean', 'oof_auc_std', 'status'
    ]
    
    display_df = df[display_cols].copy()
    
    # Format numeric columns
    display_df['test_auc'] = display_df['test_auc'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    display_df['test_f1'] = display_df['test_f1'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    display_df['oof_auc_mean'] = display_df['oof_auc_mean'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    display_df['oof_auc_std'] = display_df['oof_auc_std'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    
    print(display_df.to_string(index=False))
    print()


def save_comparison_csv(df: pd.DataFrame, output_dir: str):
    """Save comparison table to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    group_id = df['group_id'].iloc[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"comparison_{group_id}_{timestamp}.csv")
    
    df.to_csv(csv_path, index=False)
    print(f"💾 Saved comparison table to: {csv_path}")


def plot_test_auc_comparison(df: pd.DataFrame, output_dir: str):
    """Plot Test AUC comparison bar chart"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter completed runs with test_auc
    plot_df = df[df['test_auc'].notna()].copy()
    
    if len(plot_df) == 0:
        print("⚠️  No completed runs with test_auc to plot")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    bars = plt.bar(range(len(plot_df)), plot_df['test_auc'].values, color='steelblue', alpha=0.8)
    
    # Customize
    plt.xlabel('Run ID', fontsize=12)
    plt.ylabel('Test AUC', fontsize=12)
    plt.title(f'Test AUC Comparison - Group: {df["group_id"].iloc[0]}', fontsize=14, fontweight='bold')
    plt.xticks(range(len(plot_df)), plot_df['run_id'].values, rotation=45, ha='right')
    plt.ylim(0.5, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, plot_df['test_auc'].values)):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.4f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    group_id = df['group_id'].iloc[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"test_auc_comparison_{group_id}_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved Test AUC comparison plot to: {plot_path}")


def plot_oof_auc_comparison(df: pd.DataFrame, output_dir: str):
    """Plot OOF AUC mean comparison with error bars"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter completed runs with oof_auc_mean
    plot_df = df[df['oof_auc_mean'].notna()].copy()
    
    if len(plot_df) == 0:
        print("⚠️  No completed runs with oof_auc_mean to plot")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Create bar plot with error bars
    x_pos = range(len(plot_df))
    bars = plt.bar(x_pos, plot_df['oof_auc_mean'].values, 
                   yerr=plot_df['oof_auc_std'].values,
                   color='coral', alpha=0.8, capsize=5, error_kw={'linewidth': 2})
    
    # Customize
    plt.xlabel('Run ID', fontsize=12)
    plt.ylabel('OOF AUC (Mean ± Std)', fontsize=12)
    plt.title(f'OOF AUC Comparison - Group: {df["group_id"].iloc[0]}', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, plot_df['run_id'].values, rotation=45, ha='right')
    plt.ylim(0.5, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, plot_df['oof_auc_mean'].values, plot_df['oof_auc_std'].values)):
        plt.text(bar.get_x() + bar.get_width()/2, mean_val + std_val + 0.01, 
                f'{mean_val:.4f}±{std_val:.4f}', 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    group_id = df['group_id'].iloc[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"oof_auc_comparison_{group_id}_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved OOF AUC comparison plot to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare experiment results within same group')
    parser.add_argument('--group-id', type=str, required=True,
                       help='Group ID to compare')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    # Query runs
    print(f"\n🔍 Querying runs for group: {args.group_id}...")
    df = get_group_runs(args.group_id)
    
    if df is None or len(df) == 0:
        print(f"❌ No runs found for group_id: {args.group_id}")
        return
    
    print(f"✅ Found {len(df)} runs in group {args.group_id}")
    
    # Print comparison table
    print_comparison_table(df)
    
    # Save CSV
    save_comparison_csv(df, args.output_dir)
    
    # Plot Test AUC
    plot_test_auc_comparison(df, args.output_dir)
    
    # Plot OOF AUC
    plot_oof_auc_comparison(df, args.output_dir)
    
    print(f"\n✅ Comparison complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
