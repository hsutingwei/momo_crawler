import argparse
import sys
import os
import pandas as pd
from typing import List, Optional

# Mock imports to avoid heavy deps if possible, but data_loader needs pandas/psycopg2
from data_loader import load_product_level_training_set

def parse_str_list(s):
    if not s:
        return []
    return [x.strip() for x in s.split(',')]

def parse_args():
    ap = argparse.ArgumentParser()
    # Copy relevant args from train.py
    ap.add_argument("--mode", type=str, default="product_level")
    ap.add_argument("--date-cutoff", type=str, default="2025-06-25")
    ap.add_argument("--pipeline-version", type=str, default=None)
    ap.add_argument("--vocab-mode", type=str, default="global")
    ap.add_argument("--top-n", type=str, default="100")
    ap.add_argument("--exclude-products", type=str, default="8918452")
    ap.add_argument("--filter-version", type=str, default=None)
    ap.add_argument("--keyword", type=str, default=None)
    ap.add_argument("--label-delta-threshold", type=float, default=10.0)
    ap.add_argument("--label-ratio-threshold", type=float, default=None)
    ap.add_argument("--label-max-gap-days", type=float, default=14.0)
    ap.add_argument("--label-mode", type=str, default="next_batch")
    ap.add_argument("--label-strategy", type=str, default="absolute")
    ap.add_argument("--label-window-days", type=float, default=7.0)
    ap.add_argument("--align-max-gap-days", type=float, default=None)
    ap.add_argument("--min-comments", type=int, default=0)
    ap.add_argument("--keyword-blacklist", type=str, default=None)
    ap.add_argument("--keyword-whitelist", type=str, default=None)
    
    return ap.parse_args()

def main():
    args = parse_args()
    print("--- Debugging Args ---")
    print(f"label_delta_threshold: {args.label_delta_threshold}")
    print(f"label_strategy: {args.label_strategy}")
    print(f"label_ratio_threshold: {args.label_ratio_threshold}")
    print(f"exclude_products raw: {args.exclude_products}")
    
    manual_exclude = parse_str_list(args.exclude_products) or []
    # simplified exclude logic (skip DB fetch for now, assumes manual only or passed explicitly)
    excluded = [int(pid) for pid in manual_exclude if str(pid).isdigit()]
    if args.filter_version:
        print("Note: filter_version passed but not fetching DB in this debug script unless needed.")
        
    print(f"Final Excluded: {excluded}")
    
    label_params = {
        "ratio_threshold": args.label_ratio_threshold,
        "delta_threshold": args.label_delta_threshold
    }
    
    topn_list = [int(x) for x in args.top_n.split(",")]
    
    print("--- Loading Data ---")
    try:
        X_dense_df, X_tfidf, y, meta, vocab = load_product_level_training_set(
            date_cutoff=args.date_cutoff,
            top_n=topn_list[0],
            vocab_mode=args.vocab_mode,
            single_keyword=args.keyword,
            exclude_products=excluded,
            label_delta_threshold=args.label_delta_threshold,
            label_ratio_threshold=args.label_ratio_threshold,
            label_max_gap_days=args.label_max_gap_days,
            label_mode=args.label_mode,
            label_window_days=args.label_window_days,
            align_max_gap_days=args.align_max_gap_days,
            min_comments=args.min_comments,
            keyword_whitelist=parse_str_list(args.keyword_whitelist),
            keyword_blacklist=parse_str_list(args.keyword_blacklist),
            label_strategy=args.label_strategy,
            label_params=label_params
        )
        
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        print(f"Load Success. Samples: {len(y)}")
        print(f"Positives: {n_pos}, Negatives: {n_neg}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
