#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_filters.py - Generate Product-Level Filters and Store in ml_data_filters

Purpose:
  Build versioned exclusion lists for ML training:
  - min_comments: Products with comment_count < threshold before cutoff
  - empty_doc: Products with empty doc_text (for TF-IDF) after tokenization/fallback

Usage:
  # Generate min_comments filter (comment_count < 5)
  python Model/build_filters.py --mode min_comments --date-cutoff 2025-06-25 --min-comments 5
  
  # Generate empty_doc filter
  python Model/build_filters.py --mode empty_doc --date-cutoff 2025-06-25
"""

import argparse
import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime


def get_db_connection():
    """Connect to PostgreSQL database using DATABASE_URL env var"""
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL environment variable not set")
    return psycopg2.connect(db_url)


def build_min_comments_filter(cutoff: str, min_comments: int):
    """
    Generate min_comments filter: products with comment_count < threshold
    
    Args:
        cutoff: Date cutoff in YYYY-MM-DD format
        min_comments: Minimum comment threshold
    
    Returns:
        DataFrame with columns: product_id, reason, score (comment_count)
    """
    print(f"\n{'='*80}")
    print(f"🔍 Building min_comments filter (cutoff={cutoff}, min={min_comments})")
    print(f"{'='*80}")
    
    # Generate version_tag
    cutoff_clean = cutoff.replace('-', '')
    version_tag = f"v4_min_comments_prod__cutoff_{cutoff_clean}__min{min_comments}"
    print(f"  version_tag: {version_tag}")
    
    conn = get_db_connection()
    
    # Query: count comments before cutoff
    sql = """
    WITH comment_counts AS (
        SELECT
            p.id AS product_id,
            COUNT(DISTINCT pc.comment_id) AS comment_count
        FROM products p
        LEFT JOIN product_comments pc
            ON pc.product_id = p.id
           AND pc.capture_time <= %(cutoff)s::timestamp
        GROUP BY p.id
    )
    SELECT
        product_id,
        comment_count
    FROM comment_counts
    WHERE comment_count < %(min_comments)s
    ORDER BY comment_count ASC, product_id ASC;
    """
    
    df = pd.read_sql(sql, conn, params={'cutoff': cutoff, 'min_comments': min_comments})
    
    print(f"  Found {len(df)} products with comment_count < {min_comments}")
    
    if len(df) == 0:
        print("  ⚠️  No products to filter. Skipping DB write.")
        conn.close()
        return df, version_tag
    
    # Prepare data for insertion
    df['reason'] = 'min_comments'
    df['score'] = df['comment_count'].astype(float)
    
    # Insert into ml_data_filters
    insert_sql = """
    INSERT INTO ml_data_filters(version_tag, filter_level, filter_value, reason, score)
    VALUES %s
    ON CONFLICT (version_tag, filter_level, filter_value) DO NOTHING;
    """
    
    values = [
        (version_tag, 'product_id', str(row['product_id']), row['reason'], row['score'])
        for _, row in df.iterrows()
    ]
    
    cur = conn.cursor()
    execute_values(cur, insert_sql, values)
    inserted_count = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"  ✅ Inserted {inserted_count} records into ml_data_filters")
    
    return df[['product_id', 'reason', 'score']], version_tag


def build_empty_doc_filter(cutoff: str):
    """
    Generate empty_doc filter: products with empty doc_text after tokenization/fallback
    
    Uses same logic as data_loader.py:
    1. Prioritize doc_text_tokenized
    2. Fallback to aggregated_comments
    3. If both empty, doc_text = ''
    
    Args:
        cutoff: Date cutoff in YYYY-MM-DD format
    
    Returns:
        DataFrame with columns: product_id, reason, score (doc_len)
    """
    print(f"\n{'='*80}")
    print(f"🔍 Building empty_doc filter (cutoff={cutoff})")
    print(f"{'='*80}")
    
    # Generate version_tag
    cutoff_clean = cutoff.replace('-', '')
    version_tag = f"v5_empty_doc_prod__cutoff_{cutoff_clean}"
    print(f"  version_tag: {version_tag}")
    
    conn = get_db_connection()
    
    # Query: get doc_text using same logic as data_loader
    # This mirrors the SQL in data_loader.py for doc_text_tokenized and aggregated_comments
    sql = """
    WITH pre_comments AS (
        SELECT
            pc.product_id,
            pc.comment_id,
            pc.comment_text,
            pc.comment_date,
            pc.capture_time
        FROM product_comments pc
        WHERE pc.capture_time <= %(cutoff)s::timestamp
    ),
    doc_texts AS (
        SELECT
            p.id AS product_id,
            -- doc_text_tokenized: all tokenized comments before cutoff
            (
                SELECT STRING_AGG(ct.token, ' ' ORDER BY ct.token_order)
                FROM comment_tokens ct
                JOIN product_comments pc ON pc.comment_id = ct.comment_id
                WHERE pc.product_id = p.id
                  AND pc.capture_time <= %(cutoff)s::timestamp
            ) AS doc_text_tokenized,
            -- aggregated_comments: all raw comments before cutoff (fallback)
            (
                SELECT STRING_AGG(pc.comment_text, ' ')
                FROM pre_comments pc
                WHERE pc.product_id = p.id
            ) AS aggregated_comments
        FROM products p
    )
    SELECT
        product_id,
        COALESCE(doc_text_tokenized, '') AS doc_text_tokenized,
        COALESCE(aggregated_comments, '') AS aggregated_comments
    FROM doc_texts;
    """
    
    df = pd.read_sql(sql, conn, params={'cutoff': cutoff})
    
    print(f"  Loaded {len(df)} products")
    
    # Apply same fallback logic as data_loader.py
    def get_doc_text(row):
        """Mimic data_loader.py logic"""
        tokenized = row['doc_text_tokenized']
        fallback = row['aggregated_comments']
        
        if tokenized and tokenized.strip():
            return tokenized
        elif fallback and fallback.strip():
            return fallback
        else:
            return ''
    
    df['doc_text'] = df.apply(get_doc_text, axis=1)
    
    # Filter: empty after processing
    df['doc_len'] = df['doc_text'].str.len()
    empty_df = df[df['doc_text'].str.strip() == ''].copy()
    
    print(f"  Found {len(empty_df)} products with empty doc_text ({len(empty_df)/len(df)*100:.2f}%)")
    
    if len(empty_df) == 0:
        print("  ⚠️  No products to filter. Skipping DB write.")
        conn.close()
        return empty_df, version_tag
    
    # Prepare data for insertion
    empty_df['reason'] = 'empty_doc'
    empty_df['score'] = empty_df['doc_len'].astype(float)
    
    # Insert into ml_data_filters
    insert_sql = """
    INSERT INTO ml_data_filters(version_tag, filter_level, filter_value, reason, score)
    VALUES %s
    ON CONFLICT (version_tag, filter_level, filter_value) DO NOTHING;
    """
    
    values = [
        (version_tag, 'product_id', str(row['product_id']), row['reason'], row['score'])
        for _, row in empty_df.iterrows()
    ]
    
    cur = conn.cursor()
    execute_values(cur, insert_sql, values)
    inserted_count = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"  ✅ Inserted {inserted_count} records into ml_data_filters")
    
    return empty_df[['product_id', 'reason', 'score']], version_tag


def save_local_parquet(df, version_tag):
    """Save excluded products to local parquet file"""
    os.makedirs('Model/filters', exist_ok=True)
    output_path = f'Model/filters/excluded_products_{version_tag}.parquet'
    df.to_parquet(output_path, index=False)
    print(f"  💾 Saved to: {output_path}")


def print_summary(df, version_tag):
    """Print filter summary"""
    print(f"\n{'='*80}")
    print(f"📊 Summary: {version_tag}")
    print(f"{'='*80}")
    print(f"  Excluded products: {len(df)}")
    
    # Get total product count from DB
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM products;")
    total_products = cur.fetchone()[0]
    cur.close()
    conn.close()
    
    print(f"  Total products: {total_products}")
    print(f"  Exclusion ratio: {len(df)/total_products*100:.2f}%")
    
    if len(df) > 0:
        print(f"\n  Top 10 excluded products:")
        print(df.head(10).to_string(index=False))
        
        if 'score' in df.columns:
            print(f"\n  Score distribution:")
            print(df['score'].describe())


def main():
    parser = argparse.ArgumentParser(description='Build product-level filters for ML training')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['min_comments', 'empty_doc'],
                       help='Filter mode: min_comments or empty_doc')
    parser.add_argument('--date-cutoff', type=str, required=True,
                       help='Date cutoff in YYYY-MM-DD format')
    parser.add_argument('--min-comments', type=int, default=5,
                       help='Minimum comment threshold (for min_comments mode)')
    
    args = parser.parse_args()
    
    # Build filter
    if args.mode == 'min_comments':
        df, version_tag = build_min_comments_filter(args.date_cutoff, args.min_comments)
    elif args.mode == 'empty_doc':
        df, version_tag = build_empty_doc_filter(args.date_cutoff)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    # Save and summarize
    if len(df) > 0:
        save_local_parquet(df, version_tag)
        print_summary(df, version_tag)
    
    print(f"\n✅ Done! version_tag: {version_tag}")


if __name__ == '__main__':
    main()
