# -*- coding: utf-8 -*-
"""
apply_migration_001.py
執行 migration 001: 更新 split 欄位定義

使用: python Model/migrations/apply_migration_001.py
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Model.experiment_logger import get_db_connection


def apply_migration():
    """執行 migration 001"""
    print("=" * 60)
    print("Migration 001: Update split values to train_pool/test")
    print("=" * 60)
    
    # Read migration SQL
    migration_file = os.path.join(
        os.path.dirname(__file__), 
        '001_update_split_values.sql'
    )
    
    with open(migration_file, 'r', encoding='utf-8') as f:
        sql = f.read()
    
    # Connect to database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        print("\n📋 Executing migration...")
        
        # Execute migration (split by semicolon for multiple statements)
        statements = [s.strip() for s in sql.split(';') if s.strip() and not s.strip().startswith('--')]
        
        for i, statement in enumerate(statements, 1):
            if 'SELECT' in statement.upper():
                # Verification query
                print(f"\n✓ Statement {i}: Verification query")
                cursor.execute(statement)
                results = cursor.fetchall()
                print("\nCurrent split distribution:")
                for row in results:
                    print(f"  {row[0]}: split={row[1]}, count={row[2]}")
            else:
                # DDL statement
                print(f"✓ Statement {i}: {statement[:50]}...")
                cursor.execute(statement)
        
        conn.commit()
        print("\n✅ Migration completed successfully!")
        
    except Exception as e:
        conn.rollback()
        print(f"\n❌ Migration failed: {e}")
        raise
    
    finally:
        cursor.close()
        conn.close()


def verify_constraints():
    """驗證 constraints 是否正確更新"""
    print("\n" + "=" * 60)
    print("Verifying constraints...")
    print("=" * 60)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check constraint definitions
        cursor.execute("""
            SELECT 
                conname,
                pg_get_constraintdef(oid) as definition
            FROM pg_constraint
            WHERE conrelid IN (
                'experiment_samples'::regclass,
                'experiment_predictions'::regclass
            )
            AND contype = 'c'
            AND conname LIKE '%split%'
        """)
        
        results = cursor.fetchall()
        print("\n✓ Current CHECK constraints:")
        for name, definition in results:
            print(f"  {name}: {definition}")
        
        # Verify the constraints contain 'train_pool' and 'test'
        for name, definition in results:
            if 'train_pool' in definition and 'test' in definition:
                print(f"  ✅ {name} looks correct")
            else:
                print(f"  ⚠️  {name} might need review: {definition}")
        
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply migration 001')
    parser.add_argument('--verify-only', action='store_true', 
                        help='Only verify constraints without applying migration')
    args = parser.parse_args()
    
    if args.verify_only:
        verify_constraints()
    else:
        apply_migration()
        verify_constraints()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
