# -*- coding: utf-8 -*-
"""
fix_samples_constraint.py
手動修復 experiment_samples 的 split constraint

使用: python Model/migrations/fix_samples_constraint.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Model.experiment_logger import get_db_connection


def fix_constraint():
    """手動修復 experiment_samples constraint"""
    print("=" * 60)
    print("Fixing experiment_samples split constraint")
    print("=" * 60)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # 檢查現有 constraint
        print("\n1. Checking existing constraints...")
        cursor.execute("""
            SELECT conname, pg_get_constraintdef(oid)
            FROM pg_constraint
            WHERE conrelid = 'experiment_samples'::regclass
            AND conname LIKE '%split%'
        """)
        result = cursor.fetchone()
        if result:
            print(f"   Current: {result[0]}")
            print(f"   Definition: {result[1]}")
        
        # Drop old constraint
        print("\n2. Dropping old constraint...")
        cursor.execute("ALTER TABLE experiment_samples DROP CONSTRAINT IF EXISTS experiment_samples_split_check CASCADE")
        print("   ✓ Dropped")
        
        # Add new constraint
        print("\n3. Adding new constraint...")
        cursor.execute("""
            ALTER TABLE experiment_samples 
            ADD CONSTRAINT experiment_samples_split_check 
            CHECK (split IN ('train_pool', 'test'))
        """)
        print("   ✓ Added")
        
        conn.commit()
        
        # Verify
        print("\n4. Verifying new constraint...")
        cursor.execute("""
            SELECT conname, pg_get_constraintdef(oid)
            FROM pg_constraint
            WHERE conrelid = 'experiment_samples'::regclass
            AND conname LIKE '%split%'
        """)
        result = cursor.fetchone()
        if result:
            print(f"   New: {result[0]}")
            print(f"   Definition: {result[1]}")
            
            if 'train_pool' in result[1] and 'test' in result[1]:
                print("\n✅ Constraint fixed successfully!")
            else:
                print("\n⚠️  Constraint might still be incorrect")
        
    except Exception as e:
        conn.rollback()
        print(f"\n❌ Failed: {e}")
        raise
    
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    fix_constraint()
