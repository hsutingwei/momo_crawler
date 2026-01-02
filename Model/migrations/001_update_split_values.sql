-- ==========================================
-- Migration Script: Update Split Values
-- Purpose: 修正 split 欄位定義，從 train/val/test 改為 train_pool/test
-- Date: 2026-01-02
-- ==========================================

-- Step 1: Drop existing CHECK constraints
ALTER TABLE experiment_samples DROP CONSTRAINT IF EXISTS experiment_samples_split_check;
ALTER TABLE experiment_predictions DROP CONSTRAINT IF EXISTS experiment_predictions_split_check;

-- Step 2: Add new CHECK constraints with correct values
ALTER TABLE experiment_samples 
  ADD CONSTRAINT experiment_samples_split_check 
  CHECK (split IN ('train_pool','test'));

ALTER TABLE experiment_predictions 
  ADD CONSTRAINT experiment_predictions_split_check 
  CHECK (split IN ('train_pool','test'));

-- Step 3: (Optional) Migrate existing data if any
-- If you have existing records with split='train' or split='val', 
-- you need to update them:
-- UPDATE experiment_samples SET split = 'train_pool' WHERE split IN ('train', 'val');
-- UPDATE experiment_predictions SET split = 'train_pool' WHERE split IN ('train', 'val');

-- Verification
SELECT 'experiment_samples', split, COUNT(*) 
FROM experiment_samples 
GROUP BY split
UNION ALL
SELECT 'experiment_predictions', split, COUNT(*) 
FROM experiment_predictions 
GROUP BY split;
