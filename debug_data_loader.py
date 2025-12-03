from Model.data_loader import load_product_level_training_set
import pandas as pd
import traceback
import sys

def main():
    try:
        print("Loading data...")
        # load_product_level_training_set creates its own connection
        result = load_product_level_training_set(
            date_cutoff="2025-06-25",
            label_strategy="absolute"
        )
        
        X_dense, X_tfidf, y, meta, vocab = result
        
        print("Data loaded successfully.")
        print(f"X_dense shape: {X_dense.shape}")
        print(f"X_tfidf shape: {X_tfidf.shape}")
        print(f"y shape: {y.shape}")
        print(f"meta shape: {meta.shape}")
        
        # Check for Kinematics features
        kin_cols = ["kin_v_1", "kin_v_2", "kin_v_3", "kin_acc_abs", "kin_acc_rel", "kin_jerk_abs"]
        print("\nChecking Kinematics Features:")
        for col in kin_cols:
            if col in X_dense.columns:
                print(f"  {col}: Found. Mean={X_dense[col].mean():.4f}, Max={X_dense[col].max():.4f}")
            else:
                print(f"  {col}: NOT FOUND!")

        # Check for BERT features
        bert_cols = ["bert_arousal_mean", "clean_arousal_score", "intensity_score", "semantic_novelty_score", "semantic_novelty_comment"]
        print("\nChecking BERT & Novelty Features:")
        for col in bert_cols:
            if col in X_dense.columns:
                print(f"  {col}: Found. Mean={X_dense[col].mean():.4f}, Max={X_dense[col].max():.4f}")
            else:
                print(f"  {col}: NOT FOUND!")
        
        # Check aggregated_comments in the internal dataframe if accessible
        # Since load_product_level_training_set returns X_dense which is a processed dataframe, 
        # we might not have 'aggregated_comments' column in X_dense if it wasn't in dense_cols.
        # But we added it to dense_cols? No, we added 'semantic_novelty_comment'.
        # 'aggregated_comments' is an intermediate column.
        # However, we can check if 'semantic_novelty_comment' is all zeros.
        
        if "semantic_novelty_comment" in X_dense.columns:
             non_zero = (X_dense["semantic_novelty_comment"] > 0).sum()
             print(f"\nSemantic Novelty Comment Non-Zero Count: {non_zero} / {len(X_dense)}")
                
        print("\nSample Data (first 5 rows of Kinematics):")
        if all(c in X_dense.columns for c in kin_cols):
            print(X_dense[kin_cols].head())
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
