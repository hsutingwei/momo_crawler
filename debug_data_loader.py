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

        # Check for Diversity & Organic Features
        div_cols = ["feat_semantic_entropy", "feat_temporal_burstiness", "feat_lexical_diversity"]
        print("\nChecking Diversity & Organic Features:")
        for col in div_cols:
            if col in X_dense.columns:
                print(f"  {col}: Found. Mean={X_dense[col].mean():.4f}, Max={X_dense[col].max():.4f}")
            else:
                print(f"  {col}: NOT FOUND!")
        
        # Check aggregated_comments in the internal dataframe if accessible
        if "category_fit_score" in X_dense.columns:
             non_zero = (X_dense["category_fit_score"] > 0).sum()
             print(f"\nCategory Fit Score Non-Zero Count: {non_zero} / {len(X_dense)}")
             
        # Analyze comment_count_90d distribution
        if "comment_count_90d" in X_dense.columns:
            print("\nComment Count (90d) Distribution:")
            print(X_dense["comment_count_90d"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))
            
            over_20 = (X_dense["comment_count_90d"] > 20).sum()
            over_50 = (X_dense["comment_count_90d"] > 50).sum()
            over_100 = (X_dense["comment_count_90d"] > 100).sum()
            print(f"Products with > 20 comments: {over_20} ({over_20/len(X_dense):.2%})")
            print(f"Products with > 50 comments: {over_50} ({over_50/len(X_dense):.2%})")
            print(f"Products with > 100 comments: {over_100} ({over_100/len(X_dense):.2%})")
                
        print("\nSample Data (first 5 rows of Kinematics):")
        if all(c in X_dense.columns for c in kin_cols):
            print(X_dense[kin_cols].head())
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
