import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Model.data_loader import load_product_level_training_set

def main():
    print("Checking positives for product 8918452...")
    
    label_params = {
        "delta_threshold": 10,
        "ratio_threshold": 1.0
    }
    
    # Load ONLY product 8918452 by excluding everything usually, 
    # but here we can't easily "include only" with the loader, 
    # so we load all and filter in pandas.
    
    try:
        X_dense, X_tfidf, y, meta, vocab = load_product_level_training_set(
            date_cutoff="2025-06-25",
            top_n=100,
            vocab_mode="global",
            label_delta_threshold=10,
            label_strategy="hybrid",
            label_params=label_params,
            min_comments=0,
            exclude_products=[] # Include everything
        )
        
        # Check total first
        n_pos_total = int(y.sum())
        print(f"Total Positives (No Exclusions): {n_pos_total}")
        
        # Check 8918452
        if "product_id" in meta.columns:
            mask = meta["product_id"] == 8918452
            y_prod = y[mask]
            n_pos_prod = int(y_prod.sum())
            print(f"Positives for 8918452: {n_pos_prod}")
            print(f"Remaining Positives (Excluding 8918452): {n_pos_total - n_pos_prod}")
        else:
            print("Meta does not contain product_id??")
            print(meta.columns)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
