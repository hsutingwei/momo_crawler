import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Model.data_loader import load_product_level_training_set

def main():
    print("Reproducing dataset with rigorous_baseline parameters...")
    
    label_params = {
        "delta_threshold": 10,
        "ratio_threshold": 1.0,
        "use_pos_weight": True
    }
    
    try:
        X_dense, X_tfidf, y, meta, vocab = load_product_level_training_set(
            date_cutoff="2025-06-25",
            top_n=100,
            vocab_mode="global",
            label_delta_threshold=10,
            label_strategy="hybrid",
            label_params=label_params,
            label_max_gap_days=14.0, # Test Strict Mode
            min_comments=0
        )
        
        n_pos = int(y.sum())
        n_total = len(y)
        n_neg = n_total - n_pos
        
        print(f"Dataset Size: {n_total}")
        print(f"Positives: {n_pos}, Negatives: {n_neg}")
        
        spw = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
        print(f"Calculated scale_pos_weight: {spw:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
