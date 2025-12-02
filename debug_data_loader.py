from Model.data_loader import load_product_level_training_set
from config.database import DatabaseConfig
import pandas as pd
from sqlalchemy import create_engine
import os

def main():
    # Construct SQLAlchemy URL
    # Assuming config is standard
    db_config = DatabaseConfig().config
    # Handle password escaping if needed, but for now simple string
    db_url = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
    
    engine = create_engine(db_url)
    conn = engine.connect()
    
    try:
        print("Loading data...")
        # load_product_level_training_set expects a psycopg2 connection object usually, 
        # but let's see if it works with sqlalchemy connection or if we need to pass the raw connection.
        # The data_loader uses pd.read_sql(sql, conn, params=params).
        # If conn is sqlalchemy connection, it should work better.
        # However, data_loader also uses cursor() method on conn for other things?
        # Let's check data_loader usage of conn.
        # It uses `pd.read_sql(..., conn, ...)` and `fetch_top_terms(conn, ...)` which uses `conn.cursor()`.
        # SQLAlchemy connection does not have `cursor()`.
        # So we must pass a raw psycopg2 connection to `load_product_level_training_set`.
        
        # So the issue is indeed why `pd.read_sql` fails with psycopg2 connection in debug script.
        # Maybe I should just fix the data_loader to use sqlalchemy if possible, but that's a big change.
        
        # Let's try to pass the raw connection from engine.
        raw_conn = engine.raw_connection()
        
        X, y, meta = load_product_level_training_set(raw_conn, label_strategy="absolute")
        print("Data loaded successfully.")
        print("X columns:", X.columns.tolist())
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    main()
