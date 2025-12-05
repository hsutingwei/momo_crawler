import psycopg2
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.database import DatabaseConfig

def fetch_spam_refs():
    print("Connecting to database...")
    # DatabaseConfig is a class, need to instantiate it
    db = DatabaseConfig()
    conn = psycopg2.connect(**db.config)
    
    try:
        with conn.cursor() as cursor:
            print("Executing SQL query...")
            sql = """
            SELECT 
                comment_text, 
                COUNT(*) as freq
            FROM product_comments
            -- Filter 1: Ignore very short text (e.g., "Good", "Ok") -> NCD needs structure
            WHERE LENGTH(comment_text) >= 6 
            -- Filter 2: Ignore long essays (likely organic) -> Templates are usually short
              AND LENGTH(comment_text) <= 50 
            GROUP BY comment_text
            -- Filter 3: Must appear frequently to be considered a "Template"
            ORDER BY freq DESC
            LIMIT 20;
            """
            cursor.execute(sql)
            results = cursor.fetchall()
            
            print(f"\n{'='*60}")
            print(f"{'SPAM TEMPLATE ANALYSIS':^60}")
            print(f"{'='*60}\n")
            
            spam_refs = []
            for text, freq in results:
                # Clean text (remove newlines)
                clean_text = text.replace('\n', ' ').replace('\r', '').strip()
                print(f"Freq: {freq:<5} | Text: {clean_text}")
                spam_refs.append(clean_text)
            
            # Select Top 10 for the list
            top_refs = spam_refs[:10]
            
            print(f"\n{'='*60}")
            print(f"{'GENERATED PYTHON CODE':^60}")
            print(f"{'='*60}\n")
            
            print("# Generated from DB Analysis (Top frequent templates)")
            print("SPAM_REFERENCES = [")
            for ref in top_refs:
                print(f'    "{ref}",')
            print("]")

    finally:
        conn.close()

if __name__ == "__main__":
    fetch_spam_refs()
