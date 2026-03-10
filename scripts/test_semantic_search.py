
import sys
import os
import numpy as np
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# Load .env explicitly
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', '.env'))
load_dotenv(env_path)

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from app.core.engine import get_engine
from app.models.database import SessionLocal

def test_query(query, source_type="summary"):
    print(f"\nSearching for: '{query}' ({source_type})")
    engine = get_engine()
    db = SessionLocal()
    try:
        results = engine.search(db, query, top_k=10, source_type=source_type)
        
        for book in ['noli', 'elfili']:
            print(f"\n--- {book.upper()} ---")
            for i, res in enumerate(results[book]):
                print(f"{i+1}. {res['sentence_text']} (Final: {res['scores']['final']}%, Sem: {res['scores']['semantic']}%, Lex: {res['scores']['lexical']}%)")
                # print(f"   Context: {res['context_text']}")
    finally:
        db.close()

if __name__ == "__main__":
    test_query("edukasyon")
    test_query("pag-aaral")
    test_query("kamatayan")
    test_query("paglilitis")
    test_query("kababata")
