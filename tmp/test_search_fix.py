import sys
import os
sys.path.append(os.path.join(os.getcwd(), "backend"))

from app.core.engine import RizalEngine
from app.models.database import SessionLocal

def test_search():
    engine = RizalEngine()
    db = SessionLocal()
    try:
        print("\nTesting Search for 'edukasyon'...")
        results = engine.search(db, "edukasyon", top_k=5)
        
        noli_count = len(results['results']['noli'])
        fili_count = len(results['results']['elfili'])
        
        print(f"Noli Results: {noli_count}")
        print(f"Fili Results: {fili_count}")
        
        if noli_count > 0:
            print(f"Top Noli Result: {results['results']['noli'][0]['sentence_text'][:100]}...")
        if fili_count > 0:
            print(f"Top Fili Result: {results['results']['elfili'][0]['sentence_text'][:100]}...")
            
        if noli_count == 0 and fili_count == 0:
            print("FAILURE: No results found for 'edukasyon'.")
        else:
            print("SUCCESS: Results found.")
            
    finally:
        db.close()

if __name__ == "__main__":
    test_search()
