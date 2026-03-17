import os
import sys
import json

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from app.models.database import SessionLocal
from app.core.engine import get_engine

def verify_output(query):
    db = SessionLocal()
    engine = get_engine()
    
    res = engine.search(db, query)
    
    # Check structure
    print(f"Metadata keys: {list(res['metadata'].keys())}")
    
    for book in ['noli', 'elfili']:
        if res['results'][book]:
            first_res = res['results'][book][0]
            print(f"--- {book} first result ---")
            print(f"Result keys: {list(first_res.keys())}")
            print(f"Scores: {first_res['scores']}")
            print(f"Themes: {len(first_res['themes'])}")
            if first_res['themes']:
                print(f"First theme: {first_res['themes'][0]['label']} (score: {first_res['themes'][0]['score']:.4f})")
            if 'context_text' in first_res:
                print(f"Context length: {len(first_res['context_text'])}")
            else:
                print("context_text MISSING!")
    
    db.close()

if __name__ == "__main__":
    verify_output("kababaihan")
