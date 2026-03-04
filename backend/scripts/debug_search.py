import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import json

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load .env explicitly
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(env_path)

from app.models.database import Sentence, Base
from app.core.engine import RizalEngine

def debug_search(query_text, source_type="summary"):
    db_url = os.environ.get('DATABASE_URL')
    engine_db = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine_db)
    db = SessionLocal()
    
    rizal_engine = RizalEngine()
    
    results = rizal_engine.search(db, query_text, source_type=source_type, top_k=5)
    
    # Save to file
    with open('debug_search_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to debug_search_results.json")
    db.close()

if __name__ == "__main__":
    q = "kamatayan"
    st = "summary"
    if len(sys.argv) > 1: q = sys.argv[1]
    if len(sys.argv) > 2: st = sys.argv[2]
    debug_search(q, st)
