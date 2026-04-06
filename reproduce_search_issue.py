import sys
import os
from pathlib import Path

# Add backend to sys.path
sys.path.append(os.path.join(os.getcwd(), 'backend'))
os.environ['DEBUG_SEARCH'] = '1'

from app.models.database import SessionLocal, Sentence
from app.core.engine import RizalEngine
import numpy as np

def debug_search(query_text, source_type="summary"):
    print(f"--- Debugging Search for: '{query_text}' (mode: {source_type}) ---")
    db = SessionLocal()
    engine = RizalEngine()
    
    # Check counts
    count = db.query(Sentence).filter(Sentence.source_type == source_type).count()
    print(f"Database contains {count} sentences of type '{source_type}'")
    
    # Run search
    results = engine.search(db, query_text, source_type=source_type)
    
    print(f"Reason: {results.get('metadata', {}).get('reason')}")
    print(f"Result Mode: {results.get('metadata', {}).get('result_mode')}")
    
    n_noli = len(results['results']['noli'])
    n_elfili = len(results['results']['elfili'])
    print(f"Results found: Noli={n_noli}, Elfili={n_elfili}")
    
    if n_noli > 0:
        print(f"Top Noli Match: {results['results']['noli'][0]['sentence_text'][:100]}...")
    
    db.close()

if __name__ == "__main__":
    # Test common queries
    debug_search("Maria", source_type="summary")
    debug_search("Maria Clara", source_type="summary")
    debug_search("Damaso", source_type="full")
