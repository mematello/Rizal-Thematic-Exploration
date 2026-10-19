
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve()
while _repo_root.parent != _repo_root and not (_repo_root / "scripts" / "prod_guard.py").exists():
    _repo_root = _repo_root.parent
if str(_repo_root / "scripts") not in sys.path:
    sys.path.insert(0, str(_repo_root / "scripts"))
from prod_guard import verify_not_production_db
verify_not_production_db(caller_name=Path(__file__).name)

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
