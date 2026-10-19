
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve()
while _repo_root.parent != _repo_root and not (_repo_root / "scripts" / "prod_guard.py").exists():
    _repo_root = _repo_root.parent
if str(_repo_root / "scripts") not in sys.path:
    sys.path.insert(0, str(_repo_root / "scripts"))
from prod_guard import verify_not_production_db
verify_not_production_db(caller_name=Path(__file__).name)

import time
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from app.models.database import SessionLocal
from app.core.engine import get_engine

def profile_search(query):
    db = SessionLocal()
    engine = get_engine()
    
    start_total = time.time()
    
    print(f"Profiling search for: '{query}'")
    
    # We can't easily modify the engine code to add more fine-grained timing without 'replace'
    # but we can time the whole call and see if it's indeed slow.
    
    start_call = time.time()
    res = engine.search(db, query)
    end_call = time.time()
    
    print(f"Total search call took: {end_call - start_call:.4f}s")
    
    db.close()

if __name__ == "__main__":
    query = "kababaihan sa lipunan" # Example query that might trigger semantic fallback
    if len(sys.argv) > 1:
        query = sys.argv[1]
    profile_search(query)
