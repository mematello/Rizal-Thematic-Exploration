
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

from app.models.database import SessionLocal
from app.core.engine import get_engine

def run_eval():
    queries = [
        "edukasyon",
        "bakit mahalaga ang edukasyon",
        "pang-aapi ng kastila",
        "simbahan",
        "elias",
        "maria clara",
        "prayle sa pilipinas",
        "pag-aaral ng kabataan",
        "umupo sa silya",
        "tiktok",
        "namatay si elias" # test dynamic entity extraction
    ]
    
    db = SessionLocal()
    engine = get_engine()
    
    print("\n================ EVALUATING QUERY-CLASS SUGGESTIONS ================\n")
    for q in queries:
        print(f"--- Query: '{q}' ---")
        
        # Suppress prints from the internal logic for clean reporting
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            resp = engine.search(db, q, source_type="full")
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout

        meta = resp.get("metadata", {})
        suggestions = meta.get("suggestions", [])
        
        if not suggestions:
            print("  [SUPPRESSED]")
            print("  Reason: Classified as Question, Literal Fragment, or 0 Results")
        else:
            print(f"  [RENDERED]")
            for i, s in enumerate(suggestions, 1):
                print(f"    {i}. {s}")
        print("\n")
                
if __name__ == "__main__":
    run_eval()
