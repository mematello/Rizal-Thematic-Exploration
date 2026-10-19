
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.engine import RizalEngine
from app.models.database import SessionLocal
import json

def test():
    engine = RizalEngine()
    
    queries = [
        "edukasyon",
        "pang-aapi ng kastila",
        "elias",
        "maria clara",
        "simbahan"
    ]
    
    with SessionLocal() as db:
        for q in queries:
            print(f"\n--- QUERY: {q} ---")
            res = engine.search(db, q, top_k=2)
            results = res.get("results", {})
            combined = results.get('noli', []) + results.get('elfili', [])
            
            for i, r in enumerate(combined):
                themes = r.get("themes", [])
                theme_str = f"{themes[0]['label']} ({themes[0]['score']:.2f})" if themes else "None"
                print(f"Result {i+1}: {r.get('sentence_text')} -> Predicted Theme: {theme_str}")

if __name__ == "__main__":
    test()
