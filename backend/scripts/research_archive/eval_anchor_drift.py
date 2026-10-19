
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve()
while _repo_root.parent != _repo_root and not (_repo_root / "scripts" / "prod_guard.py").exists():
    _repo_root = _repo_root.parent
if str(_repo_root / "scripts") not in sys.path:
    sys.path.insert(0, str(_repo_root / "scripts"))
from prod_guard import verify_not_production_db
verify_not_production_db(caller_name=Path(__file__).name)

import os
import sys
from app.models.database import SessionLocal
from app.core.engine import RizalEngine

def run_semantic_drift_test():
    os.environ["DEBUG_SEARCH"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    test_queries = [
        "corruption",
        "power",
        "colonialism",
        "church",
        "friar",
        "priest",
        "sacrifice",
        "joy",
        "evening"
    ]
    
    engine = RizalEngine()
    db = SessionLocal()
    
    for q in test_queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {q.upper()}")
        print(f"{'='*80}")
        
        try:
            res = engine.search(db, q, top_k=3, source_type="full")
            noli_hits = res.get('results', {}).get('noli', [])
            fili_hits = res.get('results', {}).get('elfili', [])
            all_hits = noli_hits + fili_hits
            
            for i, hit in enumerate(all_hits[:3]):
                score = hit.get('scores', {}).get('final', 0)
                text = hit.get('sentence_text', '')
                print(f"[{i+1}] Score: {score:.3f}")
                print(f"    Text: {text}")
                
        except Exception as e:
            print(f"Failed: {e}")
            
    db.close()

if __name__ == "__main__":
    run_semantic_drift_test()
