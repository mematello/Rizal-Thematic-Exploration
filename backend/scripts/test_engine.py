
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
    with SessionLocal() as db:
        res = engine.search(db, "edukasyon", top_k=2)
        print(json.dumps(res, indent=2))

if __name__ == "__main__":
    test()
