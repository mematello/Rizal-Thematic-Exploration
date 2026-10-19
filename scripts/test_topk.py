
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
import json
from dotenv import load_dotenv

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
load_dotenv('backend/.env')
sys.path.insert(0, 'backend')

from app.models.database import SessionLocal
from app.core.engine import get_engine

db = SessionLocal()
engine = get_engine()

res10 = engine.search(db, 'bayan', source_type="summary", novel="both", top_k=10)
res1000 = engine.search(db, 'bayan', source_type="summary", novel="both", top_k=1000)

top10_ids = [x['id'] for x in res10['results']['elfili']]
top1000_top10_ids = [x['id'] for x in res1000['results']['elfili'][:10]]

print("top_k=10 top 10 IDs:   ", top10_ids)
print("top_k=1000 top 10 IDs: ", top1000_top10_ids)
print("Identical top 10:", top10_ids == top1000_top10_ids)
print("Full pool size with top_k=1000 in elfili:", len(res1000['results']['elfili']))
