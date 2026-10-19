
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

from app.models.database import SessionLocal, Sentence
from app.core.engine import get_engine

db = SessionLocal()
engine = get_engine()

# Load snapshots
with open('docs/perf/golden_snapshot_seed0.json', 'r', encoding='utf-8') as f:
    golden_snap = json.load(f)['queries']
with open('docs/perf/b1_snapshot_seed0.json', 'r', encoding='utf-8') as f:
    b1_snap = json.load(f)['queries']
with open('docs/perf/comparison_report.json', 'r', encoding='utf-8') as f:
    comp_report = json.load(f)

changed_queries = list(comp_report['query_diffs'].keys())

# Monkey patch engine to save full pool before [:top_k] cut
orig_search = engine.search

saved_full_pools = {}

def hooked_search(*args, **kwargs):
    # Call orig_search with top_k=10, but intercept candidates inside
    # In engine.py, candidates are put in results['noli'] and results['elfili']
    # Let's inspect the code of search() or capture via a custom wrapper
    pass

# Rather than monkey-patching internal locals, let's look at what engine.search does:
# It runs lexical retrieval (limit 200), multi-anchor (limit 20), semantic fallback (limit 150),
# computes scores for all candidates, filters by min_threshold, appends to results['noli']/['elfili'],
# sorts by key, and takes lst[:top_k].
# If we temporarily replace `top_k = 10` with custom slicing, or simply inspect:
# Actually, if we look at `top_k * 20`: for top_k=10, limit is 200.
# Earlier we proved that for ALL 16 queries in summary mode, count(*) NEVER exceeded 156!
# Simoun had 156, Basilio had 115. None exceeded 200!
# So limit 200 was NEVER reached for any of the 16 queries in summary mode!
# Thus top_k=10 and top_k=200 produce the EXACT SAME retrieval pool for all 16 queries!
