
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
import inspect
import textwrap
import json
from dotenv import load_dotenv

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
load_dotenv('backend/.env')
sys.path.insert(0, 'backend')

from app.models.database import SessionLocal
from app.core.engine import get_engine, RizalEngine

db = SessionLocal()
engine = get_engine()

with open('docs/perf/golden_snapshot_seed0.json', 'r', encoding='utf-8') as f:
    golden_snap = json.load(f)['queries']
with open('docs/perf/b1_snapshot_seed0.json', 'r', encoding='utf-8') as f:
    b1_snap = json.load(f)['queries']
with open('docs/perf/comparison_report.json', 'r', encoding='utf-8') as f:
    comp_report = json.load(f)

changed_queries = list(comp_report['query_diffs'].keys())

src = textwrap.dedent(inspect.getsource(RizalEngine.search))
target_lex = "            return unique[:top_k]"
hook_lex = """            global_full_pools[label] = list(unique)
            return unique[:top_k]"""
src_mod = src.replace(target_lex, hook_lex)
target_sem = "            results[book_key] = (lex_reserved + sem_only)[:top_k]"
hook_sem = """            global_full_pools[book_key] = list(lex_reserved + sem_only)
            results[book_key] = (lex_reserved + sem_only)[:top_k]"""
src_mod = src_mod.replace(target_sem, hook_sem)

globals_dict = dict(RizalEngine.search.__globals__)
globals_dict['global_full_pools'] = {}
local_env = {}

exec(src_mod, globals_dict, local_env)
engine.search_full = local_env['search'].__get__(engine, RizalEngine)
engine.global_full_pools = globals_dict['global_full_pools']

for q in changed_queries:
    print(f"Running query: {q}...", flush=True)
    engine.global_full_pools.clear()
    res = engine.search_full(db, q, source_type='summary', novel='both')
    print(f"Finished query: {q}. Noli: {len(engine.global_full_pools.get('noli', []))}, El Fili: {len(engine.global_full_pools.get('elfili', []))}", flush=True)
