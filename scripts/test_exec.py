
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
from dotenv import load_dotenv

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
load_dotenv('backend/.env')
sys.path.insert(0, 'backend')

from app.models.database import SessionLocal
from app.core.engine import get_engine, RizalEngine

db = SessionLocal()
engine = get_engine()

src = textwrap.dedent(inspect.getsource(RizalEngine.search))

target_lex = "            return unique[:top_k]"
hook_lex = """            global_full_pools[label] = list(unique)
            return unique[:top_k]"""
assert target_lex in src, "target_lex not found"
src_mod = src.replace(target_lex, hook_lex)

target_sem = "            results[book_key] = (lex_reserved + sem_only)[:top_k]"
hook_sem = """            global_full_pools[book_key] = list(lex_reserved + sem_only)
            results[book_key] = (lex_reserved + sem_only)[:top_k]"""
assert target_sem in src_mod, "target_sem not found"
src_mod = src_mod.replace(target_sem, hook_sem)

globals_dict = dict(RizalEngine.search.__globals__)
globals_dict['global_full_pools'] = {}
local_env = {}

exec(src_mod, globals_dict, local_env)
engine.search_full = local_env['search'].__get__(engine, RizalEngine)
engine.global_full_pools = globals_dict['global_full_pools']

res = engine.search_full(db, 'bayan', source_type='summary', novel='both')
print("Noli full pool size:   ", len(engine.global_full_pools['noli']))
print("El Fili full pool size:", len(engine.global_full_pools['elfili']))
print("Top 10 Noli IDs:       ", [x['id'] for x in res['results']['noli']])
print("Top 10 El Fili IDs:    ", [x['id'] for x in res['results']['elfili']])
