
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

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
from dotenv import load_dotenv

load_dotenv('backend/.env')
sys.path.insert(0, 'backend')

from app.models.database import SessionLocal
from app.core.engine import get_engine, RizalEngine

db = SessionLocal()
engine = get_engine()

mode = sys.argv[1] # 'set' or 'sorted'
q = sys.argv[2]

src = textwrap.dedent(inspect.getsource(RizalEngine.search))

if mode == 'sorted':
    target_cand = "cand_words = [w for w in sent_words if w in self.vocabulary and len(w) > 3 and w not in self.LOW_INFO_WORDS]"
    replace_cand = "cand_words = [w for w in sorted(sent_words) if w in self.vocabulary and len(w) > 3 and w not in self.LOW_INFO_WORDS]"
    assert target_cand in src, "target_cand not found in src"
    src = src.replace(target_cand, replace_cand)

# Hook passed candidates:
target_loop = "        for sent in candidates:"
hook_loop = """        passed_candidate_ids = []
        for sent in candidates:"""
# There are two 'for sent in candidates:' in the file, but inspect.getsource(RizalEngine.search) only has the search method!
# In search(), how many 'for sent in candidates:' are there? Just one!
src = src.replace(target_loop, hook_loop)

# Hook after validation:
target_val = "            if not has_validation:"
hook_val = """            if has_validation:
                passed_candidate_ids.append(sent.id)
            if not has_validation:"""
src = src.replace(target_val, hook_val)

# Hook return:
target_ret = "        return {\n            'results': results,"
hook_ret = """        return {
            'valid_tokens': sorted(list(valid_tokens)),
            'passed_candidate_ids': passed_candidate_ids,
            'results': results,"""
src = src.replace(target_ret, hook_ret)

globals_dict = dict(RizalEngine.search.__globals__)
local_env = {}
exec(src, globals_dict, local_env)
search_hooked = local_env['search'].__get__(engine, RizalEngine)

out = search_hooked(db, q, source_type='summary', novel='both')

print(f"QUERY: {q} | MODE: {mode} | SEED: {os.environ.get('PYTHONHASHSEED')}")
print(f"VALID_TOKENS ({len(out['valid_tokens'])}): {out['valid_tokens']}")
print(f"PASSED_IDS ({len(out['passed_candidate_ids'])}): {out['passed_candidate_ids']}")
