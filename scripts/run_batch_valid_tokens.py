import os
import sys
import inspect
import textwrap
import json

base_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(base_dir, '..', 'backend')
sys.path.insert(0, backend_dir)
sys.path.insert(0, os.getcwd())

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
from dotenv import load_dotenv

load_dotenv(os.path.join(backend_dir, '.env'))

from app.models.database import SessionLocal
from app.core.engine import get_engine, RizalEngine

db = SessionLocal()
engine = get_engine()

mode = sys.argv[1] # 'set' or 'sorted'
queries = ['kalayaan', 'edukasyon', 'pag-ibig', 'kamatayan', 'oppression', 'justice', 'love']

src = textwrap.dedent(inspect.getsource(RizalEngine.search))

if mode == 'set':
    target_cand = "cand_words = [w for w in sorted(sent_words) if len(w) > 4 and w not in self.LOW_INFO_WORDS]"
    replace_cand = "cand_words = [w for w in sent_words if len(w) > 4 and w not in self.LOW_INFO_WORDS]"
    assert target_cand in src, "target_cand not found in src"
    src = src.replace(target_cand, replace_cand)

# Hook passed candidates:
target_loop = "    for sent in candidates:"
hook_loop = """    passed_candidate_ids = []
    for sent in candidates:"""
assert target_loop in src, "target_loop not found in src"
src = src.replace(target_loop, hook_loop)

# Hook after validation:
target_val = "            if not has_validation:"
hook_val = """            if has_validation:
                passed_candidate_ids.append(sent.id)
            if not has_validation:"""
assert target_val in src, "target_val not found in src"
src = src.replace(target_val, hook_val)

# Hook return:
target_ret = '    return {\n        "results": results,'
hook_ret = """    return {
        "valid_tokens": sorted(list(valid_tokens)),
        "passed_candidate_ids": passed_candidate_ids,
        "results": results,"""
assert target_ret in src, "target_ret not found in src"
src = src.replace(target_ret, hook_ret)

globals_dict = dict(RizalEngine.search.__globals__)
local_env = {}
exec(src, globals_dict, local_env)
search_hooked = local_env['search'].__get__(engine, RizalEngine)

report = {}
for q in queries:
    out = search_hooked(db, q, source_type='summary', novel='both')
    report[q] = {
        'valid_tokens': out['valid_tokens'],
        'passed_ids': out['passed_candidate_ids']
    }

output_file = sys.argv[2]
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2)
print(f"Finished {mode} with seed {os.environ.get('PYTHONHASHSEED')} -> {output_file}", flush=True)
