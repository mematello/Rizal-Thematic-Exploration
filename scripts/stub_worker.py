
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve()
while _repo_root.parent != _repo_root and not (_repo_root / "scripts" / "prod_guard.py").exists():
    _repo_root = _repo_root.parent
if str(_repo_root / "scripts") not in sys.path:
    sys.path.insert(0, str(_repo_root / "scripts"))
from prod_guard import verify_not_production_db
verify_not_production_db(caller_name=Path(__file__).name)


import os, sys, json
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['DATABASE_URL'] = 'postgresql://rizal:rizalpass123@127.0.0.1:5433/rizal_db'
os.environ['REDIS_URL'] = 'redis://localhost:6379'
os.environ['PERF_BENCHMARK'] = '1'
sys.path.insert(0, 'backend')

from app.core.engine import get_engine, _translator_cache
from app.models.database import SessionLocal

STUB_DICT = {
    "bitcoin": "bitcoin",
    "cellphone": "telepono",
    "computer": "kompyuter",
    "covid": "covid",
    "facebook": "facebook",
    "internet": "internet",
    "online": "online",
    "tiktok": "tiktok",
    "korapsyon": "katiwalian",
    "laban sa korapsyon": "laban sa katiwalian",
    "karahasan ng mga prayle": "karahasan ng mga prayle",
    "dusa ng bilanggo": "dusa ng bilanggo",
    "hapis at hilahil": "hapis at hilahil",
    "lupit ng prayle": "lupit ng prayle",
    "panata sa simbahan": "panata sa simbahan",
    "tinig ng kunsensya": "tinig ng budhi",
    "noli me tangere. el filibusterismo.": "noli me tangere. el filibusterismo.",
    "sakit ng lipunan. kanser ng bayan.": "sakit ng lipunan. kanser ng bayan.",
}

def stub_translate(text: str):
    k = text.strip().lower()
    return STUB_DICT.get(k, k), True

_translator_cache.translate = stub_translate
engine = get_engine()
db = SessionLocal()

query_file = sys.argv[1]
reverse_order = "--reverse" in sys.argv
out_path = sys.argv[2]

with open(query_file, 'r', encoding='utf-8') as f:
    qdata = json.load(f)

first_val = next(iter(qdata['categories'].values()))
queries = []
if isinstance(first_val, list):
    for qs in qdata['categories'].values():
        queries.extend(qs)
elif isinstance(first_val, dict) and 'queries' in first_val:
    for c in qdata['categories'].values():
        queries.extend(c['queries'])

if reverse_order:
    queries = list(reversed(queries))

snap = {"queries": {}}
for q in queries:
    res = engine.search(db, q, top_k=10, source_type='summary', novel='both')
    snap["queries"][q] = {
        "noli": res['results']['noli'],
        "elfili": res['results']['elfili'],
        "precise_matches": res.get('precise_matches', []),
        "metadata": res['metadata']
    }

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(snap, f, indent=2, ensure_ascii=False)
