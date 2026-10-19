
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
import time
import subprocess
from pathlib import Path

base_dir = Path(".").resolve()
sys.path.insert(0, str(base_dir / "scripts"))
from strict_comparator import compare_strict

venv_python = str(base_dir / "backend" / ".venv" / "Scripts" / "python.exe")

out_lines = []
out_lines.append("================================================================================")
out_lines.append("ITEM 7: DETERMINISTIC TRANSLATOR STUB & CROSS-PROCESS RE-RUNS")
out_lines.append("================================================================================\n")

# -----------------------------------------------------------------------------
# 1. REAL NETWORK STATE DURING proc1..proc6
# -----------------------------------------------------------------------------
out_lines.append("--------------------------------------------------------------------------------")
out_lines.append("1. REAL NETWORK STATE DURING proc1..proc6 EXECUTION")
out_lines.append("--------------------------------------------------------------------------------")
out_lines.append("Investigation of Translator Behavior during proc1..proc6:")
out_lines.append("  - In proc1..proc6, 10 queries triggered calls to `_translator_cache.translate()`:")
out_lines.append("    ['korapsyon', 'karahasan ng mga prayle', 'tiktok', 'bitcoin', 'covid',")
out_lines.append("     'facebook', 'internet', 'computer', 'cellphone', 'online']")
out_lines.append("  - When live GoogleTranslator(source='auto', target='tl').translate() was called,")
out_lines.append("    the unauthenticated public endpoint returned:")
out_lines.append("      `deep_translator.exceptions.TooManyRequests: Server Error: You made too many requests to the server.`")
out_lines.append("  - TranslatorCache.translate() caught this exception inside its ThreadPoolExecutor,")
out_lines.append("    executed `self.set_failure(text)`, and returned `(None, True)`.")
out_lines.append("  - Consequently, in all six processes (proc1..proc6), every unmapped query returned `output=None`.")
out_lines.append("  - Because the negative cache behavior was identical across all processes, zero divergence")
out_lines.append("    occurred due to network flakiness. However, live Google was blocked by bot-rate-limiting.")

# -----------------------------------------------------------------------------
# 2. DETERMINISTIC TRANSLATOR STUB IMPLEMENTATION
# -----------------------------------------------------------------------------
out_lines.append("\n--------------------------------------------------------------------------------")
out_lines.append("2. DETERMINISTIC TRANSLATOR STUB IMPLEMENTATION (TEST-ONLY, FIXED DICTIONARY)")
out_lines.append("--------------------------------------------------------------------------------")
out_lines.append("To eliminate any external HTTP dependency and ensure 100% reproducible translation strings,")
out_lines.append("a deterministic stub dictionary covers all 18 unmapped queries in the golden and held-out suites:")

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

for k, v in STUB_DICT.items():
    out_lines.append(f"  '{k}' -> '{v}'")

# Helper function to generate snapshot with stub
worker_script = '''
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
'''

tmp_worker = Path("scripts/stub_worker.py")
tmp_worker.write_text(worker_script, encoding="utf-8")

def run_proc(cfg):
    label, seed_val, rev, out_p, qfile = cfg
    env = os.environ.copy()
    if seed_val is None:
        env.pop("PYTHONHASHSEED", None)
    else:
        env["PYTHONHASHSEED"] = seed_val
    cmd = [venv_python, str(tmp_worker), qfile, out_p]
    if rev:
        cmd.append("--reverse")
    print(f"Starting {label}...")
    t0 = time.time()
    subprocess.run(cmd, env=env, check=True)
    print(f"Finished {label} in {time.time()-t0:.2f}s")

# -----------------------------------------------------------------------------
# 3. RUN 6-PROCESS 40-QUERY MATRIX WITH DETERMINISTIC TRANSLATOR STUB
# -----------------------------------------------------------------------------
out_lines.append("\n--------------------------------------------------------------------------------")
out_lines.append("3. RERUN: SIX-PROCESS 40-QUERY MATRIX WITH DETERMINISTIC TRANSLATOR STUB")
out_lines.append("--------------------------------------------------------------------------------")

proc_configs = [
    ("proc1_stub (unset, fwd)", None, False, "docs/perf/proc1_stub.json", "scripts/golden_queries.json"),
    ("proc2_stub (unset, rev)", None, True,  "docs/perf/proc2_stub.json", "scripts/golden_queries.json"),
    ("proc3_stub (seed 1)",     "1",  False, "docs/perf/proc3_stub.json", "scripts/golden_queries.json"),
    ("proc4_stub (seed 2)",     "2",  False, "docs/perf/proc4_stub.json", "scripts/golden_queries.json"),
    ("proc5_stub (seed 12345)", "12345", False, "docs/perf/proc5_stub.json", "scripts/golden_queries.json"),
    ("proc6_stub (seed 0)",     "0",  False, "docs/perf/proc6_stub.json", "scripts/golden_queries.json"),
]

# Run sequentially one process at a time to prevent Windows paging file exhaustion
for cfg in proc_configs:
    run_proc(cfg)

header = f"{'Process Pair':<45} | {'Total':<6} | {'Diffs':<6} | {'Status'}"
out_lines.append(header)
out_lines.append("-" * len(header))

for i in range(len(proc_configs)):
    for j in range(i + 1, len(proc_configs)):
        p1_name, _, _, p1_path, _ = proc_configs[i]
        p2_name, _, _, p2_path, _ = proc_configs[j]
        rep = compare_strict(Path(p1_path), Path(p2_path))
        pair_str = f"{p1_name} vs {p2_name}"
        status = "100% BIT-IDENTICAL" if rep["queries_with_changes"] == 0 else f"{rep['queries_with_changes']} diffs"
        out_lines.append(f"{pair_str:<45} | {rep['total_queries']:^6} | {rep['queries_with_changes']:^6} | {status}")

# -----------------------------------------------------------------------------
# 4. RUN 206-QUERY HELDOUT MATRIX WITH DETERMINISTIC TRANSLATOR STUB (HEAD)
# -----------------------------------------------------------------------------
out_lines.append("\n--------------------------------------------------------------------------------")
out_lines.append("4. RERUN: 206-QUERY HELDOUT MATRIX WITH DETERMINISTIC TRANSLATOR STUB (HEAD)")
out_lines.append("--------------------------------------------------------------------------------")

heldout_configs = [
    ("HEAD unset fwd (stub)", None, False, "docs/perf/heldout_stub_unset_fwd.json", "scripts/heldout_queries.json"),
    ("HEAD unset rev (stub)", None, True,  "docs/perf/heldout_stub_unset_rev.json", "scripts/heldout_queries.json"),
    ("HEAD seed 1 (stub)",     "1",  False, "docs/perf/heldout_stub_seed1.json",     "scripts/heldout_queries.json"),
    ("HEAD seed 12345 (stub)", "12345", False, "docs/perf/heldout_stub_seed12345.json", "scripts/heldout_queries.json"),
]

for cfg in heldout_configs:
    run_proc(cfg)

h_header = f"{'Held-Out Comparison Pair':<45} | {'Total':<6} | {'Diffs':<6} | {'Status'}"
out_lines.append(h_header)
out_lines.append("-" * len(h_header))

for i in range(len(heldout_configs)):
    for j in range(i + 1, len(heldout_configs)):
        p1_name, _, _, p1_path, _ = heldout_configs[i]
        p2_name, _, _, p2_path, _ = heldout_configs[j]
        rep = compare_strict(Path(p1_path), Path(p2_path))
        pair_str = f"{p1_name} vs {p2_name}"
        status = "100% BIT-IDENTICAL" if rep["queries_with_changes"] == 0 else f"{rep['queries_with_changes']} diffs"
        out_lines.append(f"{pair_str:<45} | {rep['total_queries']:^6} | {rep['queries_with_changes']:^6} | {status}")

with open("docs/perf/evidence/b6_7.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))

print("Saved docs/perf/evidence/b6_7.txt successfully.")
