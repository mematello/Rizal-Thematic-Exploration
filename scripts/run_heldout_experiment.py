import os
import sys
import json
import subprocess
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
backend_dir = base_dir / "backend"
docs_dir = base_dir / "docs" / "perf"
scripts_dir = base_dir / "scripts"
python_exe = backend_dir / ".venv" / "Scripts" / "python.exe"

# Worker script for held-out execution
worker_code = """
import os, sys, json
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
backend_dir = base_dir / "backend"
sys.path.insert(0, str(backend_dir))
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PERF_BENCHMARK'] = '1'

from app.models.database import SessionLocal
from app.core.engine import get_engine, RizalEngine

db = SessionLocal()
engine = get_engine()

with open(base_dir / "scripts" / "heldout_queries.json", "r", encoding="utf-8") as f:
    hdata = json.load(f)

queries = []
for cat, qs in hdata["categories"].items():
    queries.extend(qs)

mode = sys.argv[1] # 'HEAD' or 'S1'
order = sys.argv[2] # 'fwd' or 'rev'
out_file = sys.argv[3]

if order == 'rev':
    queries.reverse()

if mode == 'S1':
    import inspect, textwrap
    src = textwrap.dedent(inspect.getsource(RizalEngine))
    # Revert E2: sorted(bridge_tokens) -> bridge_tokens
    src = src.replace(
        'bridge_text = \" \".join(sorted([bt for bt in bridge_tokens if len(bt) > 3]))',
        'bridge_text = \" \".join([bt for bt in bridge_tokens if len(bt) > 3])'
    )
    # Revert E3: sorted(sent_words) -> sent_words
    src = src.replace(
        'cand_words = [w for w in sorted(sent_words) if len(w) > 4 and w not in self.LOW_INFO_WORDS]',
        'cand_words = [w for w in sent_words if len(w) > 4 and w not in self.LOW_INFO_WORDS]'
    )
    globals_dict = dict(RizalEngine.search.__globals__)
    local_env = {}
    exec(src, globals_dict, local_env)
    engine.search = local_env['RizalEngine'].search.__get__(engine, RizalEngine)

results_snap = {'queries': {}}
for q in queries:
    res = engine.search(db, q, source_type='summary', novel='both')
    results_snap['queries'][q] = res

with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(results_snap, f, indent=2)
print(f"Finished {mode} {order} with seed {os.environ.get('PYTHONHASHSEED')} -> {out_file}", flush=True)
"""

worker_file = scripts_dir / "_heldout_worker.py"
with open(worker_file, "w", encoding="utf-8") as f:
    f.write(worker_code)

configurations = [
    # (mode, order, seed, filename)
    # 1. HEAD (E1+E2+E3)
    ("HEAD", "fwd", "unset", docs_dir / "heldout_head_unset_fwd.json"),
    ("HEAD", "rev", "unset", docs_dir / "heldout_head_unset_rev.json"),
    ("HEAD", "fwd", "1",     docs_dir / "heldout_head_seed1.json"),
    ("HEAD", "fwd", "12345", docs_dir / "heldout_head_seed12345.json"),
    # 2. S1 (E1 only)
    ("S1",   "fwd", "unset", docs_dir / "heldout_s1_unset_fwd.json"),
    ("S1",   "rev", "unset", docs_dir / "heldout_s1_unset_rev.json"),
    ("S1",   "fwd", "1",     docs_dir / "heldout_s1_seed1.json"),
    ("S1",   "fwd", "12345", docs_dir / "heldout_s1_seed12345.json"),
]

for mode, order, seed, out_snap in configurations:
    if out_snap.exists():
        continue
    print(f"Running {mode} {order} seed {seed}...", flush=True)
    env = dict(os.environ)
    if seed == "unset":
        env.pop("PYTHONHASHSEED", None)
    else:
        env["PYTHONHASHSEED"] = seed
    res = subprocess.run([str(python_exe), str(worker_file), mode, order, str(out_snap)], env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error in {mode} {order} seed {seed}:", res.stderr)
        sys.exit(1)

# Pairwise comparisons
sys.path.insert(0, str(scripts_dir))
from strict_comparator import compare_strict

def run_pairwise_matrix(title, file_map):
    print("\n" + "=" * 90)
    print(f"PAIRWISE STRICT COMPARISON: {title}")
    print("=" * 90)
    keys = list(file_map.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            rep = compare_strict(file_map[k1], file_map[k2])
            n_diff = rep["queries_with_changes"]
            print(f"  {k1:<20} vs {k2:<20} : {n_diff:>3d} diffs / {rep['total_queries']} queries")
            if n_diff > 0:
                # print first divergent query and field
                first_q = list(rep["diffs"].keys())[0]
                diff_obj = rep["diffs"][first_q]
                print(f"    -> First divergence in query '{first_q}': keys={list(diff_obj.keys())}")

head_files = {
    "unset (fwd)": docs_dir / "heldout_head_unset_fwd.json",
    "unset (rev)": docs_dir / "heldout_head_unset_rev.json",
    "seed 1":      docs_dir / "heldout_head_seed1.json",
    "seed 12345":  docs_dir / "heldout_head_seed12345.json",
}

s1_files = {
    "unset (fwd)": docs_dir / "heldout_s1_unset_fwd.json",
    "unset (rev)": docs_dir / "heldout_s1_unset_rev.json",
    "seed 1":      docs_dir / "heldout_s1_seed1.json",
    "seed 12345":  docs_dir / "heldout_s1_seed12345.json",
}

run_pairwise_matrix("HEAD (Commit E3 - All Edits)", head_files)
run_pairwise_matrix("S1 (Commit E1 Only - Reverted E2 and E3)", s1_files)

if worker_file.exists():
    worker_file.unlink()
