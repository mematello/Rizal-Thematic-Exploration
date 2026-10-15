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

# We can evaluate the 40 queries across the 5 states S0..S4 using subprocess runner
# For S0, S1, S2, S4 we have git commits:
# S0: 964a2c1
# S1: 17896f3
# S2: 6e16b16
# S4: 73f1d1a
# For S3, it is S1 + E3 (or S4 with E2 reverted in memory/monkeypatch)

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

# Read golden queries
with open(base_dir / "scripts" / "golden_queries.json", "r", encoding="utf-8") as f:
    gdata = json.load(f)

queries = []
for cat in gdata["categories"].values():
    queries.extend(cat["queries"])

state = sys.argv[1] # S0, S1, S2, S3, S4

# Apply in-memory state monkeypatches relative to HEAD (S4)
# HEAD has E1, E2, E3
# If S0: revert E1, E2, E3
# If S1: revert E2, E3
# If S2: revert E3
# If S3: revert E2
# If S4: keep HEAD as is

import inspect, textwrap
src = textwrap.dedent(inspect.getsource(RizalEngine))

if state in ('S0',):
    # Revert E1: sorted(candidates) -> candidates, remove secondary sort key
    src = src.replace('for w in sorted(candidates):', 'for w in candidates:')
    src = src.replace(
        'final_tokens.sort(key=lambda w: (-self.word_theme_map.get(w, {}).get(target_theme_key, 0) / self.theme_word_df.get(w, 1), w))',
        'final_tokens.sort(key=lambda w: -self.word_theme_map.get(w, {}).get(target_theme_key, 0) / self.theme_word_df.get(w, 1))'
    )

if state in ('S0', 'S1', 'S3'):
    # Revert E2: sorted(bridge_tokens) -> bridge_tokens
    src = src.replace(
        'bridge_text = \" \".join(sorted([bt for bt in bridge_tokens if len(bt) > 3]))',
        'bridge_text = \" \".join([bt for bt in bridge_tokens if len(bt) > 3])'
    )

if state in ('S0', 'S1', 'S2'):
    # Revert E3: sorted(sent_words) -> sent_words
    src = src.replace(
        'cand_words = [w for w in sorted(sent_words) if len(w) > 4 and w not in self.LOW_INFO_WORDS]',
        'cand_words = [w for w in sent_words if len(w) > 4 and w not in self.LOW_INFO_WORDS]'
    )

globals_dict = dict(RizalEngine.search.__globals__)
local_env = {}
exec(src, globals_dict, local_env)
engine.search = local_env['RizalEngine'].search.__get__(engine, RizalEngine)
engine._expand_query_with_themes = local_env['RizalEngine']._expand_query_with_themes.__get__(engine, RizalEngine)

results_snap = {'queries': {}}
for q in queries:
    res = engine.search(db, q, source_type='summary', novel='both')
    results_snap['queries'][q] = res

out_file = sys.argv[2]
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(results_snap, f, indent=2)
"""

worker_file = scripts_dir / "_ablation_worker.py"
with open(worker_file, "w", encoding="utf-8") as f:
    f.write(worker_code)

states = ["S0", "S1", "S2", "S3", "S4"]
seeds = ["unset", "1", "2", "12345"]

# Generate all snapshots
for st in states:
    for sd in seeds:
        out_snap = docs_dir / f"ablation_{st}_seed_{sd}.json"
        if out_snap.exists():
            continue
        print(f"Generating {st} under seed {sd}...", flush=True)
        env = dict(os.environ)
        if sd == "unset":
            env.pop("PYTHONHASHSEED", None)
        else:
            env["PYTHONHASHSEED"] = sd
        res = subprocess.run([str(python_exe), str(worker_file), st, str(out_snap)], env=env, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Error in {st} seed {sd}:", res.stderr)
            sys.exit(1)

# Now compare within each state across seeds
sys.path.insert(0, str(scripts_dir))
from strict_comparator import compare_strict

print("\n" + "=" * 90)
print("ABLATION STUDY: CROSS-PROCESS DETERMINISM MATRIX (40 GOLDEN QUERIES)")
print("=" * 90)

print(f"{'State':<6} | {'Description':<32} | {'unset vs 1':<12} | {'1 vs 2':<12} | {'1 vs 12345':<12} | {'Max Diffs':<10}")
print("-" * 90)

desc_map = {
    "S0": "D + B1 (No E edits)",
    "S1": "S0 + E1 (Theme expansion)",
    "S2": "S1 + E2 (Theme + Bridge text)",
    "S3": "S1 + E3 (Theme + Dynamic val)",
    "S4": "S1 + E2 + E3 (HEAD, All edits)"
}

for st in states:
    snap_unset = docs_dir / f"ablation_{st}_seed_unset.json"
    snap_1 = docs_dir / f"ablation_{st}_seed_1.json"
    snap_2 = docs_dir / f"ablation_{st}_seed_2.json"
    snap_12345 = docs_dir / f"ablation_{st}_seed_12345.json"

    rep_u1 = compare_strict(snap_unset, snap_1)
    rep_12 = compare_strict(snap_1, snap_2)
    rep_1m = compare_strict(snap_1, snap_12345)

    d_u1 = rep_u1["queries_with_changes"]
    d_12 = rep_12["queries_with_changes"]
    d_1m = rep_1m["queries_with_changes"]
    max_d = max(d_u1, d_12, d_1m)

    print(f"{st:<6} | {desc_map[st]:<32} | {d_u1:>10d}q | {d_12:>10d}q | {d_1m:>10d}q | {max_d:>8d}q")

if worker_file.exists():
    worker_file.unlink()
