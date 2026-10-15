import os
import sys
import json
import subprocess
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
scripts_dir = base_dir / "scripts"
worker = scripts_dir / "run_batch_valid_tokens.py"

runs = [
    ("set", "1", scripts_dir / "vt_seed1.json"),
    ("set", "2", scripts_dir / "vt_seed2.json"),
    ("set", "12345", scripts_dir / "vt_seed12345.json"),
    ("sorted", "1", scripts_dir / "vt_sorted.json"),
]

for mode, seed, out_path in runs:
    print(f"Running {mode} under seed {seed}...")
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    res = subprocess.run([sys.executable, str(worker), mode, str(out_path)], env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error in {mode} seed {seed}:", res.stderr)
        sys.exit(1)
    print(f"  {res.stdout.strip()}")

print("\n" + "=" * 90)
print("VALID_TOKENS & PASSING CANDIDATE IDS EXPERIMENT RESULTS")
print("=" * 90)

queries = ['kalayaan', 'edukasyon', 'pag-ibig', 'kamatayan', 'oppression', 'justice', 'love']

with open(scripts_dir / "vt_seed1.json", "r", encoding="utf-8") as f:
    d_seed1 = json.load(f)
with open(scripts_dir / "vt_seed2.json", "r", encoding="utf-8") as f:
    d_seed2 = json.load(f)
with open(scripts_dir / "vt_seed12345.json", "r", encoding="utf-8") as f:
    d_seed12345 = json.load(f)
with open(scripts_dir / "vt_sorted.json", "r", encoding="utf-8") as f:
    d_sorted = json.load(f)

for q in queries:
    vt1 = d_seed1[q]['valid_tokens']
    vt2 = d_seed2[q]['valid_tokens']
    vt12345 = d_seed12345[q]['valid_tokens']
    vts = d_sorted[q]['valid_tokens']

    ids1 = d_seed1[q]['passed_ids']
    ids2 = d_seed2[q]['passed_ids']
    ids12345 = d_seed12345[q]['passed_ids']
    idss = d_sorted[q]['passed_ids']

    vt_set_identical = (vt1 == vt2 == vt12345)
    ids_set_identical = (ids1 == ids2 == ids12345)

    vt_matches_sorted = (vt1 == vts)
    ids_matches_sorted = (ids1 == idss)

    print(f"\nQUERY: '{q}'")
    print(f"  Passed IDs count: seed1={len(ids1)}, seed2={len(ids2)}, seed12345={len(ids12345)}, sorted={len(idss)}")
    print(f"  Valid tokens count: seed1={len(vt1)}, seed2={len(vt2)}, seed12345={len(vt12345)}, sorted={len(vts)}")
    print(f"  Under raw set order (seeds 1 vs 2 vs 12345):")
    print(f"    - valid_tokens identical across seeds: {vt_set_identical}")
    print(f"    - passed_ids identical across seeds:   {ids_set_identical}")
    if not ids_set_identical:
        print(f"      Diff ids1 vs ids2: {len(set(ids1) ^ set(ids2))} differences")
    print(f"  Raw set vs sorted (Commit E3):")
    print(f"    - valid_tokens matches sorted: {vt_matches_sorted}")
    print(f"    - passed_ids matches sorted:   {ids_matches_sorted}")
