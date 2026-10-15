import os
import sys
from pathlib import Path
from strict_comparator import compare_strict

base_dir = Path(__file__).parent.parent
perf_dir = base_dir / "docs" / "perf"

files = [
    ("unset_fwd", perf_dir / "proc1_unset_fwd.json"),
    ("unset_rev", perf_dir / "proc2_unset_rev.json"),
    ("seed1",     perf_dir / "proc3_seed1.json"),
    ("seed2",     perf_dir / "proc4_seed2.json"),
    ("seed12345", perf_dir / "proc5_seed12345.json"),
    ("seed0",     perf_dir / "proc6_seed0.json"),
]

labels = [f[0] for f in files]
n = len(files)

matrix = [[0]*n for _ in range(n)]
all_diffs = {}

for i in range(n):
    for j in range(i, n):
        if i == j:
            matrix[i][j] = 0
            continue
        rep = compare_strict(files[i][1], files[j][1], allow_missing_diagnostics=False)
        matrix[i][j] = rep["queries_with_changes"]
        matrix[j][i] = rep["queries_with_changes"]
        if rep["queries_with_changes"] > 0:
            all_diffs[(labels[i], labels[j])] = rep["diffs"]

print("=" * 80)
print("PAIRWISE STRICT COMPARISON MATRIX (Queries with changes / 40)")
print("=" * 80)
header = f"{'':12}" + "".join(f"{l:>12}" for l in labels)
print(header)
for i in range(n):
    row_str = f"{labels[i]:12}" + "".join(f"{matrix[i][j]:>12}" for j in range(n))
    print(row_str)
print("=" * 80)

if not all_diffs:
    print("\n[VERDICT] ALL 6 PROCESSES ARE 100% BIT-IDENTICAL ACROSS ALL 40 QUERIES!")
    print("Cross-process determinism WITHOUT PYTHONHASHSEED PROVEN!")
else:
    print(f"\n[DIVERGENCE FOUND] {len(all_diffs)} pairs had differences:")
    for (p1, p2), diffs in all_diffs.items():
        print(f"\n--- Pair: {p1} vs {p2} ---")
        first_q = list(diffs.keys())[0]
        print(f"First divergent query: '{first_q}'")
        print(f"Diff details: {diffs[first_q]}")
