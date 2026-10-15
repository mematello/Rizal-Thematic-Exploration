import sys
import os
import json
from pathlib import Path

base_dir = Path(".").resolve()
sys.path.insert(0, str(base_dir / "scripts"))

from strict_comparator import compare_strict

out_lines = []
out_lines.append("================================================================================")
out_lines.append("ITEM 7: STRICT COMPARATOR IMPLEMENTATION & FULL RE-RUN MATRICES")
out_lines.append("================================================================================\n")

# 1. Print full source of scripts/strict_comparator.py
out_lines.append("1. FULL SOURCE OF scripts/strict_comparator.py:")
with open("scripts/strict_comparator.py", "r", encoding="utf-8") as f:
    out_lines.append(f.read())

# 2. Rerun (a): golden vs b1
out_lines.append("\n--------------------------------------------------------------------------------")
out_lines.append("2. RERUN (a): golden_snapshot_seed0.json vs b1_snapshot_seed0.json")
out_lines.append("--------------------------------------------------------------------------------")
rep_b1 = compare_strict(
    Path("docs/perf/golden_snapshot_seed0.json"),
    Path("docs/perf/b1_snapshot_seed0.json"),
    allow_missing_diagnostics=True
)
out_lines.append(f"Total queries: {rep_b1['total_queries']} | Queries with changes: {rep_b1['queries_with_changes']}")
b1_names = sorted(list(rep_b1["diffs"].keys()))
out_lines.append(f"The 16 changed queries in B1 are:")
for idx, qn in enumerate(b1_names, 1):
    out_lines.append(f"  {idx:2d}. {qn}")

# 3. Rerun (b): D+B1 under seed 1 vs seed 2 (negative control)
out_lines.append("\n--------------------------------------------------------------------------------")
out_lines.append("3. RERUN (b): NEGATIVE CONTROL D+B1 (PRE-E1) UNDER SEED 1 vs SEED 2")
out_lines.append("--------------------------------------------------------------------------------")
rep_ctrl = compare_strict(
    Path("docs/perf/ablation_S0_seed_1.json"),
    Path("docs/perf/ablation_S0_seed_2.json")
)
out_lines.append(f"Total queries: {rep_ctrl['total_queries']} | Queries with changes: {rep_ctrl['queries_with_changes']}")
ctrl_names = sorted(list(rep_ctrl["diffs"].keys()))
out_lines.append(f"Divergent queries ({len(ctrl_names)}):")
for idx, qn in enumerate(ctrl_names, 1):
    out_lines.append(f"  {idx:2d}. {qn}")

# 4. Six-process matrix on HEAD (proc1..proc6)
out_lines.append("\n--------------------------------------------------------------------------------")
out_lines.append("4. FULL 6x6 PAIRWISE COMPARISON MATRIX ACROSS PROC1..PROC6 (ON HEAD)")
out_lines.append("--------------------------------------------------------------------------------")
procs = [
    ("proc1 (unset, fwd)", "docs/perf/proc1_unset_fwd.json"),
    ("proc2 (unset, rev)", "docs/perf/proc2_unset_rev.json"),
    ("proc3 (seed 1)",     "docs/perf/proc3_seed1.json"),
    ("proc4 (seed 2)",     "docs/perf/proc4_seed2.json"),
    ("proc5 (seed 12345)", "docs/perf/proc5_seed12345.json"),
    ("proc6 (seed 0)",     "docs/perf/proc6_seed0.json")
]

header = f"{'Process Pair':<45} | {'Total':<6} | {'Diffs':<6} | {'Status'}"
out_lines.append(header)
out_lines.append("-" * len(header))

for i in range(len(procs)):
    for j in range(i + 1, len(procs)):
        p1_name, p1_path = procs[i]
        p2_name, p2_path = procs[j]
        rep = compare_strict(Path(p1_path), Path(p2_path))
        pair_str = f"{p1_name} vs {p2_name}"
        status = "100% BIT-IDENTICAL (0 diffs)" if rep["queries_with_changes"] == 0 else f"{rep['queries_with_changes']} diffs"
        out_lines.append(f"{pair_str:<45} | {rep['total_queries']:^6} | {rep['queries_with_changes']:^6} | {status}")

with open("docs/perf/evidence/b5_7.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))

print("Saved docs/perf/evidence/b5_7.txt successfully.")
