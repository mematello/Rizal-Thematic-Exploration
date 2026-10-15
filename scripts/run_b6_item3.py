import sys
import os
import json
from pathlib import Path

base_dir = Path(".").resolve()
sys.path.insert(0, str(base_dir / "scripts"))

from strict_comparator import compare_strict

out_lines = []
out_lines.append("================================================================================")
out_lines.append("ITEM 3: COMPARATOR AUDIT, DIAGNOSTIC LOCATION & FULL MATRIX RE-RUNS")
out_lines.append("================================================================================\n")

# 1. Location of query_vector_sha256 and query_expansion_text
out_lines.append("--------------------------------------------------------------------------------")
out_lines.append("1. LOCATION OF query_vector_sha256 & query_expansion_text IN CANONICAL JSON")
out_lines.append("--------------------------------------------------------------------------------")

with open("docs/perf/proc1_unset_fwd.json", "r", encoding="utf-8") as f:
    snap_data = json.load(f)

for q in ["kalayaan", "love"]:
    qdata = snap_data["queries"][q]
    meta = qdata.get("metadata", {})
    tim = meta.get("engine_timings", {})
    out_lines.append(f"Query: '{q}'")
    out_lines.append(f"  metadata.query_vector_sha256:          {meta.get('query_vector_sha256')}")
    out_lines.append(f"  engine_timings.query_vector_sha256:   {tim.get('query_vector_sha256')}")
    out_lines.append(f"  metadata.query_expansion_text:        '{meta.get('query_expansion_text')}'")
    out_lines.append(f"  engine_timings.query_expansion_text: '{tim.get('query_expansion_text')}'")
    out_lines.append(f"  metadata.bridge_tokens:                {meta.get('bridge_tokens')}")
    out_lines.append(f"  engine_timings.bridge_tokens:         {tim.get('bridge_tokens')}")

out_lines.append("\nBoth metadata and engine_timings hold these diagnostic keys. The strict comparator has been updated")
out_lines.append("to include them explicitly under engine_timings_<key> as well as top-level metadata.")

# (a) golden vs b1
out_lines.append("\n--------------------------------------------------------------------------------")
out_lines.append("2. RERUN (a): golden_snapshot_seed0.json vs b1_snapshot_seed0.json")
out_lines.append("--------------------------------------------------------------------------------")
rep_a = compare_strict(
    Path("docs/perf/golden_snapshot_seed0.json"),
    Path("docs/perf/b1_snapshot_seed0.json"),
    allow_missing_diagnostics=True
)
out_lines.append(f"Total queries: {rep_a['total_queries']} | Queries with changes: {rep_a['queries_with_changes']}")
b1_names = sorted(list(rep_a["diffs"].keys()))
out_lines.append(f"Changed queries ({len(b1_names)}):")
for idx, qn in enumerate(b1_names, 1):
    out_lines.append(f"  {idx:2d}. {qn}")

# (b) D+B1 seeds 1 vs 2
out_lines.append("\n--------------------------------------------------------------------------------")
out_lines.append("3. RERUN (b): NEGATIVE CONTROL D+B1 (S0) SEED 1 vs SEED 2")
out_lines.append("--------------------------------------------------------------------------------")
rep_b = compare_strict(
    Path("docs/perf/ablation_S0_seed_1.json"),
    Path("docs/perf/ablation_S0_seed_2.json")
)
out_lines.append(f"Total queries: {rep_b['total_queries']} | Queries with changes: {rep_b['queries_with_changes']}")
s0_names = sorted(list(rep_b["diffs"].keys()))
out_lines.append(f"Divergent queries ({len(s0_names)}):")
for idx, qn in enumerate(s0_names, 1):
    out_lines.append(f"  {idx:2d}. {qn}")

# (c) S1 vs S2 vs HEAD seeds on 40 queries
out_lines.append("\n--------------------------------------------------------------------------------")
out_lines.append("4. RERUN (c): CROSS-SEED COMPARISONS ON S1, S2, AND HEAD (40 QUERIES)")
out_lines.append("--------------------------------------------------------------------------------")

states = {
    "S1 (E1 only)": [
        ("S1 seed 1 vs S1 seed 2", "docs/perf/ablation_S1_seed_1.json", "docs/perf/ablation_S1_seed_2.json"),
        ("S1 seed 1 vs S1 seed 12345", "docs/perf/ablation_S1_seed_1.json", "docs/perf/ablation_S1_seed_12345.json"),
        ("S1 unset vs S1 seed 1", "docs/perf/ablation_S1_seed_unset.json", "docs/perf/ablation_S1_seed_1.json"),
    ],
    "S2 (E1+E2)": [
        ("S2 seed 1 vs S2 seed 2", "docs/perf/ablation_S2_seed_1.json", "docs/perf/ablation_S2_seed_2.json"),
        ("S2 seed 1 vs S2 seed 12345", "docs/perf/ablation_S2_seed_1.json", "docs/perf/ablation_S2_seed_12345.json"),
        ("S2 unset vs S2 seed 1", "docs/perf/ablation_S2_seed_unset.json", "docs/perf/ablation_S2_seed_1.json"),
    ],
    "HEAD (E1+E2+E3)": [
        ("HEAD seed 1 vs HEAD seed 2", "docs/perf/proc3_seed1.json", "docs/perf/proc4_seed2.json"),
        ("HEAD seed 1 vs HEAD seed 12345", "docs/perf/proc3_seed1.json", "docs/perf/proc5_seed12345.json"),
        ("HEAD unset vs HEAD seed 1", "docs/perf/proc1_unset_fwd.json", "docs/perf/proc3_seed1.json"),
    ]
}

for st_name, pairs in states.items():
    out_lines.append(f"\n--- State: {st_name} ---")
    for label, p1, p2 in pairs:
        rep = compare_strict(Path(p1), Path(p2))
        out_lines.append(f"{label:<35} | Total: {rep['total_queries']} | Diffs: {rep['queries_with_changes']}")
        if rep["queries_with_changes"] > 0:
            for qn in sorted(list(rep["diffs"].keys())):
                out_lines.append(f"    - {qn}")

# (d) The six-process matrix
out_lines.append("\n--------------------------------------------------------------------------------")
out_lines.append("5. RERUN (d): SIX-PROCESS PAIRWISE MATRIX (HEAD)")
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
        status = "100% BIT-IDENTICAL" if rep["queries_with_changes"] == 0 else f"{rep['queries_with_changes']} diffs"
        out_lines.append(f"{pair_str:<45} | {rep['total_queries']:^6} | {rep['queries_with_changes']:^6} | {status}")

# (e) The 206-query matrices
out_lines.append("\n--------------------------------------------------------------------------------")
out_lines.append("6. RERUN (e): 206-QUERY HELDOUT MATRICES (HEAD & S1)")
out_lines.append("--------------------------------------------------------------------------------")

heldout_pairs = [
    ("HEAD unset fwd vs HEAD unset rev", "docs/perf/heldout_head_unset_fwd.json", "docs/perf/heldout_head_unset_rev.json"),
    ("HEAD unset fwd vs HEAD seed 1",   "docs/perf/heldout_head_unset_fwd.json", "docs/perf/heldout_head_seed1.json"),
    ("HEAD unset fwd vs HEAD seed 12345", "docs/perf/heldout_head_unset_fwd.json", "docs/perf/heldout_head_seed12345.json"),
    ("HEAD seed 1 vs HEAD seed 12345",  "docs/perf/heldout_head_seed1.json", "docs/perf/heldout_head_seed12345.json"),
    ("S1 unset fwd vs S1 unset rev",     "docs/perf/heldout_s1_unset_fwd.json", "docs/perf/heldout_s1_unset_rev.json"),
    ("S1 unset fwd vs S1 seed 1",       "docs/perf/heldout_s1_unset_fwd.json", "docs/perf/heldout_s1_seed1.json"),
    ("S1 unset fwd vs S1 seed 12345",   "docs/perf/heldout_s1_unset_fwd.json", "docs/perf/heldout_s1_seed12345.json"),
    ("S1 seed 1 vs S1 seed 12345",      "docs/perf/heldout_s1_seed1.json", "docs/perf/heldout_s1_seed12345.json"),
    ("S0 seed 1 vs S0 seed 2 (neg ctrl)", "docs/perf/s0_heldout_seed1.json", "docs/perf/s0_heldout_seed2.json")
]

h_header = f"{'Held-Out Comparison Pair':<45} | {'Total':<6} | {'Diffs':<6} | {'Status'}"
out_lines.append(h_header)
out_lines.append("-" * len(h_header))
for label, p1, p2 in heldout_pairs:
    rep = compare_strict(Path(p1), Path(p2))
    status = "100% BIT-IDENTICAL" if rep["queries_with_changes"] == 0 else f"{rep['queries_with_changes']} diffs"
    out_lines.append(f"{label:<45} | {rep['total_queries']:^6} | {rep['queries_with_changes']:^6} | {status}")
    if rep["queries_with_changes"] > 0:
        for qn in sorted(list(rep["diffs"].keys()))[:10]:
            out_lines.append(f"    - {qn}")
        if rep["queries_with_changes"] > 10:
            out_lines.append(f"    ... ({rep['queries_with_changes'] - 10} more)")

with open("docs/perf/evidence/b6_3.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))

print("Saved docs/perf/evidence/b6_3.txt successfully.")
