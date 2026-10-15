import os
import sys
import json
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
docs_dir = base_dir / "docs" / "perf"
scripts_dir = base_dir / "scripts"

sys.path.insert(0, str(scripts_dir))
from strict_comparator import compare_strict

golden_path = docs_dir / "golden_snapshot_seed0.json"
b1_path = docs_dir / "b1_snapshot_seed0.json"
head_path = docs_dir / "proc1_unset_fwd.json"

with open(golden_path, "r", encoding="utf-8") as f:
    d_golden = json.load(f)
with open(b1_path, "r", encoding="utf-8") as f:
    d_b1 = json.load(f)
with open(head_path, "r", encoding="utf-8") as f:
    d_head = json.load(f)

# Old comparator simulation
def old_compare(snap1, snap2):
    q1 = snap1.get("queries", {})
    q2 = snap2.get("queries", {})
    diff_queries = []
    total_pos_changes = 0
    for q in q1:
        if q not in q2:
            diff_queries.append(q)
            continue
        has_diff = False
        for book in ["noli", "elfili"]:
            ids1 = [r["id"] for r in q1[q].get(book, [])]
            ids2 = [r["id"] for r in q2[q].get(book, [])]
            if ids1 != ids2:
                has_diff = True
                # count position changes
                for idx, i1 in enumerate(ids1):
                    if i1 in ids2:
                        idx2 = ids2.index(i1)
                        if idx != idx2:
                            total_pos_changes += 1
                    else:
                        total_pos_changes += 1
        if has_diff:
            diff_queries.append(q)
    return diff_queries, total_pos_changes

print("=" * 95)
print("CUMULATIVE BEHAVIOR DIFF: GOLDEN_SEED0 -> HEAD")
print("=" * 95)

# 1. Strict compare Golden -> HEAD
rep_gold_head = compare_strict(golden_path, head_path, allow_missing_diagnostics=True)
rep_gold_b1 = compare_strict(golden_path, b1_path, allow_missing_diagnostics=True)
rep_b1_head = compare_strict(b1_path, head_path, allow_missing_diagnostics=True)

old_diff_gh, old_pos_gh = old_compare(d_golden, d_head)
old_diff_gb1, old_pos_gb1 = old_compare(d_golden, d_b1)
old_diff_b1h, old_pos_b1h = old_compare(d_b1, d_head)

print(f"Old Comparator:    Golden -> HEAD : {len(old_diff_gh)} queries changed, {old_pos_gh} position changes")
print(f"Strict Comparator: Golden -> HEAD : {rep_gold_head['queries_with_changes']} queries changed")
print(f"Old Comparator:    Golden -> B1   : {len(old_diff_gb1)} queries changed, {old_pos_gb1} position changes")
print(f"Strict Comparator: Golden -> B1   : {rep_gold_b1['queries_with_changes']} queries changed")
print(f"Old Comparator:    B1 -> HEAD     : {len(old_diff_b1h)} queries changed, {old_pos_b1h} position changes")
print(f"Strict Comparator: B1 -> HEAD     : {rep_b1_head['queries_with_changes']} queries changed")

print("\n" + "=" * 95)
print("PER-QUERY AND PER-BOOK ATTRIBUTION TABLE (GOLDEN -> HEAD)")
print("=" * 95)
print(f"{'Query':<28} | {'Book':<7} | {'Attribution':<12} | {'Change Type':<22} | {'Entered IDs':<20} | {'Left IDs':<20}")
print("-" * 115)

all_changed_queries = sorted(list(rep_gold_head["diffs"].keys()))

for q in all_changed_queries:
    q_b1_diff = rep_gold_b1["diffs"].get(q, {})
    q_head_diff = rep_gold_head["diffs"].get(q, {})
    q_b1_to_head = rep_b1_head["diffs"].get(q, {})

    for book in ["noli", "elfili"]:
        b_diff = q_head_diff.get(book)
        if not b_diff:
            continue

        ent = b_diff.get("entered", [])
        lft = b_diff.get("left", [])
        order_only = b_diff.get("order_changed", False)
        ch_type = "Same-ID Reorder" if order_only else f"Set Change ({len(ent)} in/out)"

        # Attribution: did it change in B1, or from B1->HEAD (E1)?
        in_b1 = book in q_b1_diff
        in_e1 = book in q_b1_to_head

        if in_b1 and not in_e1:
            attr = "B1 (ORDER BY)"
        elif in_e1 and not in_b1:
            attr = "E1 (Theme Exp)"
        else:
            attr = "B1 + E1"

        print(f"{q:<28} | {book:<7} | {attr:<12} | {ch_type:<22} | {str(ent):<20} | {str(lft):<20}")
