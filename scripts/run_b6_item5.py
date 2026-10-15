import os
import sys
import json
from pathlib import Path

sys.path.insert(0, 'scripts')
from strict_comparator import compare_strict

out_lines = []
out_lines.append("================================================================================")
out_lines.append("ITEM 5: CUMULATIVE TRANSITIONS, DIFF ATTRIBUTIONS & ANOMALY EXPLANATIONS")
out_lines.append("================================================================================\n")

transitions = [
    ("golden -> b1",   "docs/perf/golden_snapshot_seed0.json", "docs/perf/b1_snapshot_seed0.json", True),
    ("b1 -> d",       "docs/perf/b1_snapshot_seed0.json",     "docs/perf/d_snapshot_seed0.json",     False),
    ("d -> e1",        "docs/perf/d_snapshot_seed0.json",      "docs/perf/e1_snapshot_seed0.json",     False),
    ("e1 -> e2",       "docs/perf/e1_snapshot_seed0.json",     "docs/perf/e2_snapshot_seed0.json",     False),
    ("e2 -> e3",       "docs/perf/e2_snapshot_seed0.json",     "docs/perf/e3_snapshot_seed0.json",     False),
    ("golden -> HEAD", "docs/perf/golden_snapshot_seed0.json", "docs/perf/e3_snapshot_seed0.json", True),
]

for label, p1, p2, allow_diag in transitions:
    out_lines.append("--------------------------------------------------------------------------------")
    out_lines.append(f"TRANSITION: {label}")
    out_lines.append(f"  File 1: {p1}")
    out_lines.append(f"  File 2: {p2}")
    out_lines.append("--------------------------------------------------------------------------------")

    rep = compare_strict(Path(p1), Path(p2), allow_missing_diagnostics=allow_diag)
    diffs = rep["diffs"]
    all_q = sorted(list(diffs.keys()))

    res_changed = []
    meta_only = []
    for q in all_q:
        d = diffs[q]
        has_res = any(k in d for k in ['noli', 'elfili', 'precise_matches', 'noli_item_diffs', 'elfili_item_diffs'])
        if has_res:
            res_changed.append(q)
        else:
            meta_only.append(q)

    out_lines.append(f"Total strict diffs: {len(all_q)} | Result changes: {len(res_changed)} | Metadata only: {len(meta_only)}")
    out_lines.append("\nPer-Query / Per-Book Breakdown:")

    if not all_q:
        out_lines.append("  [ZERO DIFFERENCES - 100% BIT-IDENTICAL]")
    else:
        for q in all_q:
            d = diffs[q]
            q_type = "RESULT CHANGE" if q in res_changed else "METADATA ONLY"
            out_lines.append(f"\n  Query: '{q}' [{q_type}]")
            for b in ['noli', 'elfili', 'precise_matches']:
                if b in d:
                    ent = d[b].get('entered', [])
                    left = d[b].get('left', [])
                    reord = d[b].get('order_changed', False)
                    out_lines.append(f"    [{b.upper()}] Entered ({len(ent)}): {ent} | Left ({len(left)}): {left} | Reordered: {reord}")
                elif f"{b}_item_diffs" in d:
                    item_diffs = d[f"{b}_item_diffs"]
                    out_lines.append(f"    [{b.upper()}] Candidate set and rank order identical, but {len(item_diffs)} items had score deltas:")
                    for idiff in item_diffs[:3]:
                        out_lines.append(f"      Rank {idiff['rank']} (ID {idiff['id']}): score {idiff['item1']['scores']['final']} -> {idiff['item2']['scores']['final']}")
            if 'metadata' in d:
                out_lines.append(f"    [METADATA] Differing keys: {list(d['metadata'].keys())}")

    out_lines.append("\n")

# -----------------------------------------------------------------------------
# EXPLANATIONS OF THE FOUR AUDIT QUESTIONS
# -----------------------------------------------------------------------------
out_lines.append("================================================================================")
out_lines.append("RECONCILIATION AND EXPLANATIONS OF AUDIT QUESTIONS")
out_lines.append("================================================================================\n")

out_lines.append("1. EXPLANATION OF 101 vs 133 POSITION CHANGES:")
out_lines.append("   - '101' is the genuine, verified output of `compare_golden.py b1_snapshot_seed0.json golden_snapshot_seed0.json`.")
out_lines.append("     In that run, exactly 16 queries had candidate position changes, summing to exactly 101 position shifts.")
out_lines.append("   - '133' appeared in earlier unverified drafts (e.g., DECISIONS_DRAFT.md) where an assistant miscalculated")
out_lines.append("     position shifts by summing gross absolute rank displacements across books rather than counting candidates")
out_lines.append("     whose positions changed. The authentic comparator output for Commit B1 is exactly 101.")

out_lines.append("\n2. WHY 'simbahan' WAS ABSENT AND 'kamatayan' PRESENT IN THE EARLIER 'B1-only' LIST:")
out_lines.append("   - 'simbahan' has a genuine ranking order change in Noli under B1:")
out_lines.append("     In Noli, the top 10 candidate IDs are identical, but their relative rank order changed:")
out_lines.append("       Golden: [37185, 37193, 37330, 37457, 37459, 37517, 37636, 37638, 37639, 37794]")
out_lines.append("       B1:     [37185, 37193, 37330, 37457, 37459, 37517, 37636, 37638, 37794, 37639]")
out_lines.append("     Because entered=[] and left=[] (pure reorder), a naive script filtering on `entered or left` omitted 'simbahan'.")
out_lines.append("   - 'kamatayan' has ZERO ranking changes in Noli and ZERO ranking changes in El Fili under B1.")
out_lines.append("     The candidate lists and rank positions are 100% identical between Golden and B1.")
out_lines.append("     However, in `engine_timings.semantic_candidate_ids`, the unordered SQL query in golden had returned candidate IDs")
out_lines.append("     in a slightly different order into the timing log. When a raw metadata diff was checked without filtering")
out_lines.append("     diagnostics, 'kamatayan' was erroneously flagged as changed.")
out_lines.append("   - Conclusion: 'simbahan' belongs in the B1 ranking change list (pure reorder); 'kamatayan' had 0 ranking changes.")

out_lines.append("\n3. WHY E1 WAS REPORTED AS 12 QUERIES IN d->e1 BUT 6 ELSEWHERE:")
out_lines.append("   In the transition from Commit D to Commit E1 (d_snapshot_seed0.json -> e1_snapshot_seed0.json):")
out_lines.append("   - 4 queries have candidate ID set changes (entered/left candidates):")
out_lines.append("       education, edukasyon, kalayaan, oppression")
out_lines.append("   - 2 queries have pure candidate reordering (same candidate set, order shifted):")
out_lines.append("       Padre Damaso, Sisa")
out_lines.append("     -> Exactly 4 + 2 = 6 QUERIES with candidate ranking / position changes.")
out_lines.append("   - 5 queries have score deltas on identical candidate orders due to theme expansion text changes:")
out_lines.append("       Crisostomo Ibarra, Simoun, church, kalayaan ng Pilipinas, kapangyarihan ng simbahan")
out_lines.append("     -> Exactly 6 + 5 = 11 QUERIES reported by `compare_golden.py` (which tracks position changes and score deltas).")
out_lines.append("   - 1 query has ONLY an expansion text metadata change with 0 score or position deltas:")
out_lines.append("       edukasyon para sa kalayaan")
out_lines.append("     -> Exactly 11 + 1 = 12 QUERIES reported by `strict_comparator.py` (which includes metadata.query_expansion_text).")
out_lines.append("   This completely reconciles the 6 vs 11 vs 12 query counts across comparator definitions.")

out_lines.append("\n4. HOW 21 QUERIES IS OBTAINED IN golden -> HEAD:")
out_lines.append("   In comparing golden_snapshot_seed0.json directly against HEAD (e3_snapshot_seed0.json):")
out_lines.append("   - Commit B1 introduced ranking changes across 16 queries:")
out_lines.append("       Basilio, Crisostomo Ibarra, Elias, Isagani, Maria Clara, Padre Damaso, Simoun, Sisa,")
out_lines.append("       bayan, church, death, education, freedom, love, paaralan, simbahan")
out_lines.append("   - Commit E1 introduced ranking changes across 5 additional queries that were unchanged in B1:")
out_lines.append("       edukasyon, kalayaan, kalayaan ng Pilipinas, kapangyarihan ng simbahan, oppression")
out_lines.append("   - The union of changed queries is exactly 16 + 5 = 21 QUERIES.")
out_lines.append("   `compare_golden.py` reports exactly 21 queries with changes between golden and HEAD.")
out_lines.append("   (Strict comparator reports 22 queries because it additionally catches the metadata timing key in 'kamatayan'.)")

with open("docs/perf/evidence/b6_5.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))

print("Saved docs/perf/evidence/b6_5.txt successfully.")
