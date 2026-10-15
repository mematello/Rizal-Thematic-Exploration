import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Set

TIMING_KEYS = {
    "engine_timings", "fastapi_layer", "db_query_count",
    "search_total_ms", "engine_stage_sum_ms", "untimed_gap_ms",
    "translator_ms", "translator_network_calls", "translator_network_call"
}

def clean_metadata(meta: Dict[str, Any], ignore_extra_diagnostic_keys: bool = False) -> Dict[str, Any]:
    cleaned = {}
    for k, v in meta.items():
        if k == "engine_timings" and isinstance(v, dict):
            # preserve semantic_candidate_ids, bridge_tokens, query_vector_sha256, query_expansion_text from engine_timings
            for tim_k in ["semantic_candidate_ids", "bridge_tokens", "query_vector_sha256", "query_expansion_text"]:
                if tim_k in v:
                    if ignore_extra_diagnostic_keys and tim_k in {"query_vector_sha256", "query_expansion_text", "bridge_tokens"}:
                        continue
                    cleaned[f"engine_timings_{tim_k}"] = v[tim_k]
            continue
        if k in TIMING_KEYS or k.endswith("_ms") or k.endswith("_count"):
            continue
        if ignore_extra_diagnostic_keys and k in {"query_vector_sha256", "query_expansion_text", "bridge_tokens", "translator_input", "translator_output"}:
            continue
        cleaned[k] = v
    return cleaned

def clean_result_item(item: Dict[str, Any]) -> Dict[str, Any]:
    # Retain all functional result fields
    res = {
        "id": item.get("id"),
        "sentence_index": item.get("sentence_index"),
        "chapter_number": item.get("chapter_number"),
        "sentence_text": item.get("sentence_text"),
        "scores": item.get("scores", {}),
        "context_text": item.get("context_text"),
        "themes": [{"id": t.get("id"), "label": t.get("label"), "score": t.get("score")}
                   for t in item.get("themes", [])]
    }
    # preserve optional rank fields if present
    for extra in ["rerank", "initial_rank"]:
        if extra in item:
            res[extra] = item[extra]
    return res

def canonical_query_obj(qdata: Dict[str, Any], ignore_extra_diagnostic_keys: bool = False) -> Dict[str, Any]:
    return {
        "noli": [clean_result_item(r) for r in qdata.get("noli", [])],
        "elfili": [clean_result_item(r) for r in qdata.get("elfili", [])],
        "precise_matches": [clean_result_item(r) for r in qdata.get("precise_matches", [])],
        "metadata": clean_metadata(qdata.get("metadata", {}), ignore_extra_diagnostic_keys=ignore_extra_diagnostic_keys)
    }

def compare_strict(snap1_path: Path, snap2_path: Path, allow_missing_diagnostics: bool = False) -> Dict[str, Any]:
    with open(snap1_path, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    with open(snap2_path, "r", encoding="utf-8") as f:
        data2 = json.load(f)

    queries1 = data1.get("queries", {})
    queries2 = data2.get("queries", {})

    diffs = {}
    all_queries = sorted(list(set(queries1.keys()) | set(queries2.keys())))

    for q in all_queries:
        if q not in queries1:
            diffs[q] = {"status": "missing_in_file1"}
            continue
        if q not in queries2:
            diffs[q] = {"status": "missing_in_file2"}
            continue

        # Check if one file has diagnostic keys and the other doesn't
        meta1 = queries1[q].get("metadata", {})
        meta2 = queries2[q].get("metadata", {})
        has_diag1 = "query_vector_sha256" in meta1
        has_diag2 = "query_vector_sha256" in meta2
        ignore_diag = allow_missing_diagnostics and (has_diag1 != has_diag2)

        c1 = canonical_query_obj(queries1[q], ignore_extra_diagnostic_keys=ignore_diag)
        c2 = canonical_query_obj(queries2[q], ignore_extra_diagnostic_keys=ignore_diag)

        s1 = json.dumps(c1, sort_keys=True, ensure_ascii=False)
        s2 = json.dumps(c2, sort_keys=True, ensure_ascii=False)

        if s1 != s2:
            q_diff = {}
            for book in ["noli", "elfili", "precise_matches"]:
                ids1 = [r["id"] for r in c1[book]]
                ids2 = [r["id"] for r in c2[book]]
                if ids1 != ids2:
                    entered = [i for i in ids2 if i not in ids1]
                    left = [i for i in ids1 if i not in ids2]
                    q_diff[book] = {
                        "ids1": ids1,
                        "ids2": ids2,
                        "entered": entered,
                        "left": left,
                        "order_changed": ids1 != ids2 and set(ids1) == set(ids2)
                    }
                else:
                    # check score or context or theme diffs
                    items_diff = []
                    for idx, (it1, it2) in enumerate(zip(c1[book], c2[book])):
                        it1_s = json.dumps(it1, sort_keys=True)
                        it2_s = json.dumps(it2, sort_keys=True)
                        if it1_s != it2_s:
                            items_diff.append({"id": it1["id"], "rank": idx+1, "item1": it1, "item2": it2})
                    if items_diff:
                        q_diff[f"{book}_item_diffs"] = items_diff

            m1_s = json.dumps(c1["metadata"], sort_keys=True)
            m2_s = json.dumps(c2["metadata"], sort_keys=True)
            if m1_s != m2_s:
                meta_diff = {}
                all_m_keys = sorted(list(set(c1["metadata"].keys()) | set(c2["metadata"].keys())))
                for mk in all_m_keys:
                    val1 = c1["metadata"].get(mk)
                    val2 = c2["metadata"].get(mk)
                    if val1 != val2:
                        meta_diff[mk] = {"val1": val1, "val2": val2}
                q_diff["metadata"] = meta_diff

            diffs[q] = q_diff

    return {
        "file1": str(snap1_path),
        "file2": str(snap2_path),
        "total_queries": len(all_queries),
        "queries_with_changes": len(diffs),
        "diffs": diffs
    }

def print_strict_report(report: Dict[str, Any]):
    print("=" * 80)
    print("STRICT CANONICAL COMPARISON REPORT")
    print("=" * 80)
    print(f"File 1: {report['file1']}")
    print(f"File 2: {report['file2']}")
    print(f"Total queries:        {report['total_queries']}")
    print(f"Queries with changes: {report['queries_with_changes']}")
    print("-" * 80)

    if report["queries_with_changes"] == 0:
        print("[SUCCESS] 0 differences! Canonical representations are 100% BIT-IDENTICAL.")
        return

    print(f"[FAIL] {report['queries_with_changes']} queries differed:")
    for q, d in report["diffs"].items():
        print(f"\nQuery: '{q}'")
        for k, v in d.items():
            if k in ["noli", "elfili", "precise_matches"]:
                ent = v.get("entered", [])
                left = v.get("left", [])
                ord_ch = v.get("order_changed", False)
                print(f"  [{k.upper()}] Entered: {ent} | Left: {left} | Reordered: {ord_ch}")
            elif "item_diffs" in k:
                print(f"  [{k}] {len(v)} items differed in scores/context/themes")
                for diff_item in v[:3]:
                    print(f"    ID {diff_item['id']} at rank {diff_item['rank']}")
            elif k == "metadata":
                print(f"  [METADATA] Differing keys: {list(v.keys())}")
                for mk, mv in v.items():
                    print(f"    {mk}: {mv['val1']} vs {mv['val2']}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/strict_comparator.py <file1.json> <file2.json> [--allow-missing-diagnostics]")
        sys.exit(1)
    f1 = Path(sys.argv[1])
    f2 = Path(sys.argv[2])
    allow_missing = "--allow-missing-diagnostics" in sys.argv
    rep = compare_strict(f1, f2, allow_missing_diagnostics=allow_missing)
    print_strict_report(rep)
    sys.exit(0 if rep["queries_with_changes"] == 0 else 1)
