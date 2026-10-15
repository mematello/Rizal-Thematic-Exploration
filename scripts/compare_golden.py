"""
Golden Snapshot Comparison Tool

Compares a new search run against the golden snapshot baseline.
Reports:
- Position changes (result reordering)
- Score deltas (semantic, lexical, final)
- Theme label changes
- New/missing results

Usage:
    python scripts/compare_golden.py <new_snapshot.json> [<golden_snapshot.json>]

Or run benchmark in compare mode:
    python scripts/benchmark.py --compare
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "perf"
GOLDEN_PATH = OUTPUT_DIR / "golden_snapshot.json"


def load_snapshot(path: Path) -> Dict[str, Any]:
    """Load a snapshot JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_results(golden: List[Dict], new: List[Dict], book: str, query: str) -> Dict[str, Any]:
    """
    Compare two result lists and report differences.

    Returns:
        {
            "position_changes": [(id, old_pos, new_pos, delta)],
            "score_deltas": [(id, score_type, old_val, new_val, delta)],
            "theme_changes": [(id, old_themes, new_themes)],
            "missing_results": [id],
            "new_results": [id]
        }
    """

    # Build lookup maps
    golden_map = {r["id"]: (idx, r) for idx, r in enumerate(golden)}
    new_map = {r["id"]: (idx, r) for idx, r in enumerate(new)}

    position_changes = []
    score_deltas = []
    theme_changes = []

    # Check for position and score changes
    for result_id, (new_pos, new_result) in new_map.items():
        if result_id in golden_map:
            old_pos, old_result = golden_map[result_id]

            # Position change
            if old_pos != new_pos:
                delta = new_pos - old_pos
                position_changes.append({
                    "id": result_id,
                    "sentence_text": new_result.get("sentence_text", "")[:50],
                    "old_position": old_pos + 1,  # 1-indexed for readability
                    "new_position": new_pos + 1,
                    "delta": delta
                })

            # Score deltas
            old_scores = old_result.get("scores", {})
            new_scores = new_result.get("scores", {})

            for score_type in ["semantic", "lexical", "final", "precision"]:
                old_val = old_scores.get(score_type, 0)
                new_val = new_scores.get(score_type, 0)

                if abs(old_val - new_val) > 0.5:  # Threshold for reporting
                    score_deltas.append({
                        "id": result_id,
                        "score_type": score_type,
                        "old_value": old_val,
                        "new_value": new_val,
                        "delta": new_val - old_val
                    })

            # Theme changes
            old_themes = set(t["label"] for t in old_result.get("themes", []))
            new_themes = set(t["label"] for t in new_result.get("themes", []))

            if old_themes != new_themes:
                theme_changes.append({
                    "id": result_id,
                    "old_themes": sorted(list(old_themes)),
                    "new_themes": sorted(list(new_themes)),
                    "added": sorted(list(new_themes - old_themes)),
                    "removed": sorted(list(old_themes - new_themes))
                })

    # Check for missing and new results
    golden_ids = set(golden_map.keys())
    new_ids = set(new_map.keys())

    missing_results = [{"id": rid, "sentence_text": golden_map[rid][1].get("sentence_text", "")[:50]}
                       for rid in sorted(golden_ids - new_ids)]
    new_results = [{"id": rid, "sentence_text": new_map[rid][1].get("sentence_text", "")[:50]}
                   for rid in sorted(new_ids - golden_ids)]

    return {
        "position_changes": sorted(position_changes, key=lambda x: abs(x["delta"]), reverse=True),
        "score_deltas": sorted(score_deltas, key=lambda x: abs(x["delta"]), reverse=True),
        "theme_changes": theme_changes,
        "missing_results": missing_results,
        "new_results": new_results
    }


def compare_snapshots(golden_path: Path, new_path: Path) -> Dict[str, Any]:
    """
    Compare two full snapshots and generate a detailed diff report.
    """
    print(f"Loading golden snapshot: {golden_path}")
    golden_data = load_snapshot(golden_path)

    print(f"Loading new snapshot: {new_path}")
    new_data = load_snapshot(new_path)

    golden_queries = golden_data["queries"]
    new_queries = new_data["queries"]

    report = {
        "golden_file": str(golden_path),
        "new_file": str(new_path),
        "golden_version": golden_data.get("created", golden_data.get("version", "unknown")),
        "new_version": new_data.get("created", new_data.get("version", "unknown")),
        "total_queries": len(golden_queries),
        "queries_with_changes": 0,
        "total_position_changes": 0,
        "total_score_changes": 0,
        "total_theme_changes": 0,
        "query_diffs": {}
    }

    for query, golden_result in golden_queries.items():
        if query not in new_queries:
            print(f"Warning: Query '{query}' missing from new snapshot")
            continue

        new_result = new_queries[query]
        query_has_changes = False
        query_diff = {
            "noli": {},
            "elfili": {},
            "precise_matches": {}
        }

        # Compare noli results
        noli_diff = compare_results(
            golden_result.get("noli", []),
            new_result.get("noli", []),
            "noli",
            query
        )
        query_diff["noli"] = noli_diff

        # Compare elfili results
        elfili_diff = compare_results(
            golden_result.get("elfili", []),
            new_result.get("elfili", []),
            "elfili",
            query
        )
        query_diff["elfili"] = elfili_diff

        # Compare precise matches
        precise_diff = compare_results(
            golden_result.get("precise_matches", []),
            new_result.get("precise_matches", []),
            "precise",
            query
        )
        query_diff["precise_matches"] = precise_diff

        # Check if any changes occurred
        for book_diff in [noli_diff, elfili_diff, precise_diff]:
            if (book_diff["position_changes"] or
                book_diff["score_deltas"] or
                book_diff["theme_changes"] or
                book_diff["missing_results"] or
                book_diff["new_results"]):
                query_has_changes = True

                report["total_position_changes"] += len(book_diff["position_changes"])
                report["total_score_changes"] += len(book_diff["score_deltas"])
                report["total_theme_changes"] += len(book_diff["theme_changes"])

        if query_has_changes:
            report["queries_with_changes"] += 1
            report["query_diffs"][query] = query_diff

    return report


def print_report(report: Dict[str, Any], verbose: bool = False):
    """Print a human-readable comparison report"""
    print("\n" + "=" * 80)
    print("GOLDEN SNAPSHOT COMPARISON REPORT")
    print("=" * 80)

    if "golden_file" in report:
        print(f"\nGolden file:     {report['golden_file']}")
        print(f"New file:        {report['new_file']}")
    print(f"Golden baseline: {report['golden_version']}")
    print(f"New snapshot:    {report['new_version']}")

    print(f"\nTotal queries compared: {report['total_queries']}")
    print(f"Queries with changes:   {report['queries_with_changes']}")

    if report['queries_with_changes'] == 0:
        print("\n[PASS] No differences detected. Results are identical to golden baseline.")
        return

    print(f"\nAggregate Changes:")
    print(f"  Position changes: {report['total_position_changes']}")
    print(f"  Score deltas:     {report['total_score_changes']}")
    print(f"  Theme changes:    {report['total_theme_changes']}")

    # Detail each query with changes
    if verbose or report['queries_with_changes'] <= 10:
        print("\n" + "-" * 80)
        print("DETAILED QUERY DIFFERENCES")
        print("-" * 80)

        for query, diff in report["query_diffs"].items():
            print(f"\nQuery: '{query}'")

            for book in ["noli", "elfili", "precise_matches"]:
                book_diff = diff[book]

                if not any([book_diff["position_changes"],
                           book_diff["score_deltas"],
                           book_diff["theme_changes"],
                           book_diff["missing_results"],
                           book_diff["new_results"]]):
                    continue

                print(f"  {book.upper()}:")

                # Position changes
                if book_diff["position_changes"]:
                    print(f"    Position changes ({len(book_diff['position_changes'])}):")
                    for change in book_diff["position_changes"][:5]:  # Top 5
                        print(f"      ID {change['id']}: #{change['old_position']} -> #{change['new_position']} (delta{change['delta']:+d})")
                        print(f"        \"{change['sentence_text']}...\"")

                # Score deltas
                if book_diff["score_deltas"]:
                    print(f"    Score changes ({len(book_diff['score_deltas'])}):")
                    for change in book_diff["score_deltas"][:5]:  # Top 5
                        print(f"      ID {change['id']} {change['score_type']}: {change['old_value']} -> {change['new_value']} (delta{change['delta']:+.1f})")

                # Theme changes
                if book_diff["theme_changes"]:
                    print(f"    Theme changes ({len(book_diff['theme_changes'])}):")
                    for change in book_diff["theme_changes"][:3]:
                        if change["added"]:
                            print(f"      ID {change['id']}: Added {change['added']}")
                        if change["removed"]:
                            print(f"      ID {change['id']}: Removed {change['removed']}")

                # Missing/new results
                if book_diff["missing_results"]:
                    print(f"    Missing results: {len(book_diff['missing_results'])} results dropped")
                if book_diff["new_results"]:
                    print(f"    New results: {len(book_diff['new_results'])} results added")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/compare_golden.py <new_snapshot.json> [<golden_snapshot.json>]")
        print("\nCompares a new snapshot against the golden baseline.")
        sys.exit(1)

    new_snapshot_path = Path(sys.argv[1])
    golden_path = Path(sys.argv[2]) if len(sys.argv) > 2 else GOLDEN_PATH

    if not golden_path.exists():
        print(f"Error: Golden snapshot not found at {golden_path}")
        print("Run benchmark.py first to create the baseline.")
        sys.exit(1)

    if not new_snapshot_path.exists():
        print(f"Error: New snapshot not found at {new_snapshot_path}")
        sys.exit(1)

    # Run comparison
    report = compare_snapshots(golden_path, new_snapshot_path)

    # Print report
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    print_report(report, verbose)

    # Save detailed report
    report_path = OUTPUT_DIR / "comparison_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[PASS] Detailed report saved: {report_path}")

    # Exit code indicates if changes detected
    sys.exit(0 if report['queries_with_changes'] == 0 else 1)


if __name__ == "__main__":
    main()
