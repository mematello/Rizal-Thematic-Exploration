"""
Verify that PerfTimer changes did not alter search ranking output.
Generates a snapshot for all 40 golden queries and compares against golden_snapshot_seed0.json.
"""
import os
import sys
import json
from pathlib import Path

# Enforce deterministic hash seed
os.environ["PYTHONHASHSEED"] = "0"
os.environ["PERF_BENCHMARK"] = "1"
os.environ["DISABLE_WORD_VEC_CACHE"] = "0"

backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.models.database import SessionLocal
from app.core.engine import get_engine
from compare_golden import compare_snapshots

GOLDEN_QUERIES_PATH = Path(__file__).parent / "golden_queries.json"
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "perf"
SEED0_SNAPSHOT_PATH = OUTPUT_DIR / "golden_snapshot_seed0.json"
TEST_SNAPSHOT_PATH = OUTPUT_DIR / "timer_fix_snapshot.json"

def serialize_result(item):
    return {
        "id": item.get("id"),
        "sentence_index": item.get("sentence_index"),
        "chapter_number": item.get("chapter_number"),
        "sentence_text": item.get("sentence_text", "")[:100],
        "scores": item.get("scores", {}),
        "themes": [{"id": t.get("id"), "label": t.get("label"), "score": round(t.get("score", 0), 4)}
                   for t in item.get("themes", [])]
    }

def main():
    with open(GOLDEN_QUERIES_PATH, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    db = SessionLocal()
    engine = get_engine()

    all_queries = []
    for cat, cdata in golden_data["categories"].items():
        for q in cdata["queries"]:
            all_queries.append(q)

    # Warmup
    print("Warming up models...")
    for q in ["kalayaan", "bayan", "Crisostomo Ibarra"]:
        _ = engine.search(db, q, source_type="summary", novel="both")

    snapshot = {}
    print(f"Running {len(all_queries)} queries for verification...")
    for idx, q in enumerate(all_queries, 1):
        res = engine.search(db, q, source_type="summary", novel="both")
        results = res.get("results", {})
        precise = res.get("precise_matches", [])
        snapshot[q] = {
            "query": q,
            "noli": [serialize_result(r) for r in results.get("noli", [])],
            "elfili": [serialize_result(r) for r in results.get("elfili", [])],
            "metadata": res.get("metadata", {})
        }

    db.close()

    with open(TEST_SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump({"version": "timer_fix_test", "queries": snapshot}, f, indent=2, ensure_ascii=False)
    print(f"Saved snapshot to {TEST_SNAPSHOT_PATH}")

    print("\nComparing against golden_snapshot_seed0.json...")
    report = compare_snapshots(SEED0_SNAPSHOT_PATH, TEST_SNAPSHOT_PATH)

    changes = report.get("queries_with_changes", 0)
    print("\n" + "=" * 60)
    print(f"Total queries with changes: {changes} / {len(all_queries)}")
    print("=" * 60)

    if changes == 0:
        print("[SUCCESS] 0 changes! Rankings and scores are 100% bit-identical.")
    else:
        print(f"[FAIL] {changes} queries changed:")
        for q, diff in report.get("diffs", {}).items():
            print(f"  {q}: {diff}")
        sys.exit(1)

if __name__ == "__main__":
    main()
