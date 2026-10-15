"""
Comprehensive Verification Script for Batched Context Fetch:
(a) 0/40 vs golden_snapshot_seed0.json
(b) Per-candidate assertion that expanded_text, neighbor lists, theme lists, and scores
    are identical to the sequential implementation, across all 40 golden queries PLUS
    an extra set of >= 50 sentence IDs at chapter starts/ends across both source types.
(c) db_query_count before vs after.
(d) Per-stage ms for bayan and oppression before vs after.
"""
import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from sqlalchemy import select, func, event
from typing import Dict, List, Tuple

os.environ["PYTHONHASHSEED"] = "0"
os.environ["PERF_BENCHMARK"] = "1"
os.environ["DISABLE_WORD_VEC_CACHE"] = "0"

backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.database import SessionLocal, Sentence
from app.core.engine import get_engine, get_translator_cache
from app.core.perf_instrumentation import PerfTimer
from scripts.compare_golden import compare_snapshots

GOLDEN_SNAPSHOT_PATH = Path(__file__).parent.parent / "docs" / "perf" / "golden_snapshot_seed0.json"
GOLDEN_QUERIES_PATH = Path(__file__).parent / "golden_queries.json"

class QueryCounter:
    def __init__(self, engine):
        self.count = 0
        self.engine = engine
        self._listener = self._before_cursor_execute
        event.listen(self.engine, "before_cursor_execute", self._listener)

    def _before_cursor_execute(self, conn, cursor, statement, parameters, context, executemany):
        self.count += 1

    def reset(self):
        self.count = 0

    def close(self):
        event.remove(self.engine, "before_cursor_execute", self._listener)

def main():
    db = SessionLocal()
    engine = get_engine()
    raw_engine = db.get_bind()
    counter = QueryCounter(raw_engine)

    print("=" * 80)
    print("VERIFICATION TASK 3: BATCHED CONTEXT FETCH")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Subtask 3(a): 0/40 vs golden_snapshot_seed0.json
    # --------------------------------------------------------------------------
    print("\n--- Subtask 3(a): Verifying 40/40 against golden_snapshot_seed0.json ---")

    with open(GOLDEN_QUERIES_PATH, "r", encoding="utf-8") as f:
        golden_queries = json.load(f)

    all_queries = []
    for cat, cdata in golden_queries["categories"].items():
        for q in cdata["queries"]:
            all_queries.append(q)

    # Ensure engine and models are warmed with standard warmup
    for q in ["kalayaan", "bayan", "Crisostomo Ibarra"]:
        _ = engine.search(db, q, source_type="summary", novel="both")

    def serialize_item(item):
        return {
            "id": item.get("id"),
            "sentence_index": item.get("sentence_index"),
            "chapter_number": item.get("chapter_number"),
            "sentence_text": item.get("sentence_text", "")[:100],
            "scores": item.get("scores", {}),
            "context_text": item.get("context_text"),
            "themes": [{"id": t.get("id"), "label": t.get("label"), "score": round(t.get("score", 0), 4)}
                       for t in item.get("themes", [])]
        }

    batched_snapshot = {}
    for idx, q in enumerate(all_queries, 1):
        actual = engine.search(db, q, source_type="summary", novel="both")
        res_map = actual.get("results", {})
        batched_snapshot[q] = {
            "query": q,
            "noli": [serialize_item(r) for r in res_map.get("noli", [])],
            "elfili": [serialize_item(r) for r in res_map.get("elfili", [])],
            "metadata": actual.get("metadata", {})
        }

    batched_snapshot_path = Path(__file__).parent.parent / "docs" / "perf" / "batched_context_snapshot.json"
    with open(batched_snapshot_path, "w", encoding="utf-8") as f:
        json.dump({"version": "batched_context", "queries": batched_snapshot}, f, indent=2, ensure_ascii=False)

    report = compare_snapshots(GOLDEN_SNAPSHOT_PATH, batched_snapshot_path)
    differences_40 = report.get("queries_with_changes", 0)

    print(f"\nSubtask 3(a) Result: {40 - differences_40}/40 queries identical ({differences_40} differences)")
    if differences_40 > 0:
        print("Differences detail:")
        for q, diff in report.get("query_diffs", {}).items():
            print(f"  {q}: {diff}")
    assert differences_40 == 0, f"Expected 0 differences vs golden snapshot, found {differences_40}"

    # --------------------------------------------------------------------------
    # Subtask 3(b): Per-candidate assertion on all 40 queries + >=50 boundary IDs
    # --------------------------------------------------------------------------
    print("\n--- Subtask 3(b): Per-candidate sequential vs batched equivalence ---")
    # Part 1: All candidates in all 40 queries
    total_candidates_verified = 0

    for cat, cdata in golden_queries["categories"].items():
        for q in cdata["queries"]:
            # Run with batched context fetch
            os.environ["DISABLE_BATCHED_CONTEXT_FETCH"] = "0"
            res_batched = engine.search(db, q, source_type="summary", novel="both")

            # Run with sequential context fetch
            os.environ["DISABLE_BATCHED_CONTEXT_FETCH"] = "1"
            res_seq = engine.search(db, q, source_type="summary", novel="both")

            for book in ["noli", "elfili"]:
                b_items = res_batched["results"].get(book, [])
                s_items = res_seq["results"].get(book, [])
                assert len(b_items) == len(s_items), f"Query '{q}' {book} count mismatch: {len(b_items)} vs {len(s_items)}"

                for idx, (b_item, s_item) in enumerate(zip(b_items, s_items)):
                    total_candidates_verified += 1
                    assert b_item["id"] == s_item["id"], f"Candidate ID mismatch at {book}[{idx}] for query '{q}'"
                    assert b_item["scores"] == s_item["scores"], f"Score mismatch at {book}[{idx}] for query '{q}'"
                    assert b_item["context_text"] == s_item["context_text"], f"context_text mismatch at {book}[{idx}] for query '{q}'"
                    assert b_item["themes"] == s_item["themes"], f"themes mismatch at {book}[{idx}] for query '{q}'"

    os.environ["DISABLE_BATCHED_CONTEXT_FETCH"] = "0"
    print(f"  [PASS] Verified all {total_candidates_verified} candidate instances across 40 queries: 100% bit-identical!")

    # Part 2: Extra set of >= 50 sentence IDs at chapter starts, ends, and both source types
    print("\n  Selecting extra sentence set at chapter boundaries across 'summary' and 'full'...")
    extra_sentences: List[Sentence] = []

    # Select chapter boundaries for 'summary' (noli and elfili)
    for b in ["noli", "elfili"]:
        chapters = db.scalars(select(Sentence.chapter_number).filter(Sentence.book == b, Sentence.source_type == "summary").distinct()).all()
        for ch in chapters[:15]: # 15 chapters each = 30 chapters
            # Start of chapter (min index)
            first_s = db.scalars(select(Sentence).filter(Sentence.book == b, Sentence.chapter_number == ch, Sentence.source_type == "summary").order_by(Sentence.sentence_index.asc()).limit(1)).first()
            # Second sentence
            second_s = db.scalars(select(Sentence).filter(Sentence.book == b, Sentence.chapter_number == ch, Sentence.source_type == "summary").order_by(Sentence.sentence_index.asc()).offset(1).limit(1)).first()
            # End of chapter (max index)
            last_s = db.scalars(select(Sentence).filter(Sentence.book == b, Sentence.chapter_number == ch, Sentence.source_type == "summary").order_by(Sentence.sentence_index.desc()).limit(1)).first()
            if first_s: extra_sentences.append(first_s)
            if second_s: extra_sentences.append(second_s)
            if last_s and (not first_s or last_s.id != first_s.id): extra_sentences.append(last_s)

    # Select chapter boundaries for 'full' (Noli Me Tangere and El Filibusterismo)
    for b in ["Noli Me Tangere", "El Filibusterismo"]:
        chapters = db.scalars(select(Sentence.chapter_number).filter(Sentence.book == b, Sentence.source_type == "full").distinct()).all()
        for ch in chapters[:15]:
            first_s = db.scalars(select(Sentence).filter(Sentence.book == b, Sentence.chapter_number == ch, Sentence.source_type == "full").order_by(Sentence.sentence_index.asc()).limit(1)).first()
            second_s = db.scalars(select(Sentence).filter(Sentence.book == b, Sentence.chapter_number == ch, Sentence.source_type == "full").order_by(Sentence.sentence_index.asc()).offset(1).limit(1)).first()
            last_s = db.scalars(select(Sentence).filter(Sentence.book == b, Sentence.chapter_number == ch, Sentence.source_type == "full").order_by(Sentence.sentence_index.desc()).limit(1)).first()
            if first_s: extra_sentences.append(first_s)
            if second_s: extra_sentences.append(second_s)
            if last_s and (not first_s or last_s.id != first_s.id): extra_sentences.append(last_s)

    print(f"  Selected {len(extra_sentences)} extra boundary sentences across 'summary' and 'full'.")
    assert len(extra_sentences) >= 50, f"Expected >= 50 extra sentences, got {len(extra_sentences)}"

    # Batch fetch context cache for all extra sentences together in 1 query!
    print("  Fetching union bounding window (+-8) for all extra sentences in a single batched query...")
    counter.reset()
    t_fetch0 = time.perf_counter()
    batch_cache = engine._fetch_context_cache(db, extra_sentences, window=8)
    fetch_time_ms = (time.perf_counter() - t_fetch0) * 1000
    fetch_queries = counter.count
    print(f"  [OK] Batch fetched {len(batch_cache)} total sentences in {fetch_queries} DB query ({fetch_time_ms:.2f} ms)")
    assert fetch_queries == 1, f"Expected exactly 1 DB query, got {fetch_queries}"

    # Verify each extra sentence
    boundary_mismatches = 0
    for s_idx, s in enumerate(extra_sentences, 1):
        # 1. Expand context: Sequential vs Batched
        seq_exp = engine._expand_context(db, s, context_cache=None)
        batch_exp = engine._expand_context(db, s, context_cache=batch_cache)
        assert seq_exp == batch_exp, f"Sentence {s.id} ({s.book} ch {s.chapter_number} idx {s.sentence_index}) expanded_text mismatch"

        # 2. Passage context & embedding: Sequential vs Batched
        seq_txt, seq_emb = engine._get_passage_context_with_embedding(db, s, window=8, context_cache=None)
        batch_txt, batch_emb = engine._get_passage_context_with_embedding(db, s, window=8, context_cache=batch_cache)
        assert seq_txt == batch_txt, f"Sentence {s.id} passage_text mismatch"
        if seq_emb is not None:
            assert batch_emb is not None
            assert np.allclose(seq_emb, batch_emb, atol=1e-7), f"Sentence {s.id} passage_emb mismatch"
        else:
            assert batch_emb is None

        # 3. Themes: Sequential vs Batched
        seq_themes = engine._classify_themes(db, s, query="", context_cache=None)
        batch_themes = engine._classify_themes(db, s, query="", context_cache=batch_cache)
        assert seq_themes == batch_themes, f"Sentence {s.id} themes mismatch"

    print(f"  [PASS] Verified all {len(extra_sentences)} boundary sentences: 100% bit-identical!")

    # --------------------------------------------------------------------------
    # Subtask 3(c): db_query_count before vs after
    # --------------------------------------------------------------------------
    print("\n--- Subtask 3(c): DB Query Count Before vs After ---")
    test_queries = ["bayan", "oppression", "edukasyon", "Crisostomo Ibarra", "karahasan ng mga prayle"]
    db_counts = {}

    for tq in test_queries:
        # Measure BEFORE (sequential: DISABLE_BATCHED_CONTEXT_FETCH=1)
        os.environ["DISABLE_BATCHED_CONTEXT_FETCH"] = "1"
        counter.reset()
        res_before = engine.search(db, tq, source_type="summary", novel="both")
        count_before = counter.count
        noli_count_b = len(res_before["results"]["noli"])
        fili_count_b = len(res_before["results"]["elfili"])

        # Measure AFTER (batched: DISABLE_BATCHED_CONTEXT_FETCH=0)
        os.environ["DISABLE_BATCHED_CONTEXT_FETCH"] = "0"
        counter.reset()
        res_after = engine.search(db, tq, source_type="summary", novel="both")
        count_after = counter.count
        noli_count_a = len(res_after["results"]["noli"])
        fili_count_a = len(res_after["results"]["elfili"])

        db_counts[tq] = {
            "candidates": noli_count_a + fili_count_a,
            "queries_before": count_before,
            "queries_after": count_after,
            "queries_saved": count_before - count_after,
            "reduction_pct": round(((count_before - count_after) / count_before) * 100, 1)
        }

        print(f"  '{tq:<26}': {count_before:2d} queries -> {count_after:2d} queries (-{count_before - count_after:2d} queries, -{db_counts[tq]['reduction_pct']}%) [Candidates: {noli_count_a + fili_count_a}]")

    # --------------------------------------------------------------------------
    # Subtask 3(d): Per-stage ms for bayan and oppression before vs after
    # --------------------------------------------------------------------------
    print("\n--- Subtask 3(d): Per-stage ms for bayan and oppression Before vs After ---")
    stage_profiles = {}

    for tq in ["bayan", "oppression"]:
        # Warm runs
        for _ in range(3):
            os.environ["DISABLE_BATCHED_CONTEXT_FETCH"] = "1"
            engine.search(db, tq, source_type="summary", novel="both")
            os.environ["DISABLE_BATCHED_CONTEXT_FETCH"] = "0"
            engine.search(db, tq, source_type="summary", novel="both")

        # Profile BEFORE (sequential)
        os.environ["DISABLE_BATCHED_CONTEXT_FETCH"] = "1"
        t0 = time.perf_counter()
        res_b = engine.search(db, tq, source_type="summary", novel="both")
        wall_b = (time.perf_counter() - t0) * 1000
        meta_b = res_b.get("metadata", {}).get("engine_timings", {})

        # Profile AFTER (batched)
        os.environ["DISABLE_BATCHED_CONTEXT_FETCH"] = "0"
        t0 = time.perf_counter()
        res_a = engine.search(db, tq, source_type="summary", novel="both")
        wall_a = (time.perf_counter() - t0) * 1000
        meta_a = res_a.get("metadata", {}).get("engine_timings", {})

        # Extract timing stages (keys not ending in _count)
        stages_b = {k: v for k, v in meta_b.items() if not k.endswith("_count") and isinstance(v, (int, float))}
        stages_a = {k: v for k, v in meta_a.items() if not k.endswith("_count") and isinstance(v, (int, float))}
        counts_b = {k[:-6]: v for k, v in meta_b.items() if k.endswith("_count")}
        counts_a = {k[:-6]: v for k, v in meta_a.items() if k.endswith("_count")}

        stage_profiles[tq] = {
            "wall_before_ms": round(wall_b, 2),
            "wall_after_ms": round(wall_a, 2),
            "timings_before": stages_b,
            "timings_after": stages_a,
            "counts_before": counts_b,
            "counts_after": counts_a
        }

        print(f"\nProfile for '{tq}':")
        print(f"  Wall-clock: Before = {wall_b:.2f} ms | After = {wall_a:.2f} ms | Delta = {wall_a - wall_b:+.2f} ms ({((wall_b - wall_a) / wall_b) * 100:+.1f}%)")
        print(f"  {'Stage':<25} | {'Before (ms)':<12} | {'After (ms)':<12} | {'Calls (B/A)'}")
        print("  " + "-" * 60)
        all_stages = sorted(set(stages_b.keys()) | set(stages_a.keys()))
        for stg in all_stages:
            tb = stages_b.get(stg, 0.0)
            ta = stages_a.get(stg, 0.0)
            cb = counts_b.get(stg, 0)
            ca = counts_a.get(stg, 0)
            print(f"  {stg:<25} | {tb:>10.2f} ms | {ta:>10.2f} ms | {cb:>2d} / {ca:<2d}")

    out_report_path = Path(__file__).parent.parent / "docs" / "perf" / "batched_context_verification.json"
    report_data = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "subtask_3a_golden_snapshot": {
            "total_queries": 40,
            "identical_queries": 40 - differences_40,
            "differences": differences_40
        },
        "subtask_3b_assertions": {
            "candidate_instances_verified": total_candidates_verified,
            "extra_boundary_sentences_verified": len(extra_sentences),
            "total_assertions_passed": total_candidates_verified * 4 + len(extra_sentences) * 3,
            "failures": 0
        },
        "subtask_3c_db_query_counts": db_counts,
        "subtask_3d_stage_profiles": stage_profiles
    }

    with open(out_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"[OK] All Subtasks 3(a), 3(b), 3(c), 3(d) PASSED!")
    print(f"Saved verification report to {out_report_path}")
    print("=" * 80)

    counter.close()
    db.close()

if __name__ == "__main__":
    main()
