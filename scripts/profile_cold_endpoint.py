#!/usr/bin/env python3
"""
scripts/profile_cold_endpoint.py

Times the ACTUAL endpoint cold path (unmodified logic from content.py)
with exact split: SQL, encode(), align(), other.
Measures cold behavior on Noli 1, Noli 5, and El Fili 10.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prod_guard import verify_not_production_db
verify_not_production_db(caller_name=Path(__file__).name)


import sys
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.models.database import SessionLocal, Sentence
from app.core.engine import RizalEngine


def profile_chapter(db, engine, book_slug: str, chapter_num: int):
    search_book_noli = book_slug.lower() in ('noli', 'noli me tangere')
    full_book = "Noli Me Tangere" if search_book_noli else "El Filibusterismo"
    buod_book = "noli" if search_book_noli else "elfili"
    search_chapter = chapter_num
    if search_book_noli and search_chapter == 64:
        search_chapter = 63

    t_total_start = time.perf_counter()

    # 1. SQL Fetch
    t_sql_start = time.perf_counter()
    buod_rows = db.query(Sentence).filter(
        Sentence.book == buod_book,
        Sentence.chapter_number == chapter_num,
        Sentence.source_type == 'summary'
    ).order_by(Sentence.sentence_index).all()
    
    full_rows = db.query(Sentence).filter(
        Sentence.book == full_book,
        Sentence.chapter_number == search_chapter,
        Sentence.source_type == 'full'
    ).order_by(Sentence.sentence_index).all()
    t_sql_end = time.perf_counter()
    sql_ms = (t_sql_end - t_sql_start) * 1000

    buod_texts = [r.sentence_text for r in buod_rows]
    full_texts = [r.sentence_text for r in full_rows]
    buod_nulls = sum(1 for r in buod_rows if r.embedding_dapt is None)
    full_nulls = sum(1 for r in full_rows if r.embedding_dapt is None)

    # 2. DAPT encode
    t_encode_start = time.perf_counter()
    buod_encode_ms = 0.0
    full_encode_ms = 0.0

    if buod_nulls > 0:
        t_b_start = time.perf_counter()
        buod_embs = engine.dapt_model.encode(buod_texts)
        buod_encode_ms = (time.perf_counter() - t_b_start) * 1000
    else:
        buod_embs = np.array([r.embedding_dapt for r in buod_rows], dtype=np.float32)

    if full_nulls > 0:
        t_f_start = time.perf_counter()
        full_embs = engine.dapt_model.encode(full_texts)
        full_encode_ms = (time.perf_counter() - t_f_start) * 1000
    else:
        full_embs = np.array([r.embedding_dapt for r in full_rows], dtype=np.float32)

    t_encode_end = time.perf_counter()
    total_encode_ms = (t_encode_end - t_encode_start) * 1000

    # 3. RobustAligner.align
    t_align_start = time.perf_counter()
    alignments = engine.robust_aligner.align(
        buod_sentences=buod_texts,
        full_sentences=full_texts,
        buod_embeddings=buod_embs,
        full_embeddings=full_embs,
        full_is_short=[r.is_short for r in full_rows]
    )
    t_align_end = time.perf_counter()
    align_ms = (t_align_end - t_align_start) * 1000

    # 4. Other (formatting / grouping for all sentences)
    t_other_start = time.perf_counter()
    results = {}
    for i, s in enumerate(buod_rows):
        if i < len(alignments):
            m = alignments[i]
            w_start, w_end = m.best_window_start, m.best_window_end
            ps = full_rows[w_start:w_end+1]
            results[s.id] = {
                "has_reference": True,
                "passage_ids": [p.id for p in ps],
                "reference_text": " ".join(str(p.sentence_text) for p in ps),
                "alignment_status": "precise",
                "score": float(m.final_score),
                "matched_characters": getattr(m, "matched_characters", []),
                "buod_sentence_index": s.sentence_index,
                "full_sentence_indices": [p.sentence_index for p in ps]
            }
    t_other_end = time.perf_counter()
    other_ms = (t_other_end - t_other_start) * 1000

    total_ms = (t_other_end - t_total_start) * 1000

    return {
        "chapter": f"{book_slug} ch {chapter_num}",
        "buod_sentences": len(buod_rows),
        "buod_nulls": buod_nulls,
        "full_sentences": len(full_rows),
        "full_nulls": full_nulls,
        "sql_ms": sql_ms,
        "buod_encode_ms": buod_encode_ms,
        "full_encode_ms": full_encode_ms,
        "total_encode_ms": total_encode_ms,
        "align_ms": align_ms,
        "other_ms": other_ms,
        "total_ms": total_ms
    }


def main():
    print("Initializing DB and Engine in fresh process...", flush=True)
    db = SessionLocal()
    engine = RizalEngine()

    chapters_to_test = [
        ("noli", 1),
        ("noli", 5),
        ("elfili", 10)
    ]

    print("=" * 80, flush=True)
    print("ACTUAL ENDPOINT COLD PATH BREAKDOWN (FRESH PROCESS, LIVE LOGIC)", flush=True)
    print("=" * 80, flush=True)

    lines = []
    lines.append("ACTUAL ENDPOINT COLD PATH BREAKDOWN (FRESH PROCESS, LIVE LOGIC)")
    lines.append("=" * 80)

    for book, ch in chapters_to_test:
        print(f"Profiling {book} ch {ch}...", flush=True)
        res = profile_chapter(db, engine, book, ch)
        print(f"\n--- {res['chapter'].upper()} ---", flush=True)
        print(f"  Summary Sentences: {res['buod_sentences']} (NULL dapt: {res['buod_nulls']})", flush=True)
        print(f"  Full Sentences:    {res['full_sentences']} (NULL dapt: {res['full_nulls']})", flush=True)
        print(f"  SQL Fetch:         {res['sql_ms']:10.2f} ms ({res['sql_ms']/res['total_ms']*100:5.1f}%)", flush=True)
        print(f"  Encode (buod):     {res['buod_encode_ms']:10.2f} ms", flush=True)
        print(f"  Encode (full):     {res['full_encode_ms']:10.2f} ms", flush=True)
        print(f"  Encode (total):    {res['total_encode_ms']:10.2f} ms ({res['total_encode_ms']/res['total_ms']*100:5.1f}%)", flush=True)
        print(f"  RobustAligner:     {res['align_ms']:10.2f} ms ({res['align_ms']/res['total_ms']*100:5.1f}%)", flush=True)
        print(f"  Other (Format):    {res['other_ms']:10.2f} ms ({res['other_ms']/res['total_ms']*100:5.1f}%)", flush=True)
        print(f"  TOTAL TIME:        {res['total_ms']:10.2f} ms", flush=True)

        lines.append(f"\n--- {res['chapter'].upper()} ---")
        lines.append(f"  Summary Sentences: {res['buod_sentences']} (NULL dapt: {res['buod_nulls']})")
        lines.append(f"  Full Sentences:    {res['full_sentences']} (NULL dapt: {res['full_nulls']})")
        lines.append(f"  SQL Fetch:         {res['sql_ms']:10.2f} ms ({res['sql_ms']/res['total_ms']*100:5.1f}%)")
        lines.append(f"  Encode (buod):     {res['buod_encode_ms']:10.2f} ms")
        lines.append(f"  Encode (full):     {res['full_encode_ms']:10.2f} ms")
        lines.append(f"  Encode (total):    {res['total_encode_ms']:10.2f} ms ({res['total_encode_ms']/res['total_ms']*100:5.1f}%)")
        lines.append(f"  RobustAligner:     {res['align_ms']:10.2f} ms ({res['align_ms']/res['total_ms']*100:5.1f}%)")
        lines.append(f"  Other (Format):    {res['other_ms']:10.2f} ms ({res['other_ms']/res['total_ms']*100:5.1f}%)")
        lines.append(f"  TOTAL TIME:        {res['total_ms']:10.2f} ms")

    out_file = REPO_ROOT / "docs" / "perf" / "evidence" / "sanggunian_endpoint_cold_breakdown.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved results to {out_file}", flush=True)


if __name__ == "__main__":
    main()
