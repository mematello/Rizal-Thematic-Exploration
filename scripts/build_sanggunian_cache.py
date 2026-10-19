#!/usr/bin/env python3
"""
scripts/build_sanggunian_cache.py
Precomputes and bundles Sanggunian cross-reference alignment responses for Rizal Thematic Explorer.
Generates static JSON bundles for:
  - Noli Me Tangere (buod: 64 chapters, full: 63 chapters)
  - El Filibusterismo (buod: 39 chapters, full: 39 chapters)

Each bundle embeds the Supabase corpus fingerprint:
  - sentence_count
  - sentence_text_md5
  - dapt_null_count
  - dapt_column_md5
  - robust_aligner_hash (SHA-256 of robust_aligner.py)
  - content_alignment_hash (SHA-256 of content.py)
  - timestamp

Checkpointing & resume:
  - Saves per-chapter progress into backend/.cache_build_scratch/sanggunian_{book}_{mode}_scratch.json
  - Safe to interrupt and resume at any time.

Usage:
    poetry run python scripts/build_sanggunian_cache.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

# Set paths
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

DATA_DIR = BACKEND_DIR / "app" / "data"
SCRATCH_DIR = BACKEND_DIR / ".cache_build_scratch"
load_dotenv(BACKEND_DIR / ".env")

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from prod_guard import verify_not_production_db
verify_not_production_db(caller_name="build_sanggunian_cache.py")

from app.models.database import SessionLocal, Sentence
from app.core.engine import get_engine

NOVEL_CONFIGS = {
    "noli": {"name": "Noli Me Tangere", "buod_chapters": 64, "full_chapters": 63},
    "elfili": {"name": "El Filibusterismo", "buod_chapters": 39, "full_chapters": 39},
}
MODES = ["buod", "full"]


def get_corpus_fingerprint(db) -> dict:
    from sqlalchemy import text
    print("Computing Supabase corpus fingerprint for Sanggunian...")
    t0 = time.time()
    
    sql = text("""
        SELECT 
            (SELECT count(*) FROM sentences) AS sentence_count,
            (SELECT md5(string_agg(sentence_text, '' ORDER BY id)) FROM sentences) AS text_md5,
            (SELECT count(*) FROM sentences WHERE embedding_dapt IS NULL) AS null_dapt,
            (SELECT md5(string_agg(coalesce(embedding_dapt::text, 'NULL'), '' ORDER BY id)) FROM sentences) AS dapt_md5
    """)
    row = db.execute(sql).fetchone()
    duration = time.time() - t0

    # Calculate robust_aligner.py content hash
    aligner_file = BACKEND_DIR / "app" / "core" / "robust_aligner.py"
    with open(aligner_file, "rb") as f:
        robust_aligner_hash = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()

    # Calculate content.py content hash
    content_file = BACKEND_DIR / "app" / "api" / "v1" / "content.py"
    with open(content_file, "rb") as f:
        content_alignment_hash = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()

    fingerprint = {
        "sentence_count": int(row[0]),
        "sentence_text_md5": str(row[1]),
        "dapt_null_count": int(row[2]),
        "dapt_column_md5": str(row[3]),
        "robust_aligner_hash": robust_aligner_hash,
        "content_alignment_hash": content_alignment_hash,
        "dapt_model_name": "mematello/rizal-xlm-r-dapt",
        "dapt_model_revision": "main",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    
    print(f"Fingerprint computed in {duration:.2f}s:")
    print(f"  Sentences:         {fingerprint['sentence_count']} rows (MD5: {fingerprint['sentence_text_md5']})")
    print(f"  NULL DAPT Count:   {fingerprint['dapt_null_count']} rows (MD5: {fingerprint['dapt_column_md5']})")
    print(f"  robust_aligner.py: SHA-256: {fingerprint['robust_aligner_hash']}")
    print(f"  content.py:        SHA-256: {fingerprint['content_alignment_hash']}")
    return fingerprint


def compute_buod_chapter_sanggunian(book: str, ch_num: int, db, engine) -> dict:
    """
    Executes the exact unmodified live logic from backend/app/api/v1/content.py:860-948
    for an entire chapter in buod mode. Returns dict of { str(sentence_id): SanggunianResponse dict }.
    """
    search_book_noli = book.lower() in ('noli', 'noli me tangere')
    full_book = "Noli Me Tangere" if search_book_noli else "El Filibusterismo"
    search_chapter = ch_num
    if search_book_noli and search_chapter == 64:
        search_chapter = 63

    buod_rows = db.query(Sentence).filter(
        Sentence.book == book,
        Sentence.chapter_number == ch_num,
        Sentence.source_type == 'summary'
    ).order_by(Sentence.sentence_index).all()

    full_rows = db.query(Sentence).filter(
        Sentence.book == full_book,
        Sentence.chapter_number == search_chapter,
        Sentence.source_type == 'full'
    ).order_by(Sentence.sentence_index).all()

    if not buod_rows or not full_rows:
        return {}

    buod_texts = [r.sentence_text for r in buod_rows]
    full_texts = [r.sentence_text for r in full_rows]

    # Check for missing embeddings and encode if necessary
    if any(r.embedding_dapt is None for r in buod_rows):
        buod_embs = engine.dapt_model.encode(buod_texts)
    else:
        buod_embs = np.array([r.embedding_dapt for r in buod_rows], dtype=np.float32)

    if any(r.embedding_dapt is None for r in full_rows):
        full_embs = engine.dapt_model.encode(full_texts)
    else:
        full_embs = np.array([r.embedding_dapt for r in full_rows], dtype=np.float32)

    # Use RobustAligner
    alignments = engine.robust_aligner.align(
        buod_sentences=buod_texts,
        full_sentences=full_texts,
        buod_embeddings=buod_embs,
        full_embeddings=full_embs,
        full_is_short=[r.is_short for r in full_rows]
    )

    chapter_refs = {}
    for i, s in enumerate(buod_rows):
        if i >= len(alignments):
            chapter_refs[str(s.id)] = {"has_reference": False}
            continue

        match = alignments[i]
        w_start, w_end = match.best_window_start, match.best_window_end
        passage_sentences = full_rows[w_start:w_end+1]

        if not passage_sentences:
            chapter_refs[str(s.id)] = {"has_reference": False}
            continue

        ref_obj = {
            "has_reference": True,
            "passage_ids": [ps.id for ps in passage_sentences],
            "reference_text": " ".join(str(ps.sentence_text) for ps in passage_sentences),
            "alignment_status": "precise",
            "score": float(match.final_score),
            "semantic_score": float(match.semantic_score),
            "lexical_score": float(match.lexical_score),
            "char_score": float(match.tauhan_score),
            "ratio_score": float(match.position_score),
            "matched_characters": match.matched_characters or [],
            "buod_sentence_index": s.original_sentence_number or s.sentence_index,
            "full_sentence_indices": [ps.original_sentence_number or ps.sentence_index for ps in passage_sentences],
            "full_is_short": [bool(ps.is_short) for ps in passage_sentences]
        }
        chapter_refs[str(s.id)] = ref_obj

    return chapter_refs


def compute_full_chapter_sanggunian(book: str, ch_num: int, db) -> dict:
    """
    Executes the exact unmodified live logic from backend/app/api/v1/content.py:837-858
    for an entire chapter in full mode. Returns dict of { str(sentence_id): SanggunianResponse dict }.
    """
    search_book_noli = book.lower() in ('noli', 'noli me tangere')
    full_book = "Noli Me Tangere" if search_book_noli else "El Filibusterismo"

    full_rows = db.query(Sentence).filter(
        Sentence.book == full_book,
        Sentence.chapter_number == ch_num,
        Sentence.source_type == 'full'
    ).order_by(Sentence.sentence_index).all()

    if not full_rows:
        return {}

    # Group by passage_id
    passage_groups = {}
    for s in full_rows:
        if s.passage_id is not None:
            passage_groups.setdefault(s.passage_id, []).append(s)

    chapter_refs = {}
    for s in full_rows:
        if s.passage_id is not None and s.passage_id in passage_groups:
            passage_sentences = passage_groups[s.passage_id]
        else:
            passage_sentences = [s]

        ref_obj = {
            "has_reference": True,
            "passage_ids": [ps.id for ps in passage_sentences],
            "reference_text": " ".join(str(ps.sentence_text) for ps in passage_sentences),
            "alignment_status": "precise",
            "score": 1.0,
            "semantic_score": 0.0,
            "lexical_score": 0.0,
            "char_score": 0.0,
            "ratio_score": 0.0,
            "matched_characters": [],
            "buod_sentence_index": s.sentence_index,
            "full_sentence_indices": [ps.sentence_index for ps in passage_sentences],
            "full_is_short": []
        }
        chapter_refs[str(s.id)] = ref_obj

    return chapter_refs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Precompute Sanggunian cross-reference bundles.")
    parser.add_argument("--book", choices=["all", "noli", "elfili"], default="all", help="Target novel (default: all)")
    parser.add_argument("--mode", choices=["all", "buod", "full"], default="all", help="Target mode (default: all)")
    parser.add_argument("--allow-prod", action="store_true", help="Allow running against production Supabase")
    args = parser.parse_args()

    verify_not_production_db(allow_prod=args.allow_prod, caller_name="build_sanggunian_cache.py")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    engine = get_engine()

    try:
        fingerprint = get_corpus_fingerprint(db)

        books_to_process = [args.book] if args.book != "all" else list(NOVEL_CONFIGS.keys())
        modes_to_process = [args.mode] if args.mode != "all" else MODES

        for book in books_to_process:
            cfg = NOVEL_CONFIGS[book]
            for mode in modes_to_process:
                target_filename = f"sanggunian_cache_{book}_{mode}.json"
                target_path = DATA_DIR / target_filename
                scratch_path = SCRATCH_DIR / f"sanggunian_{book}_{mode}_scratch.json"

                total_chapters = cfg[f"{mode}_chapters"]
                chapter_list = list(range(1, total_chapters + 1))

                # Check if target bundle already exists and is complete
                if target_path.exists():
                    try:
                        with open(target_path, "r", encoding="utf-8") as f:
                            existing_bundle = json.load(f)
                        if len(existing_bundle.get("chapters", {})) == total_chapters:
                            size_kb = target_path.stat().st_size / 1024
                            print(f"\nReusing existing chapter alignments and updating fingerprint for {target_filename} ({size_kb:.1f} KB, {total_chapters} chapters)")
                            existing_bundle["fingerprint"] = fingerprint
                            with open(target_path, "w", encoding="utf-8") as f:
                                json.dump(existing_bundle, f, ensure_ascii=False)
                            continue
                    except Exception:
                        pass

                # Check scratch file for resume
                chapters_data = {}
                if scratch_path.exists():
                    try:
                        with open(scratch_path, "r", encoding="utf-8") as f:
                            chapters_data = json.load(f)
                        print(f"\nResuming {book} ({mode}) with {len(chapters_data)}/{total_chapters} cached chapters.")
                    except Exception:
                        chapters_data = {}

                print(f"\nProcessing {book.upper()} ({mode}) - {total_chapters} chapters...")
                t_start = time.time()

                for ch in chapter_list:
                    ch_key = str(ch)
                    if ch_key in chapters_data:
                        continue

                    t0 = time.time()
                    if mode == "buod":
                        refs = compute_buod_chapter_sanggunian(book, ch, db, engine)
                    else:
                        refs = compute_full_chapter_sanggunian(book, ch, db)
                    elapsed = time.time() - t0

                    chapters_data[ch_key] = {
                        "book": book,
                        "chapter_number": ch,
                        "mode": mode,
                        "total_sentences": len(refs),
                        "references": refs
                    }

                    print(f"  [{len(chapters_data)}/{total_chapters}] {book} ch {ch} ({mode}): {len(refs)} sentences, {elapsed:.2f}s")

                    # Save checkpoint every 2 chapters or on last chapter
                    if ch % 2 == 0 or ch == chapter_list[-1]:
                        with open(scratch_path, "w", encoding="utf-8") as f:
                            json.dump(chapters_data, f, ensure_ascii=False)

                # Assemble final bundle
                bundle = {
                    "fingerprint": fingerprint,
                    "book": book,
                    "mode": mode,
                    "total_chapters": total_chapters,
                    "chapters": chapters_data
                }

                temp_path = target_path.with_suffix(".tmp")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(bundle, f, ensure_ascii=False, separators=(',', ':'))
                temp_path.replace(target_path)

                size_kb = target_path.stat().st_size / 1024
                total_duration = time.time() - t_start
                print(f"SUCCESS: Saved {target_filename} ({size_kb:.1f} KB, {total_chapters} chapters in {total_duration:.1f}s)")

                # Clean up scratch
                if scratch_path.exists():
                    try:
                        scratch_path.unlink()
                    except Exception:
                        pass

    finally:
        db.close()
        print("\nAll Sanggunian bundles built successfully.")


if __name__ == "__main__":
    main()
