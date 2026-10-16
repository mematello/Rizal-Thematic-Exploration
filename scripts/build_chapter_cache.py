#!/usr/bin/env python3
"""
Precomputes and bundles chapter content responses for Rizal Thematic Explorer.
Generates static JSON bundles for:
  - Noli Me Tangere (chapters 1-64, buod & full)
  - El Filibusterismo (chapters 1-39, buod & full)

Each bundle embeds the Supabase corpus fingerprint (sentence count, sentence text MD5,
theme count, theme content MD5, timestamp) so the backend can verify data integrity
on startup and fall back to live computation if the underlying database corpus changes.

Usage:
    poetry run python scripts/build_chapter_cache.py [--base-url http://127.0.0.1:8000]
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import urllib.error
from dotenv import load_dotenv

# Set paths
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
DATA_DIR = BACKEND_DIR / "app" / "data"
SCRATCH_DIR = BACKEND_DIR / ".cache_build_scratch"

# Load environment safely
load_dotenv(BACKEND_DIR / ".env")

NOVEL_CONFIGS = {
    "noli": {"name": "Noli Me Tangere"},
    "elfili": {"name": "El Filibusterismo"},
}
MODES = ["buod", "full"]


def get_available_chapters(base_url: str, book: str, mode: str) -> list:
    """
    Queries /api/v1/chapters?mode={mode} to get the exact list of available chapters
    for the novel, ensuring we never query non-existent chapters.
    """
    url = f"{base_url}/api/v1/chapters?mode={mode}"
    req = urllib.request.Request(url, headers={"User-Agent": "RizalEngine-CacheBuilder/1.0"})
    with urllib.request.urlopen(req, timeout=15) as response:
        data = json.loads(response.read().decode("utf-8"))
    
    # Book mapping in chapters response
    matching = []
    for item in data:
        item_book = item["book"].lower()
        if book == "noli" and ("noli" in item_book):
            matching.append(item["chapter_number"])
        elif book == "elfili" and ("fili" in item_book):
            matching.append(item["chapter_number"])
    return sorted(list(set(matching)))


def get_corpus_fingerprint() -> dict:
    """
    Connects to database using DATABASE_URL from .env and calculates
    deterministic MD5 fingerprints of sentences and themes.
    Credentials are NEVER printed or logged.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL environment variable is missing from .env!")

    from sqlalchemy import create_engine, text

    print("Computing Supabase corpus fingerprint...")
    t0 = time.time()
    engine = create_engine(db_url)
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT 
                (SELECT count(*) FROM sentences) AS sentence_count,
                (SELECT md5(string_agg(sentence_text, '' ORDER BY id)) FROM sentences) AS sentence_text_md5,
                (SELECT count(*) FROM themes) AS theme_count,
                (SELECT md5(string_agg(tagalog_title || ':' || coalesce(meaning, ''), '' ORDER BY id)) FROM themes) AS theme_content_md5
        """)).fetchone()

    duration = time.time() - t0
    fingerprint = {
        "sentence_count": int(row[0]),
        "sentence_text_md5": str(row[1]),
        "theme_count": int(row[2]),
        "theme_content_md5": str(row[3]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    print(f"Fingerprint computed in {duration:.2f}s:")
    print(f"  Sentences: {fingerprint['sentence_count']} rows (MD5: {fingerprint['sentence_text_md5']})")
    print(f"  Themes:    {fingerprint['theme_count']} rows (MD5: {fingerprint['theme_content_md5']})")
    return fingerprint


def fetch_chapter(base_url: str, book: str, chapter: int, mode: str) -> list:
    """
    Fetches chapter content from the running server.
    Returns parsed list of sentence dictionaries.
    """
    url = f"{base_url}/api/v1/chapters/{book}/{chapter}?mode={mode}"
    req = urllib.request.Request(url, headers={"User-Agent": "RizalEngine-CacheBuilder/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} when fetching {url}")
        content = response.read().decode("utf-8")
        return json.loads(content)


def build_cache(base_url: str):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Health check
    health_url = f"{base_url}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=5) as resp:
            if resp.status != 200:
                print(f"Server health check failed: HTTP {resp.status}")
                sys.exit(1)
    except Exception as e:
        print(f"Error: Unable to connect to server at {base_url}: {e}")
        print("Please ensure the FastAPI backend is running on the specified port.")
        sys.exit(1)

    # 2. Get database fingerprint
    fingerprint = get_corpus_fingerprint()

    # Pre-discover chapters for each novel and mode
    available_chapters_map = {}
    total_chapters = 0
    for book in NOVEL_CONFIGS:
        for mode in MODES:
            chs = get_available_chapters(base_url, book, mode)
            available_chapters_map[(book, mode)] = chs
            total_chapters += len(chs)

    print(f"Total chapters to process across all books and modes: {total_chapters}")
    for (b, m), chs in available_chapters_map.items():
        print(f"  {b:6} ({m:4}): {len(chs)} chapters (min={min(chs)}, max={max(chs)})")

    completed_chapters = 0
    start_total_time = time.time()
    generated_files = []

    # 3. Process each novel and mode
    for book in NOVEL_CONFIGS:
        for mode in MODES:
            target_filename = f"chapter_cache_{book}_{mode}.json"
            target_path = DATA_DIR / target_filename
            scratch_path = SCRATCH_DIR / f"{book}_{mode}_scratch.json"
            chapter_list = available_chapters_map[(book, mode)]
            max_ch = len(chapter_list)

            # If target bundle already exists and is complete, reuse it
            if target_path.exists():
                try:
                    with open(target_path, "r", encoding="utf-8") as f:
                        existing_bundle = json.load(f)
                    if len(existing_bundle.get("chapters", {})) == max_ch:
                        file_size_kb = target_path.stat().st_size / 1024
                        generated_files.append((target_filename, file_size_kb, max_ch, 0.0))
                        completed_chapters += max_ch
                        print(f"\nReusing existing complete {target_filename} ({file_size_kb:.1f} KB, {max_ch} chapters)")
                        continue
                except Exception:
                    pass

            # Check if scratch file exists for resume
            chapters_data = {}
            if scratch_path.exists():
                try:
                    with open(scratch_path, "r", encoding="utf-8") as f:
                        chapters_data = json.load(f)
                    print(f"\nResuming {book} ({mode}) with {len(chapters_data)}/{max_ch} cached chapters.")
                except Exception:
                    chapters_data = {}

            print(f"\nProcessing {book.upper()} ({mode}) - {max_ch} chapters...")
            book_mode_start = time.time()

            for ch in chapter_list:
                ch_key = str(ch)
                if ch_key in chapters_data:
                    completed_chapters += 1
                    continue

                t0 = time.time()
                retries = 3
                for attempt in range(retries):
                    try:
                        data = fetch_chapter(base_url, book, ch, mode)
                        elapsed = time.time() - t0
                        chapters_data[ch_key] = data
                        completed_chapters += 1
                        print(f"  [{completed_chapters}/{total_chapters}] {book} ch {ch}/{chapter_list[-1]} ({mode}): {len(data)} sentences, {elapsed:.2f}s")
                        break
                    except Exception as e:
                        if attempt == retries - 1:
                            print(f"  ERROR fetching {book} ch {ch} ({mode}): {e}")
                            raise
                        print(f"  Retry {attempt+1}/{retries} for {book} ch {ch} ({mode}) after error: {e}")
                        time.sleep(2)

                # Save scratch checkpoint every 5 chapters
                if ch % 5 == 0 or ch == chapter_list[-1]:
                    with open(scratch_path, "w", encoding="utf-8") as f:
                        json.dump(chapters_data, f, ensure_ascii=False)

            # Build final bundle
            bundle = {
                "fingerprint": fingerprint,
                "book": book,
                "mode": mode,
                "total_chapters": max_ch,
                "chapters": chapters_data,
            }

            # Write formatted bundle
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(bundle, f, separators=(",", ":"), ensure_ascii=False)

            file_size_kb = target_path.stat().st_size / 1024
            duration_s = time.time() - book_mode_start
            generated_files.append((target_filename, file_size_kb, len(chapters_data), duration_s))
            print(f"Saved {target_filename} ({file_size_kb:.1f} KB, {len(chapters_data)} chapters) in {duration_s:.1f}s")

            # Clean scratch file after successful bundle creation
            if scratch_path.exists():
                scratch_path.unlink()

    # Clean scratch dir
    try:
        SCRATCH_DIR.rmdir()
    except Exception:
        pass

    # Summary report
    total_elapsed = time.time() - start_total_time
    total_kb = sum(size for _, size, _, _ in generated_files)
    print("\n" + "=" * 60)
    print("CHAPTER CACHE BUILD COMPLETE")
    print("=" * 60)
    for filename, size_kb, ch_count, dur in generated_files:
        print(f"  - {filename:32} {ch_count:2} chapters | {size_kb:8.1f} KB | {dur:6.1f}s")
    print(f"Total Size:  {total_kb / 1024:.2f} MB ({total_kb:.1f} KB)")
    print(f"Total Time:  {total_elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute chapter static cache bundles.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI backend base URL")
    args = parser.parse_args()
    build_cache(args.base_url)
