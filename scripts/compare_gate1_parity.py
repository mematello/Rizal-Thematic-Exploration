#!/usr/bin/env python3
"""
Gate 1 Full Parity Verification Script:
Fetches all 205 chapter/mode combinations over HTTP sequentially from:
  - Port 8001 (main branch, live computation)
  - Port 8000 (perf/chapter-precompute, static bundle)

Writes docs/perf/evidence/chapter_parity_205.txt with:
  - Chapter, Mode, Book
  - Full 64-character SHA-256 for old and new
  - Byte counts
  - Header comparison (Content-Type, Cache-Control)
  - Match status
"""

import sys
import json
import time
import hashlib
import urllib.request
import urllib.error
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = REPO_ROOT / "docs" / "perf" / "evidence"
EVIDENCE_FILE = EVIDENCE_DIR / "chapter_parity_205.txt"

PORT_NEW = 8000  # perf/chapter-precompute
PORT_OLD = 8001  # main

NOVELS = ["noli", "elfili"]
MODES = ["buod", "full"]


def get_chapter_list(base_url, book, mode):
    url = f"{base_url}/api/v1/chapters?mode={mode}"
    req = urllib.request.Request(url, headers={"User-Agent": "ParityCheck/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    chs = []
    for item in data:
        item_book = item["book"].lower()
        if book == "noli" and "noli" in item_book:
            chs.append(item["chapter_number"])
        elif book == "elfili" and "fili" in item_book:
            chs.append(item["chapter_number"])
    return sorted(list(set(chs)))


def fetch_response(port, book, chapter, mode):
    url = f"http://127.0.0.1:{port}/api/v1/chapters/{book}/{chapter}?mode={mode}"
    req = urllib.request.Request(url, headers={"User-Agent": "ParityCheck/1.0"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read()
            elapsed = time.time() - t0
            headers = {
                "status": resp.status,
                "content-type": resp.headers.get("Content-Type", ""),
                "cache-control": resp.headers.get("Cache-Control", ""),
            }
            return body, headers, elapsed
    except urllib.error.HTTPError as e:
        elapsed = time.time() - t0
        body = e.read()
        headers = {
            "status": e.code,
            "content-type": e.headers.get("Content-Type", ""),
            "cache-control": e.headers.get("Cache-Control", ""),
        }
        return body, headers, elapsed


def run_parity():
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    print("Checking health on port 8000 and 8001...")
    for p in [PORT_NEW, PORT_OLD]:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{p}/health", timeout=5) as r:
                assert r.status == 200
        except Exception as e:
            print(f"Error: port {p} is not responding to /health: {e}")
            sys.exit(1)
    print("Both servers healthy.\n")

    # Discover all chapters
    all_tasks = []
    for book in NOVELS:
        for mode in MODES:
            chs = get_chapter_list(f"http://127.0.0.1:{PORT_NEW}", book, mode)
            for ch in chs:
                all_tasks.append((book, ch, mode))

    total = len(all_tasks)
    print(f"Total chapter/mode combinations to verify: {total}")

    # Check already verified tasks in evidence file for resume
    completed_keys = set()
    if EVIDENCE_FILE.exists():
        with open(EVIDENCE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" | ")
                if len(parts) >= 6 and not parts[0].startswith("Novel"):
                    completed_keys.add(parts[0])
        print(f"Resuming with {len(completed_keys)} already verified chapters.")
    else:
        # Write header
        with open(EVIDENCE_FILE, "w", encoding="utf-8") as f:
            f.write("# Chapter Precompute Full Parity Gate 1 (205 combinations)\n")
            f.write("# Generated: " + time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()) + "\n")
            f.write("# Port 8001 (main, live DB compute) vs Port 8000 (perf/chapter-precompute, static bundle)\n")
            f.write("-" * 180 + "\n")
            f.write(f"{'Novel/Chapter/Mode':24} | {'Old SHA-256 (port 8001)':64} | {'New SHA-256 (port 8000)':64} | {'Bytes':7} | {'Headers':7} | {'Match'}\n")
            f.write("-" * 180 + "\n")

    mismatches = []
    start_time = time.time()

    for idx, (book, ch, mode) in enumerate(all_tasks, start=1):
        task_key = f"{book.upper()} ch {ch} ({mode})"
        if task_key in completed_keys:
            continue

        # 1. Fetch old from port 8001 (live main)
        old_body, old_headers, old_time = fetch_response(PORT_OLD, book, ch, mode)
        old_sha = hashlib.sha256(old_body).hexdigest()

        # 2. Fetch new from port 8000 (precompute bundle)
        new_body, new_headers, new_time = fetch_response(PORT_NEW, book, ch, mode)
        new_sha = hashlib.sha256(new_body).hexdigest()

        # Compare headers
        headers_match = (
            old_headers.get("content-type") == new_headers.get("content-type") and
            old_headers.get("cache-control") == new_headers.get("cache-control") and
            old_headers.get("status") == new_headers.get("status")
        )
        bytes_match = (old_sha == new_sha and len(old_body) == len(new_body))

        match_str = "MATCH" if (bytes_match and headers_match) else "MISMATCH"

        line = f"{task_key:24} | {old_sha} | {new_sha} | {len(old_body):7} | {'MATCH' if headers_match else 'DIFF'} | {match_str}\n"

        with open(EVIDENCE_FILE, "a", encoding="utf-8") as f:
            f.write(line)

        print(f"[{idx:3}/{total}] {task_key:24} | Old: {old_time:5.2f}s | New: {new_time:5.3f}s | Bytes: {len(old_body):6} | {match_str}")

        if not bytes_match:
            print(f"!!! MISMATCH DETECTED ON {task_key} !!!")
            # Parse sentences and find exact difference
            try:
                old_json = json.loads(old_body.decode("utf-8"))
                new_json = json.loads(new_body.decode("utf-8"))
                print(f"Old sentence count: {len(old_json)}, New sentence count: {len(new_json)}")
                for s_idx, (s_old, s_new) in enumerate(zip(old_json, new_json)):
                    if s_old != s_new:
                        print(f"Sentence {s_idx} mismatch:")
                        print(f"  Old: {s_old}")
                        print(f"  New: {s_new}")
                        break
            except Exception as e:
                print(f"Could not parse JSON diff: {e}")
            mismatches.append(task_key)
            sys.exit(1)

        # Brief pause between live queries to avoid straining Supabase DB
        time.sleep(0.1)

    elapsed_total = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"GATE 1 PARITY COMPLETE IN {elapsed_total:.1f}s")
    print(f"Total Checked: {total} combinations")
    print(f"Mismatches:    {len(mismatches)}")
    print(f"Evidence file: {EVIDENCE_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    run_parity()
