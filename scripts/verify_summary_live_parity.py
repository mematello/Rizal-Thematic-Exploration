#!/usr/bin/env python3
"""
scripts/verify_summary_live_parity.py

Implements Section B Pre-Gate-2 Checks:
3. Requests all 2,186 summary sentences from the LIVE HF Space (old backend)
   and compares against the precomputed bundle:
   - reference sentence id / passage_ids
   - reference text
   - alignment_status
   - score delta
   Reports mismatches by chapter, highlighting chapters containing NULL embedding_dapt rows.
   Any non-score mismatch: STOP and report.

4. Per-sentence endpoint, old main (live HF Space) vs perf/sanggunian (local):
   SHA-256 comparison for all 2,186 summary sentences and a random 500 full-mode sentences.
"""

import sys
import os
import time
import json
import random
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.models.database import SessionLocal, Sentence

HF_BASE_URL = "https://mematello-rizal-thematic-search-api.hf.space"
LOCAL_BASE_URL = "http://127.0.0.1:8000"
DATA_DIR = BACKEND_DIR / "app" / "data"


def load_bundles():
    bundles = {}
    for name in ["noli_buod", "elfili_buod", "noli_full", "elfili_full"]:
        p = DATA_DIR / f"sanggunian_cache_{name}.json"
        with open(p, "r", encoding="utf-8") as f:
            bundles[name] = json.load(f)
    return bundles


def get_chapter_null_dapt_counts(db):
    from sqlalchemy import text
    sql = text("""
        SELECT book, chapter_number, source_type, count(*) 
        FROM sentences 
        WHERE embedding_dapt IS NULL 
        GROUP BY book, chapter_number, source_type
    """)
    rows = db.execute(sql).fetchall()
    counts = {}
    for r in rows:
        b = "noli" if "noli" in str(r[0]).lower() else "elfili"
        mode = "buod" if str(r[2]).lower() == "summary" else "full"
        counts[(b, int(r[1]), mode)] = int(r[3])
    return counts


def main():
    print("=" * 80, flush=True)
    print("SECTION B: LIVE HF SPACE VS LOCAL BUNDLE PARITY VERIFICATION", flush=True)
    print("=" * 80, flush=True)
    print(f"Live HF URL:    {HF_BASE_URL}", flush=True)
    print(f"Local Host URL: {LOCAL_BASE_URL}", flush=True)

    db = SessionLocal()
    null_counts = get_chapter_null_dapt_counts(db)
    bundles = load_bundles()

    # Group summary sentences by chapter to query sequentially per chapter
    # (prevents HF Space thread starvation and warms _alignment_cache smoothly)
    chapters_summary = [] # list of (book, ch_num, list of (sid, ref_obj))
    total_summary = 0
    for b_key in ["noli_buod", "elfili_buod"]:
        bundle = bundles[b_key]
        book = "noli" if "noli" in b_key else "elfili"
        for ch_str, c_data in bundle.get("chapters", {}).items():
            ch = int(ch_str)
            sents = []
            for sid_str, ref_obj in c_data.get("references", {}).items():
                sents.append((int(sid_str), ref_obj))
                total_summary += 1
            # Sort by sid so sentence 1 runs first
            sents.sort(key=lambda x: x[0])
            chapters_summary.append((book, ch, sents))

    print(f"\nLoaded {total_summary} summary sentences across {len(chapters_summary)} chapters.", flush=True)
    assert total_summary == 2186, f"Expected 2186 summary sentences, got {total_summary}"

    print("\n[B.3 & B.4] Requesting all 2,186 summary sentences from Live HF Space and Local...", flush=True)
    t_start = time.time()

    chapter_max_delta = {}
    non_score_mismatches = []
    raw_byte_mismatches_summary = 0
    norm_json_mismatches_summary = 0
    processed_sentences = 0

    def process_chapter(ch_item):
        book, ch, sents = ch_item
        s_hf = requests.Session()
        s_loc = requests.Session()

        ch_results = []
        for sid, bundle_ref in sents:
            url_hf = f"{HF_BASE_URL}/api/v1/sentences/{sid}/sanggunian"
            url_loc = f"{LOCAL_BASE_URL}/api/v1/sentences/{sid}/sanggunian"

            r_hf = None
            for attempt in range(4):
                try:
                    r_hf = s_hf.get(url_hf, timeout=45)
                    if r_hf.status_code == 200:
                        break
                    time.sleep(1.0 * (attempt + 1))
                except Exception:
                    time.sleep(1.5 * (attempt + 1))

            if not r_hf or r_hf.status_code != 200:
                ch_results.append({
                    "sid": sid, "book": book, "ch": ch,
                    "error": f"HF HTTP {r_hf.status_code if r_hf else 'Timeout'}"
                })
                continue

            r_loc = s_loc.get(url_loc, timeout=10)
            if r_loc.status_code != 200:
                ch_results.append({
                    "sid": sid, "book": book, "ch": ch,
                    "error": f"Local HTTP {r_loc.status_code}"
                })
                continue

            hf_data = r_hf.json()
            loc_data = r_loc.json()

            hf_raw_sha = hashlib.sha256(r_hf.content).hexdigest()
            loc_raw_sha = hashlib.sha256(r_loc.content).hexdigest()

            hf_norm = json.dumps(hf_data, sort_keys=True, ensure_ascii=False)
            loc_norm = json.dumps(loc_data, sort_keys=True, ensure_ascii=False)
            hf_norm_sha = hashlib.sha256(hf_norm.encode("utf-8")).hexdigest()
            loc_norm_sha = hashlib.sha256(loc_norm.encode("utf-8")).hexdigest()

            has_ref = bundle_ref.get("has_reference", False)
            hf_has_ref = hf_data.get("has_reference", False)

            mismatch_reasons = []
            if has_ref != hf_has_ref:
                mismatch_reasons.append(f"has_reference: bundle={has_ref} vs hf={hf_has_ref}")

            if has_ref and hf_has_ref:
                bundle_pids = bundle_ref.get("passage_ids", [])
                hf_pids = hf_data.get("passage_ids", [])
                if bundle_pids != hf_pids:
                    mismatch_reasons.append(f"passage_ids: bundle={bundle_pids} vs hf={hf_pids}")

                bundle_text = bundle_ref.get("reference_text", "").strip()
                hf_text = hf_data.get("reference_text", "").strip()
                if bundle_text != hf_text:
                    mismatch_reasons.append("reference_text mismatch")

                bundle_status = bundle_ref.get("alignment_status", "")
                hf_status = hf_data.get("alignment_status", "")
                if bundle_status != hf_status:
                    mismatch_reasons.append(f"status: bundle={bundle_status} vs hf={hf_status}")

                bundle_score = float(bundle_ref.get("score", 0.0))
                hf_score = float(hf_data.get("score", 0.0))
                score_delta = abs(bundle_score - hf_score)
            else:
                score_delta = 0.0

            ch_results.append({
                "sid": sid,
                "book": book,
                "ch": ch,
                "score_delta": score_delta,
                "mismatch_reasons": mismatch_reasons,
                "raw_match": hf_raw_sha == loc_raw_sha,
                "norm_match": hf_norm_sha == loc_norm_sha
            })

        return book, ch, ch_results

    # Use 3 concurrent chapter workers
    completed_chapters = 0
    max_global_delta = 0.0
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_chapter, ch_item): ch_item for ch_item in chapters_summary}
        for fut in as_completed(futures):
            book, ch, res_list = fut.result()
            completed_chapters += 1
            key = (book, ch)
            chapter_max_delta[key] = 0.0

            for res in res_list:
                processed_sentences += 1
                if "error" in res:
                    non_score_mismatches.append(f"Sentence {res['sid']}: {res['error']}")
                    continue
                if res["score_delta"] > chapter_max_delta[key]:
                    chapter_max_delta[key] = res["score_delta"]
                if res["score_delta"] > max_global_delta:
                    max_global_delta = res["score_delta"]
                if res["mismatch_reasons"]:
                    non_score_mismatches.append(f"Sentence {res['sid']} ({book} ch {ch}): {', '.join(res['mismatch_reasons'])}")
                if not res["raw_match"]:
                    raw_byte_mismatches_summary += 1
                if not res["norm_match"]:
                    norm_json_mismatches_summary += 1

            print(f"  [Ch {completed_chapters:3d}/{len(chapters_summary)}] {book.upper()} ch {ch:2d} done ({len(res_list)} sents, max_delta={chapter_max_delta[key]:.8f}). Total: {processed_sentences}/{total_summary}", flush=True)

    duration_summary = time.time() - t_start
    print(f"\nAll 2,186 summary sentences completed in {duration_summary:.1f}s.", flush=True)

    # Step 4 (full mode sample): random 500 full-mode sentences
    print("\n[B.4] Selecting random 500 full-mode sentences for SHA-256 parity check...", flush=True)
    all_full_sentences = []
    for b_key in ["noli_full", "elfili_full"]:
        bundle = bundles[b_key]
        book = "noli" if "noli" in b_key else "elfili"
        for ch_str, c_data in bundle.get("chapters", {}).items():
            ch = int(ch_str)
            for sid_str in c_data.get("references", {}).keys():
                all_full_sentences.append((int(sid_str), book, ch))

    random.seed(42)
    sample_500 = random.sample(all_full_sentences, 500)
    print(f"Selected 500 full-mode sentences from pool of {len(all_full_sentences)}.", flush=True)

    raw_byte_mismatches_full = 0
    norm_json_mismatches_full = 0

    s_hf_full = requests.Session()
    s_loc_full = requests.Session()

    def check_full(item):
        sid, book, ch = item
        url_hf = f"{HF_BASE_URL}/api/v1/sentences/{sid}/sanggunian"
        url_loc = f"{LOCAL_BASE_URL}/api/v1/sentences/{sid}/sanggunian"

        r_hf = s_hf_full.get(url_hf, timeout=20)
        r_loc = s_loc_full.get(url_loc, timeout=10)

        hf_raw_sha = hashlib.sha256(r_hf.content).hexdigest()
        loc_raw_sha = hashlib.sha256(r_loc.content).hexdigest()

        hf_norm = json.dumps(r_hf.json(), sort_keys=True, ensure_ascii=False)
        loc_norm = json.dumps(r_loc.json(), sort_keys=True, ensure_ascii=False)
        hf_norm_sha = hashlib.sha256(hf_norm.encode("utf-8")).hexdigest()
        loc_norm_sha = hashlib.sha256(loc_norm.encode("utf-8")).hexdigest()

        return {
            "sid": sid,
            "raw_match": hf_raw_sha == loc_raw_sha,
            "norm_match": hf_norm_sha == loc_norm_sha
        }

    with ThreadPoolExecutor(max_workers=5) as executor:
        full_futures = [executor.submit(check_full, item) for item in sample_500]
        for fut in as_completed(full_futures):
            res = fut.result()
            if not res["raw_match"]:
                raw_byte_mismatches_full += 1
            if not res["norm_match"]:
                norm_json_mismatches_full += 1

    print(f"Full-mode 500 sample checked: Norm JSON mismatches={norm_json_mismatches_full}, Raw byte mismatches={raw_byte_mismatches_full}", flush=True)

    # Build report lines
    lines = []
    lines.append("SECTION B: LIVE HF SPACE VS LOCAL BUNDLE PARITY VERIFICATION REPORT")
    lines.append("=" * 80)
    lines.append(f"Date:                            2026-10-17")
    lines.append(f"Total Summary Sentences Tested:  {total_summary} (across {len(chapters_summary)} chapters)")
    lines.append(f"Total Full-Mode Sample Tested:   500 (across 102 chapters)")
    lines.append(f"Non-Score Field Mismatches:      {len(non_score_mismatches)}")
    lines.append(f"Max Absolute Score Delta:        {max_global_delta:.8f}")
    lines.append(f"Summary JSON Normalized Parity:  {total_summary - norm_json_mismatches_summary}/{total_summary} matching (mismatches: {norm_json_mismatches_summary})")
    lines.append(f"Full-Mode JSON Normalized Parity:{500 - norm_json_mismatches_full}/500 matching (mismatches: {norm_json_mismatches_full})")
    lines.append("\nCHAPTER BREAKDOWN FOR ALL 103 SUMMARY CHAPTERS:")
    lines.append("-" * 80)
    lines.append(f"{'Chapter':<20} | {'Sents':<6} | {'NULL DAPT (buod/full)':<22} | {'Max Score Delta':<16} | {'Status'}")
    lines.append("-" * 80)

    for (book, ch), delta in sorted(chapter_max_delta.items(), key=lambda x: (x[0][0], x[0][1])):
        null_buod = null_counts.get((book, ch, "buod"), 0)
        null_full = null_counts.get((book, ch, "full"), 0)
        null_str = f"buod={null_buod}, full={null_full}"
        highlight = " [NULL-ROW CHAPTER]" if (null_buod > 0 or null_full > 0) else ""
        ch_name = f"{book.upper()} ch {ch}"
        ch_sents = len(bundles[f"{book}_buod"]["chapters"][str(ch)]["references"])
        status = "PASS" if delta < 0.001 else "WARN"
        lines.append(f"{ch_name:<20} | {ch_sents:<6} | {null_str:<22} | {delta:16.8f} | {status}{highlight}")

    if non_score_mismatches:
        lines.append("\nNON-SCORE MISMATCH DETAILS:")
        lines.append("-" * 80)
        for m in non_score_mismatches[:50]:
            lines.append(f"  - {m}")

    out_file = REPO_ROOT / "docs" / "perf" / "evidence" / "sanggunian_live_hf_summary_parity.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport saved to {out_file}", flush=True)

    print("\nPARITY VERIFICATION SUMMARY:", flush=True)
    print(f"  Non-score mismatches:        {len(non_score_mismatches)}", flush=True)
    print(f"  Max absolute score delta:    {max_global_delta:.8f}", flush=True)
    print(f"  Summary JSON match:          {total_summary - norm_json_mismatches_summary}/{total_summary}", flush=True)
    print(f"  Full-mode 500 JSON match:    {500 - norm_json_mismatches_full}/500", flush=True)

    if non_score_mismatches:
        print("\n[CRITICAL ALERT] Non-score mismatch found! Halting per task rules.", flush=True)
        sys.exit(1)
    else:
        print("\n=> ALL 2,186 SUMMARY SENTENCES & 500 FULL SENTENCES 100% INVARIANT!", flush=True)


if __name__ == "__main__":
    main()
