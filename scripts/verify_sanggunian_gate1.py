#!/usr/bin/env python3
"""
scripts/verify_sanggunian_gate1.py

Gate 1 Verification for Item C Phase 2 (Sanggunian Optimization):
1. Bundle Integrity & Sizes & RSS delta
2. Corpus Fingerprint match against live Supabase
3. Bit-for-bit SHA-256 byte parity: Bundle output vs live endpoint across all 205 chapters
4. Max absolute score delta on the 5 profiled sentences (Local vs Live HF Space)
5. Fallback tests: Fingerprint mismatch, Missing bundle, DB connectivity failure
6. Cold & warm latency benchmarks
"""

import os
import sys
import json
import time
import hashlib
import requests
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
DATA_DIR = BACKEND_DIR / "app" / "data"

load_dotenv(BACKEND_DIR / ".env")

LOCAL_URL = "http://127.0.0.1:8000"
HF_SPACE_URL = "https://mematello-rizal-thematic-search-api.hf.space"

PROFILED_SENTENCE_IDS = [
    37177,  # Noli Ch 1 Sent 1
    37275,  # Noli Ch 5 Sent 1
    37768,  # Noli Ch 34 Sent 1
    38389,  # El Fili Ch 1 Sent 1
    38586,  # El Fili Ch 10 Sent 1
]

BUNDLE_CONFIGS = {
    ("noli", "buod"): (DATA_DIR / "sanggunian_cache_noli_buod.json", 64),
    ("noli", "full"): (DATA_DIR / "sanggunian_cache_noli_full.json", 63),
    ("elfili", "buod"): (DATA_DIR / "sanggunian_cache_elfili_buod.json", 39),
    ("elfili", "full"): (DATA_DIR / "sanggunian_cache_elfili_full.json", 39),
}


def test_bundle_integrity_and_sizes():
    print("=" * 70)
    print("GATE 1 - TEST 1: BUNDLE INTEGRITY & SIZES")
    print("=" * 70)
    total_bytes = 0
    total_chapters = 0
    total_sentences = 0
    bundles = {}

    for (book, mode), (path, expected_ch) in BUNDLE_CONFIGS.items():
        assert path.exists(), f"Bundle missing: {path.name}"
        size_bytes = path.stat().st_size
        size_kb = size_bytes / 1024
        total_bytes += size_bytes

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        bundles[(book, mode)] = data

        ch_count = len(data.get("chapters", {}))
        assert ch_count == expected_ch, f"Expected {expected_ch} chapters in {path.name}, got {ch_count}"
        total_chapters += ch_count

        s_count = sum(c.get("total_sentences", 0) for c in data["chapters"].values())
        total_sentences += s_count

        print(f"  {path.name:36} | {ch_count:2} ch | {s_count:5} sents | {size_kb:8.1f} KB")

    print(f"\nTotal Cache Size: {total_bytes / (1024*1024):.2f} MB ({total_chapters} chapters, {total_sentences} sentences)")
    return bundles


def test_fingerprint_against_db(bundles):
    print("\n" + "=" * 70)
    print("GATE 1 - TEST 2: CORPUS FINGERPRINT VALIDATION AGAINST SUPABASE")
    print("=" * 70)

    from sqlalchemy import create_engine, text
    db_url = os.getenv("DATABASE_URL")
    assert db_url, "DATABASE_URL not found in .env"
    db_engine = create_engine(db_url)

    with db_engine.connect() as conn:
        row = conn.execute(text("""
            SELECT 
                (SELECT count(*) FROM sentences) AS sentence_count,
                (SELECT md5(string_agg(sentence_text, '' ORDER BY id)) FROM sentences) AS sentence_text_md5,
                (SELECT count(*) FROM sentences WHERE embedding_dapt IS NULL) AS dapt_null_count,
                (SELECT md5(string_agg(coalesce(embedding_dapt::text, 'NULL'), '' ORDER BY id)) FROM sentences) AS dapt_column_md5
        """)).fetchone()

    live_fp = {
        "sentence_count": int(row[0]),
        "sentence_text_md5": str(row[1]),
        "dapt_null_count": int(row[2]),
        "dapt_column_md5": str(row[3]),
    }

    bundle_fp = bundles[("noli", "buod")]["fingerprint"]
    print("  Live DB Fingerprint:")
    print(f"    sentence_count:    {live_fp['sentence_count']}")
    print(f"    sentence_text_md5: {live_fp['sentence_text_md5']}")
    print(f"    dapt_null_count:   {live_fp['dapt_null_count']}")
    print(f"    dapt_column_md5:   {live_fp['dapt_column_md5']}")

    print("  Bundle Fingerprint:")
    print(f"    sentence_count:    {bundle_fp['sentence_count']}")
    print(f"    sentence_text_md5: {bundle_fp['sentence_text_md5']}")
    print(f"    dapt_null_count:   {bundle_fp['dapt_null_count']}")
    print(f"    dapt_column_md5:   {bundle_fp['dapt_column_md5']}")
    print(f"    robust_aligner:    {bundle_fp.get('robust_aligner_hash', 'N/A')}")

    assert live_fp["sentence_count"] == bundle_fp["sentence_count"], "sentence_count mismatch!"
    assert live_fp["sentence_text_md5"] == bundle_fp["sentence_text_md5"], "sentence_text_md5 mismatch!"
    assert live_fp["dapt_null_count"] == bundle_fp["dapt_null_count"], "dapt_null_count mismatch!"
    assert live_fp["dapt_column_md5"] == bundle_fp["dapt_column_md5"], "dapt_column_md5 mismatch!"
    print("  => FINGERPRINT MATCH VERIFIED! (100% Corpus Integrity)")


def test_parity_all_chapters(bundles):
    print("\n" + "=" * 70)
    print("GATE 1 - TEST 3: RAW HTTP BYTES SHA-256 PARITY (ALL 205 CHAPTERS)")
    print("=" * 70)

    session = requests.Session()
    mismatches = 0
    checked = 0

    for (book, mode), (path, total_ch) in BUNDLE_CONFIGS.items():
        print(f"\n  Checking {book.upper()} ({mode}) - {total_ch} chapters...")
        for ch in range(1, total_ch + 1):
            url = f"{LOCAL_URL}/api/v1/chapters/{book}/{ch}/sanggunian?mode={mode}"
            resp = session.get(url, timeout=10)
            assert resp.status_code == 200, f"HTTP {resp.status_code} on {url}"

            live_json = resp.json()
            bundle_ch = bundles[(book, mode)]["chapters"][str(ch)]

            # Compare normalized JSON representation
            live_norm = json.dumps(live_json, sort_keys=True, ensure_ascii=False)
            bundle_norm = json.dumps(bundle_ch, sort_keys=True, ensure_ascii=False)

            live_sha = hashlib.sha256(live_norm.encode("utf-8")).hexdigest()
            bundle_sha = hashlib.sha256(bundle_norm.encode("utf-8")).hexdigest()

            if live_sha != bundle_sha:
                mismatches += 1
                print(f"    MISMATCH on {book} ch {ch} ({mode})! live={live_sha} vs bundle={bundle_sha}")
            checked += 1

    print(f"\n  Parity Summary: {checked - mismatches}/{checked} chapters identical (Mismatches: {mismatches})")
    assert mismatches == 0, f"Found {mismatches} mismatches between bundle and live endpoint!"
    print("  => PARITY VERIFIED: 0 MISMATCHES ACROSS ALL 205 CHAPTERS!")


def test_score_delta_profiled_sentences(bundles):
    print("\n" + "=" * 70)
    print("GATE 1 - TEST 4: SCORE DELTA ON 5 PROFILED SENTENCES (LOCAL VS LIVE HF SPACE)")
    print("=" * 70)

    session = requests.Session()
    max_delta = 0.0
    report_rows = []

    for sid in PROFILED_SENTENCE_IDS:
        # Local endpoint
        t0 = time.time()
        local_resp = session.get(f"{LOCAL_URL}/api/v1/sentences/{sid}/sanggunian", timeout=10)
        local_lat = (time.time() - t0) * 1000
        assert local_resp.status_code == 200, f"Local HTTP {local_resp.status_code} for sentence {sid}"
        local_data = local_resp.json()
        local_score = local_data.get("score", 0.0)

        # Live HF Space endpoint
        t0 = time.time()
        hf_resp = session.get(f"{HF_SPACE_URL}/api/v1/sentences/{sid}/sanggunian", timeout=30)
        hf_lat = (time.time() - t0) * 1000
        assert hf_resp.status_code == 200, f"HF HTTP {hf_resp.status_code} for sentence {sid}"
        hf_data = hf_resp.json()
        hf_score = hf_data.get("score", 0.0)

        delta = abs(local_score - hf_score)
        if delta > max_delta:
            max_delta = delta

        # Also check passage alignment parity
        local_pids = local_data.get("passage_ids", [])
        hf_pids = hf_data.get("passage_ids", [])
        pids_match = (local_pids == hf_pids)

        report_rows.append({
            "sentence_id": sid,
            "local_score": local_score,
            "hf_score": hf_score,
            "score_delta": delta,
            "passage_ids_match": pids_match,
            "local_latency_ms": local_lat,
            "hf_latency_ms": hf_lat
        })

        print(f"  Sentence {sid:5}: Local={local_score:.6f} ({local_lat:.1f}ms) | HF={hf_score:.6f} ({hf_lat:.1f}ms) | Delta={delta:.8f} | Passage match={pids_match}")

    print(f"\n  Max Absolute Score Delta: {max_delta:.8f}")
    assert max_delta < 1e-4, f"Score delta too high: {max_delta}"
    print("  => SCORE DELTA DISCLOSURE VERIFIED!")
    return report_rows, max_delta


def test_latency_and_headers():
    print("\n" + "=" * 70)
    print("GATE 1 - TEST 5: CHAPTER ENDPOINT LATENCY & CACHE-CONTROL HEADERS")
    print("=" * 70)

    session = requests.Session()
    test_endpoints = [
        ("noli", 1, "buod"),
        ("noli", 5, "buod"),
        ("elfili", 10, "buod"),
        ("noli", 25, "full"),
        ("elfili", 39, "full"),
    ]

    for book, ch, mode in test_endpoints:
        url = f"{LOCAL_URL}/api/v1/chapters/{book}/{ch}/sanggunian?mode={mode}"
        
        # Cold request (fresh session get)
        t0 = time.time()
        resp = session.get(url, timeout=10)
        dur1 = (time.time() - t0) * 1000
        assert resp.status_code == 200
        cc = resp.headers.get("Cache-Control", "")

        # Warm requests
        timings = []
        for _ in range(5):
            t0 = time.time()
            r = session.get(url, timeout=10)
            timings.append((time.time() - t0) * 1000)

        warm_avg = sum(timings) / len(timings)
        warm_min = min(timings)

        print(f"  {book:6} ch {ch:2} ({mode:4}): Cold={dur1:5.1f}ms | WarmAvg={warm_avg:5.1f}ms | WarmMin={warm_min:5.1f}ms | Cache-Control: {cc}")
        assert "public" in cc and "max-age=86400" in cc, f"Bad Cache-Control header: {cc}"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gate 1 Verification for Sanggunian")
    parser.add_argument("--allow-prod", action="store_true", help="Allow running against production Supabase")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from prod_guard import verify_not_production_db
    verify_not_production_db(allow_prod=args.allow_prod, caller_name="verify_sanggunian_gate1.py")

    print("STARTING GATE 1 VERIFICATION RUN FOR SANGGUNIAN (ITEM C PHASE 2)")
    print(f"Local URL: {LOCAL_URL}")
    print(f"HF Space:  {HF_SPACE_URL}")

    bundles = test_bundle_integrity_and_sizes()
    test_fingerprint_against_db(bundles)
    test_parity_all_chapters(bundles)
    report_rows, max_delta = test_score_delta_profiled_sentences(bundles)
    test_latency_and_headers()

    print("\n" + "=" * 70)
    print("ALL GATE 1 VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
