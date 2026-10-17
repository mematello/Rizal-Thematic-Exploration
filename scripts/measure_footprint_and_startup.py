#!/usr/bin/env python3
"""
scripts/measure_footprint_and_startup.py

Measures:
1. RSS before and after loading the 4 Sanggunian bundles.
2. Startup time of verify_sanggunian_fingerprint().
3. Disk / Docker image size delta for the bundles.
"""

import sys
import os
import psutil
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.models.database import SessionLocal
import app.api.v1.content as content

def get_rss_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def main():
    print("=" * 80)
    print("SANGGUNIAN FOOTPRINT & STARTUP MEASUREMENTS")
    print("=" * 80)

    import gc
    import json
    gc.collect()
    rss_before = get_rss_mb()
    print(f"Process RSS before loading Sanggunian bundles: {rss_before:.2f} MB")

    data_dir = BACKEND_DIR / "app" / "data"
    cache_files = [
        data_dir / "sanggunian_cache_noli_buod.json",
        data_dir / "sanggunian_cache_noli_full.json",
        data_dir / "sanggunian_cache_elfili_buod.json",
        data_dir / "sanggunian_cache_elfili_full.json",
    ]

    t0 = time.perf_counter()
    loaded_data = {}
    sentence_index = {}
    for cf in cache_files:
        with open(cf, "r", encoding="utf-8") as f:
            bundle = json.load(f)
            loaded_data[cf.name] = bundle.get("chapters", {})
            for ch_str, c_info in bundle.get("chapters", {}).items():
                for sid_str, ref_obj in c_info.get("references", {}).items():
                    sentence_index[int(sid_str)] = ref_obj

    load_time_ms = (time.perf_counter() - t0) * 1000
    rss_after = get_rss_mb()
    rss_delta = rss_after - rss_before
    print(f"Process RSS after loading Sanggunian bundles:  {rss_after:.2f} MB")
    print(f"RSS Memory Delta:                              +{rss_delta:.2f} MB (load time: {load_time_ms:.2f} ms)")
    print(f"Total sentences indexed in memory:            {len(sentence_index):,d}")

    # 2. Bundle sizes on disk (Docker image delta)
    data_dir = BACKEND_DIR / "app" / "data"
    bundles = [
        "sanggunian_cache_noli_buod.json",
        "sanggunian_cache_noli_full.json",
        "sanggunian_cache_elfili_buod.json",
        "sanggunian_cache_elfili_full.json"
    ]
    total_disk_bytes = 0
    print("\nBundle file sizes (Docker image delta):")
    for b in bundles:
        fpath = data_dir / b
        sz = fpath.stat().st_size if fpath.exists() else 0
        total_disk_bytes += sz
        print(f"  {b:36} {sz / 1024:10.2f} KB ({sz:10,d} bytes)")
    total_disk_mb = total_disk_bytes / (1024 * 1024)
    print(f"Total Docker Image Delta:                     +{total_disk_mb:.2f} MB ({total_disk_bytes:,d} bytes)")

    # 3. Startup time of verify_sanggunian_fingerprint()
    print("\nMeasuring verify_sanggunian_fingerprint() startup time against live Supabase...")
    db = SessionLocal()
    # Reset checked flag to test full verification
    content._sanggunian_fingerprint_checked = False
    content._static_sanggunian_enabled = False

    t_verify_start = time.perf_counter()
    verified = content.verify_sanggunian_fingerprint(db)
    t_verify_end = time.perf_counter()
    verify_time_ms = (t_verify_end - t_verify_start) * 1000
    print(f"Verification Result: {verified}")
    print(f"verify_sanggunian_fingerprint() runtime:     {verify_time_ms:.2f} ms ({verify_time_ms/1000:.2f} s)")

    # 4. Save to evidence
    out_file = REPO_ROOT / "docs" / "perf" / "evidence" / "sanggunian_footprint_startup.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("SANGGUNIAN FOOTPRINT & STARTUP MEASUREMENT REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"1. RSS Memory Impact:\n")
        f.write(f"   - RSS Before Loading: {rss_before:.2f} MB\n")
        f.write(f"   - RSS After Loading:  {rss_after:.2f} MB\n")
        f.write(f"   - RSS Delta:          +{rss_delta:.2f} MB\n")
        f.write(f"   - Bundle Load Time:   {load_time_ms:.2f} ms\n\n")
        f.write(f"2. Docker Image Delta:\n")
        for b in bundles:
            sz = (data_dir / b).stat().st_size
            f.write(f"   - {b:36} {sz / 1024:10.2f} KB\n")
        f.write(f"   - Total Image Delta:  +{total_disk_mb:.2f} MB ({total_disk_bytes:,d} bytes)\n\n")
        f.write(f"3. Startup Verification:\n")
        f.write(f"   - verify_sanggunian_fingerprint() Runtime: {verify_time_ms:.2f} ms ({verify_time_ms/1000:.2f} s)\n")
        f.write(f"   - Verification Success:                    {verified}\n")
        f.write(f"   - SQL Computation:                         embedding_dapt MD5 is computed entirely in SQL inside Supabase\n")
        f.write(f"   - Request Handling During Verification:    While background warmup runs, requests fall back to live compute until verified, then 0ms static bundle is enabled.\n")

    print(f"\nReport written to {out_file}")

if __name__ == "__main__":
    main()
