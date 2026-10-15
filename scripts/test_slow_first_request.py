"""
Diagnostic Script for Slow First Request Issue:
Measures the first search of 5 previously unseen Tagalog words after a fresh server start
(with word cache empty) and compares it with the immediate second (warm) search.
Reports end-to-end latency, stage breakdown, and encode call counts.
"""
import os
import sys
import time
import json
import statistics
import requests
import psutil
from pathlib import Path
from typing import Dict, List, Any

# Ensure we test with all optimizations active
os.environ["PYTHONHASHSEED"] = "0"
os.environ["PERF_BENCHMARK"] = "1"
os.environ["DISABLE_WORD_VEC_CACHE"] = "0"
os.environ["DISABLE_TRANSLATOR_CACHE"] = "0"
os.environ["DISABLE_BATCHED_CONTEXT_FETCH"] = "0"

backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

UNSEEN_TAGALOG_WORDS = [
    "kasakiman",   # greed
    "dangal",      # honor
    "takot",       # fear
    "alipin",      # slave
    "sakripisyo"   # sacrifice
]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--out", type=str, default="slow_first_request_report.json")
    args = parser.parse_args()

    api_base = f"http://127.0.0.1:{args.port}"
    session = requests.Session()

    print("=" * 90)
    print("SLOW FIRST REQUEST INVESTIGATION: 5 Unseen Tagalog Words on Fresh Server Start")
    print(f"API Base: {api_base}")
    print(f"Words tested: {', '.join(UNSEEN_TAGALOG_WORDS)}")
    print("=" * 90)

    # Check health
    try:
        r = session.get(f"{api_base}/health", timeout=5)
        if r.status_code != 200:
            print(f"[FAIL] Health check failed: {r.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"[FAIL] Could not connect to server: {e}")
        sys.exit(1)

    results = []

    for idx, word in enumerate(UNSEEN_TAGALOG_WORDS, 1):
        print(f"\n[{idx}/5] Testing unseen word: '{word}'")

        # Run 1: First request (Cold for this word)
        t0 = time.perf_counter()
        resp1 = session.get(f"{api_base}/api/v1/search", params={"q": word, "source_type": "summary", "novel": "both"}, timeout=60)
        t_cold = (time.perf_counter() - t0) * 1000
        data1 = resp1.json()
        meta1 = data1.get("metadata", {})
        eng1 = meta1.get("engine_timings", {})
        res1 = data1.get("results", {})
        noli_count1 = len(res1.get("noli", []))
        fili_count1 = len(res1.get("elfili", []))

        # Run 2: Second request (Warm for this word)
        t0 = time.perf_counter()
        resp2 = session.get(f"{api_base}/api/v1/search", params={"q": word, "source_type": "summary", "novel": "both"}, timeout=60)
        t_warm = (time.perf_counter() - t0) * 1000
        data2 = resp2.json()
        meta2 = data2.get("metadata", {})
        eng2 = meta2.get("engine_timings", {})

        speedup = ((t_cold - t_warm) / t_cold) * 100 if t_cold > 0 else 0

        print(f"  First Search (Cold):  {t_cold:7.2f} ms | Mode: {meta1.get('result_mode', 'N/A')} | Candidates: {noli_count1 + fili_count1}")
        print(f"  Second Search (Warm): {t_warm:7.2f} ms | Speedup: {speedup:+.1f}%")
        print(f"  Rerank Loop (Cold):   {eng1.get('rerank_loop', 0):7.2f} ms (Encodes: {eng1.get('encode_calls_total', 0)}, Time: {eng1.get('encode_time_total_ms', 0):.2f} ms, Misses: {eng1.get('process_cache_misses', 0)})")
        print(f"  Rerank Loop (Warm):   {eng2.get('rerank_loop', 0):7.2f} ms (Encodes: {eng2.get('encode_calls_total', 0)}, Time: {eng2.get('encode_time_total_ms', 0):.2f} ms, Misses: {eng2.get('process_cache_misses', 0)})")
        print(f"  Context Fetch/Exp:    Fetch = {eng1.get('context_fetch', 0):.2f} ms | Expand = {eng1.get('context_expansion', 0):.2f} ms | Theme = {eng1.get('theme_classification', 0):.2f} ms")

        results.append({
            "word": word,
            "cold_latency_ms": round(t_cold, 2),
            "warm_latency_ms": round(t_warm, 2),
            "speedup_pct": round(speedup, 1),
            "result_mode": meta1.get("result_mode"),
            "candidate_count": noli_count1 + fili_count1,
            "cold_timings": eng1,
            "warm_timings": eng2
        })

    out_path = Path(__file__).parent.parent / "docs" / "perf" / args.out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print("SUMMARY: UNSEEN TAGALOG WORDS FIRST REQUEST")
    print(f"{'Word':<15} | {'Cold Latency (ms)':<18} | {'Warm Latency (ms)':<18} | {'Delta':<12} | {'Result Mode'}")
    print("-" * 90)
    for r in results:
        delta = r["warm_latency_ms"] - r["cold_latency_ms"]
        print(f"{r['word']:<15} | {r['cold_latency_ms']:>18.2f} | {r['warm_latency_ms']:>18.2f} | {delta:>+10.2f} ms | {r['result_mode']}")
    print("=" * 90)

if __name__ == "__main__":
    main()
