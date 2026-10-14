"""
Diagnostic & Verification Script:
Measures the first search of 10 previously unseen Tagalog words after a fresh server start
with the precomputed word-vector cache active.
Verifies that the slow-first-request anomaly is completely resolved.
"""
import os
import sys
import time
import json
import argparse
import requests
from pathlib import Path

UNSEEN_TAGALOG_WORDS = [
    "kasakiman",    # greed (previously 20.6s cold)
    "alipin",       # slave (previously 12.6s cold)
    "dangal",       # honor (previously 5.7s cold)
    "takot",        # fear (lexical match)
    "sakripisyo",   # sacrifice (cross-lingual / unmapped)
    "katapangan",   # bravery
    "kabayanihan",  # heroism
    "kalupitan",    # cruelty
    "pagdurusa",    # suffering
    "himagsikan"    # revolution
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--out", type=str, default="unseen_10_words_report.json")
    args = parser.parse_args()

    api_base = f"http://127.0.0.1:{args.port}"
    session = requests.Session()

    print("=" * 95)
    print("VERIFICATION 3(c): 10 Unseen Tagalog Words First-Search Latency on Fresh Server")
    print(f"API Base: {api_base}")
    print(f"Words tested ({len(UNSEEN_TAGALOG_WORDS)}): {', '.join(UNSEEN_TAGALOG_WORDS)}")
    print("=" * 95)

    # Health check
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
        print(f"\n[{idx}/10] Testing unseen word: '{word}'")

        # Run 1: First request on fresh server
        t0 = time.perf_counter()
        resp1 = session.get(f"{api_base}/api/v1/search", params={"q": word, "source_type": "summary", "novel": "both"}, timeout=60)
        t_first = (time.perf_counter() - t0) * 1000
        data1 = resp1.json()
        meta1 = data1.get("metadata", {})
        eng1 = meta1.get("engine_timings", {})
        res1 = data1.get("results", {})
        noli_count1 = len(res1.get("noli", []))
        fili_count1 = len(res1.get("elfili", []))

        # Run 2: Second request (warm)
        t0 = time.perf_counter()
        resp2 = session.get(f"{api_base}/api/v1/search", params={"q": word, "source_type": "summary", "novel": "both"}, timeout=60)
        t_second = (time.perf_counter() - t0) * 1000
        data2 = resp2.json()
        meta2 = data2.get("metadata", {})
        eng2 = meta2.get("engine_timings", {})

        print(f"  First Search:  {t_first:7.2f} ms | Mode: {meta1.get('result_mode', 'N/A')} | Candidates: {noli_count1 + fili_count1}")
        print(f"  Second Search: {t_second:7.2f} ms")
        print(f"  Rerank Loop:   {eng1.get('rerank_loop', 0):7.2f} ms (Encodes: {eng1.get('encode_calls_total', 0)}, Hits: {eng1.get('process_cache_hits', 0)}, Misses: {eng1.get('process_cache_misses', 0)})")

        results.append({
            "word": word,
            "first_search_ms": round(t_first, 2),
            "second_search_ms": round(t_second, 2),
            "result_mode": meta1.get("result_mode"),
            "candidate_count": noli_count1 + fili_count1,
            "first_timings": eng1,
            "second_timings": eng2
        })

    docs_perf_dir = Path(__file__).resolve().parent.parent / "docs" / "perf"
    out_path = docs_perf_dir / args.out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 95)
    print(f"{'Word':<15} | {'First Search (ms)':<18} | {'Second Search (ms)':<18} | {'Rerank Loop (ms)':<18} | {'Result Mode'}")
    print("-" * 95)
    for r in results:
        rerank = r["first_timings"].get("rerank_loop", 0)
        print(f"{r['word']:<15} | {r['first_search_ms']:>18.2f} | {r['second_search_ms']:>18.2f} | {rerank:>18.2f} | {r['result_mode']}")
    print("=" * 95)

if __name__ == "__main__":
    main()
