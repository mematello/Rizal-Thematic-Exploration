"""
Test and report out-of-domain latency for first call (cold cache) vs repeated calls (hot cache).
Uses direct engine.search() with PerfTimer enabled.
"""
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "0"
os.environ["PERF_BENCHMARK"] = "1"
os.environ["DISABLE_WORD_VEC_CACHE"] = "0"

backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.models.database import SessionLocal
from app.core.engine import get_engine, get_translator_cache

def main():
    db = SessionLocal()
    engine = get_engine()
    cache = get_translator_cache()
    cache.clear()

    ood_queries = [
        "tiktok",
        "bitcoin",
        "covid",
        "facebook",
        "internet",
        "computer",
        "cellphone",
        "online"
    ]

    print("=" * 95)
    print("OUT-OF-DOMAIN TRANSLATOR CACHE LATENCY: First Call (Cold) vs Repeated Calls (Hot)")
    print("=" * 95)
    print(f"{'Query':<15} | {'Cold Call 1':<14} | {'Hot Call 2':<14} | {'Hot Call 3':<14} | {'Speedup':<10} | {'Cached Translation'}")
    print("-" * 95)

    results = []

    for q in ood_queries:
        # First call (Cold cache: network translation)
        t0 = time.perf_counter()
        res1 = engine.search(db, q, source_type="summary", novel="both")
        cold_ms = (time.perf_counter() - t0) * 1000

        meta1 = res1.get("metadata", {})
        eng1 = meta1.get("engine_timings", {})
        trans_ms1 = eng1.get("translator", 0.0)

        # Second call (Hot cache: memory lookup)
        t0 = time.perf_counter()
        res2 = engine.search(db, q, source_type="summary", novel="both")
        hot1_ms = (time.perf_counter() - t0) * 1000

        meta2 = res2.get("metadata", {})
        eng2 = meta2.get("engine_timings", {})
        trans_ms2 = eng2.get("translator", 0.0)

        # Third call (Hot cache: second repeat)
        t0 = time.perf_counter()
        res3 = engine.search(db, q, source_type="summary", novel="both")
        hot2_ms = (time.perf_counter() - t0) * 1000
        meta3 = res3.get("metadata", {})

        # Retrieve cached translation string
        is_hit, cached_str = cache.get(q)

        # Verify output equivalence
        assert res1["results"] == res2["results"] == res3["results"] == {"noli": [], "elfili": []}
        assert meta1["result_mode"] == meta2["result_mode"] == meta3["result_mode"] == "none"
        assert meta1["reason"] == meta2["reason"] == meta3["reason"] == "unmapped_standalone_concept"
        assert meta1["suggestions"] == meta2["suggestions"] == meta3["suggestions"]

        speedup = ((cold_ms - hot1_ms) / cold_ms) * 100

        print(f"{q:<15} | {cold_ms:>11.2f} ms | {hot1_ms:>11.2f} ms | {hot2_ms:>11.2f} ms | {speedup:>+8.1f}% | '{cached_str}'")

        results.append({
            "query": q,
            "cold_ms": round(cold_ms, 2),
            "hot1_ms": round(hot1_ms, 2),
            "hot2_ms": round(hot2_ms, 2),
            "trans_ms1": round(trans_ms1, 2),
            "trans_ms2": round(trans_ms2, 2),
            "cached_str": cached_str,
            "speedup_pct": round(speedup, 1)
        })

    cold_med = [r["cold_ms"] for r in results]
    hot1_med = [r["hot1_ms"] for r in results]
    import statistics
    print("-" * 95)
    print(f"{'MEDIAN':<15} | {statistics.median(cold_med):>11.2f} ms | {statistics.median(hot1_med):>11.2f} ms | {'-':>14} | {((statistics.median(cold_med) - statistics.median(hot1_med)) / statistics.median(cold_med) * 100):>+8.1f}% |")
    print("=" * 95)

    db.close()

if __name__ == "__main__":
    main()
