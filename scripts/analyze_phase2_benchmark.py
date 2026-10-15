"""
Analyze Phase 2 Step 1 Benchmark Results vs Phase 1 Baseline
"""

import json
import statistics
from pathlib import Path
from collections import defaultdict

PERF_DIR = Path(__file__).parent.parent / "docs" / "perf"
BASELINE_PATH = PERF_DIR / "baseline_timings.json"
PHASE2_PATH = PERF_DIR / "phase2_step1_timings.json"

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def summarize_timings(data):
    by_category = defaultdict(list)
    by_query = {}

    for item in data["queries"]:
        q = item["query"]
        cat = item["category"]
        runs = item["runs"]
        latencies = [r["http_latency_ms"] for r in runs]
        cold = latencies[0]
        warm = latencies[1:] if len(latencies) > 1 else latencies

        p50 = statistics.median(warm)
        p95 = statistics.quantiles(warm, n=20)[18] if len(warm) >= 2 else warm[0]
        mean = statistics.mean(warm)

        by_query[q] = {
            "query": q,
            "category": cat,
            "cold": cold,
            "warm_p50": p50,
            "warm_p95": p95,
            "warm_mean": mean,
            "runs": runs
        }

        by_category[cat].append({
            "cold": cold,
            "warm_p50": p50,
            "warm_p95": p95,
            "warm_mean": mean
        })

    cat_summary = {}
    for cat, samples in by_category.items():
        cat_summary[cat] = {
            "cold": statistics.mean([s["cold"] for s in samples]),
            "warm_p50": statistics.mean([s["warm_p50"] for s in samples]),
            "warm_p95": statistics.mean([s["warm_p95"] for s in samples]),
            "warm_mean": statistics.mean([s["warm_mean"] for s in samples])
        }

    return by_query, cat_summary

def main():
    if not PHASE2_PATH.exists():
        print(f"Waiting for {PHASE2_PATH}...")
        return

    base_data = load_data(BASELINE_PATH)
    p2_data = load_data(PHASE2_PATH)

    base_query, base_cat = summarize_timings(base_data)
    p2_query, p2_cat = summarize_timings(p2_data)

    print("=" * 90)
    print("PHASE 1 BASELINE vs PHASE 2 STEP 1 (PROCESS-LEVEL CACHE) COMPARISON")
    print("=" * 90)

    print("\n--- Latency by Category (HTTP end-to-end, Warm P50) ---")
    print(f"{'Category':<25} {'Baseline Cold':<15} {'Phase 2 Cold':<15} {'Baseline P50':<15} {'Phase 2 P50':<15} {'Speedup':<10}")
    print("-" * 95)
    for cat in base_cat:
        b = base_cat[cat]
        p = p2_cat.get(cat, {})
        speedup = b["warm_p50"] / p["warm_p50"] if p.get("warm_p50", 0) > 0 else 0
        print(f"{cat:<25} {b['cold']:>10.1f} ms   {p['cold']:>10.1f} ms   {b['warm_p50']:>10.1f} ms   {p['warm_p50']:>10.1f} ms   {speedup:>7.2f}x")

    print("\n--- Key Individual Queries ---")
    key_queries = [
        "edukasyon", "kalayaan", "pag-ibig", "kamatayan",
        "simbahan", "bayan", "paaralan", "education", "oppression",
        "korapsyon", "tiktok", "online"
    ]
    print(f"{'Query':<20} {'Category':<22} {'Base P50':<12} {'P2 P50':<12} {'Speedup':<10} {'Base Cold':<12} {'P2 Cold':<12}")
    print("-" * 105)
    for q in key_queries:
        if q in base_query and q in p2_query:
            b = base_query[q]
            p = p2_query[q]
            speedup = b["warm_p50"] / p["warm_p50"] if p["warm_p50"] > 0 else 0
            print(f"{q:<20} {b['category']:<22} {b['warm_p50']:>8.1f} ms   {p['warm_p50']:>8.1f} ms   {speedup:>7.2f}x   {b['cold']:>8.1f} ms   {p['cold']:>8.1f} ms")

    print("\n--- Stage-Level Breakdown for Specific Queries (bayan, tiktok, oppression) ---")
    target_queries = ["bayan", "tiktok", "oppression"]
    for q in target_queries:
        if q in p2_query:
            item = p2_query[q]
            print(f"\nQuery: '{q}' (Category: {item['category']})")
            print(f"  Warm P50 HTTP Latency: {item['warm_p50']:.1f} ms (Cold: {item['cold']:.1f} ms)")
            # Extract stage timings from warm runs
            runs = item["runs"]
            warm_runs = runs[1:] if len(runs) > 1 else runs

            # Aggregate stage timings across warm runs
            stage_keys = [
                "search_total_ms", "translator_ms", "untimed_gap_ms",
                "base_model_encode", "theme_expansion", "lexical_retrieval",
                "semantic_fallback", "rerank_loop", "context_expansion",
                "theme_classification", "translator_network_calls",
                "process_cache_hits", "process_cache_misses", "process_cache_size"
            ]
            print("  Warm Run Engine Breakdown (averages over warm runs):")
            for key in stage_keys:
                vals = []
                for r in warm_runs:
                    eng = r.get("engine_timings", {})
                    if key in eng:
                        vals.append(eng[key])
                if vals:
                    if isinstance(vals[0], bool):
                        print(f"    {key:<28}: {vals[0]}")
                    elif isinstance(vals[0], (int, float)):
                        avg_val = statistics.mean(vals)
                        med_val = statistics.median(vals)
                        print(f"    {key:<28}: avg = {avg_val:>8.2f} | p50 = {med_val:>8.2f}")

if __name__ == "__main__":
    main()
