"""
Generate accurate summary statistics from baseline_timings.json
"""

import json
import statistics
from pathlib import Path
from collections import defaultdict

TIMINGS_PATH = Path(__file__).parent.parent / "docs" / "perf" / "baseline_timings.json"

def load_timings():
    with open(TIMINGS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_query(query_data):
    """Extract cold, warm stats, and db query count for a single query"""
    runs = query_data["runs"]
    if not runs:
        return None

    cold_latency = runs[0]["http_latency_ms"]
    warm_latencies = [r["http_latency_ms"] for r in runs[1:] if "http_latency_ms" in r]

    db_count = runs[0]["engine_timings"].get("db_query_count", 0)

    result = {
        "cold": cold_latency,
        "db_queries": db_count
    }

    if warm_latencies:
        result["warm_mean"] = statistics.mean(warm_latencies)
        result["warm_p50"] = statistics.median(warm_latencies)
        result["warm_p95"] = statistics.quantiles(warm_latencies, n=20)[18] if len(warm_latencies) >= 2 else warm_latencies[0]

    return result

def main():
    data = load_timings()

    print(f"First request time: {data['first_request_time_s']:.2f}s")
    print(f"Memory delta: {data['memory_delta_mb']:.2f} MB")
    print(f"Peak memory: {data['peak_memory_mb']:.2f} MB")
    print()

    # Aggregate by category
    by_category = defaultdict(list)
    query_details = {}

    for query_data in data["queries"]:
        query = query_data["query"]
        category = query_data["category"]

        stats = analyze_query(query_data)
        if stats:
            by_category[category].append(stats)
            query_details[query] = stats

    # Print category summary
    print("=" * 100)
    print("CATEGORY SUMMARY")
    print("=" * 100)
    print(f"{'Category':<30} {'Cold (ms)':<12} {'Warm P50':<12} {'Warm P95':<12}")
    print("-" * 100)

    for category, samples in sorted(by_category.items()):
        cold_avg = statistics.mean([s["cold"] for s in samples])
        warm_p50_avg = statistics.mean([s["warm_p50"] for s in samples])
        warm_p95_avg = statistics.mean([s["warm_p95"] for s in samples])

        print(f"{category:<30} {cold_avg:>10.1f}  {warm_p50_avg:>10.1f}  {warm_p95_avg:>10.1f}")

    print()
    print("=" * 100)
    print("KEY QUERIES DETAIL")
    print("=" * 100)
    print(f"{'Query':<20} {'Cold (ms)':<12} {'Warm P50':<12} {'DB Queries':<12}")
    print("-" * 100)

    key_queries = ["kalayaan", "edukasyon", "simbahan", "bayan", "paaralan",
                   "korapsyon", "education", "oppression", "tiktok", "online"]

    for query in key_queries:
        if query in query_details:
            stats = query_details[query]
            print(f"{query:<20} {stats['cold']:>10.1f}  {stats.get('warm_p50', 0):>10.1f}  {stats['db_queries']:>10}")

if __name__ == "__main__":
    main()
