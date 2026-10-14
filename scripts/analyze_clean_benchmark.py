"""
Analyze and compare Clean Benchmark Before vs After.
Computes min/median/max and warm_median per query and per category.
Checks whether non-affected categories differ by > 15%.
Excludes out_of_domain from speedup claims.
"""
import json
import statistics
from pathlib import Path

PERF_DIR = Path(__file__).parent.parent / "docs" / "perf"
BEFORE_PATH = PERF_DIR / "clean_before_timings.json"
AFTER_PATH = PERF_DIR / "clean_after_timings.json"

def main():
    if not BEFORE_PATH.exists() or not AFTER_PATH.exists():
        print(f"Waiting for files to exist: before={BEFORE_PATH.exists()}, after={AFTER_PATH.exists()}")
        return

    with open(BEFORE_PATH, "r", encoding="utf-8") as f:
        before_data = json.load(f)

    with open(AFTER_PATH, "r", encoding="utf-8") as f:
        after_data = json.load(f)

    before_queries = {q["query"]: q for q in before_data["queries"]}
    after_queries = {q["query"]: q for q in after_data["queries"]}

    print("=" * 110)
    print("CLEAN BENCHMARK COMPARISON: Before (No Cache) vs After (With Word Cache)")
    print("=" * 110)
    print(f"System Load Before Run: CPU={before_data['system_load']['avg_cpu_during_run']}%, RAM={before_data['system_load']['load_before']['available_ram_mb']}MB free")
    print(f"System Load After Run:  CPU={after_data['system_load']['avg_cpu_during_run']}%, RAM={after_data['system_load']['load_before']['available_ram_mb']}MB free")
    print("=" * 110)

    categories = {}
    for q_text, b_info in before_queries.items():
        cat = b_info["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(q_text)

    # Detailed per-query table
    print(f"\n{'Category':<22} | {'Query':<28} | {'Before Med':<10} | {'After Med':<10} | {'Delta':<10} | {'Speedup':<9} | {'Min (B/A)':<14} | {'Max (B/A)':<14}")
    print("-" * 125)

    category_stats = {}

    for cat, q_list in categories.items():
        b_meds = []
        a_meds = []
        b_warm_meds = []
        a_warm_meds = []

        for q_text in q_list:
            b_q = before_queries[q_text]
            a_q = after_queries[q_text]

            b_med = b_q["median_ms"]
            a_med = a_q["median_ms"]
            b_warm = b_q["warm_median_ms"]
            a_warm = a_q["warm_median_ms"]

            b_min, b_max = b_q["min_ms"], b_q["max_ms"]
            a_min, a_max = a_q["min_ms"], a_q["max_ms"]

            delta = a_med - b_med
            pct = ((b_med - a_med) / b_med) * 100 if b_med > 0 else 0

            b_meds.append(b_med)
            a_meds.append(a_med)
            b_warm_meds.append(b_warm)
            a_warm_meds.append(a_warm)

            speedup_str = f"{pct:+6.1f}%" if cat != "out_of_domain" else "[N/A trans]"
            min_str = f"{b_min:.0f}/{a_min:.0f}"
            max_str = f"{b_max:.0f}/{a_max:.0f}"

            print(f"{cat:<22} | {q_text:<28} | {b_med:>8.1f} ms | {a_med:>8.1f} ms | {delta:>+8.1f} ms | {speedup_str:>9} | {min_str:>14} | {max_str:>14}")

        cat_b_med = statistics.median(b_meds)
        cat_a_med = statistics.median(a_meds)
        cat_b_warm = statistics.median(b_warm_meds)
        cat_a_warm = statistics.median(a_warm_meds)
        cat_delta = cat_a_med - cat_b_med
        cat_pct = ((cat_b_med - cat_a_med) / cat_b_med) * 100 if cat_b_med > 0 else 0

        category_stats[cat] = {
            "before_median": cat_b_med,
            "after_median": cat_a_med,
            "before_warm_median": cat_b_warm,
            "after_warm_median": cat_a_warm,
            "pct_change": cat_pct,
            "delta_ms": cat_delta
        }

    # Summary table by category
    print("\n" + "=" * 90)
    print("CATEGORY LEVEL SUMMARY (Warm Medians)")
    print("=" * 90)
    print(f"{'Category':<25} | {'Before Median':<14} | {'After Median':<14} | {'Delta (ms)':<12} | {'Change (%)':<10}")
    print("-" * 90)

    for cat, stats in category_stats.items():
        b_m = stats["before_warm_median"]
        a_m = stats["after_warm_median"]
        d = a_m - b_m
        pct = stats["pct_change"]
        pct_display = f"{pct:+6.1f}%" if cat != "out_of_domain" else "[excluded]"
        print(f"{cat:<25} | {b_m:>11.1f} ms | {a_m:>11.1f} ms | {d:>+10.1f} ms | {pct_display:>10}")

    # Check non-affected categories
    print("\n" + "=" * 90)
    print("NON-AFFECTED CATEGORIES STABILITY CHECK (Threshold <= 15%)")
    print("=" * 90)
    non_affected = ["character_names"]
    for cat in non_affected:
        if cat in category_stats:
            stats = category_stats[cat]
            diff_pct = abs(stats["pct_change"])
            status = "PASS" if diff_pct <= 15.0 else "EXCEEDS THRESHOLD"
            print(f"  {cat}: {diff_pct:.2f}% difference -> [{status}]")

    # Overall excluding out_of_domain
    all_b = [q["warm_median_ms"] for q in before_data["queries"] if q["category"] != "out_of_domain"]
    all_a = [q["warm_median_ms"] for q in after_data["queries"] if q["category"] != "out_of_domain"]
    overall_b_med = statistics.median(all_b)
    overall_a_med = statistics.median(all_a)
    overall_pct = ((overall_b_med - overall_a_med) / overall_b_med) * 100

    print("\n" + "=" * 90)
    print(f"OVERALL IN-DOMAIN SPEEDUP (excluding out_of_domain):")
    print(f"  Before Median: {overall_b_med:.1f} ms")
    print(f"  After Median:  {overall_a_med:.1f} ms")
    print(f"  Delta:         {overall_a_med - overall_b_med:+.1f} ms")
    print(f"  Speedup:       {overall_pct:+.1f}%")
    print("=" * 90)

if __name__ == "__main__":
    main()
