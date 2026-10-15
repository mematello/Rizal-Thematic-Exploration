"""
Analyzes and compares Before (cache disabled) and After (cache enabled) benchmark timings
with the fixed client (127.0.0.1 + requests.Session).

Computes per-category:
- Cold latency (Run 1)
- Warm P50 (Runs 2-10)
- Warm P95 (Runs 2-10)
Both per-query averaged over category and pooled across category samples.
"""

import json
import statistics
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

PERF_DIR = Path(__file__).parent.parent / "docs" / "perf"

def load_timings(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("queries", [])

def compute_metrics(queries: List[Dict[str, Any]]):
    # Group by category
    by_category = defaultdict(list)
    by_query = {}

    for q_item in queries:
        query = q_item["query"]
        category = q_item["category"]
        runs = q_item["runs"]

        if not runs:
            continue

        cold = runs[0]["http_latency_ms"]
        # Discard run 1 as warmup for warm statistics
        warm = [r["http_latency_ms"] for r in runs[1:]]

        warm_p50 = statistics.median(warm) if warm else cold
        if len(warm) >= 20:
            warm_p95 = statistics.quantiles(warm, n=20)[18]
        elif len(warm) >= 2:
            # For 9 warm runs, 95th percentile with linear interpolation or max
            # quantiles with n=20 gives 19 cut points
            warm_sorted = sorted(warm)
            # 95th percentile index in 9 items: 0.95 * 8 = 7.6 -> index 8 (max)
            warm_p95 = warm_sorted[-1]
        else:
            warm_p95 = warm[0] if warm else cold

        # Also get engine search_total_ms if available
        eng_warm_total = [r.get("engine_timings", {}).get("search_total_ms") for r in runs[1:] if "engine_timings" in r and "search_total_ms" in r.get("engine_timings", {})]
        eng_p50 = statistics.median(eng_warm_total) if eng_warm_total else None

        q_stat = {
            "query": query,
            "category": category,
            "cold": cold,
            "warm_samples": warm,
            "warm_p50": warm_p50,
            "warm_p95": warm_p95,
            "eng_p50": eng_p50,
        }
        by_query[query] = q_stat
        by_category[category].append(q_stat)

    cat_summary = {}
    for cat, items in by_category.items():
        # Averaged across queries in category
        cold_avg = statistics.mean([it["cold"] for it in items])
        warm_p50_avg = statistics.mean([it["warm_p50"] for it in items])
        warm_p95_avg = statistics.mean([it["warm_p95"] for it in items])

        # Also pooled across all warm runs in category
        pooled_warm = [s for it in items for s in it["warm_samples"]]
        pooled_p50 = statistics.median(pooled_warm) if pooled_warm else 0
        if len(pooled_warm) >= 20:
            pooled_p95 = statistics.quantiles(pooled_warm, n=20)[18]
        elif len(pooled_warm) >= 2:
            pooled_sorted = sorted(pooled_warm)
            idx = int(0.95 * (len(pooled_sorted) - 1))
            pooled_p95 = pooled_sorted[idx]
        else:
            pooled_p95 = pooled_warm[0] if pooled_warm else 0

        cat_summary[cat] = {
            "cold_avg": cold_avg,
            "warm_p50_avg": warm_p50_avg,
            "warm_p95_avg": warm_p95_avg,
            "pooled_p50": pooled_p50,
            "pooled_p95": pooled_p95,
            "items": items
        }

    return cat_summary, by_query

def format_comparison(before_path: Path, after_path: Path):
    before_cat, before_q = compute_metrics(load_timings(before_path))
    after_cat, after_q = compute_metrics(load_timings(after_path))

    categories = list(before_cat.keys())

    print("=" * 110)
    print("CORRECTED BEFORE vs. AFTER COMPARISON (Fixed Client: 127.0.0.1 + requests.Session)")
    print("Warm stats calculated over Runs 2-10 (Run 1 discarded as warmup)")
    print("=" * 110)
    print(f"{'Category':<25} | {'Metric':<10} | {'Before (No Cache)':<18} | {'After (Word Cache)':<18} | {'Delta':<12} | {'Change':<8}")
    print("-" * 110)

    for cat in categories:
        b = before_cat.get(cat, {})
        a = after_cat.get(cat, {})

        for metric, b_val, a_val in [
            ("Cold", b.get("cold_avg", 0), a.get("cold_avg", 0)),
            ("Warm P50", b.get("warm_p50_avg", 0), a.get("warm_p50_avg", 0)),
            ("Warm P95", b.get("warm_p95_avg", 0), a.get("warm_p95_avg", 0)),
        ]:
            delta = a_val - b_val
            pct = (delta / b_val * 100) if b_val > 0 else 0
            sign = "+" if delta > 0 else ""
            print(f"{cat:<25} | {metric:<10} | {b_val:>14.1f} ms | {a_val:>14.1f} ms | {sign}{delta:>9.1f} ms | {sign}{pct:>6.1f}%")
        print("-" * 110)

    print("\n" + "=" * 110)
    print("KEY INDIVIDUAL QUERIES (HTTP Latency)")
    print("=" * 110)
    print(f"{'Query':<15} | {'Category':<22} | {'Before Cold':<12} | {'After Cold':<12} | {'Before P50':<12} | {'After P50':<12} | {'P50 Delta':<10}")
    print("-" * 110)
    key_queries = ["kalayaan", "edukasyon", "pag-ibig", "kamatayan", "bayan", "simbahan", "oppression", "education", "tiktok", "korapsyon"]
    for q in key_queries:
        if q in before_q and q in after_q:
            bq = before_q[q]
            aq = after_q[q]
            delta = aq["warm_p50"] - bq["warm_p50"]
            sign = "+" if delta > 0 else ""
            print(f"{q:<15} | {bq['category']:<22} | {bq['cold']:>9.1f} ms | {aq['cold']:>9.1f} ms | {bq['warm_p50']:>9.1f} ms | {aq['warm_p50']:>9.1f} ms | {sign}{delta:>8.1f} ms")
    print("=" * 110)

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        b_p = Path(sys.argv[1])
        a_p = Path(sys.argv[2])
    else:
        b_p = PERF_DIR / "fixed_client_before_timings.json"
        a_p = PERF_DIR / "fixed_client_after_timings.json"

    format_comparison(b_p, a_p)
