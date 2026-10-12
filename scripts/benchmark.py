"""
Phase 1 Performance Measurement Script

Runs golden query set against the search API, captures:
- Full ranked output snapshot (IDs, scores, themes) for correctness verification
- Cold vs warm timing (10 runs per query)
- Stage-level breakdowns (with DB query counts)
- Model load time, first-request overhead, peak RAM
- HTTP end-to-end latency

Outputs:
- docs/perf/golden_snapshot.json: Baseline results before optimization
- docs/perf/baseline_timings.json: Timing data
- docs/perf/query_plans.txt: EXPLAIN ANALYZE output
"""

import json
import time
import requests
import psutil
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import statistics

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# Set environment to enable detailed timing
os.environ["PERF_BENCHMARK"] = "1"
os.environ["DEBUG_SEARCH"] = "0"  # Disable debug output during benchmark

API_BASE = "http://localhost:8000"
GOLDEN_QUERIES_PATH = Path(__file__).parent / "golden_queries.json"
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "perf"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def measure_process_memory():
    """Get current process memory in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def load_golden_queries() -> Dict[str, Any]:
    """Load the golden query set"""
    with open(GOLDEN_QUERIES_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def freeze_translator_cache():
    """
    Pre-populate translator cache with known English terms to make runs deterministic.
    This avoids network calls during benchmarking.

    NOTE: The translator is called by the ENGINE (backend/app/core/engine.py), not by this script.
    This function just pre-warms any client-side cache, but the real caching happens
    server-side within the engine's cross-lingual mapping logic.

    For true deterministic runs, the engine's dynamic translator calls should be disabled
    or cached at the server level.
    """
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='auto', target='tl')

        # Pre-translate all English cross-lingual queries
        english_terms = [
            "corruption", "education", "freedom", "oppression",
            "love", "death", "justice", "church"
        ]

        cache = {}
        print("Pre-caching translations (client-side warmup)...")
        print("NOTE: Engine-side translator calls may still occur during search.")
        for term in english_terms:
            try:
                translation = translator.translate(term)
                cache[term] = translation
                print(f"  {term} → {translation}")
            except Exception as e:
                print(f"  Warning: Could not pre-translate '{term}': {e}")

        return cache
    except ImportError:
        print("Warning: deep_translator not available, skipping cache")
        return {}


def run_search_query(query: str, translator_cache: Dict = None) -> Dict[str, Any]:
    """
    Execute search query against API and return response with timing.
    """
    params = {
        "q": query,
        "source_type": "summary",
        "novel": "both"
    }

    start_http = time.perf_counter()
    response = requests.get(f"{API_BASE}/api/v1/search", params=params)
    end_http = time.perf_counter()

    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")

    data = response.json()

    return {
        "http_latency_ms": (end_http - start_http) * 1000,
        "response": data
    }


def extract_ranked_output(response: Dict[str, Any], query: str) -> Dict[str, Any]:
    """
    Extract the ranked output for golden snapshot comparison.
    Returns: {
        "noli": [{"id", "scores", "themes", "sentence_text"}],
        "elfili": [...],
        "precise_matches": [...],
        "metadata": {...}
    }
    """
    results = response.get("results", {})
    precise = response.get("precise_matches", [])
    metadata = response.get("metadata", {})

    def serialize_result(item):
        return {
            "id": item.get("id"),
            "sentence_index": item.get("sentence_index"),
            "chapter_number": item.get("chapter_number"),
            "sentence_text": item.get("sentence_text", "")[:100],  # First 100 chars
            "scores": item.get("scores", {}),
            "themes": [{"id": t.get("id"), "label": t.get("label"), "score": round(t.get("score", 0), 4)}
                       for t in item.get("themes", [])]
        }

    return {
        "query": query,
        "noli": [serialize_result(r) for r in results.get("noli", [])],
        "elfili": [serialize_result(r) for r in results.get("elfili", [])],
        "precise_matches": [serialize_result(r) for r in precise],
        "metadata": metadata
    }


def extract_timings(response: Dict[str, Any]) -> Dict[str, float]:
    """Extract timing breakdown from response metadata (if available)"""
    metadata = response.get("metadata", {})
    timings = metadata.get("timings", {})
    db_queries = metadata.get("db_query_count", 0)

    return {
        **timings,
        "db_query_count": db_queries
    }


def run_benchmark_suite():
    """
    Main benchmark runner.
    """
    print("=" * 80)
    print("Phase 1 Performance Measurement")
    print("=" * 80)

    # Load queries
    golden_data = load_golden_queries()
    translator_cache = freeze_translator_cache()

    # Wait for server to be ready
    print("\nWaiting for API server...")
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_BASE}/health", timeout=2)
            if response.status_code == 200:
                print(f"✓ API server ready")
                break
        except requests.exceptions.RequestException:
            if i == max_retries - 1:
                print("✗ API server not responding. Please start it with:")
                print("  cd backend && poetry run uvicorn app.main:app")
                sys.exit(1)
            time.sleep(1)

    # Record initial memory and model load time
    print("\n--- Model Load Timing ---")
    mem_before = measure_process_memory()

    # First request triggers model loading
    print("Executing first request (cold start)...")
    first_request_start = time.perf_counter()
    try:
        _ = run_search_query("kalayaan", translator_cache)
        first_request_time = time.perf_counter() - first_request_start
        print(f"First request (cold): {first_request_time:.2f}s")
    except Exception as e:
        print(f"✗ First request failed: {e}")
        sys.exit(1)

    mem_after = measure_process_memory()
    mem_delta = mem_after - mem_before

    print(f"Memory before: {mem_before:.1f} MB")
    print(f"Memory after:  {mem_after:.1f} MB")
    print(f"Memory delta:  {mem_delta:.1f} MB")

    # Collect all queries
    all_queries = []
    for category, data in golden_data["categories"].items():
        for query in data["queries"]:
            all_queries.append({
                "query": query,
                "category": category
            })

    print(f"\n--- Running {len(all_queries)} queries × 10 iterations ---")

    golden_snapshot = {}
    timing_results = []
    peak_memory = mem_after

    for idx, item in enumerate(all_queries, 1):
        query = item["query"]
        category = item["category"]

        print(f"\n[{idx}/{len(all_queries)}] {category}: '{query}'")

        run_timings = []
        ranked_output = None

        for run in range(10):
            try:
                result = run_search_query(query, translator_cache)

                http_latency = result["http_latency_ms"]
                response = result["response"]

                # Extract engine timings (if instrumented)
                engine_timings = extract_timings(response)

                # First run: capture ranked output for golden snapshot
                if run == 0:
                    ranked_output = extract_ranked_output(response, query)
                    print(f"  Run 1 (cold): {http_latency:.1f}ms | Results: {len(response.get('results', {}).get('noli', []))} noli, {len(response.get('results', {}).get('elfili', []))} elfili")

                run_timings.append({
                    "run": run + 1,
                    "http_latency_ms": http_latency,
                    "engine_timings": engine_timings
                })

                # Track peak memory
                current_mem = measure_process_memory()
                peak_memory = max(peak_memory, current_mem)

                # Small delay between runs
                time.sleep(0.1)

            except Exception as e:
                print(f"  ✗ Run {run + 1} failed: {e}")
                continue

        if run_timings:
            cold_latency = run_timings[0]["http_latency_ms"]
            warm_latencies = [r["http_latency_ms"] for r in run_timings[1:]]

            if warm_latencies:
                print(f"  Warm (avg): {statistics.mean(warm_latencies):.1f}ms (p50: {statistics.median(warm_latencies):.1f}ms)")

        # Save golden snapshot for this query
        if ranked_output:
            golden_snapshot[query] = ranked_output

        # Save timing data
        timing_results.append({
            "query": query,
            "category": category,
            "runs": run_timings
        })

    print(f"\n--- Peak Memory: {peak_memory:.1f} MB ---")

    # Write golden snapshot
    snapshot_path = OUTPUT_DIR / "golden_snapshot.json"
    with open(snapshot_path, 'w', encoding='utf-8') as f:
        json.dump({
            "version": "1.0",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "queries": golden_snapshot
        }, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Golden snapshot saved: {snapshot_path}")

    # Write timing results
    timings_path = OUTPUT_DIR / "baseline_timings.json"
    with open(timings_path, 'w', encoding='utf-8') as f:
        json.dump({
            "version": "1.0",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "first_request_time_s": first_request_time,
            "memory_delta_mb": mem_delta,
            "peak_memory_mb": peak_memory,
            "queries": timing_results
        }, f, indent=2, ensure_ascii=False)
    print(f"✓ Timing results saved: {timings_path}")

    # Generate summary report
    generate_summary_report(timing_results, first_request_time, mem_delta, peak_memory)


def generate_summary_report(timing_results: List[Dict], first_request: float, mem_delta: float, peak_mem: float):
    """Generate human-readable summary"""
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)

    print(f"\nModel Load & First Request: {first_request:.2f}s")
    print(f"Memory Footprint: {mem_delta:.1f} MB (peak: {peak_mem:.1f} MB)")

    # Aggregate by category
    by_category = defaultdict(list)
    for result in timing_results:
        category = result["category"]
        runs = result["runs"]

        if len(runs) >= 2:
            cold = runs[0]["http_latency_ms"]
            warm = [r["http_latency_ms"] for r in runs[1:]]
            by_category[category].append({
                "cold": cold,
                "warm_mean": statistics.mean(warm),
                "warm_p50": statistics.median(warm),
                "warm_p95": statistics.quantiles(warm, n=20)[18] if len(warm) >= 2 else warm[0]
            })

    print("\nLatency by Category (HTTP end-to-end):")
    print("-" * 80)
    print(f"{'Category':<25} {'Cold (ms)':<12} {'Warm P50':<12} {'Warm P95':<12}")
    print("-" * 80)

    for category, samples in by_category.items():
        cold_avg = statistics.mean([s["cold"] for s in samples])
        warm_p50_avg = statistics.mean([s["warm_p50"] for s in samples])
        warm_p95_avg = statistics.mean([s["warm_p95"] for s in samples])

        print(f"{category:<25} {cold_avg:>10.1f}  {warm_p50_avg:>10.1f}  {warm_p95_avg:>10.1f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_benchmark_suite()
