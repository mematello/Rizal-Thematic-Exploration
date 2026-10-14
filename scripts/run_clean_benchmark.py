"""
Clean Benchmark Runner with Two Full Warmup Passes
Runs 40 queries x 10 runs after 2 complete in-domain warmup passes.
Captures CPU load, server RSS, free RAM, and outputs warm P50/P95 per query and per category.
Also reports Out-of-Domain split into first-seen vs repeated.
"""
import os
import sys
import time
import json
import statistics
import requests
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

GOLDEN_QUERIES_PATH = Path(__file__).parent / "golden_queries.json"
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "perf"

def get_system_load(server_proc: Optional[psutil.Process] = None):
    cpu_percent = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    server_rss = None
    if server_proc and server_proc.is_running():
        try:
            server_rss = round(server_proc.memory_info().rss / 1024 / 1024, 2)
        except Exception:
            server_rss = None

    return {
        "cpu_percent": cpu_percent,
        "available_ram_mb": round(mem.available / 1024 / 1024, 1),
        "total_ram_mb": round(mem.total / 1024 / 1024, 1),
        "percent_ram_used": mem.percent,
        "server_rss_mb": server_rss
    }

def find_server_process(port: int) -> Optional[psutil.Process]:
    try:
        for conn in psutil.net_connections(kind='inet'):
            if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                return psutil.Process(conn.pid)
    except Exception:
        pass
    return None

def run_query(session: requests.Session, api_base: str, query: str):
    params = {"q": query, "source_type": "summary", "novel": "both"}
    t0 = time.perf_counter()
    resp = session.get(f"{api_base}/api/v1/search", params=params, timeout=120)
    t1 = time.perf_counter()
    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}: {resp.text}")
    return (t1 - t0) * 1000, resp.json()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--server-pid", type=int, default=None)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--label", type=str, default="benchmark")
    args = parser.parse_args()

    api_base = f"http://127.0.0.1:{args.port}"
    print("=" * 80)
    print(f"CLEAN BENCHMARK: {args.label}")
    print(f"API Base: {api_base}")
    print(f"Output File: {args.out}")
    print("=" * 80)

    session = requests.Session()

    # Verify server is up
    print("Checking server health...")
    for _ in range(15):
        try:
            r = session.get(f"{api_base}/health", timeout=2)
            if r.status_code == 200:
                print("[OK] Server healthy!")
                break
        except Exception:
            time.sleep(1)
    else:
        print("[FAIL] Server not reachable!")
        sys.exit(1)

    server_proc = None
    if args.server_pid:
        try:
            server_proc = psutil.Process(args.server_pid)
            print(f"[OK] Found server process PID: {args.server_pid} ({server_proc.name()})")
        except Exception as e:
            print(f"[WARN] Could not attach to server PID {args.server_pid}: {e}")
    if not server_proc:
        server_proc = find_server_process(args.port)
        if server_proc:
            print(f"[OK] Discovered server process PID: {server_proc.pid} ({server_proc.name()})")
        else:
            print("[WARN] Server process could not be identified for RSS tracking.")

    with open(GOLDEN_QUERIES_PATH, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    all_queries = []
    in_domain_queries = []
    for cat, cdata in golden_data["categories"].items():
        for q in cdata["queries"]:
            item = {"query": q, "category": cat}
            all_queries.append(item)
            if cat != "out_of_domain":
                in_domain_queries.append(item)

    # Record initial system load
    load_before = get_system_load(server_proc)
    print(f"\nSystem Load Before Run:")
    print(f"  CPU Utilization: {load_before['cpu_percent']}%")
    print(f"  Available RAM:   {load_before['available_ram_mb']} MB / {load_before['total_ram_mb']} MB ({load_before['percent_ram_used']}% used)")
    if load_before['server_rss_mb'] is not None:
        print(f"  Server RSS:      {load_before['server_rss_mb']} MB")

    # Warmup Pass 1 (in-domain queries)
    print(f"\n--- Warmup Pass 1 ({len(in_domain_queries)} in-domain queries) ---")
    t_w1 = time.perf_counter()
    for idx, item in enumerate(in_domain_queries, 1):
        q = item["query"]
        ms, _ = run_query(session, api_base, q)
        print(f"  [W1 {idx:02d}/{len(in_domain_queries)}] {q:<30}: {ms:7.1f} ms")
    print(f"Warmup Pass 1 completed in {(time.perf_counter() - t_w1):.1f}s")

    # Warmup Pass 2 (in-domain queries)
    print(f"\n--- Warmup Pass 2 ({len(in_domain_queries)} in-domain queries) ---")
    t_w2 = time.perf_counter()
    for idx, item in enumerate(in_domain_queries, 1):
        q = item["query"]
        ms, _ = run_query(session, api_base, q)
        print(f"  [W2 {idx:02d}/{len(in_domain_queries)}] {q:<30}: {ms:7.1f} ms")
    print(f"Warmup Pass 2 completed in {(time.perf_counter() - t_w2):.1f}s")

    # System load after warmups
    load_warm = get_system_load(server_proc)
    print(f"\nSystem Load After Warmups:")
    print(f"  CPU Utilization: {load_warm['cpu_percent']}%")
    print(f"  Available RAM:   {load_warm['available_ram_mb']} MB ({load_warm['percent_ram_used']}% used)")
    if load_warm['server_rss_mb'] is not None:
        print(f"  Server RSS:      {load_warm['server_rss_mb']} MB")

    # Official Measurement Pass: 40 queries x 10 runs
    print(f"\n--- Official Measurement Runs (40 queries x 10 runs) ---")
    results = []
    cpu_samples = []
    server_rss_samples = []

    for idx, item in enumerate(all_queries, 1):
        q = item["query"]
        cat = item["category"]
        print(f"\n[{idx:02d}/40] {cat}: '{q}'")
        runs = []

        for r in range(10):
            ms, data = run_query(session, api_base, q)
            meta = data.get("metadata", {})
            eng_timings = meta.get("engine_timings", {})
            runs.append({
                "run": r + 1,
                "http_latency_ms": ms,
                "engine_timings": eng_timings
            })
            cpu_samples.append(psutil.cpu_percent(interval=None))
            if server_proc and server_proc.is_running():
                try:
                    server_rss_samples.append(round(server_proc.memory_info().rss / 1024 / 1024, 2))
                except Exception:
                    pass
            time.sleep(0.05)

        latencies = [run["http_latency_ms"] for run in runs]
        # Run 1 is cold / first-seen within measurement block; runs 2-10 are warm / repeated
        cold_latency = latencies[0]
        warm_latencies = latencies[1:]
        min_lat = min(latencies)
        med_lat = statistics.median(latencies)
        max_lat = max(latencies)
        warm_med = float(np.percentile(warm_latencies, 50))
        warm_p95 = float(np.percentile(warm_latencies, 95))

        print(f"  Cold (Run 1): {cold_latency:6.1f} ms | Warm P50: {warm_med:6.1f} ms | Warm P95: {warm_p95:6.1f} ms | Min: {min_lat:6.1f} ms | Max: {max_lat:6.1f} ms")

        results.append({
            "query": q,
            "category": cat,
            "cold_run1_ms": round(cold_latency, 2),
            "min_ms": round(min_lat, 2),
            "median_ms": round(med_lat, 2),
            "max_ms": round(max_lat, 2),
            "warm_p50_ms": round(warm_med, 2),
            "warm_p95_ms": round(warm_p95, 2),
            "runs": runs
        })

    load_after = get_system_load(server_proc)
    avg_cpu = statistics.mean(cpu_samples) if cpu_samples else load_after['cpu_percent']
    avg_server_rss = statistics.mean(server_rss_samples) if server_rss_samples else load_after['server_rss_mb']

    # Per-category warm aggregation (Runs 2-10 pooled across all queries in category)
    categories = golden_data["categories"].keys()
    cat_summary = {}
    for cat in categories:
        cat_queries = [r for r in results if r["category"] == cat]
        pooled_warm = []
        for cq in cat_queries:
            pooled_warm.extend([run["http_latency_ms"] for run in cq["runs"][1:]])

        cat_summary[cat] = {
            "query_count": len(cat_queries),
            "warm_sample_count": len(pooled_warm),
            "warm_p50_ms": round(float(np.percentile(pooled_warm, 50)), 2),
            "warm_p95_ms": round(float(np.percentile(pooled_warm, 95)), 2),
            "min_warm_ms": round(min(pooled_warm), 2),
            "max_warm_ms": round(max(pooled_warm), 2)
        }

    # Out-of-Domain detailed split: first-seen (Run 1) vs repeated (Runs 2-10)
    ood_queries = [r for r in results if r["category"] == "out_of_domain"]
    ood_first_seen = [r["runs"][0]["http_latency_ms"] for r in ood_queries]
    ood_repeated = []
    for r in ood_queries:
        ood_repeated.extend([run["http_latency_ms"] for run in r["runs"][1:]])

    ood_split = {
        "first_seen_run1": {
            "sample_count": len(ood_first_seen),
            "p50_ms": round(float(np.percentile(ood_first_seen, 50)), 2),
            "p95_ms": round(float(np.percentile(ood_first_seen, 95)), 2),
            "min_ms": round(min(ood_first_seen), 2),
            "max_ms": round(max(ood_first_seen), 2),
            "per_query": {r["query"]: round(r["runs"][0]["http_latency_ms"], 2) for r in ood_queries}
        },
        "repeated_runs2_10": {
            "sample_count": len(ood_repeated),
            "p50_ms": round(float(np.percentile(ood_repeated, 50)), 2),
            "p95_ms": round(float(np.percentile(ood_repeated, 95)), 2),
            "min_ms": round(min(ood_repeated), 2),
            "max_ms": round(max(ood_repeated), 2),
            "per_query_p50": {
                r["query"]: round(float(np.percentile([run["http_latency_ms"] for run in r["runs"][1:]], 50)), 2)
                for r in ood_queries
            }
        }
    }

    out_path = OUTPUT_DIR / args.out
    payload = {
        "label": args.label,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "server_rss": {
            "initial_mb": load_before["server_rss_mb"],
            "post_warmup_mb": load_warm["server_rss_mb"],
            "post_benchmark_mb": load_after["server_rss_mb"],
            "avg_during_run_mb": round(avg_server_rss, 2) if avg_server_rss else None
        },
        "free_ram": {
            "initial_available_mb": load_before["available_ram_mb"],
            "post_warmup_available_mb": load_warm["available_ram_mb"],
            "post_benchmark_available_mb": load_after["available_ram_mb"],
            "total_ram_mb": load_before["total_ram_mb"]
        },
        "system_load": {
            "load_before": load_before,
            "load_warm": load_warm,
            "load_after": load_after,
            "avg_cpu_during_run": round(avg_cpu, 1)
        },
        "category_summary": cat_summary,
        "out_of_domain_split": ood_split,
        "queries": results
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"[OK] Saved results to {out_path}")
    print(f"Average CPU during runs: {avg_cpu:.1f}%")
    print(f"Free RAM (available): {load_after['available_ram_mb']} MB / {load_after['total_ram_mb']} MB")
    if load_after['server_rss_mb'] is not None:
        print(f"Server Process RSS:   {load_after['server_rss_mb']} MB")
    print("=" * 80)

    # Print Category Table
    print("\n--- PER-CATEGORY WARM SUMMARY (Runs 2-10, Excluded Run 1) ---")
    print(f"{'Category':<25} | {'Warm P50 (ms)':<15} | {'Warm P95 (ms)':<15} | {'Min Warm (ms)':<15} | {'Max Warm (ms)':<15}")
    print("-" * 90)
    for cat, data in cat_summary.items():
        print(f"{cat:<25} | {data['warm_p50_ms']:>15.2f} | {data['warm_p95_ms']:>15.2f} | {data['min_warm_ms']:>15.2f} | {data['max_warm_ms']:>15.2f}")

    print("\n--- OUT-OF-DOMAIN BREAKDOWN: First-Seen vs Repeated ---")
    print(f"First-Seen (Run 1):      P50 = {ood_split['first_seen_run1']['p50_ms']:.2f} ms | P95 = {ood_split['first_seen_run1']['p95_ms']:.2f} ms | Min = {ood_split['first_seen_run1']['min_ms']:.2f} ms | Max = {ood_split['first_seen_run1']['max_ms']:.2f} ms")
    print(f"Repeated (Runs 2-10):    P50 = {ood_split['repeated_runs2_10']['p50_ms']:.2f} ms | P95 = {ood_split['repeated_runs2_10']['p95_ms']:.2f} ms | Min = {ood_split['repeated_runs2_10']['min_ms']:.2f} ms | Max = {ood_split['repeated_runs2_10']['max_ms']:.2f} ms")
    print("=" * 80)

if __name__ == "__main__":
    main()
