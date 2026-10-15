#!/usr/bin/env python3
"""Targeted benchmark: 5 queries x 5 runs each, PERF_BENCHMARK=1 required."""
import os, sys, json, time, requests
os.environ["PERF_BENCHMARK"] = "1"
queries = {
    "kalayaan": {"mode":"semantic_fallback","expected_slow":True},
    "edukasyon": {"mode":"semantic_fallback","expected_slow":True},
    "bayan": {"mode":"lexical","expected_slow":False},
    "tiktok": {"mode":"early_reject","expected_slow":False},
    "oppression": {"mode":"lexical","expected_slow":False}
}
results = {}
for q, info in queries.items():
    runs = []
    for i in range(5):
        t0 = time.perf_counter()
        r = requests.get(f"http://127.0.0.1:8000/api/v1/search?q={q}", timeout=60)
        lat = (time.perf_counter() - t0) * 1000
        d = r.json()
        m = d.get("metadata", {})
        runs.append({"run": i+1, "ms": round(lat,1), "mode": m.get("result_mode"), "db_queries": m.get("db_query_count"), "timings": m.get("engine_timings")})
    results[q] = {"runs": runs, "expected_slow": info["expected_slow"]}
    # Print concise per-query
    avg = sum(x["ms"] for x in runs) / len(runs)
    print(f"{q}: avg={avg:.0f}ms mode={runs[0]['mode']} db={runs[0]['db_queries']} timings={list(runs[0]['timings'].keys()) if runs[0]['timings'] else 'NONE'}")
with open("docs/perf/targeted_benchmark.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("[OK] Wrote docs/perf/targeted_benchmark.json")
