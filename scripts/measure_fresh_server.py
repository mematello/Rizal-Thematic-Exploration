"""
Fresh Server Startup & 10 Unseen Words Latency Measurement
Measures:
  - Startup time to health check ready
  - Server process RSS immediately at startup
  - First-search latency on fresh boot for 10 unseen Tagalog words
  - Server process RSS after all searches
  - Clean shutdown
"""
import os
import sys
import time
import json
import subprocess
import requests
import psutil
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent / "backend"
docs_perf_dir = Path(__file__).resolve().parent.parent / "docs" / "perf"
sys.path.insert(0, str(backend_dir))

UNSEEN_TAGALOG_WORDS = [
    "kasakiman",    # greed (previously 20,603 ms cold!)
    "alipin",       # slave (previously 12,583 ms cold!)
    "dangal",       # honor (previously 5,706 ms cold!)
    "takot",        # fear (lexical match)
    "sakripisyo",   # sacrifice (cross-lingual / unmapped)
    "katapangan",   # bravery
    "kabayanihan",  # heroism
    "kalupitan",    # cruelty
    "pagdurusa",    # suffering
    "himagsikan"    # revolution
]

def find_server_process(port: int = 8002):
    for conn in psutil.net_connections(kind='inet'):
        if conn.status == psutil.CONN_LISTEN and conn.laddr.port == port:
            try:
                proc = psutil.Process(conn.pid)
                if proc.is_running():
                    return proc
            except Exception:
                pass
    return None

def main():
    port = 8002
    api_base = f"http://127.0.0.1:{port}"

    print("=" * 95)
    print("STEP A VERIFICATION (c) & (d): Fresh Server Boot, RSS, and 10 Unseen Words")
    print(f"API Base: {api_base}")
    print("=" * 95)

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["PERF_BENCHMARK"] = "1"
    env["DISABLE_WORD_VEC_CACHE"] = "0"
    env["DISABLE_TRANSLATOR_CACHE"] = "0"
    env["DISABLE_BATCHED_CONTEXT_FETCH"] = "0"

    uvicorn_exe = str(backend_dir / ".venv" / "Scripts" / "uvicorn.exe")

    print("\n[1/4] Starting fresh Uvicorn server...")
    t_start = time.perf_counter()
    server_proc = subprocess.Popen(
        [uvicorn_exe, "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(backend_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    session = requests.Session()
    server_ready = False
    startup_time_s = 0.0

    # Poll until ready
    while time.perf_counter() - t_start < 30:
        try:
            r = session.get(f"{api_base}/health", timeout=1)
            if r.status_code == 200:
                startup_time_s = time.perf_counter() - t_start
                server_ready = True
                break
        except Exception:
            pass
        time.sleep(0.1)

    if not server_ready:
        print("[FAIL] Server failed to start within 30 seconds!")
        server_proc.kill()
        sys.exit(1)

    print(f"[PASS] Server ready in {startup_time_s:.2f} s")

    # Locate child process listening on port 8002
    actual_proc = find_server_process(port) or server_proc
    initial_rss_mb = round(actual_proc.memory_info().rss / 1024 / 1024, 2)
    print(f"[INFO] Server PID: {actual_proc.pid} | Initial RSS: {initial_rss_mb:.2f} MB")

    # Run search on first search of 10 unseen Tagalog words
    print(f"\n[2/4] Testing first search of 10 unseen Tagalog words on freshly booted server...")
    results = []

    for idx, word in enumerate(UNSEEN_TAGALOG_WORDS, 1):
        # First search (Cold for this query on fresh server)
        t0 = time.perf_counter()
        r1 = session.get(f"{api_base}/api/v1/search", params={"q": word, "source_type": "summary", "novel": "both"}, timeout=60)
        t_cold = (time.perf_counter() - t0) * 1000
        d1 = r1.json()
        meta1 = d1.get("metadata", {})
        eng1 = meta1.get("engine_timings", {})
        res1 = d1.get("results", {})
        c_count = len(res1.get("noli", [])) + len(res1.get("elfili", []))

        # Immediate second search (Warm)
        t0 = time.perf_counter()
        r2 = session.get(f"{api_base}/api/v1/search", params={"q": word, "source_type": "summary", "novel": "both"}, timeout=60)
        t_warm = (time.perf_counter() - t0) * 1000
        d2 = r2.json()
        meta2 = d2.get("metadata", {})
        eng2 = meta2.get("engine_timings", {})

        rerank_ms = eng1.get("rerank_loop", 0)
        hits = eng1.get("process_cache_hits", 0)
        misses = eng1.get("process_cache_misses", 0)
        encodes = eng1.get("encode_calls_total", 0)

        print(f"  [{idx:02d}/10] {word:<15} | Cold: {t_cold:7.2f} ms | Warm: {t_warm:7.2f} ms | Rerank: {rerank_ms:6.2f} ms | Hits: {hits:3d} | Misses: {misses:2d} | Mode: {meta1.get('result_mode')}")

        results.append({
            "word": word,
            "cold_latency_ms": round(t_cold, 2),
            "warm_latency_ms": round(t_warm, 2),
            "result_mode": meta1.get("result_mode"),
            "candidate_count": c_count,
            "cold_rerank_ms": round(rerank_ms, 2),
            "process_cache_hits": hits,
            "process_cache_misses": misses,
            "encode_calls_total": encodes,
            "cold_timings": eng1,
            "warm_timings": eng2
        })

    post_search_rss_mb = round(actual_proc.memory_info().rss / 1024 / 1024, 2)
    print(f"\n[3/4] Server RSS after 10 searches: {post_search_rss_mb:.2f} MB (Delta: +{post_search_rss_mb - initial_rss_mb:.2f} MB)")

    # Clean shutdown
    print("[4/4] Shutting down server...")
    try:
        server_proc.terminate()
        server_proc.wait(timeout=5)
    except Exception:
        server_proc.kill()

    summary_data = {
        "startup_time_seconds": round(startup_time_s, 2),
        "initial_rss_mb": initial_rss_mb,
        "post_search_rss_mb": post_search_rss_mb,
        "results": results
    }

    out_file = docs_perf_dir / "step_a_unseen_10_words.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to {out_file}")
    print("\n" + "=" * 95)
    print(f"{'Word':<15} | {'Cold Latency (ms)':<18} | {'Warm Latency (ms)':<18} | {'Rerank Loop (ms)':<18} | {'Result Mode'}")
    print("-" * 95)
    for r in results:
        print(f"{r['word']:<15} | {r['cold_latency_ms']:>18.2f} | {r['warm_latency_ms']:>18.2f} | {r['cold_rerank_ms']:>18.2f} | {r['result_mode']}")
    print("=" * 95)

if __name__ == "__main__":
    main()
