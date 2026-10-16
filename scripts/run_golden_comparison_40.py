import urllib.request
import urllib.parse
import json
import time
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "http://127.0.0.1:8000"
GOLDEN_QUERIES_PATH = REPO_ROOT / "scripts" / "golden_queries.json"
DETERMINISTIC_SNAPSHOT_PATH = REPO_ROOT / "docs" / "perf" / "golden_snapshot_deterministic.json"
OUT_SNAPSHOT_PATH = REPO_ROOT / "docs" / "perf" / "golden_test_snapshot.json"

def load_all_40_queries():
    with open(GOLDEN_QUERIES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = []
    for cat_name, cat in data.get("categories", {}).items():
        queries.extend(cat.get("queries", []))
    assert len(queries) == 40, f"Expected 40 queries, got {len(queries)}"
    return queries

def fetch_search(query, retries=5):
    encoded = urllib.parse.quote(query)
    url = f"{BASE_URL}/api/v1/search?q={encoded}&novel=both"
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data
        except Exception as e:
            if attempt == retries:
                raise e
            print(f"[Retry {attempt+1}/{retries} on '{query}': {e}]", flush=True)
            time.sleep(3 * (attempt + 1))

def main():
    print("=== RUNNING ALL 40 GOLDEN SEARCH QUERIES ===")
    queries = load_all_40_queries()
    snapshot_queries = {}

    cache_file = OUT_SNAPSHOT_PATH.parent / "golden_partial_cache.json"
    if cache_file.exists():
        try:
            snapshot_queries = json.loads(cache_file.read_text(encoding="utf-8"))
            print(f"Loaded {len(snapshot_queries)} cached queries.")
        except Exception:
            snapshot_queries = {}

    for idx, q in enumerate(queries, 1):
        if q in snapshot_queries:
            noli_res = snapshot_queries[q].get("noli", [])
            fili_res = snapshot_queries[q].get("elfili", [])
            print(f"[{idx:02d}/40] '{q}': noli={len(noli_res)}, elfili={len(fili_res)} (cached)", flush=True)
            continue

        t0 = time.perf_counter()
        data = fetch_search(q)
        dur = (time.perf_counter() - t0) * 1000
        results = data.get("results", {})
        noli_res = results.get("noli", [])
        fili_res = results.get("elfili", [])
        
        snapshot_queries[q] = {
            "noli": noli_res,
            "elfili": fili_res
        }
        cache_file.write_text(json.dumps(snapshot_queries, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{idx:02d}/40] '{q}': noli={len(noli_res)}, elfili={len(fili_res)} ({dur:.1f}ms)", flush=True)

    new_snapshot = {
        "version": "golden_verification_run",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries": len(snapshot_queries),
        "queries": snapshot_queries
    }

    OUT_SNAPSHOT_PATH.write_text(json.dumps(new_snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSnapshot written to {OUT_SNAPSHOT_PATH}")

    # Now execute scripts/compare_golden.py
    print("\n=== RUNNING compare_golden.py AGAINST golden_snapshot_deterministic.json ===")
    cmd = [
        sys.executable,
        r"c:\Users\marcu\Desktop\thesis\scripts\compare_golden.py",
        str(OUT_SNAPSHOT_PATH),
        str(DETERMINISTIC_SNAPSHOT_PATH)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)

    assert res.returncode == 0, f"compare_golden.py exited with code {res.returncode}"
    print("\nALL 40 GOLDEN QUERIES INVARIANCE CHECK PASSED!")

if __name__ == "__main__":
    main()
