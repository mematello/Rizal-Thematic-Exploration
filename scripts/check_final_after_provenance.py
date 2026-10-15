import json
import numpy as np
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "docs" / "perf" / "final_after_timings.json"

with open(json_path, "r", encoding="utf-8") as f:
    d = json.load(f)

print("=" * 85)
print("PROVENANCE ANALYSIS OF final_after_timings.json")
print("=" * 85)

# Check top-level metadata
print("Top-level keys in JSON:", list(d.keys()))
for k in ["commit", "flags", "environment", "git_commit", "metadata"]:
    if k in d:
        print(f"  {k}: {d[k]}")

# Collect process_cache_size and RSS
cache_sizes = []
first_run_cache = {}
rss_values = []
flag_fields = set()

for q in d.get("queries", []):
    q_name = q.get("query")
    runs = q.get("runs", [])
    for idx, r in enumerate(runs):
        for k in r.keys():
            if "disable" in k.lower() or "flag" in k.lower() or "commit" in k.lower():
                flag_fields.add(k)

        cs = r.get("process_cache_size")
        if cs is not None:
            cache_sizes.append(cs)
            if idx == 0:
                first_run_cache[q_name] = cs

        rss = r.get("server_rss_mb") or r.get("rss_mb") or r.get("server_rss")
        if rss is not None:
            rss_values.append(rss)

print(f"\nFlag/commit fields in run objects: {flag_fields if flag_fields else 'NONE'}")
if cache_sizes:
    print(f"process_cache_size across all runs: min={min(cache_sizes)}, max={max(cache_sizes)}")
else:
    print("process_cache_size field: NOT PRESENT in runs")

print(f"\nFirst-run process_cache_size for key queries:")
for q_name in ["kalayaan", "edukasyon", "pag-ibig", "kamatayan", "bayan", "church", "oppression", "tiktok"]:
    print(f"  {q_name:<20}: {first_run_cache.get(q_name, 'N/A')}")

# RSS analysis
print(f"\nRSS Values analysis:")
if rss_values:
    mean_rss = np.mean(rss_values)
    print(f"  Total RSS data points: {len(rss_values)}")
    print(f"  Mean RSS: {mean_rss:.2f} MB")
    print(f"  Min RSS:  {min(rss_values):.2f} MB")
    print(f"  Max RSS:  {max(rss_values):.2f} MB")
else:
    print("  No per-run RSS values found. Checking query level...")
    q_rss = [q.get("server_rss_mb") for q in d.get("queries", []) if q.get("server_rss_mb") is not None]
    if q_rss:
        print(f"  Found {len(q_rss)} query-level server_rss_mb values:")
        print(f"  Mean query-level RSS: {np.mean(q_rss):.2f} MB")
        print(f"  Min: {min(q_rss):.2f} MB, Max: {max(q_rss):.2f} MB")
    else:
        print("  Checking top-level system_info / rss...")
        print("  top-level:", {k: v for k, v in d.items() if "rss" in k.lower() or "mem" in k.lower()})

# Let's search all files in docs/perf for 756.27 and 864.43
print(f"\nSearching docs/ for '756.27' and '864.43':")
for p in (base_dir / "docs").rglob("*.md"):
    with open(p, "r", encoding="utf-8") as f:
        txt = f.read()
    if "756.27" in txt or "756" in txt:
        print(f"  Found 756 reference in {p.name}")
    if "864.43" in txt or "864" in txt:
        print(f"  Found 864 reference in {p.name}")
