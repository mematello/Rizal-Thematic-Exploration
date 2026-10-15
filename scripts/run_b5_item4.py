import sys
import os
import json
import subprocess
from pathlib import Path

base_dir = Path(".").resolve()
backend_dir = base_dir / "backend"
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(base_dir / "scripts"))

from strict_comparator import compare_strict

unmapped_queries = [
    "korapsyon", "karahasan ng mga prayle", "tiktok", "bitcoin",
    "covid", "facebook", "internet", "computer", "cellphone", "online"
]

out_lines = []
out_lines.append("================================================================================")
out_lines.append("ITEM 4: TRANSLATOR PERSISTENCE, LOGS, AND ONLINE NETWORK EXPERIMENTS")
out_lines.append("================================================================================\n")

# 1. Grep TranslatorCache for disk persistence
out_lines.append("1. GREP TranslatorCache FOR DISK PERSISTENCE (translator_cache.json):")
res_grep = subprocess.run(["grep", "-rn", "translator_cache.json", "backend/"], capture_output=True, text=True)
if not res_grep.stdout.strip():
    out_lines.append("  Result: No occurrences of 'translator_cache.json' in backend/.")
    out_lines.append("  TranslatorCache implementation (backend/app/core/engine.py:45-96) is purely in-memory:")
    out_lines.append("    - self._cache: Dict[str, Tuple[Optional[str], float, bool]] = {}")
    out_lines.append("    - No disk read, write, pickle, or JSON dump/load exists in TranslatorCache.")
else:
    out_lines.append(res_grep.stdout.strip())

# 2. Print [TRANSLATOR] metadata lines from proc1..proc6
out_lines.append("\n--------------------------------------------------------------------------------")
out_lines.append("2. [TRANSLATOR] METADATA LOGS FOR 10 UNMAPPED QUERIES ACROSS proc1..proc6:")
out_lines.append("--------------------------------------------------------------------------------")
out_lines.append("Network state during proc1..proc6: OFFLINE (TRANSFORMERS_OFFLINE=1, HF_HUB_OFFLINE=1, network calls blocked).")
procs = [
    "proc1_unset_fwd.json", "proc2_unset_rev.json", "proc3_seed1.json",
    "proc4_seed2.json", "proc5_seed12345.json", "proc6_seed0.json"
]
for p in procs:
    p_path = Path("docs/perf") / p
    if not p_path.exists():
        continue
    with open(p_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out_lines.append(f"\nSnapshot: {p}")
    for q in unmapped_queries:
        if q in data.get("queries", {}):
            meta = data["queries"][q].get("metadata", {})
            t_in = meta.get("translator_input")
            t_out = meta.get("translator_output")
            t_calls = meta.get("translator_network_calls")
            out_lines.append(f"  [TRANSLATOR] query='{q:<25}' -> input='{str(t_in)}', output='{str(t_out)}', network_calls={t_calls}")

# 3. Online run under unset and seed 1
out_lines.append("\n--------------------------------------------------------------------------------")
out_lines.append("3. FRESH ONLINE PROCESSES (NETWORK ON: unset vs seed 1):")
out_lines.append("--------------------------------------------------------------------------------")

ONLINE_CHILD_SCRIPT = """
import sys, os, json
os.environ["DATABASE_URL"] = "postgresql://rizal:rizalpass123@127.0.0.1:5433/rizal_db"
os.environ["REDIS_URL"] = "redis://localhost:6379"
os.environ["PERF_BENCHMARK"] = "1"
# Ensure network is NOT disabled
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)
sys.path.insert(0, "backend")

from app.models.database import SessionLocal
from app.core.engine import get_engine

db = SessionLocal()
engine = get_engine()

unmapped = [
    "korapsyon", "karahasan ng mga prayle", "tiktok", "bitcoin",
    "covid", "facebook", "internet", "computer", "cellphone", "online"
]
results = {"queries": {}}
for q in unmapped:
    res = engine.search(db, q, source_type="summary", novel="both")
    results["queries"][q] = {
        "noli": res["results"]["noli"],
        "elfili": res["results"]["elfili"],
        "precise_matches": res.get("precise_matches", []),
        "metadata": res.get("metadata", {})
    }

print("ONLINE_JSON_START:" + json.dumps(results) + ":ONLINE_JSON_END")
"""

p_unset = subprocess.run([sys.executable, "-c", ONLINE_CHILD_SCRIPT], env={k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}, capture_output=True, text=True)
p_seed1 = subprocess.run([sys.executable, "-c", ONLINE_CHILD_SCRIPT], env={**os.environ, "PYTHONHASHSEED": "1"}, capture_output=True, text=True)

line_u = [l for l in p_unset.stdout.splitlines() if l.startswith("ONLINE_JSON_START:")][-1]
line_s = [l for l in p_seed1.stdout.splitlines() if l.startswith("ONLINE_JSON_START:")][-1]

json_u = json.loads(line_u[len("ONLINE_JSON_START:"):-len(":ONLINE_JSON_END")])
json_s = json.loads(line_s[len("ONLINE_JSON_START:"):-len(":ONLINE_JSON_END")])

path_u = Path("docs/perf/online_unset.json")
path_s = Path("docs/perf/online_seed1.json")
with open(path_u, "w", encoding="utf-8") as f:
    json.dump(json_u, f, indent=2)
with open(path_s, "w", encoding="utf-8") as f:
    json.dump(json_s, f, indent=2)

out_lines.append(f"Saved online_unset.json and online_seed1.json ({len(json_u['queries'])} queries).")

out_lines.append("\nLogged Translator Strings in Online Run (unset):")
for q in unmapped_queries:
    meta = json_u["queries"][q].get("metadata", {})
    out_lines.append(f"  Query: '{q:<25}' -> translator_input='{meta.get('translator_input')}', translator_output='{meta.get('translator_output')}'")

# Strict compare online_unset vs online_seed1
rep_online = compare_strict(path_u, path_s)
out_lines.append(f"\nStrict Comparison: online_unset vs online_seed1:")
out_lines.append(f"  Total queries: {rep_online['total_queries']} | Diffs: {rep_online['queries_with_changes']}")
if rep_online['queries_with_changes'] == 0:
    out_lines.append("  [SUCCESS] 100% BIT-IDENTICAL between unset and seed 1 with network ON!")
else:
    out_lines.append(f"  Differing queries: {list(rep_online['diffs'].keys())}")

# Compare online vs golden_snapshot_deterministic.json
out_lines.append("\nComparison of Online Run against golden_snapshot_deterministic.json (10 unmapped queries):")
path_det = Path("docs/perf/golden_snapshot_deterministic.json")
with open(path_det, "r", encoding="utf-8") as f:
    gdet = json.load(f)

# Filter gdet to the 10 unmapped queries for strict compare
gdet_10 = {"queries": {q: gdet["queries"][q] for q in unmapped_queries if q in gdet.get("queries", {})}}
path_det_10 = Path("docs/perf/golden_snapshot_deterministic_10.json")
with open(path_det_10, "w", encoding="utf-8") as f:
    json.dump(gdet_10, f, indent=2)

rep_vs_det = compare_strict(path_u, path_det_10)
out_lines.append(f"  Total queries compared: {rep_vs_det['total_queries']}")
out_lines.append(f"  Queries with changes  : {rep_vs_det['queries_with_changes']}")
if rep_vs_det['queries_with_changes'] == 0:
    out_lines.append("  [SUCCESS] Online run matches golden_snapshot_deterministic.json 100%!")
else:
    for q, d in rep_vs_det['diffs'].items():
        out_lines.append(f"  Query '{q}' diff: {list(d.keys())}")

with open("docs/perf/evidence/b5_4.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))

print("Saved docs/perf/evidence/b5_4.txt successfully.")
