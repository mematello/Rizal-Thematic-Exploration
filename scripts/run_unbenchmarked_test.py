import os
import sys
import json
import subprocess
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
backend_dir = base_dir / "backend"
docs_dir = base_dir / "docs" / "perf"
scripts_dir = base_dir / "scripts"
python_exe = backend_dir / ".venv" / "Scripts" / "python.exe"

worker_code = """
import os, sys, json
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
backend_dir = base_dir / "backend"
sys.path.insert(0, str(backend_dir))
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
# Ensure PERF_BENCHMARK is UNSET
os.environ.pop('PERF_BENCHMARK', None)

from app.models.database import SessionLocal
from app.core.engine import get_engine

db = SessionLocal()
engine = get_engine()

with open(base_dir / "scripts" / "golden_queries.json", "r", encoding="utf-8") as f:
    gdata = json.load(f)

queries = []
for cat in gdata["categories"].values():
    queries.extend(cat["queries"])

results_snap = {'queries': {}}
for q in queries:
    res = engine.search(db, q, source_type='summary', novel='both')
    results_snap['queries'][q] = res

out_file = sys.argv[1]
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(results_snap, f, indent=2)
print(f"Finished unbenchmarked pass -> {out_file}", flush=True)
"""

worker_file = scripts_dir / "_unbenchmarked_worker.py"
with open(worker_file, "w", encoding="utf-8") as f:
    f.write(worker_code)

snap_a = docs_dir / "unbenchmarked_proc_a.json"
snap_b = docs_dir / "unbenchmarked_proc_b.json"

print("=" * 80)
print("TEST WITH PERF_BENCHMARK UNSET (TWO FRESH PROCESSES, 40 GOLDEN QUERIES)")
print("=" * 80)

# Process A (unset hash seed)
print("Running Process A (PERF_BENCHMARK unset, PYTHONHASHSEED unset)...")
env_a = dict(os.environ)
env_a.pop("PERF_BENCHMARK", None)
env_a.pop("PYTHONHASHSEED", None)
subprocess.run([str(python_exe), str(worker_file), str(snap_a)], env=env_a, check=True)

# Process B (unset hash seed)
print("Running Process B (PERF_BENCHMARK unset, PYTHONHASHSEED unset)...")
env_b = dict(os.environ)
env_b.pop("PERF_BENCHMARK", None)
env_b.pop("PYTHONHASHSEED", None)
subprocess.run([str(python_exe), str(worker_file), str(snap_b)], env=env_b, check=True)

# Strict compare
sys.path.insert(0, str(scripts_dir))
from strict_comparator import compare_strict, print_strict_report

rep = compare_strict(snap_a, snap_b, allow_missing_diagnostics=True)
print_strict_report(rep)

if worker_file.exists():
    worker_file.unlink()
