import os
import sys
import time
import json
import subprocess
from pathlib import Path
import requests

base_dir = Path(__file__).resolve().parent.parent
backend_dir = base_dir / "backend"
docs_dir = base_dir / "docs" / "perf"
python_exe = backend_dir / ".venv" / "Scripts" / "python.exe"

def wait_for_server(port=8000, timeout=60):
    start = time.time()
    url = f"http://127.0.0.1:{port}/health"
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

def run_pass(seed, out_json):
    print(f"\n--- Starting server with PYTHONHASHSEED={seed} ---")
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(seed)
    env["PERF_BENCHMARK"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"

    server_cmd = [
        str(python_exe), "-c",
        "import sys, os; print(f'[SERVER] hash_randomization={sys.flags.hash_randomization}, PYTHONHASHSEED={os.environ.get(\"PYTHONHASHSEED\")}'); import uvicorn; uvicorn.run('app.main:app', host='127.0.0.1', port=8000, log_level='warning')"
    ]

    proc = subprocess.Popen(server_cmd, cwd=str(backend_dir), env=env)
    try:
        if not wait_for_server(8000, timeout=60):
            raise RuntimeError(f"Server with seed {seed} failed to start within 60s")
        print(f"Server ready. Fetching snapshot to {out_json.name}...")

        # Run fetch
        fetch_script = base_dir / "scripts" / "fetch_snapshot.py"
        res = subprocess.run([str(python_exe), str(fetch_script), str(out_json), "8000"], cwd=str(backend_dir), capture_output=True, text=True)
        if res.returncode != 0:
            print("Fetch error:", res.stderr)
            raise RuntimeError("Fetch failed")
        print("Fetch completed successfully.")
    finally:
        print("Terminating server process...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print("Server process stopped.")

out_seed1 = docs_dir / "neg_ctrl_seed1.json"
out_seed2 = docs_dir / "neg_ctrl_seed2.json"

print("=" * 80)
print("NEGATIVE CONTROL (b): Commit D (before E1), seeds 1 vs 2")
print("=" * 80)

run_pass(1, out_seed1)
run_pass(2, out_seed2)

# Now run strict comparison
sys.path.insert(0, str(base_dir / "scripts"))
from strict_comparator import compare_strict, print_strict_report

rep = compare_strict(out_seed1, out_seed2)
print_strict_report(rep)
