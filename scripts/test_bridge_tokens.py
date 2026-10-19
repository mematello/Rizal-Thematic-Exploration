
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve()
while _repo_root.parent != _repo_root and not (_repo_root / "scripts" / "prod_guard.py").exists():
    _repo_root = _repo_root.parent
if str(_repo_root / "scripts") not in sys.path:
    sys.path.insert(0, str(_repo_root / "scripts"))
from prod_guard import verify_not_production_db
verify_not_production_db(caller_name=Path(__file__).name)

import os
import sys
import subprocess
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
backend_dir = base_dir / "backend"

script = """
import os, sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
backend_dir = base_dir / "backend"
sys.path.insert(0, str(backend_dir))
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from app.core.engine import get_engine
engine = get_engine()

words = ['oppression', 'love', 'justice', 'corruption']
seed = os.environ.get('PYTHONHASHSEED', 'unset')
print(f"=== PYTHONHASHSEED = {seed} ===")
for w in words:
    raw_set = set(engine.EN_TL_DICTIONARY.get(w, []))
    bt_before = " ".join([bt for bt in raw_set if len(bt) > 3])
    bt_after = " ".join(sorted([bt for bt in raw_set if len(bt) > 3]))
    print(f"  {w:<12} | elements: {list(raw_set)} | before_E2: '{bt_before}' | after_E2: '{bt_after}'")
"""

worker_script = Path(__file__).resolve().parent / "_bt_worker.py"
with open(worker_script, "w", encoding="utf-8") as f:
    f.write(script)

for seed in ["1", "2"]:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    res = subprocess.run([sys.executable, str(worker_script)], capture_output=True, text=True, env=env)
    print(res.stdout)
    if res.stderr and "Error" in res.stderr:
        print("STDERR:", res.stderr)

if worker_script.exists():
    worker_script.unlink()
