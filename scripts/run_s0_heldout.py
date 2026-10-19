
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve()
while _repo_root.parent != _repo_root and not (_repo_root / "scripts" / "prod_guard.py").exists():
    _repo_root = _repo_root.parent
if str(_repo_root / "scripts") not in sys.path:
    sys.path.insert(0, str(_repo_root / "scripts"))
from prod_guard import verify_not_production_db
verify_not_production_db(caller_name=Path(__file__).name)

import sys
import os
import json
import subprocess
from pathlib import Path
import threading

base_dir = Path(".").resolve()
sys.path.insert(0, str(base_dir / "scripts"))
from strict_comparator import compare_strict

S0_CHILD_SCRIPT = """
import sys, os, json, time, re
import numpy as np

os.environ["DATABASE_URL"] = "postgresql://rizal:rizalpass123@127.0.0.1:5433/rizal_db"
os.environ["REDIS_URL"] = "redis://localhost:6379"
os.environ["PERF_BENCHMARK"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, "backend")

from app.models.database import SessionLocal
from app.core.engine import get_engine, extract_words
from app.core.config import get_settings
settings = get_settings()

db = SessionLocal()
engine = get_engine()

# Monkeypatch engine._expand_query_with_themes to S0 (pre-E1):
# Iterates unsorted candidates set, sorts final_tokens without secondary key
orig_expand = engine._expand_query_with_themes
def s0_expand(query_vec, is_safely_mapped=False):
    if engine.theme_matrix is None:
        return "", set()
    query_norm = np.linalg.norm(query_vec)
    if query_norm > 0:
        query_vec = query_vec / query_norm
    else:
        return "", set()
    scores = np.dot(engine.theme_matrix, query_vec)
    top_indices = np.argsort(scores)[-3:][::-1]
    candidates = set()
    best_title = ""
    target_theme_key = ""
    for i, idx in enumerate(top_indices):
        score = scores[idx]
        thresh = 0.35 if is_safely_mapped else settings.TEST_GATE_THRESHOLD_OOV
        if score < thresh: continue
        theme = engine.theme_cache[idx]
        if i == 0:
            best_title = theme['tagalog_title']
            target_theme_key = best_title.strip().lower()
        meaning_words = extract_words(theme['meaning'].lower())
        candidates.update(meaning_words)
    if not candidates or not target_theme_key:
        return "", set()
    final_tokens = []
    # Pre-E1: unsorted set iteration!
    for w in candidates:
        if len(w) <= 3: continue
        df = engine.theme_word_df.get(w, 0)
        if df < 1: continue
        theme_counts = engine.word_theme_map.get(w, {})
        count_in_target = theme_counts.get(target_theme_key, 0)
        if count_in_target == 0: continue
        specificity = count_in_target / df
        if specificity >= 0.30:
            final_tokens.append(w)
    # Pre-E1: NO secondary key 'w'!
    final_tokens.sort(key=lambda w: -engine.word_theme_map.get(w, {}).get(target_theme_key, 0) / engine.theme_word_df.get(w, 1))
    top_tokens = final_tokens[:12]
    if not top_tokens:
        return "", []
    expanded_text = " ".join(top_tokens)
    return expanded_text, top_tokens

engine._expand_query_with_themes = s0_expand

with open("scripts/heldout_queries.json", "r", encoding="utf-8") as f:
    hq = json.load(f)

qlist = []
for cat, qs in hq["categories"].items():
    qlist.extend(qs)

results = {"queries": {}}
for q in qlist:
    res = engine.search(db, q, source_type="summary", novel="both")
    results["queries"][q] = {
        "noli": res["results"]["noli"],
        "elfili": res["results"]["elfili"],
        "precise_matches": res.get("precise_matches", []),
        "metadata": res.get("metadata", {})
    }

out_file = sys.argv[1]
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"Wrote {len(results['queries'])} queries to {out_file}")
"""

def run_proc(seed, out_file):
    cmd = [sys.executable, "-c", S0_CHILD_SCRIPT, out_file]
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(f"Seed {seed} finished with code {p.returncode}")
    return p.returncode

if __name__ == "__main__":
    out1 = "docs/perf/s0_heldout_seed1.json"
    out2 = "docs/perf/s0_heldout_seed2.json"

    t1 = threading.Thread(target=run_proc, args=(1, out1))
    t2 = threading.Thread(target=run_proc, args=(2, out2))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    rep = compare_strict(Path(out1), Path(out2))
    print(f"Total queries: {rep['total_queries']}")
    print(f"Queries with changes: {rep['queries_with_changes']}")
    for q in sorted(list(rep['diffs'].keys())):
        print(f"  - {q}")
