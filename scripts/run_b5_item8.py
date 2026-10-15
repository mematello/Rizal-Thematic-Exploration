import sys
import os
import json
import hashlib
import subprocess
from pathlib import Path

base_dir = Path(".").resolve()
sys.path.insert(0, str(base_dir / "scripts"))

from strict_comparator import compare_strict

out_lines = []
out_lines.append("================================================================================")
out_lines.append("ITEM 8: HELDOUT SUITE (206 QUERIES), PER-PROCESS RESULTS & S0 NEGATIVE CONTROL")
out_lines.append("================================================================================\n")

# 1. Cat scripts/heldout_queries.json
out_lines.append("1. SCRIPTS/HELDOUT_QUERIES.JSON CONTENT:")
with open("scripts/heldout_queries.json", "r", encoding="utf-8") as f:
    heldout_text = f.read()
out_lines.append(heldout_text)

# 2. SHA256 and path of every per-process result file
heldout_files = [
    "docs/perf/heldout_head_unset_fwd.json",
    "docs/perf/heldout_head_unset_rev.json",
    "docs/perf/heldout_head_seed1.json",
    "docs/perf/heldout_head_seed12345.json",
    "docs/perf/heldout_s1_unset_fwd.json",
    "docs/perf/heldout_s1_unset_rev.json",
    "docs/perf/heldout_s1_seed1.json",
    "docs/perf/heldout_s1_seed12345.json"
]

out_lines.append("\n--------------------------------------------------------------------------------")
out_lines.append("2. PER-PROCESS HELDOUT RESULT FILES & NON-EMPTY COUNTS:")
out_lines.append("--------------------------------------------------------------------------------")
out_lines.append(f"{'File Path':<42} | {'SHA256 Hash':<64} | {'Total':<6} | {'Non-Empty'}")
out_lines.append("-" * 125)

for hf in heldout_files:
    p = Path(hf)
    if p.exists():
        data = p.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        jdata = json.loads(data.decode("utf-8"))
        queries = jdata.get("queries", {})
        non_empty = 0
        for q, qd in queries.items():
            noli_len = len(qd.get("noli", []))
            fili_len = len(qd.get("elfili", []))
            if noli_len > 0 or fili_len > 0:
                non_empty += 1
        out_lines.append(f"{hf:<42} | {sha:<64} | {len(queries):^6} | {non_empty:^9}")

# 3. Negative control: run S0 on held-out set under seed 1 and seed 2
out_lines.append("\n--------------------------------------------------------------------------------")
out_lines.append("3. NEGATIVE CONTROL: ABLATION S0 ON HELDOUT SET (SEED 1 vs SEED 2):")
out_lines.append("--------------------------------------------------------------------------------")

S0_CHILD_SCRIPT = """
import sys, os, json
os.environ["DATABASE_URL"] = "postgresql://rizal:rizalpass123@127.0.0.1:5433/rizal_db"
os.environ["REDIS_URL"] = "redis://localhost:6379"
os.environ["PERF_BENCHMARK"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, "backend")

from app.models.database import SessionLocal
from app.core.engine import get_engine

db = SessionLocal()
engine = get_engine()

# Monkeypatch engine._expand_query_by_theme to S0 behavior (pre-E1: set iteration, no secondary key)
def s0_expand(query: str):
    if not engine.theme_cache or not engine.word_theme_map:
        return "", set()
    query_words = [w.lower() for w in query.split()]
    candidates = set()
    for w in query_words:
        if w in engine.word_theme_map:
            for tid in engine.word_theme_map[w]:
                candidates.add(tid)
    if not candidates:
        return "", set()
    theme_scores = {}
    for tid in candidates:
        theme = engine.theme_cache.get(tid)
        if not theme:
            continue
        score = 0.0
        words = theme.get("keywords", [])
        for w in query_words:
            if w in words:
                score += 1.0
        theme_scores[tid] = score
    if not theme_scores:
        return "", set()
    top_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    final_tokens = []
    for tid, _ in top_themes:
        theme = engine.theme_cache.get(tid)
        if theme:
            final_tokens.extend(theme.get("keywords", [])[:3])
    return " ".join(final_tokens), set(final_tokens)

engine._expand_query_by_theme = s0_expand

with open("scripts/heldout_queries.json", "r", encoding="utf-8") as f:
    hq = json.load(f)

results = {"queries": {}}
for q in hq:
    res = engine.search(db, q, source_type="summary", novel="both")
    results["queries"][q] = {
        "noli": res["results"]["noli"],
        "elfili": res["results"]["elfili"],
        "precise_matches": res.get("precise_matches", []),
        "metadata": res.get("metadata", {})
    }

print("S0_HELDOUT_START:" + json.dumps(results) + ":S0_HELDOUT_END")
"""

p_s0_1 = subprocess.run([sys.executable, "-c", S0_CHILD_SCRIPT], env={**os.environ, "PYTHONHASHSEED": "1"}, capture_output=True, text=True)
p_s0_2 = subprocess.run([sys.executable, "-c", S0_CHILD_SCRIPT], env={**os.environ, "PYTHONHASHSEED": "2"}, capture_output=True, text=True)

line_1 = [l for l in p_s0_1.stdout.splitlines() if l.startswith("S0_HELDOUT_START:")][-1]
line_2 = [l for l in p_s0_2.stdout.splitlines() if l.startswith("S0_HELDOUT_START:")][-1]

s0_h1 = json.loads(line_1[len("S0_HELDOUT_START:"):-len(":S0_HELDOUT_END")])
s0_h2 = json.loads(line_2[len("S0_HELDOUT_START:"):-len(":S0_HELDOUT_END")])

path_s0_h1 = Path("docs/perf/s0_heldout_seed1.json")
path_s0_h2 = Path("docs/perf/s0_heldout_seed2.json")
with open(path_s0_h1, "w", encoding="utf-8") as f:
    json.dump(s0_h1, f, indent=2)
with open(path_s0_h2, "w", encoding="utf-8") as f:
    json.dump(s0_h2, f, indent=2)

rep_s0_heldout = compare_strict(path_s0_h1, path_s0_h2)
out_lines.append(f"Ablation S0 Held-Out Comparison (seed 1 vs seed 2):")
out_lines.append(f"  Total queries: {rep_s0_heldout['total_queries']}")
out_lines.append(f"  Queries with changes: {rep_s0_heldout['queries_with_changes']}")
out_lines.append(f"  Divergent queries list ({len(rep_s0_heldout['diffs'])}):")
for idx, qn in enumerate(sorted(list(rep_s0_heldout['diffs'].keys())), 1):
    out_lines.append(f"    {idx:2d}. {qn}")

with open("docs/perf/evidence/b5_8.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))

print("Saved docs/perf/evidence/b5_8.txt successfully.")
