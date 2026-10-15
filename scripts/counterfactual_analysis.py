import os
import sys
import json
import inspect
import textwrap
from pathlib import Path
import numpy as np

base_dir = Path(__file__).parent.parent
backend_dir = base_dir / "backend"
sys.path.insert(0, str(backend_dir))

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from app.models.database import SessionLocal, Sentence
from app.core.engine import get_engine, RizalEngine

db = SessionLocal()
engine = get_engine()

# Read golden queries
with open(base_dir / "scripts" / "golden_queries.json", "r", encoding="utf-8") as f:
    gdata = json.load(f)

queries = []
for cat in gdata["categories"].values():
    queries.extend(cat["queries"])

# Let's inspect the engine.search source to hook unrounded scores
src = textwrap.dedent(inspect.getsource(RizalEngine.search))

# In engine.search, find where result_item is built:
# 'id': sent.id,
# We can attach unrounded_final, unrounded_lex, unrounded_sem
target = "'id': sent.id,"
replacement = """'id': sent.id,
                'unrounded_final': final_score,
                'unrounded_lex': lex_score,
                'unrounded_sem': sem_score,
                'unrounded_prec': precision_score,"""
assert target in src, "target not found"
src = src.replace(target, replacement, 1)

# Also capture the raw evaluated candidates per book before any finalize/rerank
target_top = "    if not self._validate_query_semantics(db, query):"
replacement_top = """    self._last_raw_pools = {'noli': [], 'elfili': []}
    if not self._validate_query_semantics(db, query):"""
assert target_top in src, "target_top not found"
src = src.replace(target_top, replacement_top, 1)

target_finalize = "    def finalize(lst, label):"
replacement_finalize = """    self._last_raw_pools = {'noli': list(results['noli']), 'elfili': list(results['elfili'])}
    def finalize(lst, label):"""
assert target_finalize in src, "target_finalize not found"
src = src.replace(target_finalize, replacement_finalize, 1)

globals_dict = dict(RizalEngine.search.__globals__)
local_env = {}
exec(src, globals_dict, local_env)
search_hooked = local_env["search"].__get__(engine, RizalEngine)

print("Running counterfactual analysis for all 40 queries...")

report_data = []

for q in queries:
    out = search_hooked(db, q, source_type="summary", novel="both")
    raw_pools = engine._last_raw_pools

    for book in ["noli", "elfili"]:
        pool = raw_pools[book]
        if not pool:
            report_data.append({
                "query": q,
                "book": book,
                "overlap": 0,
                "overlap_pct": 0.0,
                "mean_sem_a": 0.0,
                "mean_sem_b": 0.0,
                "top10_a_ids": [],
                "top10_b_ids": []
            })
            continue

        # (a) Current key: (scores['final'] + scores.get('precision', 0) * 0.5) descending, then text dedupe
        pool_a = list(pool)
        pool_a.sort(key=lambda x: (x['scores']['final'] + x['scores'].get('precision', 0) * 0.5), reverse=True)
        seen_text_a = set()
        top10_a = []
        for x in pool_a:
            t = x['sentence_text'].strip()
            if t not in seen_text_a:
                seen_text_a.add(t)
                top10_a.append(x)
                if len(top10_a) == 10:
                    break

        # (b) Counterfactual key: unrounded final desc, then lexical desc, then id asc
        pool_b = list(pool)
        pool_b.sort(key=lambda x: (-x['unrounded_final'], -x['unrounded_lex'], x['id']))
        seen_text_b = set()
        top10_b = []
        for x in pool_b:
            t = x['sentence_text'].strip()
            if t not in seen_text_b:
                seen_text_b.add(t)
                top10_b.append(x)
                if len(top10_b) == 10:
                    break

        ids_a = [x['id'] for x in top10_a]
        ids_b = [x['id'] for x in top10_b]
        overlap = len(set(ids_a) & set(ids_b))
        overlap_pct = (overlap / max(1, len(ids_a))) * 100.0

        # Mean semantic score: unrounded_sem * 100 or scores['semantic']
        mean_sem_a = np.mean([x['unrounded_sem'] * 100 for x in top10_a]) if top10_a else 0.0
        mean_sem_b = np.mean([x['unrounded_sem'] * 100 for x in top10_b]) if top10_b else 0.0

        report_data.append({
            "query": q,
            "book": book,
            "overlap": overlap,
            "overlap_pct": overlap_pct,
            "mean_sem_a": round(float(mean_sem_a), 2),
            "mean_sem_b": round(float(mean_sem_b), 2),
            "top10_a_ids": ids_a,
            "top10_b_ids": ids_b
        })

print(f"{'Query':<30} | {'Book':<7} | {'Overlap':<9} | {'Mean Sem A':<12} | {'Mean Sem B':<12} | {'Delta Sem':<10}")
print("-" * 90)
for r in report_data:
    delta_sem = r['mean_sem_b'] - r['mean_sem_a']
    print(f"{r['query']:<30} | {r['book']:<7} | {r['overlap']:>2d}/10 ({r['overlap_pct']:>3.0f}%) | {r['mean_sem_a']:>10.2f} | {r['mean_sem_b']:>10.2f} | {delta_sem:>+9.2f}")

# Save full json report
out_path = base_dir / "docs" / "perf" / "counterfactual_report.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=2)
print(f"\nSaved full counterfactual report to {out_path}")
