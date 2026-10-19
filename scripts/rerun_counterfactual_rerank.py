
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prod_guard import verify_not_production_db
verify_not_production_db(caller_name=Path(__file__).name)

import os
import sys
import json
from pathlib import Path
import numpy as np

base_dir = Path(__file__).resolve().parent.parent
backend_dir = base_dir / "backend"
sys.path.insert(0, str(backend_dir))

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from app.models.database import SessionLocal
from app.core.engine import get_engine, RizalEngine

db = SessionLocal()
engine = get_engine()

fallback_queries = ['kalayaan', 'edukasyon', 'pag-ibig', 'kamatayan', 'corruption', 'oppression', 'love', 'justice']

orig_rerank = engine._rerank_candidates
captured_pools = {}

def hooked_rerank(query, candidates, query_vec=None):
    if not candidates:
        captured_pools.setdefault(query, []).append([])
        return []

    if query_vec is None:
        q_emb = engine.base_model.encode(query, show_progress_bar=False)
    else:
        q_emb = query_vec
    q_norm = np.linalg.norm(q_emb)
    if q_norm > 0:
        q_emb = q_emb / q_norm

    # compute unrounded rerank score on copy of candidates
    pool_copy = []
    for i, cand in enumerate(candidates):
        c = dict(cand)
        c['scores'] = dict(cand['scores'])
        if 'embedding' in cand and cand['embedding'] is not None:
            text_emb = np.array(cand['embedding'])
        else:
            text = cand['sentence_text']
            text_emb = engine.base_model.encode(text, show_progress_bar=False)
        t_norm = np.linalg.norm(text_emb)
        if t_norm > 0:
            text_emb = text_emb / t_norm

        rerank_val = float(np.dot(q_emb, text_emb))
        c['unrounded_rerank'] = rerank_val
        c['scores']['rerank'] = round(rerank_val * 100)
        c['scores']['initial_rank'] = i + 1
        pool_copy.append(c)

    captured_pools.setdefault(query, []).append(pool_copy)
    return orig_rerank(query, candidates, query_vec)

engine._rerank_candidates = hooked_rerank

print("=" * 95)
print("COUNTERFACTUAL RE-RUN (b) ON RERANK KEY FOR SEMANTIC_FALLBACK QUERIES")
print("=" * 95)
print(f"{'Query':<15} | {'Book':<7} | {'Overlap':<10} | {'Mean Sem A':<11} | {'Mean Sem B':<11} | {'Delta Sem':<10}")
print("-" * 95)

results_data = []

for q in fallback_queries:
    captured_pools[q] = []
    res = engine.search(db, q, source_type="summary", novel="both")

    analysis = engine.query_analyzer.analyze_query_words(q)
    sig_words = [item['word'] for item in analysis if not item['is_stopword'] or item['semantic_weight'] > 0.5]

    # captured_pools[q][0] is noli, [1] is elfili
    for b_idx, book in enumerate(["noli", "elfili"]):
        pool = captured_pools[q][b_idx]

        # (a) Current engine result
        top10_a = res['results'][book]
        ids_a = [x['id'] for x in top10_a]

        # (b) Counterfactual key on rerank:
        # Sort candidates by:
        # 1. unrounded rerank descending
        # 2. lexical score descending
        # 3. Sentence.id ascending
        # With the same lexical-priority reservation applied by engine lines 1555-1567

        # Apply sort to pool:
        pool_b = list(pool)
        pool_b.sort(key=lambda x: (-x['unrounded_rerank'], -x['scores']['lexical'], x['id']))

        # Post-rerank lexical reservation
        lex_reserved = []
        sem_only = []
        for item in pool_b:
            txt_lower = item['sentence_text'].lower()
            has_real_lex = any(sw in txt_lower for sw in sig_words) if sig_words else False
            if has_real_lex:
                lex_reserved.append(item)
            else:
                sem_only.append(item)
        top10_b = (lex_reserved + sem_only)[:10]
        ids_b = [x['id'] for x in top10_b]

        overlap = len(set(ids_a) & set(ids_b))
        overlap_pct = (overlap / max(1, len(ids_a))) * 100.0

        mean_sem_a = np.mean([x['scores']['semantic'] for x in top10_a]) if top10_a else 0.0
        mean_sem_b = np.mean([x['scores']['semantic'] for x in top10_b]) if top10_b else 0.0
        delta_sem = mean_sem_b - mean_sem_a

        results_data.append({
            "query": q,
            "book": book,
            "overlap": overlap,
            "overlap_pct": overlap_pct,
            "mean_sem_a": round(float(mean_sem_a), 2),
            "mean_sem_b": round(float(mean_sem_b), 2),
            "delta_sem": round(float(delta_sem), 2),
            "top10_a_ids": ids_a,
            "top10_b_ids": ids_b
        })

        print(f"{q:<15} | {book:<7} | {overlap:>2d}/10 ({overlap_pct:>3.0f}%) | {mean_sem_a:>11.2f} | {mean_sem_b:>11.2f} | {delta_sem:>+10.2f}")

out_path = base_dir / "docs" / "perf" / "counterfactual_rerank_report.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results_data, f, indent=2)
print(f"\nSaved report to {out_path}")
