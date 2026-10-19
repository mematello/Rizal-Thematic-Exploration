
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
import json
from dotenv import load_dotenv

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
load_dotenv('backend/.env')
sys.path.insert(0, 'backend')

from app.models.database import SessionLocal
from app.core.engine import get_engine

db = SessionLocal()
engine = get_engine()

# Load snapshots
with open('docs/perf/golden_snapshot_seed0.json', 'r', encoding='utf-8') as f:
    golden_snap = json.load(f)['queries']
with open('docs/perf/b1_snapshot_seed0.json', 'r', encoding='utf-8') as f:
    b1_snap = json.load(f)['queries']
with open('docs/perf/comparison_report.json', 'r', encoding='utf-8') as f:
    comp_report = json.load(f)

changed_queries = list(comp_report['query_diffs'].keys())

print("=" * 80)
print("ACCURATE RECOMPUTATION OF TIE STATISTICS (FULL EVALUATED POOL)")
print("=" * 80)

for q in changed_queries:
    captured = {}
    def trace_fn(frame, event, arg):
        if event == 'line' and frame.f_code.co_name == 'search' and 'engine.py' in frame.f_code.co_filename:
            if frame.f_lineno == 1474:
                res = frame.f_locals.get('results')
                rm = frame.f_locals.get('result_mode')
                sig_w = frame.f_locals.get('sig_words')
                if res:
                    captured['noli'] = [dict(x) for x in res.get('noli', [])]
                    captured['elfili'] = [dict(x) for x in res.get('elfili', [])]
                    captured['result_mode'] = rm
                    captured['sig_words'] = sig_w
        return trace_fn

    sys.settrace(trace_fn)
    search_res = engine.search(db, q, source_type="summary", novel="both")
    sys.settrace(None)

    rm = captured.get('result_mode')
    print(f"\n================================================================================")
    print(f"QUERY: '{q}' | Mode: {rm}")
    print(f"================================================================================")

    for book in ['noli', 'elfili']:
        pool = captured.get(book, [])
        g_list = golden_snap.get(q, {}).get(book, [])
        b_list = b1_snap.get(q, {}).get(book, [])
        g_ids = [x['id'] for x in g_list]
        b_ids = [x['id'] for x in b_list]
        entered = [i for i in b_ids if i not in g_ids]
        left = [i for i in g_ids if i not in b_ids]

        # In finalize(), pool is sorted by:
        # key = (x['scores']['final'] + x['scores'].get('precision', 0) * 0.5)
        # Deduplication by text is applied
        # Then if semantic_fallback, lex_hits + sem_only [:20] are sent to _rerank_candidates,
        # which sorts candidates by x['scores']['rerank']

        if rm == "semantic_fallback":
            # For semantic_fallback, the final ranking order is rerank score
            # Let's see pool size and sorting
            # Pool entering rerank
            pool.sort(key=lambda x: (x['scores']['final'] + x['scores'].get('precision', 0) * 0.5), reverse=True)
            seen_text = set()
            unique = []
            for itm in pool:
                txt = itm['sentence_text'].strip()
                if txt not in seen_text:
                    seen_text.add(txt)
                    unique.append(itm)

            sig_w = captured.get('sig_words', [])
            lex_hits = []
            sem_only = []
            for itm in unique:
                txt_lower = itm['sentence_text'].lower()
                has_real_lex = any(sw in txt_lower for sw in sig_w) if sig_w else False
                if has_real_lex:
                    lex_hits.append(itm)
                else:
                    sem_only.append(itm)
            merged = (lex_hits + sem_only)[:20]
            # Now let's inspect the actual final evaluated pool in search_res
            final_evaluated = search_res['results'][book]
            sort_keys = [x['scores'].get('rerank', x['scores']['final']) for x in final_evaluated]
            pool_for_stats = final_evaluated
        else:
            # Lexical mode: sort by (final + precision * 0.5)
            pool.sort(key=lambda x: (x['scores']['final'] + x['scores'].get('precision', 0) * 0.5), reverse=True)
            seen_text = set()
            unique = []
            for itm in pool:
                txt = itm['sentence_text'].strip()
                if txt not in seen_text:
                    seen_text.add(txt)
                    unique.append(itm)
            pool_for_stats = unique
            sort_keys = [(x['scores']['final'] + x['scores'].get('precision', 0) * 0.5) for x in pool_for_stats]

        distinct_keys = sorted(list(set(sort_keys)), reverse=True)
        print(f"\n  [{book.upper()}] Full Evaluated Pool Size: {len(pool_for_stats)}")
        print(f"  Distinct sort-key values in pool: {len(distinct_keys)}")

        if len(pool_for_stats) == 0:
            print("  Pool is empty (0 candidates).")
            continue

        if len(pool_for_stats) < 10:
            print(f"  Pool has fewer than 10 candidates ({len(pool_for_stats)}).")
            last_key = sort_keys[-1]
            tied_count = sum(1 for k in sort_keys if k == last_key)
            print(f"  Sort key of last candidate (rank {len(pool_for_stats)}): {last_key}")
            print(f"  Pool candidates with sort key == {last_key}: {tied_count}")
            print(f"  Golden IDs: {g_ids}")
            print(f"  New IDs:    {b_ids}")
            if entered or left:
                print(f"  Entered ({len(entered)}): {entered} | Left ({len(left)}): {left}")
            else:
                print("  IDs unchanged (pure reordering)")
            continue

        key_10th = sort_keys[9]
        tied_in_pool = sum(1 for k in sort_keys if k == key_10th)
        strictly_above = sum(1 for k in sort_keys if k > key_10th)
        slots_decided_by_tie = 10 - strictly_above
        tied_in_top10 = sum(1 for k in sort_keys[:10] if k == key_10th)

        print(f"  Sort key of 10th candidate: {key_10th}")
        print(f"  Candidates in FULL pool with sort key == {key_10th}: {tied_in_pool}")
        print(f"  Candidates strictly above sort key {key_10th}: {strictly_above}")
        print(f"  Top-10 slots decided by tie order: {slots_decided_by_tie} (tied candidates competing: {tied_in_pool})")
        print(f"  Golden top-10 IDs: {g_ids}")
        print(f"  New    top-10 IDs: {b_ids}")
        if entered or left:
            print(f"  Entered ({len(entered)}): {entered}")
            print(f"  Left    ({len(left)}): {left}")
        else:
            print("  IDs unchanged (pure reordering within top-10)")
