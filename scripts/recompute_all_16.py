import os
import sys
import inspect
import textwrap
import json
from dotenv import load_dotenv

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
load_dotenv('backend/.env')
sys.path.insert(0, 'backend')

from app.models.database import SessionLocal
from app.core.engine import get_engine, RizalEngine

db = SessionLocal()
engine = get_engine()

# Load snapshots
with open('docs/perf/golden_snapshot_seed0.json', 'r', encoding='utf-8') as f:
    golden_snap = json.load(f)['queries']
with open('docs/perf/b1_snapshot_seed0.json', 'r', encoding='utf-8') as f:
    b1_snap = json.load(f)['queries']

changed_queries = [
    'simbahan', 'bayan', 'paaralan', 'education', 'freedom', 'love', 'death', 'church',
    'Crisostomo Ibarra', 'Maria Clara', 'Simoun', 'Elias', 'Padre Damaso', 'Basilio', 'Sisa', 'Isagani'
]

src = textwrap.dedent(inspect.getsource(RizalEngine.search))
target_lex = "            return unique[:top_k]"
hook_lex = """            global_full_pools[label] = list(unique)
            return unique[:top_k]"""
src_mod = src.replace(target_lex, hook_lex)
target_sem = "            results[book_key] = (lex_reserved + sem_only)[:top_k]"
hook_sem = """            global_full_pools[book_key] = list(lex_reserved + sem_only)
            results[book_key] = (lex_reserved + sem_only)[:top_k]"""
src_mod = src_mod.replace(target_sem, hook_sem)

globals_dict = dict(RizalEngine.search.__globals__)
globals_dict['global_full_pools'] = {}
local_env = {}

exec(src_mod, globals_dict, local_env)
engine.search_full = local_env['search'].__get__(engine, RizalEngine)
engine.global_full_pools = globals_dict['global_full_pools']

print("=" * 80, flush=True)
print("ACCURATE RECOMPUTED TIE STATISTICS FOR ALL 16 CHANGED QUERIES", flush=True)
print("=" * 80, flush=True)

summary_rows = []

for q in changed_queries:
    engine.global_full_pools.clear()
    res = engine.search_full(db, q, source_type='summary', novel='both')
    meta = res.get('metadata', {})
    rm = meta.get('result_mode')

    print(f"\n================================================================================", flush=True)
    print(f"QUERY: '{q}' (mode: {rm})", flush=True)
    print(f"================================================================================", flush=True)

    for book in ['noli', 'elfili']:
        pool = engine.global_full_pools.get(book, [])
        g_list = golden_snap.get(q, {}).get(book, [])
        b_list = b1_snap.get(q, {}).get(book, [])
        g_ids = [x['id'] for x in g_list]
        b_ids = [x['id'] for x in b_list]
        entered = [i for i in b_ids if i not in g_ids]
        left = [i for i in g_ids if i not in b_ids]

        sort_keys = [(x['scores']['final'] + x['scores'].get('precision', 0) * 0.5) for x in pool]
        distinct_keys = sorted(list(set(sort_keys)), reverse=True)

        pool_size = len(pool)

        if pool_size == 0:
            print(f"  [{book.upper()}] Evaluated Pool Size: 0 (No candidates)", flush=True)
            continue

        if pool_size < 10:
            last_key = sort_keys[-1]
            tied_in_pool = sum(1 for k in sort_keys if k == last_key)
            strictly_above = sum(1 for k in sort_keys if k > last_key)
            slots_tied = pool_size - strictly_above
            print(f"  [{book.upper()}] Pool size: {pool_size} (<10). Distinct sort-key values: {len(distinct_keys)}", flush=True)
            print(f"    Sort key of last candidate (rank {pool_size}): {last_key}", flush=True)
            print(f"    Candidates in pool with sort key == {last_key}: {tied_in_pool}", flush=True)
            print(f"    Slots decided by tie order: {slots_tied}", flush=True)
            print(f"    Entered: {entered} | Left: {left}", flush=True)
            summary_rows.append({
                'query': q,
                'book': book,
                'pool_size': pool_size,
                'distinct_keys': len(distinct_keys),
                'key_10th': last_key,
                'tied_in_pool': tied_in_pool,
                'slots_decided_by_tie': slots_tied,
                'entered_count': len(entered),
                'left_count': len(left)
            })
            continue

        key_10th = sort_keys[9]
        tied_in_pool = sum(1 for k in sort_keys if k == key_10th)
        strictly_above = sum(1 for k in sort_keys if k > key_10th)
        slots_decided_by_tie = 10 - strictly_above

        print(f"  [{book.upper()}] Full Evaluated Pool Size: {pool_size}", flush=True)
        print(f"    Distinct sort-key values: {len(distinct_keys)}", flush=True)
        print(f"    Sort key of 10th candidate: {key_10th}", flush=True)
        print(f"    Candidates in pool with sort key == {key_10th}: {tied_in_pool}", flush=True)
        print(f"    Candidates strictly above {key_10th}: {strictly_above}", flush=True)
        print(f"    Top-10 slots decided by tie order: {slots_decided_by_tie} (tied candidates competing: {tied_in_pool})", flush=True)
        print(f"    Golden top-10 IDs: {g_ids}", flush=True)
        print(f"    New    top-10 IDs: {b_ids}", flush=True)
        print(f"    Entered IDs ({len(entered)}): {entered}", flush=True)
        print(f"    Left IDs    ({len(left)}): {left}", flush=True)

        summary_rows.append({
            'query': q,
            'book': book,
            'pool_size': pool_size,
            'distinct_keys': len(distinct_keys),
            'key_10th': key_10th,
            'tied_in_pool': tied_in_pool,
            'slots_decided_by_tie': slots_decided_by_tie,
            'entered_count': len(entered),
            'left_count': len(left)
        })

print("\n\n" + "=" * 80, flush=True)
print("SUMMARY TABLE OF TIE STATISTICS ACROSS 16 CHANGED QUERIES", flush=True)
print("=" * 80, flush=True)
print(f"{'Query':<20} | {'Book':<6} | {'Pool':<5} | {'Distinct':<8} | {'10th Key':<8} | {'Tied in Pool':<12} | {'Tied Slots':<10} | {'Entered/Left':<12}", flush=True)
print("-" * 95, flush=True)
for r in summary_rows:
    ent_left = f"+{r['entered_count']}/-{r['left_count']}"
    print(f"{r['query']:<20} | {r['book']:<6} | {r['pool_size']:<5} | {r['distinct_keys']:<8} | {r['key_10th']:<8} | {r['tied_in_pool']:<12} | {r['slots_decided_by_tie']:<10} | {ent_left:<12}", flush=True)
