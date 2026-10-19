
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prod_guard import verify_not_production_db
verify_not_production_db(caller_name=Path(__file__).name)

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv('backend/.env')
sys.path.insert(0, 'backend')

from app.models.database import SessionLocal, Sentence
from sqlalchemy import select, func, or_
from app.core.engine import get_engine, extract_words

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
print(f"ANALYSIS OF 16 CHANGED QUERIES (MECHANISM & TIE-BREAKS)")
print("=" * 80)

for q in changed_queries:
    print(f"\n================================================================================")
    print(f"QUERY: '{q}'")
    print(f"================================================================================")

    # Run query through engine to capture exact lexical count(*) vs limit
    res = engine.search(db, q, source_type="summary", novel="both")
    meta = res.get('metadata', {})
    print(f"Result Mode: {meta.get('result_mode')}, Reason: {meta.get('reason')}")

    # Query analysis
    query_words = extract_words(q.lower())
    query_analysis = engine.query_analyzer.analyze_query_words(q)
    sig_items = [item for item in query_analysis if not item['is_stopword']]
    sig_words = [item['word'].lower() for item in sig_items]
    sig_words = [w for w in sig_words if w not in engine.EN_STOPWORDS]
    native_tokens = [w for w in sig_words if w in engine.vocabulary and w not in engine.LOW_INFO_WORDS]
    foreign_tokens = [w for w in sig_words if w not in engine.vocabulary and len(w) > 2]
    is_cross_lingual = len(foreign_tokens) > 0
    is_single_word = len(query_words) == 1

    bridge_tokens = set()
    if is_cross_lingual:
        for w in foreign_tokens:
            if w in engine.EN_TL_DICTIONARY:
                bridge_tokens.update(engine.EN_TL_DICTIONARY[w])
        # If any foreign tokens not in EN_TL_DICTIONARY, dynamic translation might be used
        # Check metadata['bridge_tokens']
        if 'bridge_tokens' in meta:
            bridge_tokens.update(meta['bridge_tokens'])

    lex_word = sig_words[0] if sig_words else (query_words[0] if query_words else q)

    # Check each book
    for book in ['noli', 'elfili']:
        book_name = "noli" if book == "noli" else "elfili"

        stmt = select(Sentence).filter(Sentence.book == book_name).filter(Sentence.source_type == "summary")
        if is_cross_lingual:
            for nt in native_tokens:
                stmt = stmt.filter(Sentence.sentence_text.ilike(f"%{nt}%"))
            if bridge_tokens:
                filters = [Sentence.sentence_text.ilike(f"%{bt}%") for bt in bridge_tokens if len(bt) > 3]
                if filters:
                    stmt = stmt.filter(or_(*filters))
            elif foreign_tokens:
                stmt = stmt.filter(Sentence.sentence_text.ilike(f"%{lex_word}%"))
        elif not is_single_word and sig_words:
            for w in sig_words:
                stmt = stmt.filter(Sentence.sentence_text.ilike(f"%{w}%"))
        else:
            stmt = stmt.filter(Sentence.sentence_text.ilike(f"%{lex_word}%"))

        total_lex_matches = db.scalar(select(func.count()).select_from(stmt.subquery()))
        limit_applied = 200 # top_k * 20 = 10 * 20
        is_truncated = total_lex_matches > limit_applied

        print(f"\n  [{book.upper()}] Lexical Count(*): {total_lex_matches} vs Limit: {limit_applied} -> Pool Truncated: {is_truncated}")

        # Top 10 IDs in Golden vs B1
        g_list = golden_snap.get(q, {}).get(book, [])
        b_list = b1_snap.get(q, {}).get(book, [])

        g_ids = [x['id'] for x in g_list]
        b_ids = [x['id'] for x in b_list]

        print(f"  Golden top-10 IDs: {g_ids}")
        print(f"  New    top-10 IDs: {b_ids}")

        entered_ids = [id_ for id_ in b_ids if id_ not in g_ids]
        left_ids = [id_ for id_ in g_ids if id_ not in b_ids]

        if entered_ids or left_ids:
            print(f"  Entered IDs: {entered_ids}")
            for eid in entered_ids:
                item = next(x for x in b_list if x['id'] == eid)
                print(f"    + ID {eid}: final={item['scores'].get('final')}, sem={item['scores'].get('semantic')}, lex={item['scores'].get('lexical')}, text='{item['sentence_text'][:60]}'")
            print(f"  Left IDs: {left_ids}")
            for lid in left_ids:
                item = next(x for x in g_list if x['id'] == lid)
                print(f"    - ID {lid}: final={item['scores'].get('final')}, sem={item['scores'].get('semantic')}, lex={item['scores'].get('lexical')}, text='{item['sentence_text'][:60]}'")
        else:
            print("  IDs unchanged (pure reordering of existing candidates)")

        if b_list:
            score_10th = b_list[-1]['scores'].get('final')
            ties_in_top10 = sum(1 for x in b_list if x['scores'].get('final') == score_10th)
            all_res_book = res.get('results', {}).get(book, [])
            total_tied_pool = sum(1 for x in all_res_book if (x.get('scores', {}).get('final') if isinstance(x, dict) else x.scores.get('final')) == score_10th)
            print(f"  10th-ranked final score: {score_10th}")
            print(f"  Candidates in top-10 with score == {score_10th}: {ties_in_top10}")
            print(f"  Total evaluated candidates with score == {score_10th}: {total_tied_pool}")
