import os
import sys
import json

base_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(base_dir, '..', 'backend')
sys.path.insert(0, backend_dir)

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from app.core.engine import get_engine, extract_words

with open(os.path.join(base_dir, 'golden_queries.json'), 'r', encoding='utf-8') as f:
    gdata = json.load(f)

engine = get_engine()

all_q = []
for cat, cdata in gdata['categories'].items():
    for q in cdata['queries']:
        all_q.append((cat, q))

print(f"Total golden queries: {len(all_q)}")
for cat, q in all_q:
    query_words = extract_words(q.lower())
    query_analysis = engine.query_analyzer.analyze_query_words(q)
    sig_items = [item for item in query_analysis if not item['is_stopword']]
    sig_words = [item['word'].lower() for item in sig_items]
    discarded_words = [w for w in sig_words if w in engine.EN_STOPWORDS]
    sig_words = [w for w in sig_words if w not in engine.EN_STOPWORDS]
    native_tokens = [w for w in sig_words if w in engine.vocabulary and w not in engine.LOW_INFO_WORDS]
    foreign_tokens = [w for w in sig_words if w not in engine.vocabulary and len(w) > 2]

    is_cross = len(foreign_tokens) > 0
    dict_bridges = []
    unmapped = []
    if is_cross:
        for w in foreign_tokens:
            if w in engine.EN_TL_DICTIONARY:
                dict_bridges.extend(engine.EN_TL_DICTIONARY[w])
            else:
                unmapped.append(w)

    triggers_translator = len(unmapped) > 0
    if is_cross or triggers_translator:
        print(f"[{cat}] '{q}': foreign={foreign_tokens}, dict_bridges={dict_bridges}, unmapped={unmapped}, triggers_translator={triggers_translator}")
