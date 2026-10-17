import requests
import json
import time

sids = [
    37666, 38246, 38639, 38607, 38608, 38609, 38519, 38693, 38671, 38713,
    38752, 38798, 38859, 38881, 38882, 38901, 38823, 38824, 38827, 38923,
    38959, 38960, 39083, 39131, 39180, 39181, 39296, 39297
]

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
noli_bundle = json.load(open(ROOT / 'backend/app/data/sanggunian_cache_noli_buod.json', encoding='utf-8'))
fili_bundle = json.load(open(ROOT / 'backend/app/data/sanggunian_cache_elfili_buod.json', encoding='utf-8'))

all_refs = {}
for b in [noli_bundle, fili_bundle]:
    for ch in b['chapters'].values():
        all_refs.update(ch['references'])

results = []
s = requests.Session()
for sid in sids:
    url_hf = f'https://mematello-rizal-thematic-search-api.hf.space/api/v1/sentences/{sid}/sanggunian'
    ref = all_refs.get(str(sid), {})
    try:
        r = s.get(url_hf, timeout=45)
        if r.status_code == 200:
            d = r.json()
            score_delta = abs(float(ref.get('score', 0)) - float(d.get('score', 0)))
            pids_match = ref.get('passage_ids') == d.get('passage_ids')
            status_match = ref.get('alignment_status') == d.get('alignment_status')
            text_match = ref.get('reference_text', '').strip() == d.get('reference_text', '').strip()
            item = (sid, 'OK', score_delta, pids_match, status_match, text_match)
            results.append(item)
            print(f'Sentence {sid}: {item}', flush=True)
        else:
            item = (sid, f'HTTP {r.status_code}', None, None, None, None)
            results.append(item)
            print(f'Sentence {sid}: {item}', flush=True)
    except Exception as e:
        item = (sid, f'Error: {e}', None, None, None, None)
        results.append(item)
        print(f'Sentence {sid}: {item}', flush=True)

print(f'Tested {len(results)} sentences:')
for res in results:
    print(res)
