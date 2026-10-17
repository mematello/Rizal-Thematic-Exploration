import requests
import json
import random
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
bundles = {
    'noli_full': json.load(open(ROOT / 'backend/app/data/sanggunian_cache_noli_full.json', encoding='utf-8')),
    'elfili_full': json.load(open(ROOT / 'backend/app/data/sanggunian_cache_elfili_full.json', encoding='utf-8'))
}

all_full = []
for b_key in ['noli_full', 'elfili_full']:
    for ch, cdata in bundles[b_key]['chapters'].items():
        for sid in cdata['references'].keys():
            all_full.append(int(sid))

random.seed(42)
sample_500 = random.sample(all_full, 500)

s_hf = requests.Session()
s_loc = requests.Session()

def check(sid):
    url_hf = f'https://mematello-rizal-thematic-search-api.hf.space/api/v1/sentences/{sid}/sanggunian'
    url_loc = f'http://127.0.0.1:8000/api/v1/sentences/{sid}/sanggunian'
    r_hf = s_hf.get(url_hf, timeout=15)
    r_loc = s_loc.get(url_loc, timeout=10)
    sha_hf = hashlib.sha256(r_hf.content).hexdigest()
    sha_loc = hashlib.sha256(r_loc.content).hexdigest()
    return sid, sha_hf == sha_loc

matches = 0
mismatches = 0
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(check, sid) for sid in sample_500]
    for fut in as_completed(futures):
        sid, ok = fut.result()
        if ok:
            matches += 1
        else:
            mismatches += 1

print(f'Full mode 500 sample verification: matches={matches}, mismatches={mismatches}')
