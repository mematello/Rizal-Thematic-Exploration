import hashlib
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
content_p = repo_root / 'backend/app/api/v1/content.py'
aligner_p = repo_root / 'backend/app/core/robust_aligner.py'
bundle_p = repo_root / 'backend/app/data/sanggunian_cache_noli_buod.json'

def lf_hash(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read().replace(b'\r\n', b'\n')).hexdigest()

c_hash = lf_hash(content_p)
a_hash = lf_hash(aligner_p)

print('FILE HASHES (LF-normalized):')
print(f'content.py:        {c_hash}')
print(f'robust_aligner.py: {a_hash}')

bundle_names = [
    'sanggunian_cache_noli_buod.json',
    'sanggunian_cache_noli_full.json',
    'sanggunian_cache_elfili_buod.json',
    'sanggunian_cache_elfili_full.json',
]

for b_name in bundle_names:
    p = repo_root / 'backend/app/data' / b_name
    with open(p, 'r', encoding='utf-8') as f:
        b = json.load(f)
    fp = b.get('fingerprint', {})
    print(f'\nBUNDLE: {b_name}')
    print('  content_alignment_hash:     ' + str(fp.get('content_alignment_hash')))
    print('  robust_aligner_hash:        ' + str(fp.get('robust_aligner_hash')))
    print('  content match:', c_hash == fp.get('content_alignment_hash'))
    print('  aligner match:', a_hash == fp.get('robust_aligner_hash'))

