
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

base_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(base_dir, '..', 'backend')
sys.path.insert(0, backend_dir)

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from app.models.database import SessionLocal
from app.core.engine import get_engine, extract_words
from app.core.config import get_settings
settings = get_settings()
import numpy as np

db = SessionLocal()
engine = get_engine()
engine._ensure_themes_loaded(db)

queries = ['kalayaan', 'edukasyon']

def option1_expand(self, query_vec: np.ndarray, is_safely_mapped: bool = False):
    if self.theme_matrix is None:
        return "", set()
    query_norm = np.linalg.norm(query_vec)
    if query_norm > 0:
        query_vec = query_vec / query_norm
    else:
        return "", set()

    scores = np.dot(self.theme_matrix, query_vec)
    top_indices = np.argsort(scores)[-3:][::-1]

    candidates = set()
    best_title = ""
    target_theme_key = ""

    for i, idx in enumerate(top_indices):
        score = scores[idx]
        thresh = 0.35 if is_safely_mapped else settings.TEST_GATE_THRESHOLD_OOV
        if score < thresh: continue

        theme = self.theme_cache[idx]
        if i == 0:
            best_title = theme['tagalog_title']
            target_theme_key = best_title.strip().lower()

        meaning_words = extract_words(theme['meaning'].lower())
        candidates.update(meaning_words)

    if not candidates or not target_theme_key:
        return "", set()

    final_tokens = []
    # Option 1: iterate sorted(candidates)
    for w in sorted(candidates):
        if len(w) <= 3: continue
        df = self.theme_word_df.get(w, 0)
        if df < 1: continue
        theme_counts = self.word_theme_map.get(w, {})
        count_in_target = theme_counts.get(target_theme_key, 0)
        if count_in_target == 0: continue
        specificity = count_in_target / df
        if specificity >= 0.30:
            final_tokens.append(w)

    # Option 1: secondary sort key (word)
    final_tokens.sort(key=lambda w: (-self.word_theme_map.get(w, {}).get(target_theme_key, 0) / self.theme_word_df.get(w, 1), w))
    top_tokens = final_tokens[:12]

    if not top_tokens:
        return "", set()
    expanded_text = " ".join(top_tokens)
    return expanded_text, set(top_tokens)

print("=" * 70)
print("THEME EXPANSION: CURRENT (SEED-DEPENDENT) VS OPTION 1 (SORTED)")
print("=" * 70)

for q in queries:
    q_emb = engine.base_model.encode(q, show_progress_bar=False)
    current_exp, _ = engine._expand_query_with_themes(q_emb)
    opt1_exp, _ = option1_expand(engine, q_emb)
    print(f"Query: '{q}'")
    print(f"  Current expansion:  '{current_exp}'")
    print(f"  Option 1 expansion: '{opt1_exp}'")
