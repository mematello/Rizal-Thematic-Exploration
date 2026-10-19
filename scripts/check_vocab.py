
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve()
while _repo_root.parent != _repo_root and not (_repo_root / "scripts" / "prod_guard.py").exists():
    _repo_root = _repo_root.parent
if str(_repo_root / "scripts") not in sys.path:
    sys.path.insert(0, str(_repo_root / "scripts"))
from prod_guard import verify_not_production_db
verify_not_production_db(caller_name=Path(__file__).name)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from app.core.engine import get_engine
import os

engine = get_engine()
word = "kamatayan"
if word in engine.vocabulary:
    print(f"'{word}' found in vocabulary.")
else:
    print(f"'{word}' NOT found in vocabulary.")
    
# Also check exact match behavior
query = "kamatayan ni Basilio"
print(f"Validating query: {query}")
try:
    # We need a dummy db session if we call search, but let's just use invalid session to test valid_words extraction
    # Actually checking valid_words logic from engine.py:
    from app.core.analyzer import extract_words
    words = extract_words(query.lower())
    unknown = [w for w in words if w not in engine.vocabulary]
    print(f"Unknown words: {unknown}")
except Exception as e:
    print(f"Error: {e}")
