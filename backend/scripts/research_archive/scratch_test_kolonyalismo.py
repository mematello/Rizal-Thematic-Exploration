import os
import sys
import numpy as np

sys.path.append(os.path.abspath("c:/Users/marcu/Desktop/thesis/backend"))
from app.core.engine import RizalEngine
from app.models.database import SessionLocal

os.environ["DEBUG_SEARCH"] = "1"

engine = RizalEngine()
db = SessionLocal()

q = "kolonyalismo"
print(f"\n--- Testing Query: {q} ---")
engine._ensure_themes_loaded(db)
q_vec = engine.base_model.encode(q)
q_norm = np.linalg.norm(q_vec)
q_vec = q_vec / q_norm if q_norm > 0 else q_vec

scores = np.dot(engine.theme_matrix, q_vec)
max_idx = np.argmax(scores)
max_score = scores[max_idx]
best_theme = engine.theme_cache[max_idx]["tagalog_title"]

print(f"ThemeProximity / DomainAlignment Score: {max_score:.4f}")
print(f"Best Theme: {best_theme}")

print("\n--- Running Full Search ---")
res = engine.search(db, q, top_k=3, source_type="full")

print("\n--- Results ---")
if "results" in res:
    noli = res["results"].get("noli", [])
    fili = res["results"].get("elfili", [])
    all_res = noli + fili
    all_res.sort(key=lambda x: x.get("scores", {}).get("final", 0), reverse=True)
    
    for i, r in enumerate(all_res[:3]):
        scores = r.get("scores", {})
        print(f"Rank {i+1}:")
        print(f"  Snippet: {r.get('sentence_text')[:80]}...")
        print(f"  Final Score (Formula 1): {scores.get('final', 0)}")
        print(f"  Lexical: {scores.get('lexical', 0)}, Semantic: {scores.get('semantic', 0)}, Hybrid: {scores.get('hybrid', 0)}, Precision: {scores.get('precision', 0)}")
else:
    print("No results or blocked.")
    print("Metadata:", res.get("metadata"))
