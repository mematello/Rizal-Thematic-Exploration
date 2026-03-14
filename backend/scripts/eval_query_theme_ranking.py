import sys
import os
import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.core.engine import RizalEngine
from app.models.database import SessionLocal

def normalize(vec):
    try:
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec
    except:
        return np.zeros(768)

def main():
    queries = [
        "edukasyon",
        "pang-aapi ng kastila",
        "simbahan",
        "kalayaan",
        "elias",
        "maria clara"
    ]
    
    output_lines = ["# Query-to-Theme Embedding Ranking Diagnostic\n"]
    engine = RizalEngine()
    
    with SessionLocal() as db:
        engine._ensure_themes_loaded(db)
        
        for q in queries:
            output_lines.append(f"## Query: `{q}`\n")
            
            # Embed the query
            q_emb_raw = engine.base_model.encode(q, show_progress_bar=False)
            if isinstance(q_emb_raw, list): q_emb_raw = np.array(q_emb_raw)
            query_vec = normalize(q_emb_raw)
            
            # Compute similarity against all themes
            q_sims = np.dot(engine.theme_matrix, query_vec)
            
            # Rank all themes
            ranked_indices = np.argsort(q_sims)[::-1]
            
            output_lines.append(f"**Top 5 Restricted Pool Candidate Subset:**")
            for rank, idx in enumerate(ranked_indices[:5]):
                theme_label = engine.theme_cache[idx]['tagalog_title']
                score = q_sims[idx]
                output_lines.append(f"{rank + 1}. **{theme_label}** ({score:.3f})")
                
            output_lines.append(f"\n**Remaining Themes (Rank 6-N):**")
            for rank, idx in enumerate(ranked_indices[5:]):
                theme_label = engine.theme_cache[idx]['tagalog_title']
                score = q_sims[idx]
                output_lines.append(f"{rank + 6}. {theme_label} ({score:.3f})")
                
            output_lines.append("\n---\n")

    artifact_path = '/Users/marcusoliver/.gemini/antigravity/brain/b1ab1c8a-66f1-4193-942b-b09ee6af16fe/query_theme_ranking_diagnostic.md'
    with open(artifact_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
        
    print(f"Done. Wrote to {artifact_path}")

if __name__ == "__main__":
    main()
