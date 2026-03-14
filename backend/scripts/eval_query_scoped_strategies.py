import sys
import os
import numpy as np
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.core.engine import RizalEngine
from app.models.database import SessionLocal, Sentence
from sqlalchemy import select

def get_passage_context(db, center, window=8):
    range_start, range_end = center.sentence_index - window, center.sentence_index + window
    candidates = db.scalars(select(Sentence).filter(
        Sentence.book == center.book, 
        Sentence.chapter_number == center.chapter_number, 
        Sentence.source_type == center.source_type,
        Sentence.sentence_index >= range_start, 
        Sentence.sentence_index <= range_end
    ).order_by(Sentence.sentence_index)).all()
    
    return " ".join([s.sentence_text for s in candidates])

def normalize(vec):
    try:
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec
    except:
        return np.zeros(768)

def format_theme(theme_dict):
    if not theme_dict: return "None (0.00)"
    try:
        return f"{theme_dict['label']} ({abs(round(float(theme_dict['score']), 2))})"
    except:
        return "Error Formatting Theme"

def determine_query_type(query):
    chars = ["elias", "maria clara", "ibarra", "simoun"]
    if query.lower() in chars:
        return "Character"
    
    exact_themes = ["edukasyon", "kalayaan", "simbahan"]
    if query.lower() in exact_themes:
        return "Theme Keyword"
        
    return "Abstract Phrase"

def main():
    queries = [
        "edukasyon",
        "pang-aapi ng kastila",
        "simbahan",
        "kalayaan",
        "elias",
        "maria clara"
    ]
    
    output_lines = ["# Query-Scoped Theme Restriction Evaluation\n"]
    engine = RizalEngine()
    
    with SessionLocal() as db:
        engine._ensure_themes_loaded(db)
        
        for q in queries:
            q_type = determine_query_type(q)
            output_lines.append(f"## Query: `{q}`\n*Type: {q_type}*\n")
            try:
                res = engine.search(db=db, query=q, source_type="full")
                results = res.get("results", {})
                combined = results.get("noli", []) + results.get("elfili", [])
                
                if not combined:
                    output_lines.append("*No results found.*\n")
                    continue
                
                # --- NEW LOGIC: Pre-compute the Query-Scoped Theme Pool ---
                q_emb_raw = engine.base_model.encode(q, show_progress_bar=False)
                if isinstance(q_emb_raw, list): q_emb_raw = np.array(q_emb_raw)
                query_vec = normalize(q_emb_raw)
                
                q_sims = np.dot(engine.theme_matrix, query_vec)
                
                # Get indices of top 5 themes for this query
                target_pool_size = 5
                top_theme_indices = np.argsort(q_sims)[-target_pool_size:][::-1]
                
                # Print the restricted candidate pool
                pool_labels = [engine.theme_cache[idx]['tagalog_title'] for idx in top_theme_indices]
                output_lines.append(f"**Restricted Theme Pool (Top {target_pool_size} vs Query):**")
                for i, idx in enumerate(top_theme_indices):
                    output_lines.append(f"- {pool_labels[i]} ({q_sims[idx]:.2f})")
                output_lines.append("")
                
                # Take top 2 results
                for i, item in enumerate(combined[:2]):
                    sent_id = item.get("id")
                    if not sent_id: continue
                    center_sent = db.scalars(select(Sentence).filter(Sentence.id == sent_id)).first()
                    if not center_sent: continue
                    
                    exact_text = center_sent.sentence_text
                    passage_text = get_passage_context(db, center_sent, window=8)
                    
                    raw_emb = center_sent.embedding
                    if raw_emb is None: continue
                    if isinstance(raw_emb, list): raw_emb = np.array(raw_emb)
                    
                    p_emb_raw = engine.base_model.encode(passage_text, show_progress_bar=False)
                    if isinstance(p_emb_raw, list): p_emb_raw = np.array(p_emb_raw)
                    passage_emb = normalize(p_emb_raw)
                    
                    sent_sims = np.dot(engine.theme_matrix, normalize(raw_emb))
                    pass_sims = np.dot(engine.theme_matrix, passage_emb)
                    
                    unrestricted_candidates = []
                    restricted_candidates = []
                    
                    # 1. First build the UNRESTRICTED candidates (Baseline from Phase 29)
                    for tidx, theme in enumerate(engine.theme_cache):
                        meaning = theme.get('meaning', '')
                        s_score = max(0.0, float(sent_sims[tidx]))
                        p_score = max(0.0, float(pass_sims[tidx]))
                        lex_score = engine._compute_simple_lexical(passage_text, meaning)
                        sent_lex = engine._compute_simple_lexical(exact_text, meaning)
                        
                        final_s_score = (0.5 * s_score) + (0.5 * sent_lex)
                        final_p_score = (0.5 * p_score) + (0.5 * lex_score)
                        
                        score = 0
                        if q_type == "Character": score = final_p_score
                        elif q_type == "Theme Keyword": score = (0.7 * final_s_score) + (0.3 * final_p_score)
                        else: score = (0.4 * final_s_score) + (0.6 * final_p_score)
                            
                        # Add to unrestricted pool
                        unrestricted_candidates.append({'label': theme['tagalog_title'], 'score': score})
                        
                        # Add to restricted pool ONLY if it's in the top N target indices
                        if tidx in top_theme_indices:
                            restricted_candidates.append({'label': theme['tagalog_title'], 'score': score})
                    
                    unrestricted_candidates.sort(key=lambda x: x['score'], reverse=True)
                    restricted_candidates.sort(key=lambda x: x['score'], reverse=True)
                    
                    output_lines.append(f"**Result {i+1}:**")
                    output_lines.append(f"> {exact_text}")
                    output_lines.append(f"- **Unrestricted Prediction:** {format_theme(unrestricted_candidates[0])}")
                    output_lines.append(f"- **Restricted Pool Prediction:** {format_theme(restricted_candidates[0])}")
                    output_lines.append("\n---\n")
                    
            except Exception as e:
                output_lines.append(f"*Error during search: {e}*")

    artifact_path = '/Users/marcusoliver/.gemini/antigravity/brain/b1ab1c8a-66f1-4193-942b-b09ee6af16fe/query_scoped_theme_eval.md'
    with open(artifact_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
        
    print(f"Done. Wrote to {artifact_path}")

if __name__ == "__main__":
    main()
