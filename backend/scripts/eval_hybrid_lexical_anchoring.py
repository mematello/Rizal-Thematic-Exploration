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

def generate_hybrid_theme_pool(engine, query, target_pool_size=5):
    """
    Implements the final `Lexical Anchor Injection + Embedding Fill` architecture.
    """
    pool_indices = set()
    anchors_triggered = []
    
    q_lower = query.lower()
    
    # 1. Lexical Anchor Pass (Hard-coded aliases for evaluation, normally would map explicitly)
    lexical_aliases = {
        "kalayaan": ["Kalayaan at Pagmamahal sa Bayan"],
        "bayan": ["Kalayaan at Pagmamahal sa Bayan"],
        "edukasyon": ["Edukasyon", "Edukasyon at Kalayaan"],
        "paaralan": ["Edukasyon"],
        "simbahan": ["Katiwalian", "Kolonyal na Kaisipan at Paghahangad Umasenso", "Kapangyarihan at Kawalang-Katarungan"],
        "prayle": ["Katiwalian", "Ipokrasya at Pang-aaping Kolonyal"],
        "pang-aapi": ["Kawalang-Katarungan at Katarungan", "Kapangyarihan at Kawalang-Katarungan", "Pang-aapi sa Kababaihan"],
        "kababaihan": ["Pang-aapi sa Kababaihan"],
        "pag-ibig": ["Pag-ibig, Kadalisayan, at Katapatan"]
    }
    
    # Pre-compute the query semantic ranking.
    q_emb_raw = engine.base_model.encode(query, show_progress_bar=False)
    if isinstance(q_emb_raw, list): q_emb_raw = np.array(q_emb_raw)
    query_vec = normalize(q_emb_raw)
    q_sims = np.dot(engine.theme_matrix, query_vec)
    ranked_indices = np.argsort(q_sims)[::-1]
    
    # Check if query matches exact theme titles
    for i, theme in enumerate(engine.theme_cache):
        title_tag = theme['tagalog_title']
        title_lower = title_tag.lower()
        # Direct word match against the theme title mapping
        if q_lower in lexical_aliases:
            if title_tag in lexical_aliases[q_lower]:
                pool_indices.add(i)
                anchors_triggered.append(title_tag)
        # Catch explicit subsets
        elif title_lower == q_lower or q_lower in title_lower.split():
             pool_indices.add(i)
             anchors_triggered.append(title_tag)
                
    # 2. Embedding Pool Fill Pass
    # Fill remaining slots up to target_pool_size using semantic vectors
    for idx in ranked_indices:
        if len(pool_indices) >= target_pool_size:
            break
        pool_indices.add(idx)
        
    final_pool = list(pool_indices)
    return final_pool, anchors_triggered

def main():
    queries = [
        "edukasyon",
        "pang-aapi ng kastila",
        "simbahan",
        "kalayaan",
        "elias",
        "maria clara"
    ]
    
    output_lines = ["# Hybrid Lexical Anchoring Theme Classification Evaluation\n"]
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
                
                # --- NEW HYBRID POOL LOGIC ---
                pool_indices, anchors = generate_hybrid_theme_pool(engine, q, target_pool_size=5)
                
                if anchors:
                    output_lines.append(f"**Lexical Anchors Triggered:** {', '.join(anchors)}")
                else:
                    output_lines.append(f"**Lexical Anchors Triggered:** *None (Semantic Fill Only)*")
                    
                pool_labels = [engine.theme_cache[i]['tagalog_title'] for i in pool_indices]
                output_lines.append(f"**Final Candidate Pool:**\n- " + "\n- ".join(pool_labels) + "\n")
                
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
                    
                    candidates = []
                    
                    # Score ONLY against the Hybrid Pool
                    for tidx in pool_indices:
                        theme = engine.theme_cache[tidx]
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
                            
                        candidates.append({'label': theme['tagalog_title'], 'score': score})
                        
                    candidates.sort(key=lambda x: x['score'], reverse=True)
                    
                    output_lines.append(f"**Result {i+1}:**")
                    output_lines.append(f"> {exact_text}")
                    output_lines.append(f"- **Final Selected Theme:** {format_theme(candidates[0])}")
                    output_lines.append("\n*Judgment*: ")
                    output_lines.append("---\n")
                    
            except Exception as e:
                output_lines.append(f"*Error during search: {e}*")

    artifact_path = '/Users/marcusoliver/.gemini/antigravity/brain/b1ab1c8a-66f1-4193-942b-b09ee6af16fe/lexical_anchor_evaluation.md'
    with open(artifact_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
        
    print(f"Done. Wrote to {artifact_path}")

if __name__ == "__main__":
    main()
