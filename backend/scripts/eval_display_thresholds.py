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
    if not theme_dict: return "None", 0.0
    try:
        return theme_dict['label'], float(theme_dict['score'])
    except:
        return "Error", 0.0

def determine_query_type(query):
    chars = ["elias", "maria clara"]
    if query.lower() in chars:
        return "Character"
    
    exact_themes = ["edukasyon", "kalayaan", "simbahan"]
    if query.lower() in exact_themes:
        return "Theme Keyword"
        
    return "Abstract Phrase"

def generate_hybrid_theme_pool(engine, query, target_pool_size=5):
    pool_indices = set()
    q_lower = query.lower()
    
    lexical_aliases = {
        "kalayaan": ["Kalayaan at Pagmamahal sa Bayan"],
        "edukasyon": ["Edukasyon", "Edukasyon at Kalayaan"],
        "simbahan": ["Katiwalian", "Kolonyal na Kaisipan at Paghahangad Umasenso", "Kapangyarihan at Kawalang-Katarungan"],
        "pang-aapi": ["Kawalang-Katarungan at Katarungan", "Kapangyarihan at Kawalang-Katarungan", "Pang-aapi sa Kababaihan"]
    }
    
    q_emb_raw = engine.base_model.encode(query, show_progress_bar=False)
    if isinstance(q_emb_raw, list): q_emb_raw = np.array(q_emb_raw)
    query_vec = normalize(q_emb_raw)
    q_sims = np.dot(engine.theme_matrix, query_vec)
    ranked_indices = np.argsort(q_sims)[::-1]
    
    for i, theme in enumerate(engine.theme_cache):
        title_tag = theme['tagalog_title']
        title_lower = title_tag.lower()
        if q_lower in lexical_aliases:
            if title_tag in lexical_aliases[q_lower]:
                pool_indices.add(i)
        elif title_lower == q_lower or q_lower in title_lower.split():
             pool_indices.add(i)
                
    for idx in ranked_indices:
        if len(pool_indices) >= target_pool_size:
            break
        pool_indices.add(idx)
        
    return list(pool_indices)

def main():
    queries = [
        "edukasyon",
        "pang-aapi ng kastila",
        "simbahan",
        "kalayaan",
        "elias",
        "maria clara"
    ]
    
    engine = RizalEngine()
    
    results_store = []
    
    with SessionLocal() as db:
        engine._ensure_themes_loaded(db)
        
        for q in queries:
            q_type = determine_query_type(q)
            try:
                res = engine.search(db=db, query=q, source_type="full")
                results = res.get("results", {})
                combined = results.get("noli", []) + results.get("elfili", [])
                
                if not combined: continue
                
                pool_indices = generate_hybrid_theme_pool(engine, q, target_pool_size=5)
                
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
                    label, score = format_theme(candidates[0])
                    
                    results_store.append({
                        "query": q,
                        "text": exact_text,
                        "label": label,
                        "score": score
                    })
                    
            except Exception as e:
                 print(f"Error on {q}: {e}")

    # Format the Markdown Table Output
    output_lines = ["# Theme Display-Readiness Confidence Evaluation\n"]
    
    output_lines.append("This evaluation analyzes how many and which theme pills survive at various confidence thresholds before going to the UI.\n")
    
    thresholds = [0.45, 0.50, 0.55, 0.60]
    
    for t in thresholds:
        output_lines.append(f"## Threshold: `>= {t}`")
        survivors = [r for r in results_store if r["score"] >= t]
        
        output_lines.append(f"**Surviving Labels:** {len(survivors)} / {len(results_store)} ({(len(survivors)/len(results_store)*100):.1f}% survival rate)\n")
        
        if survivors:
            for r in survivors:
                output_lines.append(f"- **Query:** `{r['query']}`")
                output_lines.append(f"  > \"{r['text']}\"")
                output_lines.append(f"  **Theme:** {r['label']} ({r['score']:.2f})")
                output_lines.append(f"  *Judgment:* [Fill post-review]\n")
        else:
            output_lines.append("*All labels suppressed at this threshold.*\n")

    output_lines.append("## Proposed Final Display Strategy")
    output_lines.append("> [To be filled out after reviewing the threshold tiers above.]\n")

    artifact_path = '/Users/marcusoliver/Desktop/Rizal-Thematic-Exploration/display_readiness_evaluation.txt'
    with open(artifact_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
        
    print(f"Done. Wrote to {artifact_path}")

if __name__ == "__main__":
    main()
