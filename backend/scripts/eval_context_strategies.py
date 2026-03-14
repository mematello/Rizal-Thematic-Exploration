import sys
import os
import numpy as np

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
    
    text = " ".join([s.sentence_text for s in candidates])
    return text

def normalize(vec):
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec

def format_theme(theme_dict):
    if not theme_dict: return "None (0.00)"
    return f"{theme_dict['label']} ({abs(round(float(theme_dict['score']), 2))})"

def format_themes_list(themes):
    if not themes: return "None (0.00)"
    return ", ".join([format_theme(t) for t in themes[:1]]) # Just show top 1


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
    output_lines = ["# Context and Query Conditioning Strategy Comparison\n"]
    
    with SessionLocal() as db:
        engine._ensure_themes_loaded(db)
        
        for q in queries:
            output_lines.append(f"## Query: `{q}`\n")
            try:
                res = engine.search(db=db, query=q, source_type="full")
                results = res.get("results", {})
                combined = results.get("noli", []) + results.get("elfili", [])
                
                if not combined:
                    output_lines.append("*No results found.*\n")
                    continue
                
                # Encode Query
                query_vec = normalize(engine.base_model.encode(q, show_progress_bar=False))
                # Compute query-theme similarity statically for all themes
                query_theme_sims = np.dot(engine.theme_matrix, query_vec)
                    
                # Take top 2 results
                for i, item in enumerate(combined[:2]):
                    sent_id = item.get("id")
                    if not sent_id: continue
                    
                    center_sent = db.scalars(select(Sentence).filter(Sentence.id == sent_id)).first()
                    if not center_sent: continue
                    
                    exact_text = center_sent.sentence_text
                    
                    # 1. Sentence-only classification (baseline)
                    themes_exact = engine._classify_themes(db, center_sent.embedding, exact_text)
                    
                    # 2. Passage-block classification (current experiment)
                    passage_text = get_passage_context(db, center_sent, window=8)
                    passage_emb = normalize(engine.base_model.encode(passage_text, show_progress_bar=False))
                    themes_passage = engine._classify_themes(db, passage_emb, passage_text)
                    
                    # Get raw vector scores for manual blends
                    sent_sims = np.dot(engine.theme_matrix, normalize(np.array(center_sent.embedding)))
                    pass_sims = np.dot(engine.theme_matrix, passage_emb)
                    
                    # 3. Blended scoring (0.4 sentence + 0.6 passage) - using lexical score of passage for simplicity
                    candidates_blend = []
                    candidates_q_cond = []
                    candidates_q_gate = []
                    
                    for tidx, theme in enumerate(engine.theme_cache):
                        lex_score = engine._compute_simple_lexical(passage_text, theme['meaning'])
                        
                        s_score = max(0.0, float(sent_sims[tidx]))
                        p_score = max(0.0, float(pass_sims[tidx]))
                        q_score = max(0.0, float(query_theme_sims[tidx]))
                        
                        # Existing _classify_themes uses: (0.5 * sem_sim) + (0.5 * lex_score)
                        # We reconstruct the base sentence and passage final scores to blend them properly
                        sent_lex = engine._compute_simple_lexical(exact_text, theme['meaning'])
                        final_s_score = (0.5 * s_score) + (0.5 * sent_lex)
                        final_p_score = (0.5 * p_score) + (0.5 * lex_score)
                        
                        # Strategy 3: Blend (e.g., 40% sentence, 60% passage to keep grounded)
                        blend_score = (0.4 * final_s_score) + (0.6 * final_p_score)
                        candidates_blend.append({'label': theme['tagalog_title'], 'score': blend_score})
                        
                        # Strategy 4: Query-conditioned (Rerank passage themes based on query sim)
                        # Combine passage semantic score with query semantic score
                        q_cond_score = (0.5 * final_p_score) + (0.5 * q_score)
                        candidates_q_cond.append({'label': theme['tagalog_title'], 'score': q_cond_score})
                        
                        # Strategy 5: Query gate (Use passage themes but suppress if query similarity is extremely low)
                        # If a theme has almost 0 relationship to the query, heavily penalize it
                        p_gated_score = final_p_score
                        if q_score < 0.20:
                            p_gated_score = p_gated_score * 0.3 # Severe penalty if irrelevant to query
                        candidates_q_gate.append({'label': theme['tagalog_title'], 'score': p_gated_score})
                    
                    candidates_blend.sort(key=lambda x: x['score'], reverse=True)
                    candidates_q_cond.sort(key=lambda x: x['score'], reverse=True)
                    candidates_q_gate.sort(key=lambda x: x['score'], reverse=True)
                    
                    output_lines.append(f"**Result {i+1}:**")
                    output_lines.append(f"> {exact_text}")
                    output_lines.append(f"- **1. Sentence-only (baseline):** {format_themes_list(themes_exact)}")
                    output_lines.append(f"- **2. Passage-block:** {format_themes_list(themes_passage)}")
                    output_lines.append(f"- **3. Blended scoring (40S / 60P):** {format_themes_list(candidates_blend)}")
                    output_lines.append(f"- **4. Query-conditioned rerank:** {format_themes_list(candidates_q_cond)}")
                    output_lines.append(f"- **5. Passage w/ Query Gate (sim < 0.2):** {format_themes_list(candidates_q_gate)}")
                    output_lines.append("\n*Judgment*: [Fill in post-review]")
                    output_lines.append("\n---\n")
                    
            except Exception as e:
                import traceback
                output_lines.append(f"*Error during search: {e}*\n ```\n{traceback.format_exc()}\n```\n")
                
    artifact_path = '/Users/marcusoliver/.gemini/antigravity/brain/b1ab1c8a-66f1-4193-942b-b09ee6af16fe/context_strategy_comparison.md'
    with open(artifact_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
        
    print(f"Done. Wrote to {artifact_path}")

if __name__ == "__main__":
    main()
