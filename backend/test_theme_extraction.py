import os
import sys
import numpy as np
from sqlalchemy.orm import Session
from app.models.database import SessionLocal
from app.core.engine import RizalEngine

def run_theme_extraction_test():
    engine = RizalEngine()
    db = SessionLocal()
    engine._ensure_themes_loaded(db)
    
    test_words = ["power", "freedom", "fear", "poverty", "truth", "disease", "wealth"]
    
    print(f"{'Query':<15} | {'MaxSim':<6} | {'Matched Theme'} | {'Extracted Keywords'}")
    print("-" * 80)
    
    for q in test_words:
        q_vec = engine.base_model.encode(q, show_progress_bar=False)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0: q_vec = q_vec / q_norm
        
        # Manually compute similarity against theme matrix
        theme_scores = np.dot(engine.theme_matrix, q_vec)
        top_indices = np.argsort(theme_scores)[::-1][:2]
        
        max_sim = theme_scores[top_indices[0]]
        
        # Simulate extraction with threshold = 0.25 (to see what it grabs)
        candidates = set()
        best_title = "None"
        for i, idx in enumerate(top_indices):
            score = theme_scores[idx]
            if score < 0.25: continue
            
            theme = engine.theme_cache[idx]
            if i == 0:
                best_title = theme['tagalog_title']
                target_theme_key = best_title.strip().lower()
                
            from app.core.analyzer import extract_words
            meaning_words = engine.query_analyzer.analyze_query_words(theme['meaning'].lower())
            meaning_words = [item['word'] for item in meaning_words if not item['is_stopword']]
            candidates.update(meaning_words)
            
        final_tokens = []
        if candidates and best_title != "None":
            for w in candidates:
                if len(w) <= 3: continue
                df = engine.theme_word_df.get(w, 0)
                if df < 1: continue 
                theme_counts = engine.word_theme_map.get(w, {})
                count_in_target = theme_counts.get(target_theme_key, 0)
                if count_in_target == 0: continue
                
                specificity = count_in_target / df
                if specificity >= 0.30: 
                    final_tokens.append(w)
                    
            final_tokens.sort(key=lambda w: engine.word_theme_map.get(w, {}).get(target_theme_key, 0) / engine.theme_word_df.get(w, 1), reverse=True)
        
        print(f"{q:<15} | {max_sim:.3f} | {best_title[:20]:<20} | {final_tokens[:5]}")

    db.close()

if __name__ == "__main__":
    run_theme_extraction_test()
