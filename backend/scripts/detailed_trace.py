import sys
import os
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.database import SessionLocal, Sentence
from app.core.engine import get_engine
from app.core.analyzer import QueryAnalyzer, extract_words
from app.core.tagalog_parser import TagalogRoleParser

def run_trace():
    db = SessionLocal()
    engine = get_engine()
    query = "pagpapasakop ng bayan sa dayuhang kapangyarihan"
    
    print("="*60)
    print(f"QUERY: {query}")
    print("="*60)
    
    # 1. Query Validation Output
    print("\n--- 1. Query Validation Output ---")
    modern_terms = engine.MODERN_TERMS
    pass_blocklist = not any(term in query.lower() for term in modern_terms)
    print(f"Passed modern-term blocklist check? {'Yes' if pass_blocklist else 'No'}")
    
    query_words = extract_words(query.lower())
    engine.query_analyzer.STOPWORDS = engine.query_analyzer._load_official_stopwords()
    analysis = engine.query_analyzer.analyze_query_words(query)
    stopword_ratio = engine.query_analyzer.get_stopword_ratio(query)
    print("wordfreq and linguistic validation:")
    for a in analysis:
        print(f" - {a['word']}: freq={a['frequency']}, is_stop={a['is_stopword']}")
    print(f"Valid word ratio (not stopword): {1.0 - stopword_ratio:.2f}")
    
    # Check Theme Proximity (Domain Alignment)
    engine._ensure_themes_loaded(db)
    q_vec_base = engine.base_model.encode(query, show_progress_bar=False)
    q_vec_base_norm = q_vec_base / np.linalg.norm(q_vec_base) if np.linalg.norm(q_vec_base) > 0 else q_vec_base
    theme_scores = np.dot(engine.theme_matrix, q_vec_base_norm)
    max_theme_score = float(np.max(theme_scores))
    print(f"DomainAlignment/ThemeProximity score: {max_theme_score:.4f}")
    
    threshold = 0.50 if any(w not in engine.vocabulary for w in query_words) else 0.40
    print(f"Threshold for Domain Alignment: {threshold:.2f}")
    validation_pass = len(query_words) > 0 and pass_blocklist
    print(f"Did it pass or fail validation? {'Pass' if validation_pass else 'Fail'}")

    # 2. Query Preprocessing Output
    print("\n--- 2. Query Preprocessing Output ---")
    total_query_weight = 0
    print("Word-by-word breakdown:")
    for a in analysis:
        word_type = "stopword" if a['is_stopword'] else "content word"
        print(f" - {a['word']}: {word_type}, Semantic Weight: {a['semantic_weight']}")
        total_query_weight += a['semantic_weight']
    print(f"Total query weight: {total_query_weight:.2f}")
    
    sig_words = [item['word'].lower() for item in analysis if not item['is_stopword']]
    is_single_word = len(sig_words) < 2
    
    print(f"Query embedding generated? Yes (Base Shape: {q_vec_base.shape})")
    
    try:
        # To get the exact trace, we can mock or use another script to copy engine search
        # Oh wait we need exactly Stage A and Stage B counts
        pass
    except Exception as e:
        print(e)
        
    db.close()

if __name__ == '__main__':
    run_trace()
