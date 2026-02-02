from app.models.database import SessionLocal, Sentence
from app.core.engine import get_engine
import numpy as np

def debug_theme_scoring():
    db = SessionLocal()
    engine = get_engine()
    
    # Get a sentence about education
    sentence = db.query(Sentence).filter(Sentence.sentence_text.ilike("%edukasyon%")).first()
    if not sentence:
        print("No sentence found")
        return

    print(f"Sentence: {sentence.sentence_text}")
    print(f"Vector Norm: {np.linalg.norm(np.array(sentence.embedding))}")
    
    # Run classification
    # Force load cache
    if not hasattr(engine, 'theme_cache'):
        from app.models.database import Theme
        themes = db.query(Theme).all()
        engine.theme_cache = [
            {
                'id': t.id,
                'tagalog_title': t.tagalog_title,
                'meaning': t.meaning,
                'embedding': np.array(t.embedding) / np.linalg.norm(np.array(t.embedding)),
                'meaning_len': len(t.meaning.split())
            }
            for t in themes
        ]
        
    sent_vec = np.array(sentence.embedding)
    norm = np.linalg.norm(sent_vec)
    sent_vec = sent_vec / norm
    
    print("\n--- Scores ---")
    for theme in engine.theme_cache:
        sem_sim = float(np.dot(sent_vec, theme['embedding']))
        lex_score = engine._compute_simple_lexical(sentence.sentence_text, theme['meaning'])
        
        lambda_lex, lambda_sem = engine._compute_dynamic_weights_by_length(theme['meaning_len'], len(sentence.sentence_text.split()))
        theme_score = (lambda_sem * sem_sim) + (lambda_lex * lex_score)
        
        if sem_sim > 0.3: # Only print relevant-ish ones
            print(f"Theme: {theme['tagalog_title']}")
            print(f"  Sem: {sem_sim:.4f}")
            print(f"  Lex: {lex_score:.4f}")
            print(f"  Final: {theme_score:.4f}")

if __name__ == "__main__":
    debug_theme_scoring()
