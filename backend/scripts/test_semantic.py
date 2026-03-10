import numpy as np
from sqlalchemy.orm import Session
from app.models.database import SessionLocal, Sentence
from app.core.engine import get_engine
from sqlalchemy import select

def test_semantic_threshold():
    db = SessionLocal()
    engine = get_engine()
    
    queries = ["edukasyon", "bitcoin", "korapsyon", "katiwalian", "paaralan", "tiktok", "elon musk"]
    
    print("Testing Semantic Fallback Embeddings")
    
    for q in queries:
        print(f"\n--- Query: {q} ---")
        emb = engine.base_model.encode(q, show_progress_bar=False)
        emb_list = emb.tolist()
        
        # Get top 5 semantic matches from Noli full text
        results = db.scalars(
            select(Sentence)
            .filter(Sentence.book == "Noli Me Tangere")
            .filter(Sentence.source_type == "full")
            .order_by(Sentence.embedding.cosine_distance(emb_list))
            .limit(5)
        ).all()
        
        for r in results:
            v_sent = np.array(r.embedding)
            v_sent = v_sent / np.linalg.norm(v_sent) if np.linalg.norm(v_sent) > 0 else v_sent
            v_query = emb / np.linalg.norm(emb)
            sim = float(np.dot(v_query, v_sent))
            print(f"Score: {sim:.3f} | Text: {r.sentence_text}")

if __name__ == "__main__":
    test_semantic_threshold()
