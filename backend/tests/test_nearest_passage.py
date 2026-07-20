import os
import sys
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models.database import SessionLocal, Sentence
from app.core.analyzer import extract_words
from app.core.engine import RizalEngine

def extract_frequent_words(sentences):
    # Quick utility to find most frequent words
    words = {}
    blacklist = {
        'pag', 'upang', 'ngunit', 'bilang', 'ginamit', 'kanyang', 'kaniyang',
        'dahil', 'isang', 'siya', 'naging', 'mga', 'ang', 'sa', 'ng', 'na',
        'at', 'ito', 'niya', 'nang', 'ay', 'si', 'sila', 'kami', 'tayo',
        'asa', 'para', 'lang', 'muli', 'kahit', 'mga', 'isang', 'pa',
        'hindi', 'walang', 'wala', 'din', 'rin', 'naman', 'muna', 'muli',
        'aking', 'siyang', 'kanilang', 'iyong', 'ating', 'inyong', 'namin',
        'gayon', 'ano', 'kapag', 'lamang', 'sana', 'nito', 'nila'
    }
    
    for s in sentences:
        txt = s.sentence_text.lower()
        w_list = extract_words(txt)
        for w in w_list:
            if len(w) > 3 and w not in blacklist:
                words[w] = words.get(w, 0) + 1
                
    # Sort by frequency
    sorted_words = sorted(words.items(), key=lambda x: x[1], reverse=True)
    return [w[0] for w in sorted_words[:3]]

def run_nearest_passage():
    engine = RizalEngine()
    db = SessionLocal()
    
    test_words = ["power", "freedom", "fear", "poverty", "truth", "disease", "wealth"]
    
    print(f"{'Query':<15} | {'Top 5 Sentence Words (Mined Anchors)'}")
    print("-" * 50)
    
    for q in test_words:
        q_vec = engine.base_model.encode(q, show_progress_bar=False)
        q_vec = q_vec / np.linalg.norm(q_vec)
        
        # Get top 5 nearest Tagalog sentences from Noli
        results = db.scalars(
            select(Sentence)
            .filter(Sentence.book == "Noli Me Tangere")
            .filter(Sentence.source_type == "full")
            .order_by(Sentence.embedding.cosine_distance(q_vec.tolist()))
            .limit(10)
        ).all()
        
        mined_anchors = extract_frequent_words(results)
        
        print(f"{q:<15} | {mined_anchors}")

    db.close()

if __name__ == "__main__":
    run_nearest_passage()
