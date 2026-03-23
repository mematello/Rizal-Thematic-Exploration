import os
import sys
import numpy as np
import re
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add backend root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.models.database import Sentence
from app.core.config import get_settings
from app.core.engine import RizalEngine

def debug_ch7():
    settings = get_settings()
    engine_db = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine_db)
    db = SessionLocal()
    rizal_engine = RizalEngine()
    
    # Kabanata 7 Summary Sentence
    buod = db.query(Sentence).filter(
        Sentence.book == 'noli',
        Sentence.chapter_number == 7,
        Sentence.source_type == 'summary'
    ).first()
    
    if not buod:
        print("Summary sentence not found.")
        return

    buod_text = buod.sentence_text.lower()
    print(f"BUOD: {buod_text}")
    
    # Extract characters
    book_key = "noli"
    patterns = rizal_engine.char_patterns.get(book_key, [])
    buod_chars = set()
    for canon_name, pattern in patterns:
        if pattern.search(buod_text):
            buod_chars.add(canon_name)
    print(f"BUOD CHARS: {buod_chars}")

    # Search Logic (similar to API)
    full_sentences = db.query(Sentence).filter(
        Sentence.book == 'Noli Me Tangere',
        Sentence.chapter_number == 7,
        Sentence.source_type == 'full'
    ).order_by(Sentence.sentence_index).all()
    
    print(f"Full sentences in chapter: {len(full_sentences)}")
    
    # Calculate search window
    buod_sentences = db.query(Sentence).filter(
        Sentence.book == 'noli',
        Sentence.chapter_number == 7,
        Sentence.source_type == 'summary'
    ).order_by(Sentence.sentence_index).all()
    
    total_full = len(full_sentences)
    total_buod = len(buod_sentences)
    chapter_ratio = total_full / total_buod
    dynamic_buffer = chapter_ratio * 4.0 # Larger buffer for debug
    
    buod_idx = next((i for i, bs in enumerate(buod_sentences) if bs.id == buod.id), 0)
    position_ratio = buod_idx / total_buod
    search_center = position_ratio * total_full
    
    print(f"Window: {search_center - dynamic_buffer:.1f} to {search_center + dynamic_buffer:.1f}")
    
    candidates = [fs for i, fs in enumerate(full_sentences) if (search_center - dynamic_buffer) <= i <= (search_center + dynamic_buffer)]
    print(f"Candidates in window: {len(candidates)}")

    for fs in candidates:
        fs_text = (fs.sentence_text or "").lower()
        
        # Passage-level check (NEW)
        passage_text = fs_text
        if fs.passage_id is not None:
            passage_sents = db.query(Sentence).filter(
                Sentence.book == fs.book,
                Sentence.chapter_number == fs.chapter_number,
                Sentence.source_type == 'full',
                Sentence.passage_id == fs.passage_id
            ).order_by(Sentence.sentence_index).all()
            passage_text = " ".join([(s.sentence_text or "").lower() for s in passage_sents])
        
        fs_chars = set()
        for canon_name, pattern in patterns:
            if pattern.search(passage_text):
                fs_chars.add(canon_name)
        
        char_pass = True
        if buod_chars:
            if not buod_chars.issubset(fs_chars):
                char_pass = False
        
        # DEBUG PRINT
        if fs.passage_id is not None and "maria clara" in passage_text:
             # Only print once per passage to avoid spam
             if fs.sentence_index == min(s.sentence_index for s in passage_sents):
                 print(f"DEBUG Passage ID {fs.passage_id} (Length {len(passage_text)}): {passage_text[:150]}...")
                 print(f"DEBUG Detected Chars in Passage: {fs_chars}")
        
        # Scoring
        buod_emb = np.array(buod.embedding)
        fs_emb = np.array(fs.embedding)
        num = float(np.dot(buod_emb, fs_emb))
        den = (np.linalg.norm(buod_emb) * np.linalg.norm(fs_emb))
        semantic_score = num / den if den > 0 else 0.0
        lexical_score = rizal_engine._compute_simple_lexical(buod_text, fs_text)
        final_score = (0.55 * lexical_score) + (0.45 * semantic_score)
        
        indicator = ""
        if char_pass:
            if final_score >= 0.45:
                indicator = "[PASS]"
            else:
                indicator = "[PASS_FILTER_LOW_SCORE]"
        else:
            indicator = "[FAIL_STAGE2]"
            
        if char_pass or final_score > 0.4:
            print(f"{indicator} Score: {final_score:.4f} (L:{lexical_score:.2f} S:{semantic_score:.2f}) | Text: {fs.sentence_text[:60]}...")

if __name__ == "__main__":
    debug_ch7()
