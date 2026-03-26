import os
import sys
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.models.database import engine, Sentence
from app.core.analyzer import extract_words

def merge_short_sentences():
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    print("Fetching all sentences...")
    # Process by book, chapter, and source_type to keep them together
    groups = session.query(
        Sentence.book, 
        Sentence.chapter_number, 
        Sentence.source_type
    ).distinct().all()
    
    total_merged = 0
    
    for book, ch_num, s_type in groups:
        sentences = session.query(Sentence).filter(
            Sentence.book == book,
            Sentence.chapter_number == ch_num,
            Sentence.source_type == s_type
        ).order_by(Sentence.sentence_index).all()
        
        if not sentences:
            continue
            
        i = 0
        while i < len(sentences):
            s = sentences[i]
            text = (s.sentence_text or "").strip()
            words = extract_words(text)
            
            # Dialogue marker check: if it starts with a dash or quote, it might be a new speaker
            is_new_speaker = text.startswith('-') or text.startswith('—') or text.startswith('"') or text.startswith("'")
            
            if len(words) <= 3 and len(sentences) > 1:
                # Decide which neighbor to merge with
                # Prefer merging FORWARD if it looks like a new speaker fragment
                if is_new_speaker and i < len(sentences) - 1:
                    # Merge with next (forward) to keep with the same speaker
                    target = sentences[i+1]
                    target.sentence_text = text + " " + target.sentence_text.strip()
                    session.delete(s)
                    sentences.pop(i)
                    total_merged += 1
                elif i > 0:
                    # Default: Merge with previous (backward)
                    target = sentences[i-1]
                    target.sentence_text = target.sentence_text.strip() + " " + text
                    session.delete(s)
                    sentences.pop(i)
                    total_merged += 1
                elif i < len(sentences) - 1:
                    # Fallback: Merge with next
                    target = sentences[i+1]
                    target.sentence_text = text + " " + target.sentence_text.strip()
                    session.delete(s)
                    sentences.pop(i)
                    total_merged += 1
                else:
                    i += 1
            else:
                i += 1
        
        # Re-index remaining sentences in this group to ensure continuity
        for idx, s in enumerate(sentences):
            s.sentence_index = idx
            
    print(f"Merging completed. Total fragments merged: {total_merged}")
    session.commit()
    session.close()

if __name__ == "__main__":
    merge_short_sentences()
