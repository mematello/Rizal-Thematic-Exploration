import sys
import os
import numpy as np
from sqlalchemy.orm import sessionmaker

# Append backend root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.models.database import engine, Sentence
from app.core.config import get_settings

def compute_cosine_similarity(v1, v2):
    if v1 is None or v2 is None: return 0.0
    v1 = np.array(v1)
    v2 = np.array(v2)
    if v1.size == 0 or v2.size == 0: return 0.0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)

def segment_chapters():
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    print("Ensuring passage_id column exists...")
    from sqlalchemy import text
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE sentences ADD COLUMN IF NOT EXISTS passage_id INTEGER;"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_sentences_passage_id ON sentences (passage_id);"))
            conn.commit()
        except Exception as e:
            print(f"Migration error (might already exist): {e}")

    print("Updating Noli Buod Ch64 -> Ch63 hard mapping")
    # For buod (source_type == 'summary'), if book is 'noli' and Ch=64
    session.query(Sentence).filter(
        (Sentence.book == 'noli') | (Sentence.book == 'Noli Me Tangere'),
        Sentence.source_type == 'summary',
        Sentence.chapter_number == 64
    ).update({"chapter_number": 63})
    session.commit()
    
    print("Assigning passage_ids to buod sentences (1:1 mapping)")
    # For summary, each sentence is a single passage
    next_passage_id = 1
    
    summaries = session.query(Sentence).filter(Sentence.source_type == 'summary').order_by(Sentence.id).all()
    for s in summaries:
        s.passage_id = next_passage_id
        next_passage_id += 1
    session.commit()
    
    print("Segmenting full text chapters...")
    # Fetch all full text books & chapters
    chapters = session.query(Sentence.book, Sentence.chapter_number).filter(
        Sentence.source_type == 'full'
    ).distinct().all()
    
    T_cohesion = 0.50 # default parameter (swept in Phase 2)
    
    for book, ch_num in chapters:
        sentences = session.query(Sentence).filter(
            Sentence.book == book,
            Sentence.chapter_number == ch_num,
            Sentence.source_type == 'full'
        ).order_by(Sentence.sentence_index).all()
        
        if not sentences: continue
        
        # Calculate lengths and prepare embeddings
        lengths = [len(str(s.sentence_text).split()) for s in sentences]
        mean_len = np.mean(lengths) if lengths else 0
        
        current_passage = []
        for i, sentence in enumerate(sentences):
            current_passage.append(sentence)
            
            # Constraints: max 10
            if len(current_passage) >= 10:
                for p_sent in current_passage:
                    p_sent.passage_id = next_passage_id
                next_passage_id += 1
                current_passage = []
                continue
                
            # Constraints: min 3
            if len(current_passage) < 3:
                continue
                
            # Try to find a natural boundary using Triple-Signal
            if i + 1 < len(sentences):
                sn = sentences[i]
                sn_next = sentences[i+1]
                sn_prev = sentences[i-1] if i > 0 else None
                
                # Signal 1: Cohesion
                sim_n = compute_cosine_similarity(sn_prev.embedding, sn.embedding) if sn_prev else 1.0
                sim_next = compute_cosine_similarity(sn.embedding, sn_next.embedding)
                cohesion = (sim_n + sim_next) / 2.0
                
                # Signal 2: Dialogue transition
                has_quote_n = '"' in sn.sentence_text or '“' in sn.sentence_text or '”' in sn.sentence_text
                has_quote_next = '"' in sn_next.sentence_text or '“' in sn_next.sentence_text or '”' in sn_next.sentence_text
                dialogue_transition = (has_quote_n != has_quote_next)
                
                # Signal 3: Length transition
                len_n = lengths[i]
                len_next1 = lengths[i+1] if i+1 < len(lengths) else 0
                len_next2 = lengths[i+2] if i+2 < len(lengths) else 0
                is_long = len_n > mean_len
                is_short_next = (len_next1 > 0 and len_next1 < 0.5 * mean_len) and (len_next2 > 0 and len_next2 < 0.5 * mean_len)
                length_transition = is_long and is_short_next
                
                candidate = False
                if dialogue_transition and cohesion < T_cohesion:
                    candidate = True
                elif length_transition:
                    candidate = True
                elif cohesion < (T_cohesion - 0.15): # Strong drop
                    candidate = True
                    
                if candidate:
                    for p_sent in current_passage:
                        p_sent.passage_id = next_passage_id
                    next_passage_id += 1
                    current_passage = []
                    
        # Flush remaining sentences
        if current_passage:
            # If remaining < 3 and we have previous passages, merge with previous
            if len(current_passage) < 3 and next_passage_id > 1:
                # Assuming previous passage exists in the same chapter
                # But it's simpler to just reuse next_passage_id - 1
                target_id = next_passage_id - 1
                # Must make sure we don't merge into buod or another chapter!
                # Since passage_id is global and monotonic, if we just finished a full passage in THIS chapter, we can merge.
                # A safer way: check if the first sentence of current_passage's index > 0.
                if current_passage[0].sentence_index > 0:
                    for p_sent in current_passage:
                        p_sent.passage_id = target_id
                else:
                    for p_sent in current_passage:
                        p_sent.passage_id = next_passage_id
                    next_passage_id += 1
            else:
                for p_sent in current_passage:
                    p_sent.passage_id = next_passage_id
                next_passage_id += 1
                
        session.commit()
        print(f"Segmented {book} Chapter {ch_num}")
    
    session.close()
    print(f"Segmentation completed. Total passages: {next_passage_id - 1}")

if __name__ == "__main__":
    segment_chapters()
