import pandas as pd
from app.models.database import SessionLocal, Sentence
import difflib
import os

def backfill_original_indices():
    db = SessionLocal()
    books = [("noli", "../csvFiles/fullversion_noli.csv"), ("fili", "../csvFiles/fullversion_fili.csv")]
    
    for book_key, csv_path in books:
        if not os.path.exists(csv_path):
            print(f"CSV not found: {csv_path}")
            continue
            
        print(f"Backfilling {book_key} from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Get all sentences of this book in DB
        # To avoid massive memory usage, let's process chapter by chapter
        unique_chapters = db.query(Sentence.chapter_number).filter(
            Sentence.book.ilike(f"%{book_key}%"),
            Sentence.source_type == 'full'
        ).distinct().all()
        unique_chapters = [c[0] for c in unique_chapters]
        
        for ch_num in sorted(unique_chapters):
            db_sentences = db.query(Sentence).filter(
                Sentence.chapter_number == ch_num,
                Sentence.book.ilike(f"%{book_key}%"),
                Sentence.source_type == 'full'
            ).order_by(Sentence.sentence_index).all()
            
            csv_rows = df[df['chapter_number'] == ch_num]
            csv_list = csv_rows.to_dict('records')
            
            if not csv_list:
                print(f"No CSV data for {book_key} Ch {ch_num}")
                continue
                
            print(f"Processing {book_key} Ch {ch_num} ({len(db_sentences)} DB vs {len(csv_list)} CSV)")
            
            csv_pointer = 0
            match_count = 0
            
            for db_s in db_sentences:
                best_match = None
                best_score = -1
                
                # Broaden look-ahead for robust matching
                look_ahead = 20
                search_range = range(max(0, csv_pointer - 5), min(len(csv_list), csv_pointer + look_ahead))
                
                db_text = str(db_s.sentence_text or "")[:100]
                
                for i in search_range:
                    csv_text = str(csv_list[i].get('sentence_text', ""))[:100]
                    score = difflib.SequenceMatcher(None, db_text, csv_text).ratio()
                    if score > best_score:
                        best_score = score
                        best_match = csv_list[i]
                
                if best_match and best_score > 0.6: # Relaxed for rough segments
                    db_s.original_sentence_number = int(best_match['sentence_number'])
                    csv_pointer = csv_list.index(best_match)
                    match_count += 1
                else:
                    # Fallback to the previous known pointer if no match found
                    if csv_pointer < len(csv_list):
                        db_s.original_sentence_number = int(csv_list[csv_pointer]['sentence_number'])
            
            db.commit()
            print(f"  Matched {match_count} sentences for Ch {ch_num}")

if __name__ == "__main__":
    backfill_original_indices()
