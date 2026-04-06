import pandas as pd
from app.models.database import SessionLocal, Sentence
from app.core.analyzer import is_short_sentence
import os

def resegment_db():
    db = SessionLocal()
    books = [("noli", "../csvFiles/fullversion_noli.csv"), ("fili", "../csvFiles/fullversion_elfili.csv")]
    
    for book_key, csv_path in books:
        if not os.path.exists(csv_path):
            print(f"CSV not found: {csv_path}")
            continue
            
        print(f"Resegmenting {book_key} from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Get all chapters
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
            
            csv_rows = df[df['chapter_number'] == ch_num].to_dict('records')
            if not csv_rows: continue
            
            print(f"  Chapter {ch_num}: {len(db_sentences)} DB vs {len(csv_rows)} CSV")
            
            new_sentences_data = []
            db_pointer = 0
            
            # For each CSV sentence, try to find it in the DB collection
            # We want to reconstruct the sequence based on CSV structure
            for csv_s in csv_rows:
                csv_text = str(csv_s['sentence_text']).strip()
                if not csv_text: continue
                
                # Find which DB row currently contains this text
                found = False
                for i in range(len(db_sentences)):
                    db_s = db_sentences[i]
                    if csv_text in db_s.sentence_text:
                        # This DB row contains our CSV fragment
                        # We'll create a new record for this fragment
                        new_s = Sentence(
                            book=db_s.book,
                            chapter_number=db_s.chapter_number,
                            chapter_title=db_s.chapter_title,
                            source_type=db_s.source_type,
                            sentence_text=csv_text,
                            original_sentence_number=int(csv_s['sentence_number']),
                            is_short=is_short_sentence(csv_text),
                            embedding=db_s.embedding, # Duplicate embedding to keep context
                            passage_id=db_s.passage_id
                        )
                        new_sentences_data.append(new_s)
                        found = True
                        break
                
                if not found:
                    # If text not found in DB rows, it might be truly missing or highly different
                    # We'll create it as a new row anyway if it's important
                    # Let's just log it for now
                    # print(f"    Warning: CSV {csv_s['sentence_number']} not found in DB")
                    pass

            if new_sentences_data:
                # To maintain ordering, we re-assign sentence_index sequentially
                for idx, s in enumerate(new_sentences_data):
                    s.sentence_index = idx
                
                # Now, wipe the old DB rows for this chapter and insert new ones
                # Safest: Use a transaction
                try:
                    db.query(Sentence).filter(
                        Sentence.chapter_number == ch_num,
                        Sentence.book.ilike(f"%{book_key}%"),
                        Sentence.source_type == 'full'
                    ).delete()
                    
                    db.bulk_save_objects(new_sentences_data)
                    db.commit()
                    print(f"    Success: Chapter {ch_num} re-segmented into {len(new_sentences_data)} sentences.")
                except Exception as e:
                    db.rollback()
                    print(f"    Error in Chapter {ch_num}: {e}")

if __name__ == "__main__":
    resegment_db()
