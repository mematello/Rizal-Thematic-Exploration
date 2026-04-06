import pandas as pd
from app.models.database import SessionLocal, Sentence
import difflib

def main():
    db = SessionLocal()
    # Let's try Chapter 24 (SA GUBAT)
    chapter_num = 24
    book = "noli"
    
    # DB data
    db_sentences = db.query(Sentence).filter(
        Sentence.chapter_number == chapter_num,
        Sentence.source_type == 'full'
    ).order_by(Sentence.sentence_index).all()
    
    # CSV data
    csv_path = "../csvFiles/fullversion_noli.csv"
    df = pd.read_csv(csv_path)
    csv_rows = df[(df['chapter_number'] == chapter_num) & (df['book_title'].str.contains('Noli', case=False))]
    
    print(f"DB count: {len(db_sentences)}")
    print(f"CSV count: {len(csv_rows)}")
    
    # Attempt matching
    matches = []
    csv_idx = 0
    csv_list = csv_rows.to_dict('records')
    
    for db_s in db_sentences:
        best_match = None
        best_score = -1
        
        # Look ahead a bit in CSV to find a match
        look_ahead = 10
        for i in range(max(0, csv_idx - 5), min(len(csv_list), csv_idx + look_ahead)):
            db_text = str(db_s.sentence_text or "")[:100]
            csv_text = str(csv_list[i].get('sentence_text', ""))[:100]
            score = difflib.SequenceMatcher(None, db_text, csv_text).ratio()
            if score > best_score:
                best_score = score
                best_match = csv_list[i]
        
        if best_score > 0.8:
            # print(f"MATCH: DB {db_s.sentence_index} -> CSV {best_match['sentence_number']} (Score: {best_score})")
            # Update csv_idx to move forward
            csv_idx = csv_list.index(best_match) + 1
            matches.append((db_s.id, best_match['sentence_number']))
        else:
            print(f"NO MATCH: DB {db_s.sentence_index}: {db_s.sentence_text[:50]}...")

    print(f"Total matches: {len(matches)}")
    
    # If we have enough matches, we could proceed.
    # But wait, if CSV count < DB count, many DB sentences will have NO matching original number.
    # This usually means the DB sentences were split from one CSV sentence.

if __name__ == "__main__":
    main()
