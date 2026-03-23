import os
import sys
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add backend root to path
sys.path.append(os.path.join(os.getcwd()))

from app.models.database import Sentence
from app.core.config import get_settings
from app.api.v1.content import get_sentence_sanggunian
from app.core.engine import RizalEngine

def evaluate_40():
    settings = get_settings()
    engine = create_engine(settings.DATABASE_URL)
    Session = sessionmaker(bind=engine)
    db = Session()
    
    # Load RizalEngine
    print("Loading RizalEngine...")
    rizal_engine = RizalEngine()
    
    # Selected 40 IDs
    ids = [
        # Noli (25)
        1, 46, 47, 66, 154, 203, 248, 370, 472, 526, 
        99, 288, 447, 607, 798, 978, 1161, 1162, 3358, 
        171, 187, 228, 238, 267, 871,
        # Fili (15)
        1213, 1222, 1248, 1274, 1298, 1410, 1531, 1611, 
        1733, 4094, 1924, 1941, 1955, 1977, 4226
    ]
    
    results = []
    
    print(f"Starting evaluation of {len(ids)} sentences...")
    print("=" * 80)
    
    for sid in ids:
        buod_sentence = db.query(Sentence).filter(Sentence.id == sid).first()
        if not buod_sentence:
            print(f"Sentence ID {sid} not found!")
            continue
            
        book_label = "Noli" if buod_sentence.book == 'noli' else "Fili"
        print(f"\n--- [{book_label} Ch {buod_sentence.chapter_number}] ID {sid}: {buod_sentence.sentence_text[:60]}... ---")
        
        try:
            # Call the production matching logic
            res = get_sentence_sanggunian(buod_sentence.id, db, rizal_engine)
            
            summary = {
                "id": sid,
                "book": book_label,
                "chapter": buod_sentence.chapter_number,
                "buod_text": buod_sentence.sentence_text,
                "has_reference": res.has_reference,
                "reference_text": res.reference_text if res.has_reference else "N/A",
                "alignment": res.alignment_status if res.has_reference else "Walang Sanggunian",
                "score": res.score if res.has_reference else 0.0
            }
            results.append(summary)
            
            print(f"Status: {summary['alignment'].upper()}")
            if res.has_reference:
                print(f"Score: {res.score:.4f}")
                print(f"First Sentence: {res.reference_text.split('.')[0]}...")
            else:
                print("Result: Walang Sanggunian")
                
        except Exception as e:
            print(f"Error testing ID {sid}: {e}")
            
    # Final Summary
    total = len(results)
    matches = len([r for r in results if r["has_reference"]])
    precise = len([r for r in results if r["alignment"] == "precise"])
    expanded = len([r for r in results if r["alignment"] == "expanded"])
    
    print("\n" + "=" * 50)
    print("FINAL EVALUATION SUMMARY (40 SENTENCES)")
    print("=" * 50)
    print(f"Total Tested: {total}")
    print(f"Matches Found: {matches}")
    print(f"Precise Matches: {precise}")
    print(f"Expanded Matches: {expanded}")
    print(f"Match Rate: {(matches/total)*100:.1f}%")
    print("=" * 50)

if __name__ == "__main__":
    evaluate_40()
