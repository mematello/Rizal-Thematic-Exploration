import os
import sys
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add backend root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.models.database import Sentence
from app.core.config import get_settings
from app.api.v1.content import get_sentence_sanggunian

def evaluate():
    settings = get_settings()
    engine = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    from app.core.engine import RizalEngine
    rizal_engine = RizalEngine()
    
    # Target Chapters (Noli)
    target_chapters = [1, 2, 3, 4, 7, 10, 13, 20, 26, 30]
    results = []
    
    for ch in target_chapters:
        # Get first summary sentence of the chapter
        buod_sentence = db.query(Sentence).filter(
            Sentence.book == 'noli', # Corrected book name for summary
            Sentence.chapter_number == ch,
            Sentence.source_type == 'summary'
        ).first()
        
        if not buod_sentence:
            continue
            
        print(f"\n--- Testing Kabanata {ch}: {buod_sentence.sentence_text[:60]}... ---")
        
        try:
            sanggunian = get_sentence_sanggunian(buod_sentence.id, db, rizal_engine)
            status = "Correct" if sanggunian.has_reference else "Walang Sanggunian"
            
            results.append({
                "chapter": ch,
                "buod_id": buod_sentence.id,
                "buod_text": buod_sentence.sentence_text,
                "has_reference": sanggunian.has_reference,
                "reference_text": sanggunian.reference_text if sanggunian.has_reference else "N/A",
                "alignment": sanggunian.alignment_status if sanggunian.has_reference else "N/A",
                "score": sanggunian.score if sanggunian.has_reference else 0.0
            })
            
            print(f"Status: {status}")
            if sanggunian.has_reference:
                print(f"Alignment: {sanggunian.alignment_status.upper()}")
                print(f"Score: {sanggunian.score:.4f}")
                print(f"Reference: {sanggunian.reference_text[:200]}...")
            else:
                print("Result: Walang Sanggunian")
                
        except Exception as e:
            print(f"Error testing ID {buod_sentence.id}: {e}")
            
    # Summary stats
    total = len(results)
    matches = sum(1 for r in results if r['has_reference'])
    precise = sum(1 for r in results if r['alignment'] == 'precise')
    expanded = sum(1 for r in results if r['alignment'] == 'expanded')
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total Tested: {total}")
    print(f"Matches Found: {matches}")
    print(f"Precise Matches: {precise}")
    print(f"Expanded Matches: {expanded}")
    print(f"Match Rate: {matches/total:.1%}")
    print("="*50)

if __name__ == "__main__":
    evaluate()
