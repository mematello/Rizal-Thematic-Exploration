import sys
import os
from sqlalchemy.orm import Session
from pathlib import Path

# Add the backend directory to sys.path to allow importing from 'app'
backend_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(backend_dir))

from app.models.database import SessionLocal, Sentence
from app.core.analyzer import is_short_sentence

def backfill():
    db: Session = SessionLocal()
    try:
        print("Starting backfill for is_short column...")
        sentences = db.query(Sentence).all()
        total = len(sentences)
        print(f"Found {total} sentences to process.")
        
        count = 0
        for s in sentences:
            s.is_short = is_short_sentence(str(s.sentence_text))
            count += 1
            if count % 500 == 0:
                print(f"Processed {count}/{total}...")
                db.commit()
        
        db.commit()
        print(f"Successfully updated {count} sentences.")
    except Exception as e:
        print(f"Error during backfill: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    backfill()
