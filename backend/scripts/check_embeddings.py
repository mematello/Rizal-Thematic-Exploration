import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.core.engine import RizalEngine
from app.models.database import SessionLocal, Sentence

def main():
    with SessionLocal() as db:
        # Check summary sentences
        sum_with_emb = db.query(Sentence).filter(Sentence.source_type == "summary", Sentence.embedding.is_not(None)).count()
        sum_no_emb = db.query(Sentence).filter(Sentence.source_type == "summary", Sentence.embedding.is_(None)).count()
        print(f"Summary: {sum_with_emb} with embedding, {sum_no_emb} without")
        
        # Check full sentences
        full_with_emb = db.query(Sentence).filter(Sentence.source_type == "full", Sentence.embedding.is_not(None)).count()
        full_no_emb = db.query(Sentence).filter(Sentence.source_type == "full", Sentence.embedding.is_(None)).count()
        print(f"Full: {full_with_emb} with embedding, {full_no_emb} without")

if __name__ == "__main__":
    main()
