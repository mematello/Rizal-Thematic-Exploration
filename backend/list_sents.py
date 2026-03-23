import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add backend root to path
sys.path.append(os.path.join(os.getcwd()))

from app.models.database import Sentence
from app.core.config import get_settings

def list_sents():
    settings = get_settings()
    engine = create_engine(settings.DATABASE_URL)
    Session = sessionmaker(bind=engine)
    db = Session()
    
    noli_chs = [1, 2, 3, 4, 7, 10, 13, 20, 26, 30, 5, 15, 25, 35, 45, 55, 61, 62, 63, 64, 8, 9, 11, 12, 14]
    fili_chs = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 31, 32, 33, 34, 35]
    
    print("ID,BOOK,CH,TEXT")
    for b, chs in [('noli', noli_chs), ('elfili', fili_chs)]:
        for ch in chs:
            sent = db.query(Sentence).filter(Sentence.book==b, Sentence.chapter_number==ch, Sentence.source_type=='summary').first()
            if sent:
                # Truncate text for display
                txt = sent.sentence_text[:50].replace('\n', ' ')
                print(f"{sent.id},{b},{ch},{txt}")

if __name__ == "__main__":
    list_sents()
