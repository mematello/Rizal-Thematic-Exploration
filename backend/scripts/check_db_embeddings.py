from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session
import sys
import os

# Add the parent directory to sys.path to allow imports from app
sys.path.append(os.path.join(os.getcwd()))

from app.models.database import Theme, Sentence
from app.core.config import get_settings

def check_db():
    settings = get_settings()
    engine = create_engine(settings.DATABASE_URL)
    
    with Session(engine) as session:
        # Check themes
        themes = session.query(Theme).all()
        print(f"Checking {len(themes)} themes...")
        for t in themes:
            if t.embedding is None:
                print(f"Theme {t.tagalog_title} (ID: {t.id}) has NO embedding!")
            elif len(t.embedding) == 0:
                print(f"Theme {t.tagalog_title} (ID: {t.id}) has EMPTY embedding!")
        
        # Check sentences
        sentences = session.query(Sentence).limit(10).all()
        print(f"Checking first 10 sentences...")
        for s in sentences:
            if s.embedding is None:
                print(f"Sentence {s.id} in {s.book} has NO embedding!")
            
    print("Check complete.")

if __name__ == "__main__":
    check_db()
