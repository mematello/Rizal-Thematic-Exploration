import os, sys
sys.path.append(os.path.abspath('..'))
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.database import Sentence
from app.core.config import get_settings

try:
    settings = get_settings()
    engine = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    target_texts = [
        "Si Don Rafael ay hindi makapangyarihan",
        "Ang bayan ng San Diego ay matatagpuan sa gitna",
        "Naglalayag sa ilog Pasig ang Bapor Tabo",
        "Sinadya ni Isagani ang opisina ng manananggol"
    ]
    for partial_text in target_texts:
        matches = db.query(Sentence).filter(Sentence.sentence_text.ilike(f"%{partial_text}%")).all()
        for m in matches:
            print(f"FOUND ID={m.id} | Book={m.book} Ch={m.chapter_number} | {m.sentence_text[:50]}")
except Exception as e:
    print(f"Error: {e}")
