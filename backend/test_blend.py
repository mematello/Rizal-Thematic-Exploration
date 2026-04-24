import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.engine import RizalEngine
from app.core.config import get_settings
import numpy as np

settings = get_settings()
db_engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

rizal = RizalEngine()
db = SessionLocal()
rizal._ensure_themes_loaded(db)

def test_blend(query, translated):
    q1 = rizal.base_model.encode(query)
    q2 = rizal.base_model.encode(translated)
    blended = (q1 + q2) / 2.0
    blended = blended / np.linalg.norm(blended)
    
    scores = np.dot(rizal.theme_matrix, blended)
    print(f"Query: {query:15} | Trans: {translated:15} | Max Score: {np.max(scores):.3f}")

test_blend("betrayal", "pagtataksil")
test_blend("revenge", "paghihiganti")
test_blend("tiktok is cool", "astig ang tiktok")
test_blend("bitcoin", "bitcoin")
