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

for word in ["betrayal", "revenge", "tiktok", "bitcoin"]:
    q_vec = rizal.base_model.encode(word)
    q_norm = np.linalg.norm(q_vec)
    if q_norm > 0: q_vec = q_vec / q_norm
    scores = np.dot(rizal.theme_matrix, q_vec)
    print(f"Word: {word:10} | Max Score: {np.max(scores):.3f}")
