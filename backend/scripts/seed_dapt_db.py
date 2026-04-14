"""
seed_dapt_db.py
Seeds the embedding_dapt column for ALL sentences (both summary and full)
using the custom rizal-xlm-r-dapt model.
This column is used exclusively by the Sanggunian (cross-modal reference) system.
"""
import sys
import os
import logging
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(env_path)

from app.models.database import Sentence, engine
from app.core.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SessionLocal = sessionmaker(bind=engine)

def seed_dapt_embeddings():
    # --- Locate DAPT model ---
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dapt_path_1 = os.path.join(base_path, 'models', 'rizal-xlm-r-dapt')
    dapt_path_2 = os.path.join(base_path, 'app', 'models', 'rizal-xlm-r-dapt')
    dapt_path = dapt_path_1 if os.path.exists(dapt_path_1) else (dapt_path_2 if os.path.exists(dapt_path_2) else None)

    if not dapt_path:
        logger.error(f"DAPT model not found at {dapt_path_1} or {dapt_path_2}.")
        logger.error("Please ensure rizal-xlm-r-dapt is in backend/app/models/.")
        sys.exit(1)

    logger.info(f"Loading DAPT model from {dapt_path}")
    model = SentenceTransformer(dapt_path)

    session = SessionLocal()

    try:
        # Process ALL sentences (both summary and full)
        total = session.query(Sentence).count()
        logger.info(f"Found {total} sentences total to update with DAPT embeddings...")

        batch_size = 64
        offset = 0
        updated = 0

        while True:
            batch = session.query(Sentence).offset(offset).limit(batch_size).all()
            if not batch:
                break

            texts = []
            for s in batch:
                # Use chapter title + sentence text for richer context (same as seed_db.py)
                combined = f"{s.chapter_title} {s.sentence_text}" if s.chapter_title else s.sentence_text
                texts.append(combined)

            embeddings = model.encode(texts, show_progress_bar=False)

            for i, s in enumerate(batch):
                s.embedding_dapt = embeddings[i].tolist()

            session.commit()
            updated += len(batch)
            offset += batch_size

            if updated % 500 == 0 or updated >= total:
                logger.info(f"  Updated {updated}/{total} sentences with DAPT embeddings...")

        logger.info(f"SUCCESS: Done. {updated} sentences seeded with embedding_dapt.")

    finally:
        session.close()


if __name__ == "__main__":
    seed_dapt_embeddings()
