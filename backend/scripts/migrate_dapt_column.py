"""
migrate_dapt_column.py
Adds the embedding_dapt column to the sentences table.
This column stores DAPT model embeddings used exclusively by Sanggunian.
"""
import sys
import os
from sqlalchemy import text

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.database import engine
from app.core.config import get_settings

def migrate():
    with engine.connect() as conn:
        # Ensure pgvector extension is loaded
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

        try:
            conn.execute(text(
                "ALTER TABLE sentences ADD COLUMN embedding_dapt vector(768);"
            ))
            conn.commit()
            print("SUCCESS: Column 'embedding_dapt' added to 'sentences' table.")
        except Exception as e:
            if 'already exists' in str(e).lower() or 'DuplicateColumn' in str(type(e).__name__):
                print("INFO: Column 'embedding_dapt' already exists. Skipping.")
            else:
                raise

if __name__ == "__main__":
    migrate()
