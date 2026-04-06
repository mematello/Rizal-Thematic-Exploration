from sqlalchemy import create_engine, text
import sys
import os
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(backend_dir))

from app.core.config import get_settings

def migrate():
    settings = get_settings()
    engine = create_engine(settings.DATABASE_URL)
    
    with engine.connect() as conn:
        print("Checking if is_short column exists...")
        result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='sentences' AND column_name='is_short';"))
        if not result.fetchone():
            print("Adding is_short column to sentences table...")
            conn.execute(text("ALTER TABLE sentences ADD COLUMN is_short BOOLEAN DEFAULT FALSE;"))
            conn.commit()
            print("Column added successfully.")
        else:
            print("is_short column already exists.")

if __name__ == "__main__":
    migrate()
