
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from prod_guard import verify_not_production_db
verify_not_production_db(caller_name=Path(__file__).name)

from sqlalchemy import create_engine, text
import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load .env explicitly
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(env_path)

try:
    from app.core.config import get_settings
    settings = get_settings()
    engine = create_engine(settings.DATABASE_URL)
    
    with engine.connect() as conn:
        # Add source_type column to sentences table if it doesn't exist
        # This syntax is for PostgreSQL
        conn.execute(text("ALTER TABLE sentences ADD COLUMN IF NOT EXISTS source_type VARCHAR(20) DEFAULT 'summary'"))
        conn.commit()
        print("Column 'source_type' added to 'sentences' table successfully.")
except Exception as e:
    print(f"Error during migration: {e}")
