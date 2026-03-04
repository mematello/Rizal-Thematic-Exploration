import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load .env explicitly
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(env_path)

db_url = os.environ.get('DATABASE_URL')
if not db_url:
    print("DATABASE_URL not found")
    sys.exit(1)

try:
    engine = create_engine(db_url)
    with engine.connect() as conn:
        res = conn.execute(text("SELECT count(1) FROM sentences WHERE source_type = 'full'"))
        count = res.scalar()
        print(f"Full sentences count: {count}")
except Exception as e:
    print(f"Error: {e}")
