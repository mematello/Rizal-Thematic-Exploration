import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load .env explicitly
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(env_path)

db_url = os.environ.get('DATABASE_URL')
try:
    engine = create_engine(db_url)
    with engine.connect() as conn:
        res = conn.execute(text("SELECT book, source_type, count(1) FROM sentences GROUP BY book, source_type"))
        for row in res:
            print(f"Book: {row[0]}, Source: {row[1]}, Count: {row[2]}")
except Exception as e:
    print(f"Error: {e}")
