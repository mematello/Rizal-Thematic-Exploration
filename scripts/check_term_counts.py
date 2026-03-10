
import os
import sys
from sqlalchemy import select, func
from dotenv import load_dotenv

# Load .env
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', '.env'))
load_dotenv(env_path)

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from app.models.database import SessionLocal, Sentence

db = SessionLocal()

terms = ['hukuman', 'litis', 'pari', 'sentensya', 'kasalanan', 'kaibigan', 'kalaro', 'bata']

for t in terms:
    count = db.scalar(select(func.count()).select_from(Sentence).filter(Sentence.sentence_text.ilike(f"%{t}%")))
    print(f"Term '{t}': {count} matches")

db.close()
