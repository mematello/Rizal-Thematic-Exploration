
import os
import sys
import numpy as np
from sqlalchemy import select, or_
from dotenv import load_dotenv

# Load .env
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', '.env'))
load_dotenv(env_path)

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from app.core.engine import get_engine
from app.models.database import SessionLocal, Sentence

engine = get_engine()
db = SessionLocal()

def debug_query(query):
    print(f"\n--- DEBUG: {query} ---")
    query_words = [query]
    synonyms = {
        'paglilitis': ['hukuman', 'litis', 'pari', 'sentensya', 'kasalanan'],
        'kababata': ['kaibigan', 'kalaro', 'bata']
    }
    if query in synonyms:
        query_words.extend(synonyms[query])
    
    for qw in query_words:
        res = db.scalars(select(Sentence).filter(Sentence.sentence_text.ilike(f"%{qw}%")).limit(1)).first()
        if res:
            print(f"Word '{qw}' found in: {res.sentence_text[:50]}...")
            # Check embedding
            v_q = engine.base_model.encode(query)
            v_sent = np.array(res.embedding)
            sim = np.dot(v_q, v_sent) / (np.linalg.norm(v_q) * np.linalg.norm(v_sent))
            print(f"  Similarity to '{query}': {sim}")
        else:
            print(f"Word '{qw}' NOT found")

debug_query("paglilitis")
debug_query("kababata")
