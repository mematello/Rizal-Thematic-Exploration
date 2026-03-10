
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

query = "edukasyon"
v_q = engine.base_model.encode(query).tolist()

print(f"Top 10 semantic results for '{query}':")
results = db.scalars(
    select(Sentence)
    .filter(Sentence.book == 'noli')
    .filter(Sentence.source_type == 'summary')
    .order_by(Sentence.embedding.cosine_distance(v_q))
    .limit(10)
).all()

for i, s in enumerate(results):
    dist = np.dot(s.embedding, v_q) / (np.linalg.norm(s.embedding) * np.linalg.norm(v_q))
    print(f"{i+1}. {s.sentence_text} (Sim: {dist})")

print(f"\nSearching for 'pag-aaral' sentences for '{query}':")
pag_aaral_results = db.scalars(
    select(Sentence)
    .filter(Sentence.book == 'noli')
    .filter(Sentence.source_type == 'summary')
    .filter(Sentence.sentence_text.ilike('%pag-aaral%'))
).all()

for s in pag_aaral_results:
    dist = np.dot(s.embedding, v_q) / (np.linalg.norm(s.embedding) * np.linalg.norm(v_q))
    if dist > 0.1:
        print(f"PAG-AARAL: {s.sentence_text} (Sim: {dist})")
