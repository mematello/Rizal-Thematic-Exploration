
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
v_q = engine.base_model.encode(query)
v_q_norm = v_q / np.linalg.norm(v_q)

results = db.scalars(
    select(Sentence)
    .filter(Sentence.book == 'noli')
    .filter(Sentence.source_type == 'summary')
).all()

count_gt_019 = 0
for s in results:
    dist = np.dot(s.embedding, v_q_norm) / np.linalg.norm(s.embedding)
    if dist > 0.19:
        count_gt_019 += 1

print(f"Number of sentences with similarity > 0.19 to '{query}': {count_gt_019}")
print(f"Total sentences: {len(results)}")
