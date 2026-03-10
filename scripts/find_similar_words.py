
import os
import sys
import numpy as np
from dotenv import load_dotenv

# Load .env
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', '.env'))
load_dotenv(env_path)

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from app.core.engine import get_engine

engine = get_engine()
model = engine.base_model
vocab = list(engine.vocabulary)

q = "edukasyon"
v_q = model.encode(q)

print(f"Encoding {len(vocab)} words...")
# Encode in batches to save memory
batch_size = 500
vocab_vecs = []
for i in range(0, len(vocab), batch_size):
    batch = vocab[i:i+batch_size]
    vocab_vecs.append(model.encode(batch))

vocab_vecs = np.vstack(vocab_vecs)
vocab_vecs = vocab_vecs / np.linalg.norm(vocab_vecs, axis=1, keepdims=True)
v_q_norm = v_q / np.linalg.norm(v_q)

sims = np.dot(vocab_vecs, v_q_norm)
top_indices = np.argsort(sims)[-20:][::-1]

print(f"\nTop 20 words for '{q}':")
for idx in top_indices:
    print(f"{vocab[idx]}: {sims[idx]}")
