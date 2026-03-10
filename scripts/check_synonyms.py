
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

def find_sim(q):
    print(f"\nSearching for similar words for '{q}':")
    v_q = model.encode(q)
    v_q_norm = v_q / np.linalg.norm(v_q)
    
    # We'll just encode 5000 random words from vocab to save time if it's too slow,
    # but let's try all of them first.
    batch_size = 1000
    vocab_vecs = []
    for i in range(0, len(vocab), batch_size):
        batch = vocab[i:i+batch_size]
        vocab_vecs.append(model.encode(batch))
    
    vocab_vecs = np.vstack(vocab_vecs)
    vocab_vecs = vocab_vecs / np.linalg.norm(vocab_vecs, axis=1, keepdims=True)
    
    sims = np.dot(vocab_vecs, v_q_norm)
    top_indices = np.argsort(sims)[-10:][::-1]
    for idx in top_indices:
        print(f"{vocab[idx]}: {sims[idx]}")

find_sim("kamatayan")
find_sim("paglilitis")
find_sim("kababata")
