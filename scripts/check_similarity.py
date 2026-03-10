
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

q = "edukasyon"
s1 = "Sisikapin niyang paunladin ang sistema ng edukasyon sa kanilang lugar."
s2 = "Ibinilin din nito na gamitin ang kayamanang ito sa pag-aaral."

v_q = model.encode(q)
v_s1 = model.encode(s1)
v_s2 = model.encode(s2)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"Similarity ('{q}', '{s1}'): {cosine_sim(v_q, v_s1)}")
print(f"Similarity ('{q}', '{s2}'): {cosine_sim(v_q, v_s2)}")

# MaxSim
tokens_s2 = model.encode([s2], output_value='token_embeddings')[0]
sims = np.dot(tokens_s2, v_q) / (np.linalg.norm(tokens_s2, axis=1) * np.linalg.norm(v_q))
print(f"MaxSim ('{q}', '{s2}'): {np.max(sims)}")
