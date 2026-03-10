
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

w1 = "edukasyon"
w2 = "pag-aaral"

v1 = model.encode(w1)
v2 = model.encode(w2)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"Similarity ('{w1}', '{w2}'): {cosine_sim(v1, v2)}")
