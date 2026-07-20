import os
import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load model directly
print("Loading model...")
model = SentenceTransformer("sentence-transformers/paraphrase-xlm-r-multilingual-v1")

# Load themes
df = pd.read_csv("c:/Users/marcu/Desktop/thesis/theme_anchors.csv")
themes = df['theme_tagalog'].unique()

print(f"Loaded {len(themes)} unique themes.")
theme_embs = model.encode(themes)
theme_norms = np.linalg.norm(theme_embs, axis=1, keepdims=True)
theme_embs = np.divide(theme_embs, theme_norms, out=np.zeros_like(theme_embs), where=theme_norms!=0)

queries = [
    "prayle",
    "kura",
    "pag-ibig sa bayan",
    "utang na loob",
    "corruption ng mga prayle",
    "freedom para sa Pilipinas",
    "paano ipinaglaban ni ibarra ang edukasyon ng mga kabataan",
    "ang lihim na yaman ni simoun at ang kanyang paghihiganti",
    "cellphone at internet connection",
    "spaceship navigation system"
]

for q in queries:
    q_vec = model.encode(q)
    q_norm = np.linalg.norm(q_vec)
    if q_norm > 0:
        q_vec = q_vec / q_norm

    scores = np.dot(theme_embs, q_vec)
    max_idx = np.argmax(scores)

    print(f"\nQuery: {q}")
    print(f"Max ThemeProximity Score: {scores[max_idx]:.4f}")
    print(f"Matched Theme: {themes[max_idx]}")

    # Top 3 themes
    print("Top 3 Themes:")
    top_indices = np.argsort(scores)[::-1][:3]
    for idx in top_indices:
        print(f"{themes[idx]}: {scores[idx]:.4f}")
