import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading model...")
model = SentenceTransformer("sentence-transformers/paraphrase-xlm-r-multilingual-v1")

q = "kolonyalismo"
s1 = "Nagpapacabicabila ang ilang mga kastila at pinagsasabihan ang bawa't kanilang nasasalubong."
s2 = "Tangi lamang, kung ang nilolooban ay isang kombento o isang kastila, ay saka lumalabas ang mahahabang salaysay na nagsisiwalat ng mga kakilakilabot na pangyayari at hinihingi ang estado de sitio, mga matitinding panupil, iba pa."

q_vec = model.encode(q)
q_norm = np.linalg.norm(q_vec)
if q_norm > 0: q_vec = q_vec / q_norm

s1_vec = model.encode(s1)
s1_norm = np.linalg.norm(s1_vec)
if s1_norm > 0: s1_vec = s1_vec / s1_norm

s2_vec = model.encode(s2)
s2_norm = np.linalg.norm(s2_vec)
if s2_norm > 0: s2_vec = s2_vec / s2_norm

print(f"S1 Semantic Score: {np.dot(q_vec, s1_vec):.4f}")
print(f"S2 Semantic Score: {np.dot(q_vec, s2_vec):.4f}")
