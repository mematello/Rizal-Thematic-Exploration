import sys
import os
import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from app.models.database import engine, Sentence
from sqlalchemy.orm import sessionmaker

SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

base_dir = Path(__file__).resolve().parent.parent
backend_data_dir = base_dir / "backend" / "app" / "data"

with open(backend_data_dir / 'theme_bank.pkl', 'rb') as f:
    theme_bank = pickle.load(f)

unique_themes = {}
for book in ['noli', 'elfili']:
    if book in theme_bank:
        for row in theme_bank[book]:
            label = row['label']
            if label not in unique_themes:
                unique_themes[label] = []
            unique_themes[label].append(row['embedding'])

theme_centroids = {}
for label, embs in unique_themes.items():
    theme_centroids[label] = np.mean(embs, axis=0)

sentences = db.query(Sentence.id, Sentence.sentence_text, Sentence.embedding).filter(Sentence.source_type == 'full', Sentence.embedding.isnot(None)).all()

sent_embs_mat = np.array([np.array(s.embedding) for s in sentences])

used_ids = set()
results = {}

for theme, centroid in theme_centroids.items():
    centroid_mat = np.array([centroid])
    sims = cosine_similarity(centroid_mat, sent_embs_mat)[0]
    
    sorted_idx = np.argsort(sims)[::-1]
    
    top_5 = []
    for idx in sorted_idx:
        s = sentences[idx]
        if s.id not in used_ids and len(s.sentence_text.split()) > 10:
            top_5.append(s)
            used_ids.add(s.id)
            if len(top_5) == 5:
                break
                
    results[theme] = []
    for s in top_5:
        results[theme].append({
            'corpus_sentence_id': s.id,
            'corpus_sentence_text': s.sentence_text.replace('\n', ' ').strip()
        })

with open('C:/tmp/top5_anchors.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Done")
