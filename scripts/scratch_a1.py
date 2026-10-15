import os
import sys
from dotenv import load_dotenv

load_dotenv('backend/.env')
sys.path.insert(0, 'backend')

from app.models.database import SessionLocal, Sentence
from app.core.engine import get_engine, extract_words
import numpy as np

db = SessionLocal()
engine = get_engine()

# Inspect candidates for bayan in elfili
q = 'bayan'
query_words = extract_words(q.lower())
query_analysis = engine.query_analyzer.analyze_query_words(q)
stopword_ratio = engine.query_analyzer.get_stopword_ratio(q)
sig_words = [item['word'].lower() for item in query_analysis if not item['is_stopword']]

# Let's inspect sentences 38978 and 39299 directly
for sid in [38978, 39299]:
    sent = db.get(Sentence, sid)
    lex_score = engine._compute_lexical_score(q, sent.sentence_text, query_analysis, stopword_ratio)

    # Query embedding
    q_emb = engine.base_model.encode(q, show_progress_bar=False)
    q_vec = q_emb / np.linalg.norm(q_emb)
    s_vec = np.array(sent.embedding)
    s_vec = s_vec / np.linalg.norm(s_vec)
    sem_score = float(np.dot(q_vec, s_vec))

    # Text length adjustment
    word_count = len(sent.sentence_text.split())
    if word_count < 5:
        sem_score *= 0.5

    final_score = (lex_score * 0.95) + (sem_score * 0.05)

    print(f"ID {sid}:")
    print(f"  sentence_text: {sent.sentence_text[:60]}")
    print(f"  unrounded lex_score: {lex_score:.6f}")
    print(f"  unrounded sem_score: {sem_score:.6f}")
    print(f"  unrounded final_score: {final_score:.6f} = ({lex_score:.6f} * 0.95) + ({sem_score:.6f} * 0.05)")
    print(f"  round(sem_score * 100): {round(sem_score * 100)}")
    print(f"  round(lex_score * 100): {round(lex_score * 100)}")
    print(f"  round(final_score * 100): {round(final_score * 100)}")
