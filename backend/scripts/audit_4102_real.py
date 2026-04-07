"""
Audit script for Section 4.10.2: Extract REAL production alignment data
for 4 representative cases from the 40-sentence audit.

Cases selected to cover all 4 Sanggunian scenario types:
  - Character-Assisted (Tauhan active, high tauhan contribution)
  - Semantic-Driven (Tauhan inactive / N/A, semantic dominant)
  - Lexical-Driven (Tauhan inactive, high lexical score)
  - High-Confidence Multi-Signal (All signals strong)
"""
import os, sys, json
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
import logging

log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audit_4102_log.txt')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s', encoding='utf-8')

def log(msg):
    print(msg)
    logging.info(msg)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.database import Sentence
from app.core.config import get_settings
from app.core.engine import RizalEngine

settings = get_settings()
db_engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(bind=db_engine)
db = SessionLocal()

log("Loading RizalEngine (this takes a moment)...")
rizal = RizalEngine()

# The 4 sentence IDs chosen from figure5_audit_40_sentences.json:
# 1. ID 228 - Character-Assisted: Don Rafael, tauhan=1.0, score=0.716 (highest with char)
# 2. ID 203 - Semantic-Driven (No Characters): San Diego description, tauhan_active=0, score=0.600
# 3. ID 1213 - Lexical-Driven: Bapor Tabo, tauhan_active=0, lexical=0.334 (highest lex no char)
# 4. ID 1531 - High-Confidence Multi-Signal: Isagani + Senyor Pasta, tauhan=1.0, score=0.683

TARGET_TEXTS = [
    "Si Don Rafael ay hindi makapangyarihan",
    "Ang bayan ng San Diego ay matatagpuan sa gitna",
    "Naglalayag sa ilog Pasig ang Bapor Tabo",
    "Sinadya ni Isagani ang opisina ng manananggol"
]

results = []

for partial_text in TARGET_TEXTS:
    sentence = db.query(Sentence).filter(Sentence.sentence_text.ilike(f"%{partial_text}%")).first()
    if not sentence:
        log(f"ERROR: Sentence with text '{partial_text}' not found!")
        continue
    sid = sentence.id
    
    log(f"\n{'='*70}")
    log(f"  Sentence ID: {sid}")
    log(f"  Book: {sentence.book} | Chapter: {sentence.chapter_number}")
    log(f"  Buod: {sentence.sentence_text}")
    
    search_book_noli = sentence.book.lower() in ('noli', 'noli me tangere')
    full_book = "Noli Me Tangere" if search_book_noli else "El Filibusterismo"
    search_chapter = sentence.chapter_number
    if search_book_noli and search_chapter == 64:
        search_chapter = 63
    
    # Fetch buod and full sentences
    buod_rows = db.query(Sentence).filter(
        Sentence.book == sentence.book,
        Sentence.chapter_number == sentence.chapter_number,
        Sentence.source_type == 'summary'
    ).order_by(Sentence.sentence_index).all()
    
    full_rows = db.query(Sentence).filter(
        Sentence.book == full_book,
        Sentence.chapter_number == search_chapter,
        Sentence.source_type == 'full'
    ).order_by(Sentence.sentence_index).all()
    
    if not buod_rows or not full_rows:
        log(f"  ERROR: Missing buod or full rows. buod_rows: {len(buod_rows)}, full_rows: {len(full_rows)}")
        continue
    
    buod_texts = [r.sentence_text for r in buod_rows]
    full_texts = [r.sentence_text for r in full_rows]
    
    buod_embs = np.array([r.embedding for r in buod_rows], dtype=np.float32)
    full_embs = np.array([r.embedding for r in full_rows], dtype=np.float32)
    full_is_short = [r.is_short for r in full_rows]
    
    log(f"  Buod count: {len(buod_texts)} | Full count: {len(full_texts)}")
    log(f"  Embedding dim: {buod_embs.shape[1]}")
    
    aligned, debug = rizal.robust_aligner.align(
        buod_sentences=buod_texts,
        full_sentences=full_texts,
        buod_embeddings=buod_embs,
        full_embeddings=full_embs,
        full_is_short=full_is_short,
        return_debug=True
    )
    
    buod_idx = next((i for i, r in enumerate(buod_rows) if r.id == sid), None)
    if buod_idx is None or buod_idx >= len(aligned):
        log(f"  ERROR: Could not find buod_idx for ID {sid}")
        continue
    
    match = aligned[buod_idx]
    
    w_start = match.best_window_start
    w_end = match.best_window_end
    passage_sentences = full_rows[w_start:w_end+1]
    
    log(f"\n  RESULTS:")
    log(f"  Window: [{w_start}:{w_end}] ({w_end - w_start + 1} sentences)")
    log(f"  Lexical Score:  {match.lexical_score:.4f}")
    log(f"  Semantic Score: {match.semantic_score:.4f}")
    log(f"  Tauhan Score:   {match.tauhan_score:.4f}")
    log(f"  Position Score: {match.position_score:.4f}")
    log(f"  FINAL SCORE:    {match.final_score:.4f}")
    log(f"  Matched Chars:  {match.matched_characters}")
    log(f"\n  PASSAGE WINDOW:")
    for i, s in enumerate(passage_sentences):
        idx_label = w_start + i
        short_flag = " [SHORT]" if s.is_short else ""
        log(f"    [{idx_label}]{short_flag} {s.sentence_text}")
    
    case_data = {
        "sentence_id": sid,
        "novel": sentence.book,
        "chapter": sentence.chapter_number,
        "buod_text": sentence.sentence_text,
        "buod_index_in_chapter": buod_idx,
        "total_buod_in_chapter": len(buod_texts),
        "total_full_in_chapter": len(full_texts),
        "embedding_dim": int(buod_embs.shape[1]),
        "window_start": w_start,
        "window_end": w_end,
        "window_size": w_end - w_start + 1,
        "lexical_score": round(float(match.lexical_score), 4),
        "semantic_score": round(float(match.semantic_score), 4),
        "tauhan_score": round(float(match.tauhan_score), 4),
        "position_score": round(float(match.position_score), 4),
        "final_score": round(float(match.final_score), 4),
        "matched_characters": match.matched_characters,
        "tauhan_active": match.tauhan_score != -1.0,
        "passage_sentences": [
            {
                "index": w_start + i,
                "text": s.sentence_text,
                "is_short": s.is_short,
                "original_index": s.original_sentence_number or s.sentence_index
            }
            for i, s in enumerate(passage_sentences)
        ]
    }
    results.append(case_data)

# Save results
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audit_4102_real_results.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

log(f"\n\nResults saved to {out_path}")
log(f"Total cases processed: {len(results)}")

db.close()
