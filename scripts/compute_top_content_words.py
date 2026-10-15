import os
import sys
from pathlib import Path
from collections import Counter

base_dir = Path(__file__).parent.parent
backend_dir = base_dir / "backend"
sys.path.insert(0, str(backend_dir))

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from app.models.database import SessionLocal, Sentence
from app.core.engine import get_engine, extract_words
from sqlalchemy import select, func

db = SessionLocal()
engine = get_engine()

# Extended function words / particles to distinguish grammatical clitics from semantic content words
GRAMMATICAL_PARTICLES = {
    'lang', 'nang', 'pang', 'nito', 'niyang', 'itong', 'nasa', 'nilang',
    'noon', 'mula', 'doon', 'saka', 'kasi', 'tila', 'umano', 'ayon',
    'wala', 'walang', 'hindi', 'bawat', 'lahat', 'ibang', 'dapat', 'maaari',
    'nais', 'upang', 'dahil', 'kahit', 'ngunit', 'subalit', 'datapwat', 'habang',
    'bago', 'pagkatapos', 'samantala', 'kaya', 'tulad', 'para', 'gamit', 'buong',
    'mismo', 'lamang', 'paulit-ulit', 'muli', 'agad', 'palagi', 'minsan'
}

base_stopwords = set(engine.LOW_INFO_WORDS) | set(engine.EN_STOPWORDS) | GRAMMATICAL_PARTICLES

for stype in ['summary', 'full']:
    print("\n" + "=" * 80)
    print(f"DATABASE SOURCE_TYPE: '{stype}'")
    print("=" * 80)

    noli_book = 'noli' if stype == 'summary' else 'Noli Me Tangere'
    elfili_book = 'elfili' if stype == 'summary' else 'El Filibusterismo'

    noli_sents = db.scalars(select(Sentence.sentence_text).filter(Sentence.book == noli_book, Sentence.source_type == stype)).all()
    elfili_sents = db.scalars(select(Sentence.sentence_text).filter(Sentence.book == elfili_book, Sentence.source_type == stype)).all()

    print(f"Sentences: Noli={len(noli_sents)}, El Fili={len(elfili_sents)}")

    def get_cands(sents):
        c = Counter()
        for s in sents:
            words = extract_words(s.lower())
            for w in set(words):
                if len(w) > 3 and w not in base_stopwords:
                    c[w] += 1
        return c

    noli_cands = get_cands(noli_sents)
    elfili_cands = get_cands(elfili_sents)

    # Check top 60 candidates with ILIKE count(*)
    n_counts = []
    for w, _ in noli_cands.most_common(60):
        cnt = db.scalar(select(func.count(Sentence.id)).filter(
            Sentence.book == noli_book,
            Sentence.source_type == stype,
            Sentence.sentence_text.ilike(f"%{w}%")
        ))
        n_counts.append((w, cnt))
    n_counts.sort(key=lambda x: x[1], reverse=True)

    e_counts = []
    for w, _ in elfili_cands.most_common(60):
        cnt = db.scalar(select(func.count(Sentence.id)).filter(
            Sentence.book == elfili_book,
            Sentence.source_type == stype,
            Sentence.sentence_text.ilike(f"%{w}%")
        ))
        e_counts.append((w, cnt))
    e_counts.sort(key=lambda x: x[1], reverse=True)

    print(f"\n--- TOP 30 PURE CONTENT WORDS IN NOLI ({stype}) by ILIKE count(*) ---")
    for r, (w, cnt) in enumerate(n_counts[:30], 1):
        ex = " [EXCEEDS 200]" if cnt > 200 else ""
        print(f"{r:2d}. {w:<20} : {cnt:>4d}{ex}")
    exceed_n = [w for w, c in n_counts if c > 200]
    print(f"Words exceeding 200 in Noli ({len(exceed_n)}): {exceed_n}")

    print(f"\n--- TOP 30 PURE CONTENT WORDS IN EL FILI ({stype}) by ILIKE count(*) ---")
    for r, (w, cnt) in enumerate(e_counts[:30], 1):
        ex = " [EXCEEDS 200]" if cnt > 200 else ""
        print(f"{r:2d}. {w:<20} : {cnt:>4d}{ex}")
    exceed_e = [w for w, c in e_counts if c > 200]
    print(f"Words exceeding 200 in El Fili ({len(exceed_e)}): {exceed_e}")
