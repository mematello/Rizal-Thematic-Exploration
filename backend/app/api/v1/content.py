from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from app.models.database import SessionLocal, Sentence, Theme
from app.core.engine import get_engine, RizalEngine
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy import select, text
import threading
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import math
import json
import re
from app.core.config import get_settings
config = get_settings()

# Cache for CSV data
_csv_cache = {}

# Cache for chapter-level buod-to-full alignments  (key: (full_book, full_chapter, buod_book, buod_chapter))
_alignment_cache: dict = {}

# Process-level cache for static themes response
_themes_cache: Optional[List[dict]] = None
_themes_lock = threading.Lock()

# Process-level cache for static chapters list response (keyed by source_type: "summary" | "full")
_chapters_cache: Dict[str, List[Any]] = {}
_chapters_lock = threading.Lock()

# Bounded in-memory LRU cache for chapter content (keyed by (book, chapter_number, mode))
from collections import OrderedDict
MAX_CHAPTER_CONTENT_CACHE_ENTRIES = 64
_chapter_content_cache: OrderedDict[Tuple[str, int, str], List[Any]] = OrderedDict()
_chapter_content_lock = threading.Lock()

# Static precomputed chapter content cache (keyed by (book, mode) -> {chapter_number_str: [sentences]})
_static_chapter_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
_static_cache_enabled: bool = False
_static_cache_fingerprint: Optional[Dict[str, Any]] = None
_static_cache_lock = threading.Lock()
_static_cache_loaded: bool = False
_fingerprint_checked: bool = False

def load_static_chapter_cache() -> None:
    """
    Loads precomputed chapter static cache bundles from backend/app/data/
    into process memory. Fast operation (~3.1 MB parsed in <15ms).
    """
    global _static_chapter_cache, _static_cache_fingerprint, _static_cache_loaded
    with _static_cache_lock:
        if _static_cache_loaded:
            return
        data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        cache_files = {
            ("noli", "buod"): data_dir / "chapter_cache_noli_buod.json",
            ("noli", "full"): data_dir / "chapter_cache_noli_full.json",
            ("elfili", "buod"): data_dir / "chapter_cache_elfili_buod.json",
            ("elfili", "full"): data_dir / "chapter_cache_elfili_full.json",
        }
        loaded = {}
        bundle_fp = None
        for key, fpath in cache_files.items():
            if not fpath.exists():
                import logging
                logging.warning(f"Static chapter cache bundle missing: {fpath.name}")
                return
            with open(fpath, "r", encoding="utf-8") as f:
                bundle = json.load(f)
            loaded[key] = bundle.get("chapters", {})
            if bundle_fp is None and "fingerprint" in bundle:
                bundle_fp = bundle["fingerprint"]

        _static_chapter_cache = loaded
        _static_cache_fingerprint = bundle_fp
        _static_cache_loaded = True

def verify_corpus_fingerprint(db: Session) -> bool:
    """
    Queries live database to verify corpus integrity against static cache fingerprint.
    Returns True if row counts and deterministic sentence/theme MD5 hashes match.
    Returns False and logs warning on mismatch or DB error.
    DOES NOT CRASH.
    """
    global _static_cache_enabled, _fingerprint_checked
    with _static_cache_lock:
        if _fingerprint_checked:
            return _static_cache_enabled
        if not _static_cache_loaded or not _static_cache_fingerprint:
            load_static_chapter_cache()
            if not _static_cache_fingerprint:
                _fingerprint_checked = True
                _static_cache_enabled = False
                return False

        try:
            sql = text("""
                SELECT 
                    (SELECT count(*) FROM sentences) AS sentence_count,
                    (SELECT md5(string_agg(sentence_text, '' ORDER BY id)) FROM sentences) AS sentence_text_md5,
                    (SELECT count(*) FROM themes) AS theme_count,
                    (SELECT md5(string_agg(tagalog_title || ':' || coalesce(meaning, ''), '' ORDER BY id)) FROM themes) AS theme_content_md5
            """)
            row = db.execute(sql).fetchone()
            if not row:
                _fingerprint_checked = True
                _static_cache_enabled = False
                return False

            live_fp = {
                "sentence_count": int(row[0]),
                "sentence_text_md5": str(row[1]),
                "theme_count": int(row[2]),
                "theme_content_md5": str(row[3]),
            }

            fp = _static_cache_fingerprint
            if (
                live_fp["sentence_count"] == fp.get("sentence_count")
                and live_fp["sentence_text_md5"] == fp.get("sentence_text_md5")
                and live_fp["theme_count"] == fp.get("theme_count")
                and live_fp["theme_content_md5"] == fp.get("theme_content_md5")
            ):
                _static_cache_enabled = True
                _fingerprint_checked = True
                import logging
                logging.info("Chapter static cache verified against Supabase corpus (205 chapters active).")
                return True
            else:
                _static_cache_enabled = False
                _fingerprint_checked = True
                import logging
                logging.warning(
                    f"Chapter static cache fingerprint mismatch! Live DB: {live_fp} vs Cache: {fp}. "
                    "Disabling static chapter cache, falling back to live computation. "
                    "Run scripts/build_chapter_cache.py to regenerate."
                )
                return False
        except Exception as e:
            _static_cache_enabled = False
            _fingerprint_checked = True
            import logging
            logging.warning(f"Error checking corpus fingerprint: {e}. Falling back to live computation.")
            return False

# Initialize cache at module load
try:
    load_static_chapter_cache()
except Exception as e:
    import logging
    logging.warning(f"Failed to load static chapter cache at module load: {e}")


def get_csv_data(book: str) -> Optional[pd.DataFrame]:
    if book in _csv_cache:
        return _csv_cache[book]
    
    file_name = "fullversion_noli.csv" if book == "noli" else "fullversion_elfili.csv"
    # Root dir relative to backend/app/api/v1/content.py
    base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
    file_path = base_dir / "csvFiles" / file_name
    
    if not file_path.exists():
        return None
        
    df = pd.read_csv(file_path)
    _csv_cache[book] = df
    return df


router = APIRouter()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Models
class ChapterResponse(BaseModel):
    book: str
    chapter_number: int
    chapter_title: str


@router.get("/chapters", response_model=List[ChapterResponse])
def get_chapters(
    response: Response = None,
    mode: Optional[str] = "buod",
    db: Session = Depends(get_db)
):
    """
    Fetch unique chapters from the sentences table for the given mode.
    Cached in process memory; includes Cache-Control header.
    """
    if response:
        response.headers["Cache-Control"] = "public, max-age=86400, stale-while-revalidate=604800"

    source_type = "summary" if (mode == "buod" or not mode) else "full"

    if source_type in _chapters_cache:
        return _chapters_cache[source_type]

    with _chapters_lock:
        if source_type in _chapters_cache:
            return _chapters_cache[source_type]

        results = []
        try:
            chapters = db.query(
                Sentence.book, 
                Sentence.chapter_number, 
                Sentence.chapter_title
            ).filter(
                Sentence.source_type == source_type
            ).distinct(
                Sentence.book, 
                Sentence.chapter_number
            ).order_by(
                Sentence.book, 
                Sentence.chapter_number
            ).all()

            results = [
                ChapterResponse(
                    book=c.book, 
                    chapter_number=c.chapter_number, 
                    chapter_title=c.chapter_title
                ) for c in chapters
            ]
        except Exception:
            # Bundled static fallback
            cache_file = "chapters_cache_buod.json" if source_type == "summary" else "chapters_cache_full.json"
            bundled_path = Path(__file__).resolve().parent.parent.parent / "data" / cache_file
            if bundled_path.exists():
                with open(bundled_path, "r", encoding="utf-8") as f:
                    cached_raw = json.load(f)
                results = [ChapterResponse(**item) for item in cached_raw]
            else:
                raise

        _chapters_cache[source_type] = results
        return _chapters_cache[source_type]


def warm_chapters_cache():
    """
    Loads static chapter cache bundles and validates corpus fingerprint against DB
    in a background thread at startup. Also warms the chapters list cache.
    Startup is never delayed.
    """
    db = SessionLocal()
    try:
        load_static_chapter_cache()
        verify_corpus_fingerprint(db)
        get_chapters(response=None, mode="buod", db=db)
        get_chapters(response=None, mode="full", db=db)
        import logging
        logging.info("Chapters cache initialization complete.")
    except Exception as e:
        import logging
        logging.warning(f"Failed to initialize chapters cache at startup: {e}")
    finally:
        db.close()

class ThemeContextResponse(BaseModel):
    book: str
    chapter_number: int
    chapter_title: str
    sentence_text: str
    sentence_index: int

class ThemeResponse(BaseModel):
    tagalog_title: str
    meaning: str
    best_match: Optional[ThemeContextResponse] = None

@router.get("/themes", response_model=List[ThemeResponse])
def get_themes(response: Response, db: Session = Depends(get_db)):
    """
    Fetch unique themes and their best matching sentence context.
    Cached in process memory; set-based LATERAL join eliminates N+1 query loop.
    """
    global _themes_cache
    if response:
        response.headers["Cache-Control"] = "public, max-age=3600, stale-while-revalidate=86400"

    if _themes_cache is not None:
        return _themes_cache

    with _themes_lock:
        if _themes_cache is not None:
            return _themes_cache

        results = []
        try:
            # Set-based LATERAL query: 1 single query instead of 26 sequential round-trips
            sql = text("""
                WITH unique_t AS (
                    SELECT DISTINCT ON (tagalog_title) id, tagalog_title, meaning, embedding
                    FROM themes
                    ORDER BY tagalog_title, id
                )
                SELECT u.id, u.tagalog_title, u.meaning, s.book, s.chapter_number, s.chapter_title, s.sentence_text, s.sentence_index
                FROM unique_t u
                LEFT JOIN LATERAL (
                    SELECT s.book, s.chapter_number, s.chapter_title, s.sentence_text, s.sentence_index
                    FROM sentences s
                    WHERE u.embedding IS NOT NULL
                    ORDER BY s.embedding <=> u.embedding, s.id
                    LIMIT 1
                ) s ON true
                ORDER BY u.id
            """)
            rows = db.execute(sql).fetchall()
            for r in rows:
                best_match = None
                if r[3] is not None:  # s.book
                    best_match = ThemeContextResponse(
                        book=r[3],
                        chapter_number=r[4],
                        chapter_title=r[5] or "",
                        sentence_text=r[6],
                        sentence_index=r[7]
                    )
                results.append(ThemeResponse(
                    tagalog_title=r[1],
                    meaning=r[2],
                    best_match=best_match
                ))
        except Exception as e:
            # Fallback 1: sequential queries with explicit Theme.id and Sentence.id ordering
            try:
                all_themes = db.query(Theme).order_by(Theme.id).all()
                unique_themes = {}
                for t in all_themes:
                    if t.tagalog_title not in unique_themes:
                        unique_themes[t.tagalog_title] = t

                for title, theme_obj in unique_themes.items():
                    best_match = None
                    if theme_obj.embedding is not None:
                        closest_sentence = db.scalars(
                            db.query(Sentence)
                            .order_by(Sentence.embedding.cosine_distance(theme_obj.embedding), Sentence.id)
                            .limit(1)
                        ).first()

                        if closest_sentence:
                            best_match = ThemeContextResponse(
                                book=closest_sentence.book,
                                chapter_number=closest_sentence.chapter_number,
                                chapter_title=closest_sentence.chapter_title or "",
                                sentence_text=closest_sentence.sentence_text,
                                sentence_index=closest_sentence.sentence_index
                            )

                    results.append(ThemeResponse(
                        tagalog_title=title,
                        meaning=theme_obj.meaning,
                        best_match=best_match
                    ))
            except Exception:
                # Fallback 2: bundled static cache if database is unreachable
                bundled_path = Path(__file__).resolve().parent.parent.parent / "data" / "themes_cache.json"
                if bundled_path.exists():
                    with open(bundled_path, "r", encoding="utf-8") as f:
                        cached_raw = json.load(f)
                    results = [ThemeResponse(**item) for item in cached_raw]

        _themes_cache = results
        return _themes_cache


def warm_themes_cache():
    """
    Warms the process-level _themes_cache in a background thread at startup.
    Uses get_themes with SessionLocal so the first visitor hits warm cache immediately.
    """
    global _themes_cache
    if _themes_cache is not None:
        return
    db = SessionLocal()
    try:
        get_themes(response=None, db=db)
        import logging
        logging.info("Themes cache successfully warmed at startup.")
    except Exception as e:
        import logging
        logging.warning(f"Failed to warm themes cache at startup: {e}")
    finally:
        db.close()

class CharacterChapterResponse(BaseModel):
    book: str
    chapter_number: int
    chapter_title: str
    score: float = 0.0
    preview_text: Optional[str] = None
    sentence_index: Optional[int] = None

from sqlalchemy import or_

@router.get("/characters/chapters", response_model=List[CharacterChapterResponse])
def get_character_chapters(
    name: str, # Can be comma-separated list of names/aliases
    sort_by: str = "number", # "number" or "relevance"
    mode: Optional[str] = "buod",
    db: Session = Depends(get_db),
    engine: RizalEngine = Depends(get_engine)
):
    """
    Get chapters where a character appears, sorted by number or relevance.
    Accepts comma-separated names for aliases (e.g. "Maria Clara,Maria,Clara").
    """
    aliases = [n.strip() for n in name.split(',') if n.strip()]
    primary_name = aliases[0] if aliases else name
    
    source_type = "summary" if mode == "buod" else "full"

    # 1. Identify relevant sentences
    relevant_sentences = []
    
    if sort_by == "relevance":
        # Semantic search
        query_embedding = engine.base_model.encode(primary_name, show_progress_bar=False).tolist()
        
        # Fetch top matches filtered by source_type
        sentences = db.query(Sentence).filter(
            Sentence.source_type == source_type
        ).order_by(
            Sentence.embedding.cosine_distance(query_embedding)
        ).limit(500).all()
        
        # Calculate scores
        for s in sentences:
            emb_vec = np.array(s.embedding)
            q_vec = np.array(query_embedding)
            score = float(np.dot(emb_vec, q_vec))
            relevant_sentences.append((s, score))
            
    else: # sort_by == "number" or default
        # Text search with OR logic for aliases and filter by source_type
        filters = [Sentence.sentence_text.ilike(f"%{alias}%") for alias in aliases]
        
        sentences = db.query(Sentence).filter(
            Sentence.source_type == source_type,
            or_(*filters)
        ).all()
        
        for s in sentences:
            relevant_sentences.append((s, 1.0)) # Dummy score

    # 2. Group by Book + Chapter
    chapter_map = {} # (book, chapter_num) -> {title, max_score, count, best_snippet, sentence_index}
    
    for s, score in relevant_sentences:
        # Standardize book name for UI response (noli/elfili)
        ui_book = s.book
        if 'noli' in s.book.lower(): ui_book = 'noli'
        elif 'fili' in s.book.lower(): ui_book = 'elfili'

        key = (ui_book, s.chapter_number)
        if key not in chapter_map:
            chapter_map[key] = {
                'book': ui_book,
                'chapter_number': s.chapter_number,
                'chapter_title': s.chapter_title or "",
                'max_score': -1.0,
                'preview_text': s.sentence_text,
                'sentence_index': s.sentence_index
            }
        
        # Update stats
        if score > chapter_map[key]['max_score']:
            chapter_map[key]['max_score'] = score
            chapter_map[key]['preview_text'] = s.sentence_text # Update snippet to best match
            chapter_map[key]['sentence_index'] = s.sentence_index # Update index to best match

    # 3. Convert to list and sort
    results = []
    for info in chapter_map.values():
        results.append(CharacterChapterResponse(
            book=info['book'],
            chapter_number=info['chapter_number'],
            chapter_title=info['chapter_title'],
            score=info['max_score'],
            preview_text=info['preview_text'],
            sentence_index=info['sentence_index']
        ))
        
    if sort_by == "relevance":
        results.sort(key=lambda x: x.score, reverse=True)
    else:
        # Sort by book (noli first), then chapter number
        results.sort(key=lambda x: (0 if x.book == 'noli' else 1, x.chapter_number))
        
    return results

class ThemeMatch(BaseModel):
    id: str
    label: str
    score: float
    explanation: Optional[str] = ""

class ChapterContentResponse(BaseModel):
    id: int
    is_short: bool = False
    sentence_index: int
    sentence_text: str
    themes: List[ThemeMatch] = []

class ReferenceRequest(BaseModel):
    sentence_text: str
    book: str
    chapter_number: int
    target_mode: str  # 'buod' or 'full'

@router.post("/chapters/reference")
def get_sentence_reference(
    request: ReferenceRequest,
    db: Session = Depends(get_db),
    engine: RizalEngine = Depends(get_engine)
):
    """
    Find the most similar sentence in the other mode (summary vs full) WITHIN THE SAME CHAPTER.
    """
    target_source_type = "summary" if request.target_mode == "buod" else "full"
    
    # Map book name
    db_book = request.book
    if request.target_mode == "full":
        if request.book.lower() == "noli": db_book = "Noli Me Tangere"
        elif request.book.lower() in ["elfili", "fili"]: db_book = "El Filibusterismo"
    else:
        if request.book.lower() == "noli": db_book = "noli"
        elif request.book.lower() in ["elfili", "fili"]: db_book = "elfili"

    # Encode the source sentence
    query_embedding = engine.base_model.encode(request.sentence_text, show_progress_bar=False).tolist()
    
    # Find the most similar sentence in the target book, source_type, AND SAME CHAPTER
    closest_sentence = db.query(Sentence).filter(
        Sentence.book == db_book,
        Sentence.source_type == target_source_type,
        Sentence.chapter_number == request.chapter_number
    ).order_by(
        Sentence.embedding.cosine_distance(query_embedding)
    ).first()
    
    if not closest_sentence:
        raise HTTPException(status_code=404, detail="No reference found in the target version.")
        
    return {
        "sentence_text": closest_sentence.sentence_text,
        "chapter_number": closest_sentence.chapter_number,
        "chapter_title": closest_sentence.chapter_title,
        "book": request.book,
        "mode": request.target_mode
    }

@router.get("/chapters/{book}/{chapter_number}", response_model=List[ChapterContentResponse])
def get_chapter_content(
    book: str, 
    chapter_number: int, 
    response: Response = None,
    mode: Optional[str] = "buod",
    db: Session = Depends(get_db),
    engine: RizalEngine = Depends(get_engine)
):
    """
    Fetch all sentences for a specific chapter, ordered by index.
    Includes theme classification for each sentence.
    Cached in process memory (bounded LRU, max 64 entries); includes Cache-Control header.
    """
    if response:
        response.headers["Cache-Control"] = "public, max-age=86400, stale-while-revalidate=604800"

    book_norm = book.lower()
    if "noli" in book_norm:
        book_key = "noli"
    elif "fili" in book_norm:
        book_key = "elfili"
    else:
        book_key = book_norm

    mode_key = "full" if (mode and mode.lower() == "full") else "buod"
    ch_key = str(chapter_number)

    # Ensure fingerprint check is run
    if not _fingerprint_checked:
        verify_corpus_fingerprint(db)

    # 1. Level 1: Static precomputed cache (0ms latency, zero DB load, zero CPU)
    if _static_cache_enabled:
        bundle = _static_chapter_cache.get((book_key, mode_key))
        if bundle and ch_key in bundle:
            return bundle[ch_key]

    # 2. Level 2: Bounded in-memory LRU cache
    cache_key = (book_norm, chapter_number, mode_key)
    with _chapter_content_lock:
        if cache_key in _chapter_content_cache:
            _chapter_content_cache.move_to_end(cache_key)
            return _chapter_content_cache[cache_key]

    # Mapping book names for consistency
    db_book = book
    if mode == "full":
        if book.lower() == "noli": db_book = "Noli Me Tangere"
        elif book.lower() in ["elfili", "fili"]: db_book = "El Filibusterismo"
    else:
        if book.lower() == "noli": db_book = "noli"
        elif book.lower() in ["elfili", "fili"]: db_book = "elfili"

    source_type = "summary" if mode == "buod" else "full"

    sentences = db.query(Sentence).filter(
        Sentence.book == db_book,
        Sentence.chapter_number == chapter_number,
        Sentence.source_type == source_type
    ).order_by(Sentence.sentence_index).all()
    
    if not sentences:
        # Fallback for full version if not in DB but maybe in CSV
        if mode == "full":
            df = get_csv_data(book)
            if df is not None:
                chapter_df = df[df['chapter_number'] == chapter_number]
                if not chapter_df.empty:
                    chapter_df = chapter_df.sort_values('sentence_number')
                    response_data = []
                    for _, row in chapter_df.iterrows():
                        response_data.append(ChapterContentResponse(
                            id=0, # Fallback ID for CSV data
                            is_short=len(str(row['sentence_text']).split()) <= 4,
                            sentence_index=row['sentence_number'],
                            sentence_text=row['sentence_text'],
                            themes=[]
                        ))
                    with _chapter_content_lock:
                        if len(_chapter_content_cache) >= MAX_CHAPTER_CONTENT_CACHE_ENTRIES:
                            _chapter_content_cache.popitem(last=False)
                        _chapter_content_cache[cache_key] = response_data
                    return response_data
        
        raise HTTPException(status_code=404, detail=f"Chapter {chapter_number} not found for {book} in {mode} mode")
    
    # Deduplicate by sentence_index (keep first occurrence)
    seen_indices = set()
    unique_sentences = []
    for s in sentences:
        if s.sentence_index not in seen_indices:
            seen_indices.add(s.sentence_index)
            unique_sentences.append(s)
    
    # Process themes using engine._fetch_context_cache to eliminate N+1 queries
    context_cache = engine._fetch_context_cache(db, unique_sentences, window=8)
    response_data = []
    for s in unique_sentences:
        # Classify themes using the engine with pre-fetched context_cache
        themes = engine._classify_themes(db, s, "", context_cache=context_cache)
        
        # Convert to Pydantic model format
        theme_matches = [
            ThemeMatch(
                id=t['id'],
                label=t['label'],
                score=t['score'],
                explanation=t.get('explanation', "")
            ) for t in themes
        ]
        
        response_data.append(ChapterContentResponse(
            id=s.id,
            is_short=s.is_short,
            sentence_index=s.original_sentence_number or s.sentence_index,
            sentence_text=s.sentence_text,
            themes=theme_matches
        ))
    
    with _chapter_content_lock:
        if len(_chapter_content_cache) >= MAX_CHAPTER_CONTENT_CACHE_ENTRIES:
            _chapter_content_cache.popitem(last=False)
        _chapter_content_cache[cache_key] = response_data

    return response_data

class ThemeResult(BaseModel):
    label: str
    confidence: float
    evidence: Optional[str] = None

class PaksaResponse(BaseModel):
    has_theme: bool
    passage_ids: List[int] = []
    themes: List[ThemeResult] = []

@router.get("/sentences/{id}/paksa", response_model=PaksaResponse)
def get_sentence_paksa(id: int, db: Session = Depends(get_db), engine: RizalEngine = Depends(get_engine)):
    sentence = db.query(Sentence).filter(Sentence.id == id).first()
    if not sentence:
        raise HTTPException(status_code=404, detail="Sentence not found")
        
    book_key = "noli" if sentence.book.lower() in ("noli", "noli me tangere") else "elfili"
    
    if sentence.source_type == 'summary':
        passage_sentences = [sentence]
    else:
        if sentence.passage_id is None:
            passage_sentences = [sentence]
        else:
            passage_sentences = db.query(Sentence).filter(
                Sentence.book == sentence.book,
                Sentence.chapter_number == sentence.chapter_number,
                Sentence.source_type == 'full',
                Sentence.passage_id == sentence.passage_id
            ).order_by(Sentence.sentence_index).all()
            
    passage_text = " ".join([str(s.sentence_text) for s in passage_sentences]).lower()
    
    engine._ensure_themes_loaded(db)
    
    # 1. Character Detection (String matching via engine default logic)
    char_candidates = engine._get_character_theme_candidates(passage_text, book_key)
    candidate_themes = char_candidates if char_candidates is not None else set(engine.theme_grouped.keys())

    if not candidate_themes:
        return PaksaResponse(has_theme=False)

    # 2. Candidate Theme Scoring
    weighted_embeddings = []
    total_weight = 0.0
    for s in passage_sentences:
        if s.embedding is None: continue
        word_count = len(str(s.sentence_text).split())
        weight = math.log(word_count + 1)
        weighted_embeddings.append(np.array(s.embedding) * weight)
        total_weight += weight
        
    if total_weight == 0:
        return PaksaResponse(has_theme=False)
        
    consensus = np.sum(weighted_embeddings, axis=0) / total_weight
    consensus = consensus / (np.linalg.norm(consensus) + 1e-9)

    final_passed = []
    
    for theme_title in candidate_themes:
        if theme_title not in engine.theme_grouped:
            continue
            
        theme_data = engine.theme_grouped[theme_title]
        title_emb = theme_data['title_embedding']
        meaning_emb = theme_data['avg_meaning_embedding']
        
        # Semantic Score: 0.30 Title + 0.70 Meaning
        title_sim = float(np.dot(consensus, title_emb))
        meaning_sim = float(np.dot(consensus, meaning_emb))
        theme_score = (0.30 * title_sim) + (0.70 * meaning_sim)
        
        # Lexical Score (Overlap Passge vs Theme Title)
        lexical_score = engine._compute_simple_lexical(passage_text, theme_title)
        
        # Final Score: 0.80 Semantic + 0.20 Lexical
        final_score = (0.80 * theme_score) + (0.20 * lexical_score)
        final_score = min(max(final_score, 0.0), 1.0)
        
        if final_score >= 0.70:
            final_passed.append(ThemeResult(
                label=theme_title,
                confidence=final_score,
                evidence=None
            ))
            
    if not final_passed:
        return PaksaResponse(has_theme=False)
        
    # Sort and limit to top 2
    final_passed.sort(key=lambda x: x.confidence, reverse=True)
    final_passed = final_passed[:2]
    
    return PaksaResponse(
        has_theme=True,
        passage_ids=[s.id for s in passage_sentences],
        themes=final_passed
    )

class SanggunianResponse(BaseModel):
    has_reference: bool
    passage_ids: List[int] = []
    reference_text: str = ""
    alignment_status: str = ""
    score: float = 0.0
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    char_score: float = 0.0
    ratio_score: float = 0.0
    matched_characters: List[str] = []
    buod_sentence_index: Optional[int] = None
    full_sentence_indices: List[int] = []
    full_is_short: List[bool] = []


@router.get("/sentences/{id}/sanggunian", response_model=SanggunianResponse)
def get_sentence_sanggunian(id: int, db: Session = Depends(get_db), engine: RizalEngine = Depends(get_engine)):
    sentence = db.query(Sentence).filter(Sentence.id == id).first()
    if not sentence:
        raise HTTPException(status_code=404, detail="Sentence not found")

    # ── Full-text sentence: return its own passage ────────────────────────────
    if sentence.source_type == 'full':
        if sentence.passage_id is not None:
            passage_sentences = db.query(Sentence).filter(
                Sentence.book == sentence.book,
                Sentence.chapter_number == sentence.chapter_number,
                Sentence.source_type == 'full',
                Sentence.passage_id == sentence.passage_id
            ).order_by(Sentence.sentence_index).all()
        else:
            passage_sentences = [sentence]

        return SanggunianResponse(
            has_reference=True,
            passage_ids=[s.id for s in passage_sentences],
            reference_text=" ".join(str(s.sentence_text) for s in passage_sentences),
            alignment_status="precise",
            score=1.0,
            matched_characters=[],
            buod_sentence_index=sentence.sentence_index,
            full_sentence_indices=[s.sentence_index for s in passage_sentences]
        )

    # ── Buod sentence: run Buod-to-Full alignment ────────────────────────────
    search_book_noli = sentence.book.lower() in ('noli', 'noli me tangere')
    full_book = "Noli Me Tangere" if search_book_noli else "El Filibusterismo"

    search_chapter = sentence.chapter_number
    # Noli chapter 64 maps to 63 in the full text
    if search_book_noli and search_chapter == 64:
        search_chapter = 63

    # ── Run alignment (with per-chapter cache) ───────────────────────────────
    cache_key = (full_book, search_chapter, sentence.book, sentence.chapter_number)
    if cache_key not in _alignment_cache:
        # 1. Fetch ALL buod and full sentences for the chapter
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
            return SanggunianResponse(has_reference=False)

        # 2. Extract texts and embeddings
        buod_texts = [r.sentence_text for r in buod_rows]
        full_texts = [r.sentence_text for r in full_rows]
        
        # Check for missing embeddings and encode if necessary
        if any(r.embedding_dapt is None for r in buod_rows):
            buod_embs = engine.dapt_model.encode(buod_texts)
        else:
            buod_embs = np.array([r.embedding_dapt for r in buod_rows], dtype=np.float32)
            
        if any(r.embedding_dapt is None for r in full_rows):
            full_embs = engine.dapt_model.encode(full_texts)
        else:
            full_embs = np.array([r.embedding_dapt for r in full_rows], dtype=np.float32)

        # 3. Use RobustAligner
        _alignment_cache[cache_key] = {
            "v": "2.0", # Versioning to force refresh
            "alignments": engine.robust_aligner.align(
                buod_sentences=buod_texts,
                full_sentences=full_texts,
                buod_embeddings=buod_embs,
                full_embeddings=full_embs,
                full_is_short=[r.is_short for r in full_rows]
            ),
            "full_sentences": full_rows,
            "buod_rows": buod_rows
        }

    cached_data = _alignment_cache[cache_key]
    alignments = cached_data["alignments"]
    full_chapter_sentences = cached_data["full_sentences"]
    buod_rows = cached_data["buod_rows"]
    
    # Find the position of THIS sentence within the buod list
    buod_idx = next((i for i, s in enumerate(buod_rows) if s.id == sentence.id), None)
    if buod_idx is None or buod_idx >= len(alignments):
        return SanggunianResponse(has_reference=False)

    match = alignments[buod_idx]
    w_start, w_end = match.best_window_start, match.best_window_end
    
    passage_sentences = full_chapter_sentences[w_start:w_end+1]
    if not passage_sentences:
        return SanggunianResponse(has_reference=False)

    return SanggunianResponse(
        has_reference=True,
        passage_ids=[s.id for s in passage_sentences],
        reference_text=" ".join(str(s.sentence_text) for s in passage_sentences),
        alignment_status="precise",
        score=match.final_score,
        semantic_score=match.semantic_score,
        lexical_score=match.lexical_score,
        char_score=match.tauhan_score,
        ratio_score=match.position_score,
        matched_characters=match.matched_characters,
        buod_sentence_index=sentence.original_sentence_number or sentence.sentence_index,
        full_sentence_indices=[s.original_sentence_number or s.sentence_index for s in passage_sentences],
        full_is_short=[s.is_short for s in passage_sentences]
    )


class CharacterThemeResult(BaseModel):
    label: str
    description: str

class CharacterPaksaResponse(BaseModel):
    characterName: str
    themes: List[CharacterThemeResult]

@router.get("/characters/{name}/paksa", response_model=CharacterPaksaResponse)
def get_character_paksa(
    name: str, 
    book: str, 
    db: Session = Depends(get_db), 
    engine: RizalEngine = Depends(get_engine)
):
    book_key = "noli" if book.lower() in ("noli", "noli me tangere") else "elfili"
    
    backend_data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    char_aliases_path = backend_data_dir / "character_aliases.json"
    char_aliases = json.loads(char_aliases_path.read_text(encoding='utf-8')) if char_aliases_path.exists() else []
    
    canonical_name = name
    name_lower = name.lower().strip()
    
    for c in char_aliases:
        c_name = c.get('name', '')
        aliases = c.get('aliases', [])
        if name_lower == c_name.lower() or name_lower in [a.lower() for a in aliases]:
            canonical_name = c_name
            break
            
    char_theme_map_path = backend_data_dir / f"character_theme_map_{book_key}.json"
    char_theme_map = json.loads(char_theme_map_path.read_text(encoding='utf-8')) if char_theme_map_path.exists() else {}
    
    associated_themes = char_theme_map.get(canonical_name, [])
    
    engine._ensure_themes_loaded(db)
    
    results = []
    for theme_label in associated_themes:
        meanings = [t['meaning'] for t in getattr(engine, 'theme_cache', []) if t['tagalog_title'] == theme_label]
        
        best_meaning = ""
        if meanings:
            explicit_meanings = [m for m in meanings if re.search(r'\b' + re.escape(canonical_name) + r'\b', m, re.IGNORECASE)]
            if explicit_meanings:
                best_meaning = explicit_meanings[0]
            else:
                best_meaning = meanings[0]
                
        results.append(CharacterThemeResult(
            label=theme_label,
            description=best_meaning
        ))
        
    return CharacterPaksaResponse(
        characterName=canonical_name,
        themes=results
    )
