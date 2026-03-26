from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.models.database import SessionLocal, Sentence, Theme
from app.core.engine import get_engine, RizalEngine
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import select
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
    mode: Optional[str] = "buod",
    db: Session = Depends(get_db)
):
    """
    Fetch unique chapters from the sentences table for the given mode.
    """
    source_type = "summary" if mode == "buod" else "full"
    
    # We query for distinct book, chapter_number, and chapter_title
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
    
    return [
        ChapterResponse(
            book=c.book, 
            chapter_number=c.chapter_number, 
            chapter_title=c.chapter_title
        ) for c in chapters
    ]

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
def get_themes(db: Session = Depends(get_db)):
    """
    Fetch unique themes and their best matching sentence context.
    """
    # 1. Fetch all themes
    all_themes = db.query(Theme).all()
    
    # 2. Deduplicate by tagalog_title
    unique_themes = {}
    for t in all_themes:
        if t.tagalog_title not in unique_themes:
            unique_themes[t.tagalog_title] = t
            
    results = []
    
    # 3. Find best context for each unique theme
    for title, theme_obj in unique_themes.items():
        best_match = None
        
        # If theme has embedding, find closest sentence
        if theme_obj.embedding is not None:
            # Using cosine distance
            closest_sentence = db.scalars(
                db.query(Sentence)
                .order_by(Sentence.embedding.cosine_distance(theme_obj.embedding))
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
        
    return results

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
    mode: Optional[str] = "buod",
    db: Session = Depends(get_db),
    engine: RizalEngine = Depends(get_engine)
):
    """
    Fetch all sentences for a specific chapter, ordered by index.
    Includes theme classification for each sentence.
    """
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
                            sentence_index=row['sentence_number'],
                            sentence_text=row['sentence_text'],
                            themes=[]
                        ))
                    return response_data
        
        raise HTTPException(status_code=404, detail=f"Chapter {chapter_number} not found for {book} in {mode} mode")
    
    # Deduplicate by sentence_index (keep first occurrence)
    seen_indices = set()
    unique_sentences = []
    for s in sentences:
        if s.sentence_index not in seen_indices:
            seen_indices.add(s.sentence_index)
            unique_sentences.append(s)
    
    # Process themes for each sentence
    response_data = []
    for s in unique_sentences:
        # Classify themes using the engine
        themes = engine._classify_themes(db, s, "")
        
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
            sentence_index=s.sentence_index,
            sentence_text=s.sentence_text,
            themes=theme_matches
        ))
        
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

_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load(config.SANGGUNIAN_SPACY_MODEL)
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", config.SANGGUNIAN_SPACY_MODEL])
            _nlp = spacy.load(config.SANGGUNIAN_SPACY_MODEL)
    return _nlp

@router.get("/sentences/{id}/sanggunian", response_model=SanggunianResponse)
def get_sentence_sanggunian(id: int, db: Session = Depends(get_db), engine: RizalEngine = Depends(get_engine)):
    sentence = db.query(Sentence).filter(Sentence.id == id).first()
    if not sentence:
        raise HTTPException(status_code=404, detail="Sentence not found")
        
    if sentence.source_type == 'full':
        passage_sentences = []
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
            reference_text=" ".join([str(s.sentence_text) for s in passage_sentences]),
            alignment_status="precise",
            score=1.0,
            matched_characters=[],
            buod_sentence_index=sentence.sentence_index,
            full_sentence_indices=[s.sentence_index for s in passage_sentences]
        )
        
    search_chapter = sentence.chapter_number
    search_book_noli = sentence.book.lower() in ('noli', 'noli me tangere')
    if search_book_noli and search_chapter == 64:
        search_chapter = 63
        
    full_book = "Noli Me Tangere" if search_book_noli else "El Filibusterismo"
    book_key = "noli" if search_book_noli else "elfili"
    
    buod_text = (sentence.sentence_text or "").lower()
    
    # Extract ALL characters from buod using engine mapping
    patterns = getattr(engine, 'char_patterns', {}).get(book_key, [])
    buod_chars = set()
    for canon_name, pattern in patterns:
        if pattern.search(buod_text):
            buod_chars.add(canon_name)
            
    buod_emb = np.array(sentence.embedding) if sentence.embedding is not None else None
    
    def _jaccard(b_tokens: set, window_sentences: list) -> float:
        """Window-level Jaccard: B vs union of all tokens in the window."""
        from app.core.analyzer import extract_words
        w_tokens: set = set()
        for s in window_sentences:
            w_tokens.update(extract_words((s.sentence_text or "").lower()))
        if not b_tokens and not w_tokens:
            return 0.0
        return len(b_tokens & w_tokens) / len(b_tokens | w_tokens)

    def _avg_embedding(sents: list):
        embs = [np.array(s.embedding) for s in sents if s.embedding is not None]
        if not embs:
            return None
        return np.mean(embs, axis=0)

    def _cosine(a, b) -> float:
        if a is None or b is None:
            return 0.0
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))

    def _tauhan_boost(window_sents: list, char_freq: dict) -> tuple:
        """Returns (boost_value, list_of_matched_chars)."""
        if not buod_chars:
            return 0.0, []
        matched = []
        boost = 0.0
        window_text = " ".join((s.sentence_text or "").lower() for s in window_sents)
        for canon_name, pattern in patterns:
            if canon_name in buod_chars and pattern.search(window_text):
                distinctiveness = 1.0 - char_freq.get(canon_name, 0.0)
                boost += distinctiveness * 0.20
                matched.append(canon_name)
        # Cap at 0.20 (the weight slot for tauhan)
        return min(boost, 0.20), matched

    def _ratio_score(actual_idx: float, expected_idx: float, total_full: int) -> float:
        if total_full == 0:
            return 0.0
        return float(np.clip(1.0 - abs(actual_idx - expected_idx) / total_full, 0.0, 1.0))

    def _combine(jaccard: float, semantic: float, tauhan: float, ratio: float) -> float:
        return (jaccard * 0.30) + (semantic * 0.40) + (tauhan * 0.20) + (ratio * 0.10)

    def _get_window(sent_list: list, center_i: int, half: int) -> list:
        lo = max(0, center_i - half)
        hi = min(len(sent_list) - 1, center_i + half)
        return sent_list[lo:hi + 1]

    def search_in_chapter(chap_num, anchor_shift: int = 0):
        from app.core.analyzer import extract_words

        full_sentences_raw = db.query(Sentence).filter(
            Sentence.book == full_book,
            Sentence.chapter_number == chap_num,
            Sentence.source_type == 'full'
        ).order_by(Sentence.sentence_index).all()

        buod_sentences_raw = db.query(Sentence).filter(
            Sentence.book == sentence.book,
            Sentence.chapter_number == sentence.chapter_number,
            Sentence.source_type == 'summary'
        ).order_by(Sentence.sentence_index).all()

        if not full_sentences_raw or not buod_sentences_raw:
            return None

        # Deduplicate to prevent bloated totals if DB has dupes
        seen_full = set()
        full_sentences = []
        for fs in full_sentences_raw:
            if fs.sentence_index not in seen_full:
                seen_full.add(fs.sentence_index)
                full_sentences.append(fs)

        seen_buod = set()
        buod_sentences = []
        for bs in buod_sentences_raw:
            if bs.sentence_index not in seen_buod:
                seen_buod.add(bs.sentence_index)
                buod_sentences.append(bs)

        total_full = len(full_sentences)
        total_buod = max(1, len(buod_sentences))

        # ── Pre-computation ──────────────────────────────────────────────────
        buod_idx = next((i for i, bs in enumerate(buod_sentences) if bs.id == sentence.id), 0)
        buod_ratio = buod_idx / total_buod
        full_expected_idx = buod_ratio * total_full + anchor_shift
        full_expected_idx = max(0.0, min(full_expected_idx, total_full - 1))

        adaptive_window = max(3, round((total_full / total_buod) * 1.5))

        if buod_emb is None:
            return None

        buod_tokens = set(extract_words(buod_text))

        # Build character frequency map: canon_name -> fraction of full sentences containing it
        char_freq: dict = {}
        for canon_name, pattern in patterns:
            count = sum(1 for fs in full_sentences if pattern.search((fs.sentence_text or "").lower()))
            char_freq[canon_name] = count / total_full if total_full > 0 else 0.0

        # ── PASS 1: Large-window Jaccard scan ────────────────────────────────
        # Generate 5 candidate region centers around the expected position
        expected_int = int(round(full_expected_idx))
        spacing = max(1, adaptive_window // 2)
        region_centers = sorted(set(
            max(0, min(total_full - 1, expected_int + offset))
            for offset in [-2 * spacing, -spacing, 0, spacing, 2 * spacing]
        ))

        best_zone_center = expected_int
        best_p1_score = -1.0

        for center in region_centers:
            window = _get_window(full_sentences, center, adaptive_window)
            if not window:
                continue
            j_large  = _jaccard(buod_tokens, window)
            sem_large = _cosine(buod_emb, _avg_embedding(window))
            tau_large, _ = _tauhan_boost(window, char_freq)
            rat_large = _ratio_score(center, full_expected_idx, total_full)
            # For tauhan in combine we need the raw boost value not capped — convert to 0..1
            tau_norm = tau_large / 0.20 if tau_large > 0 else 0.0
            p1_score = _combine(j_large, sem_large, tau_norm, rat_large)
            if p1_score > best_p1_score:
                best_p1_score = p1_score
                best_zone_center = center

        # Candidate zone: adaptive_window sentences around best P1 winner
        zone = _get_window(full_sentences, best_zone_center, adaptive_window)
        # Map back to indices in full_sentences for ratio computation
        zone_start = max(0, best_zone_center - adaptive_window)

        # ── PASS 2: Sentence-level refinement ────────────────────────────────
        best_fs = None
        best_score = -1.0
        best_chars: list = []
        best_semantic = 0.0
        best_lexical = 0.0
        best_char_score = 0.0
        best_ratio_score = 0.0
        best_idx = -1

        for zone_i, fs in enumerate(zone):
            if fs.embedding is None or not (fs.sentence_text or "").strip():
                continue

            actual_full_i = zone_start + zone_i  # position in full_sentences list

            # ±3 window around this sentence (within full chapter)
            small_window = _get_window(full_sentences, actual_full_i, 3)

            # Jaccard: window-level (±3 window, not single sentence)
            j_small = _jaccard(buod_tokens, small_window)

            # Semantic: single-sentence cosine (narrows at Pass 2)
            fs_emb = np.array(fs.embedding)
            sem_sent = _cosine(buod_emb, fs_emb)

            # Tauhan boost over ±3 window
            tau_val, matched_chars = _tauhan_boost(small_window, char_freq)
            tau_norm = tau_val / 0.20 if tau_val > 0 else 0.0

            # Ratio
            rat = _ratio_score(actual_full_i, full_expected_idx, total_full)

            final = _combine(j_small, sem_sent, tau_norm, rat)

            if final > best_score:
                best_score = final
                best_fs = fs
                best_chars = matched_chars if matched_chars else list(buod_chars)
                best_semantic = sem_sent
                best_lexical = j_small
                best_char_score = tau_norm
                best_ratio_score = rat
                best_idx = actual_full_i

        # ── Fallback ─────────────────────────────────────────────────────────
        THRESHOLD = 0.35
        if best_score < THRESHOLD:
            return None

        # ── Dynamic Window Expansion ─────────────────────────────────────────
        # GUARDS:
        # 1. Hard cap at 15 sentences maximum.
        # 2. Tightened threshold (0.50) for per-sentence gating.
        MAX_PASSAGE_SIZE = 15
        EXPANSION_THRESHOLD = 0.50
        
        final_indices = [best_idx]
        
        # Expand Left
        curr = best_idx - 1
        while curr >= 0 and len(final_indices) < MAX_PASSAGE_SIZE:
            fs = full_sentences[curr]
            sim = _cosine(buod_emb, np.array(fs.embedding))
            if sim >= EXPANSION_THRESHOLD or len(final_indices) < 2:
                final_indices.insert(0, curr)
                curr -= 1
            else:
                break
                
        # Expand Right
        curr = best_idx + 1
        while curr < len(full_sentences) and len(final_indices) < MAX_PASSAGE_SIZE:
            fs = full_sentences[curr]
            sim = _cosine(buod_emb, np.array(fs.embedding))
            if sim >= EXPANSION_THRESHOLD or len(final_indices) < 3:
                final_indices.append(curr)
                curr += 1
            else:
                break

        passage_sentences = [full_sentences[i] for i in final_indices]
        return passage_sentences, best_score, best_chars, best_semantic, best_lexical, best_char_score, best_ratio_score
        
    res = search_in_chapter(search_chapter)
    alignment = "precise"
    
    if not res:
        res = search_in_chapter(search_chapter - 1)
        if res: alignment = "expanded"
        if not res:
            res = search_in_chapter(search_chapter + 1)
            if res: alignment = "expanded"
            
    if not res:
        return SanggunianResponse(has_reference=False)
        
    passage_sentences, best_score, best_chars, sem_score, lex_score, char_score, ratio_score = res
        
    return SanggunianResponse(
        has_reference=True,
        passage_ids=[s.id for s in passage_sentences],
        reference_text=" ".join([str(s.sentence_text) for s in passage_sentences]),
        alignment_status=alignment,
        score=best_score,
        semantic_score=sem_score,
        lexical_score=lex_score,
        char_score=char_score,
        ratio_score=ratio_score,
        matched_characters=best_chars,
        buod_sentence_index=sentence.sentence_index,
        full_sentence_indices=[s.sentence_index for s in passage_sentences]
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
