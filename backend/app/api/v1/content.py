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

@router.post("/sentences/batch/paksa", response_model=dict[int, PaksaResponse])
def get_batch_paksa(ids: List[int], db: Session = Depends(get_db)):
    """
    Batch fetch theme data for multiple sentences.
    """
    results = {}
    for sid in ids:
        try:
            results[sid] = _get_paksa_data(sid, db)
        except HTTPException:
            continue
    return results

def _get_paksa_data(id: int, db: Session) -> PaksaResponse:
    sentence = db.query(Sentence).filter(Sentence.id == id).first()
    if not sentence:
        raise HTTPException(status_code=404, detail="Sentence not found")
        
    book_key = "noli" if sentence.book.lower() in ("noli", "noli me tangere") else "elfili"
    
    backend_data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    theme_bank_path = backend_data_dir / "theme_bank.pkl"
    if not theme_bank_path.exists():
        return PaksaResponse(has_theme=False)
        
    with open(theme_bank_path, 'rb') as f:
        theme_bank_full = pickle.load(f)
        
    theme_bank = []
    if isinstance(theme_bank_full, dict) and book_key in theme_bank_full:
        theme_bank = theme_bank_full[book_key]
        
    char_theme_map_path = backend_data_dir / f"character_theme_map_{book_key}.json"
    theme_kw_path = backend_data_dir / f"theme_keywords_{book_key}.json"
    char_aliases_path = backend_data_dir / "character_aliases.json"
    
    char_theme_map = json.loads(char_theme_map_path.read_text(encoding='utf-8')) if char_theme_map_path.exists() else {}
    theme_kw_map = json.loads(theme_kw_path.read_text(encoding='utf-8')) if theme_kw_path.exists() else {}
    char_aliases = json.loads(char_aliases_path.read_text(encoding='utf-8')) if char_aliases_path.exists() else []
    
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
    
    # Gate 1: Character-Theme Affinity Filter
    found_chars = set()
    for c in char_aliases:
        name = c.get('name', '')
        aliases = c.get('aliases', [])
        if name and name not in aliases: aliases.append(name)
        for a in aliases:
            if not a: continue
            if re.search(r'\b' + re.escape(a.lower()) + r'\b', passage_text):
                found_chars.add(name)
                break
                
    candidate_themes = set()
    has_chars = len(found_chars) > 0
    if has_chars:
        for c in found_chars:
            if c in char_theme_map:
                candidate_themes.update(char_theme_map[c])
    else:
        candidate_themes = set([row['label'] for row in theme_bank])
        
    if not candidate_themes:
        return PaksaResponse(has_theme=False)
        
    # Gate 2: Keyword Density Filter
    passing_gate_2 = set()
    theme_kw_densities = {}
    
    req_kws = config.PAKSA_MIN_KWS_WITH_CHARS if has_chars else config.PAKSA_MIN_KWS_NO_CHARS
    passage_words = set(re.findall(r'\w+', passage_text))
    
    for theme in candidate_themes:
        kws = theme_kw_map.get(theme, [])
        matches = sum(1 for kw in kws if kw.lower() in passage_words)
        
        if matches >= req_kws:
            passing_gate_2.add(theme)
            density_score = min(1.0, matches / max(1, req_kws))
            theme_kw_densities[theme] = density_score
            
    if not passing_gate_2:
        return PaksaResponse(has_theme=False)
        
    # Final Scoring
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
    
    final_passed = {}
    wd = config.PAKSA_WEIGHT_KEYWORD_DENSITY
    ws = config.PAKSA_WEIGHT_SEMANTIC
    
    for row in theme_bank:
        label = row['label']
        if label not in passing_gate_2: continue
            
        semantic_score = float(np.dot(consensus, np.array(row['embedding'])))
        density_score = theme_kw_densities[label]
        final_score = (density_score * wd) + (semantic_score * ws)
        
        if final_score >= config.PAKSA_THEME_THRESHOLD:
            if label not in final_passed or semantic_score > final_passed[label]['sem_score']:
                final_passed[label] = {
                    'confidence': final_score,
                    'sem_score': semantic_score,
                    'evidence': row['evidence']
                }
                
    if not final_passed:
        return PaksaResponse(has_theme=False)
        
    sorted_themes = sorted([
        ThemeResult(label=k, confidence=v['confidence'], evidence=v['evidence'])
        for k, v in final_passed.items()
    ], key=lambda x: x.confidence, reverse=True)
    
    return PaksaResponse(
        has_theme=True,
        passage_ids=[s.id for s in passage_sentences],
        themes=sorted_themes
    )

@router.get("/sentences/{id}/paksa", response_model=PaksaResponse)
def get_sentence_paksa(id: int, db: Session = Depends(get_db)):
    return _get_paksa_data(id, db)

class SanggunianResponse(BaseModel):
    has_reference: bool
    passage_ids: List[int] = []
    reference_text: str = ""
    alignment_status: str = ""
    score: float = 0.0
    matched_characters: List[str] = []

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

@router.post("/sentences/batch/sanggunian", response_model=dict[int, SanggunianResponse])
def get_batch_sanggunian(ids: List[int], db: Session = Depends(get_db)):
    """
    Batch fetch reference data for multiple sentences.
    """
    results = {}
    for sid in ids:
        try:
            results[sid] = _get_sanggunian_data(sid, db)
        except HTTPException:
            continue
    return results

def _get_sanggunian_data(id: int, db: Session) -> SanggunianResponse:
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
            matched_characters=[]
        )
        
    search_chapter = sentence.chapter_number
    search_book_noli = sentence.book.lower() in ('noli', 'noli me tangere')
    if search_book_noli and search_chapter == 64:
        search_chapter = 63
        
    full_book = "Noli Me Tangere" if search_book_noli else "El Filibusterismo"
    backend_data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    
    char_aliases_path = backend_data_dir / "character_aliases.json"
    char_aliases = json.loads(char_aliases_path.read_text(encoding='utf-8')) if char_aliases_path.exists() else []
    
    buod_text = (sentence.sentence_text or "").lower()
    buod_chars = set()
    for c in char_aliases:
        name = c.get('name', '')
        aliases = c.get('aliases', [])
        if name and name not in aliases: aliases.append(name)
        for a in aliases:
            if not a: continue
            if re.search(r'\b' + re.escape(a.lower()) + r'\b', buod_text):
                buod_chars.add(name)
                break
                
    nlp = get_nlp()
    doc = nlp(buod_text)
    action_words = set([token.lemma_.lower() for token in doc if token.pos_ in ["VERB", "NOUN"] and len(token.text) > 2])
    buod_emb = np.array(sentence.embedding) if sentence.embedding is not None else None
    buod_words_all = set(re.findall(r'\w+', buod_text))
    
    def search_in_chapter(chap_num):
        full_sentences = db.query(Sentence).filter(
            Sentence.book == full_book,
            Sentence.chapter_number == chap_num,
            Sentence.source_type == 'full'
        ).order_by(Sentence.sentence_index).all()
        
        buod_sentences = db.query(Sentence).filter(
            Sentence.book == sentence.book,
            Sentence.chapter_number == sentence.chapter_number,
            Sentence.source_type == 'summary'
        ).order_by(Sentence.sentence_index).all()
        
        if not full_sentences or not buod_sentences: return None
        
        total_full = len(full_sentences)
        total_buod = len(buod_sentences)
        
        chapter_ratio = total_full / max(1, total_buod)
        dynamic_buffer = chapter_ratio * config.SANGGUNIAN_DYNAMIC_BFR_MULTIPLIER
        
        buod_idx = next((i for i, bs in enumerate(buod_sentences) if bs.id == sentence.id), 0)
        search_center = (buod_idx / max(1, total_buod)) * total_full
        
        candidates = [fs for i, fs in enumerate(full_sentences) if (search_center - dynamic_buffer) <= i <= (search_center + dynamic_buffer)]
        if not candidates or buod_emb is None: return None
        
        best_fs = None
        best_score = -1.0
        best_chars = []
        
        for fs in candidates:
            fs_text = (fs.sentence_text or "").lower()
            if not fs_text or fs.embedding is None: continue
            
            # Gate 1: Character Overlap Filter
            fs_chars = set()
            for c in char_aliases:
                name = c.get('name', '')
                aliases = c.get('aliases', [])
                if name and name not in aliases: aliases.append(name)
                for a in aliases:
                    if not a: continue
                    if re.search(r'\b' + re.escape(a.lower()) + r'\b', fs_text):
                        fs_chars.add(name)
                        break
                        
            if buod_chars:
                overlap = buod_chars.intersection(fs_chars)
                if not overlap: continue # Hard reject
            else:
                overlap = set()
                
            # Gate 2: Action Keyword Match
            fs_doc = nlp(fs_text)
            fs_action_words = set([token.lemma_.lower() for token in fs_doc if token.pos_ in ["VERB", "NOUN"] and len(token.text) > 2])
            
            if action_words and not action_words.intersection(fs_action_words):
                continue # Hard reject
                
            # Hybrid Scoring
            fs_emb = np.array(fs.embedding)
            semantic_score = float(np.dot(buod_emb, fs_emb) / (np.linalg.norm(buod_emb) * np.linalg.norm(fs_emb) + 1e-9))
            
            fs_words_all = set(re.findall(r'\w+', fs_text))
            lexical_overlap = len(buod_words_all.intersection(fs_words_all))
            lexical_score = lexical_overlap / max(1, len(buod_words_all))
            
            char_boost = min(len(overlap) * config.SANGGUNIAN_CHAR_BOOST_PER_MATCH, config.SANGGUNIAN_CHAR_BOOST_MAX)
            lexical_score += char_boost
            
            final_score = (lexical_score * config.SANGGUNIAN_WEIGHT_LEXICAL) + (semantic_score * config.SANGGUNIAN_WEIGHT_SEMANTIC)
            
            if final_score > best_score:
                best_score = final_score
                best_fs = fs
                best_chars = list(overlap) if overlap else list(fs_chars)
                
        if best_score < config.SANGGUNIAN_FALLBACK_THRESHOLD:
            return None
            
        return best_fs, best_score, best_chars
        
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
        
    best_fs, best_score, best_chars = res
    
    passage_sentences = []
    if best_fs.passage_id is not None:
        passage_sentences = db.query(Sentence).filter(
            Sentence.book == best_fs.book,
            Sentence.chapter_number == best_fs.chapter_number,
            Sentence.source_type == 'full',
            Sentence.passage_id == best_fs.passage_id
        ).order_by(Sentence.sentence_index).all()
    else:
        passage_sentences = [best_fs]
        
    return SanggunianResponse(
        has_reference=True,
        passage_ids=[s.id for s in passage_sentences],
        reference_text=" ".join([str(s.sentence_text) for s in passage_sentences]),
        alignment_status=alignment,
        score=best_score,
        matched_characters=best_chars
    )

@router.get("/sentences/{id}/sanggunian", response_model=SanggunianResponse)
def get_sentence_sanggunian(id: int, db: Session = Depends(get_db)):
    return _get_sanggunian_data(id, db)
