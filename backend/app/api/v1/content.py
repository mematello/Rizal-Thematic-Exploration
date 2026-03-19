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

class PaksaResponse(BaseModel):
    has_theme: bool
    passage_ids: List[int] = []
    themes: List[ThemeResult] = []

@router.get("/sentences/{id}/paksa", response_model=PaksaResponse)
def get_sentence_paksa(id: int, db: Session = Depends(get_db)):
    sentence = db.query(Sentence).filter(Sentence.id == id).first()
    if not sentence:
        raise HTTPException(status_code=404, detail="Sentence not found")
        
    # Load theme bank
    backend_data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    theme_bank_path = backend_data_dir / "theme_bank.pkl"
    if not theme_bank_path.exists():
        return PaksaResponse(has_theme=False)
        
    with open(theme_bank_path, 'rb') as f:
        theme_bank = pickle.load(f)
        
    # Fetch passage
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
            
    # Compute Significance-Weighted Consensus Embedding
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
    
    T_theme = 0.65
    T_override = 0.85
    
    # Multi-Example Theme Scoring
    theme_scores = {}
    for theme_name, examples in theme_bank.items():
        scores = [float(np.dot(consensus, np.array(ex))) for ex in examples]
        scores.sort(reverse=True)
        top_3_avg = np.mean(scores[:3]) if len(scores) >= 3 else np.mean(scores)
        theme_scores[theme_name] = top_3_avg
        
    earned_themes = {t: s for t, s in theme_scores.items() if s >= T_theme}
    
    # Stability Check (for full text passages)
    if sentence.source_type == 'full' and len(passage_sentences) > 1 and earned_themes:
        sentence_top_themes = []
        for s in passage_sentences:
            if s.embedding is None: continue
            s_emb = np.array(s.embedding)
            s_scores = {}
            for t_name, examples in theme_bank.items():
                s_s = [float(np.dot(s_emb, np.array(ex))) for ex in examples]
                s_s.sort(reverse=True)
                s_scores[t_name] = np.mean(s_s[:3]) if len(s_s) >= 3 else np.mean(s_s)
            
            top_3 = sorted(s_scores.keys(), key=lambda k: s_scores[k], reverse=True)[:3]
            sentence_top_themes.append(set(top_3))
            
        if sentence_top_themes:
            intersection = sentence_top_themes[0].intersection(*sentence_top_themes[1:])
            union = sentence_top_themes[0].union(*sentence_top_themes[1:])
            jaccard = len(intersection) / len(union) if len(union) > 0 else 0
            
            if jaccard <= 0.6:
                earned_themes = {} # Stability failed
                
    # Sharp Sentence Override
    if sentence.embedding is not None:
        s_emb = np.array(sentence.embedding)
        for t_name, examples in theme_bank.items():
            s_s = [float(np.dot(s_emb, np.array(ex))) for ex in examples]
            s_s.sort(reverse=True)
            s_score = np.mean(s_s[:3]) if len(s_s) >= 3 else float(np.mean(s_s))
            if s_score >= T_override:
                earned_themes[t_name] = max(earned_themes.get(t_name, 0.0), s_score)
                
    if not earned_themes:
        return PaksaResponse(has_theme=False)
        
    sorted_themes = sorted([{"label": k, "confidence": float(v)} for k, v in earned_themes.items()], 
                          key=lambda x: x["confidence"], reverse=True)
                          
    return PaksaResponse(
        has_theme=True,
        passage_ids=[s.id for s in passage_sentences],
        themes=sorted_themes
    )

class SanggunianResponse(BaseModel):
    has_reference: bool
    passage_ids: List[int] = []
    reference_text: str = ""
    alignment_status: str = ""
    score: float = 0.0
    matched_characters: List[str] = []

@router.get("/sentences/{id}/sanggunian", response_model=SanggunianResponse)
def get_sentence_sanggunian(id: int, db: Session = Depends(get_db)):
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
    character_aliases = []
    if char_aliases_path.exists():
        with open(char_aliases_path, 'r', encoding='utf-8') as f:
            character_aliases = json.load(f)
            
    buod_chars_found = []
    if sentence.sentence_text:
        text_lower = sentence.sentence_text.lower()
        for c in character_aliases:
            aliases = c.get('aliases', [])
            if c.get('name'):
                aliases.append(c.get('name'))
            for alias in aliases:
                if not alias: continue
                if re.search(r'\b' + re.escape(alias.lower()) + r'\b', text_lower):
                    buod_chars_found.append(c)
                    break
                
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
        
        chapter_ratio = total_full / total_buod
        dynamic_buffer = chapter_ratio * 2
        
        buod_idx_in_chap = 0
        for i, bs in enumerate(buod_sentences):
            if bs.id == sentence.id:
                buod_idx_in_chap = i
                break
                
        position_ratio = buod_idx_in_chap / total_buod
        search_center = position_ratio * total_full
        
        candidates = []
        for i, fs in enumerate(full_sentences):
            if (search_center - dynamic_buffer) <= i <= (search_center + dynamic_buffer):
                candidates.append(fs)
                
        if not candidates:
            return None
            
        filtered_candidates = []
        if buod_chars_found:
            for fs in candidates:
                fs_matched_chars = []
                if not fs.sentence_text: continue
                fs_text_lower = fs.sentence_text.lower()
                for c in buod_chars_found:
                    aliases = c.get('aliases', [])
                    if c.get('name'): aliases.append(c.get('name'))
                    for alias in aliases:
                        if alias and re.search(r'\b' + re.escape(alias.lower()) + r'\b', fs_text_lower):
                            fs_matched_chars.append(c.get('name'))
                            break
                if fs_matched_chars:
                    filtered_candidates.append((fs, list(set(fs_matched_chars))))
            if not filtered_candidates:
                return None
        else:
            filtered_candidates = [(fs, []) for fs in candidates]
            
        best_fs = None
        best_score = -1.0
        best_chars = []
        
        if sentence.embedding is None or not sentence.sentence_text:
            return None
            
        buod_emb = np.array(sentence.embedding)
        buod_words = set(re.findall(r'\w+', sentence.sentence_text.lower()))
        
        for fs, fs_chars in filtered_candidates:
            if fs.embedding is None or not fs.sentence_text: continue
            fs_emb = np.array(fs.embedding)
            
            semantic_score = np.dot(buod_emb, fs_emb) / (np.linalg.norm(buod_emb) * np.linalg.norm(fs_emb) + 1e-9)
            
            fs_words = set(re.findall(r'\w+', fs.sentence_text.lower()))
            overlap = len(buod_words.intersection(fs_words))
            lexical_score = overlap / max(len(buod_words), 1)
            
            if fs_chars:
                lexical_score *= 2.0
                
            final_score = float(0.55 * lexical_score + 0.45 * semantic_score)
            if final_score > best_score:
                best_score = final_score
                best_fs = fs
                best_chars = fs_chars
                
        T_fallback = 0.45
        if best_score < T_fallback:
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
