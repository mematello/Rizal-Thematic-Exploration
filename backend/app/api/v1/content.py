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
def get_chapters(db: Session = Depends(get_db)):
    """
    Fetch unique chapters from the sentences table.
    """
    # We query for distinct book, chapter_number, and chapter_title
    chapters = db.query(
        Sentence.book, 
        Sentence.chapter_number, 
        Sentence.chapter_title
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
    db: Session = Depends(get_db),
    engine: RizalEngine = Depends(get_engine)
):
    """
    Get chapters where a character appears, sorted by number or relevance.
    Accepts comma-separated names for aliases (e.g. "Maria Clara,Maria,Clara").
    """
    aliases = [n.strip() for n in name.split(',') if n.strip()]
    primary_name = aliases[0] if aliases else name

    # 1. Identify relevant sentences
    relevant_sentences = []
    
    if sort_by == "relevance":
        # Semantic search
        # Use the primary name for the embedding query as it captures the essence best usually.
        # Alternatively, we could average embeddings of all aliases, but primary name is usually sufficient for semantic context.
        query_embedding = engine.model.encode(primary_name, show_progress_bar=False).tolist()
        
        # Fetch top matches
        sentences = db.scalars(
            select(Sentence)
            .order_by(Sentence.embedding.cosine_distance(query_embedding))
            .limit(500)
        ).all()
        
        # Calculate scores
        for s in sentences:
            emb_vec = np.array(s.embedding)
            q_vec = np.array(query_embedding)
            score = float(np.dot(emb_vec, q_vec))
            relevant_sentences.append((s, score))
            
    else: # sort_by == "number" or default
        # Text search with OR logic for aliases
        filters = [Sentence.sentence_text.ilike(f"%{alias}%") for alias in aliases]
        
        sentences = db.query(Sentence).filter(
            or_(*filters)
        ).all()
        
        for s in sentences:
            relevant_sentences.append((s, 1.0)) # Dummy score

    # 2. Group by Book + Chapter
    chapter_map = {} # (book, chapter_num) -> {title, max_score, count, best_snippet, sentence_index}
    
    for s, score in relevant_sentences:
        key = (s.book, s.chapter_number)
        if key not in chapter_map:
            chapter_map[key] = {
                'book': s.book,
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

# ... existing endpoints ...

class ThemeMatch(BaseModel):
    id: str
    label: str
    score: float
    explanation: str

class ChapterContentResponse(BaseModel):
    sentence_index: int
    sentence_text: str
    themes: List[ThemeMatch] = []

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
    if mode == "full":
        df = get_csv_data(book)
        if df is None:
            raise HTTPException(status_code=404, detail="Full version not available for this chapter.")
            
        chapter_df = df[df['chapter_number'] == chapter_number]
        if chapter_df.empty:
            raise HTTPException(status_code=404, detail="Full version not available for this chapter.")
            
        chapter_df = chapter_df.sort_values('sentence_number')
        
        response_data = []
        for _, row in chapter_df.iterrows():
            response_data.append(ChapterContentResponse(
                sentence_index=row['sentence_number'],
                sentence_text=row['sentence_text'],
                themes=[]
            ))
        return response_data

    sentences = db.query(Sentence).filter(
        Sentence.book == book,
        Sentence.chapter_number == chapter_number
    ).order_by(Sentence.sentence_index).all()
    
    if not sentences:
        raise HTTPException(status_code=404, detail="Chapter not found")
    
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
        # Note: This might be slow for long chapters if not cached or optimized
        # But for now, we follow the user's request to use the model embedding
        themes = engine._classify_themes(db, s.embedding, s.sentence_text)
        
        # Convert to Pydantic model format
        theme_matches = [
            ThemeMatch(
                id=t['id'],
                label=t['label'],
                score=t['score'],
                explanation=t['explanation']
            ) for t in themes
        ]
        
        response_data.append(ChapterContentResponse(
            sentence_index=s.sentence_index,
            sentence_text=s.sentence_text,
            themes=theme_matches
        ))
        
    return response_data
