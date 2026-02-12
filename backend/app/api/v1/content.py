from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.models.database import SessionLocal, Sentence, Theme
from app.core.engine import get_engine, RizalEngine
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import select

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

class CharacterAppearance(BaseModel):
    book: str
    chapter_number: int
    chapter_title: str
    sentence_text: str
    sentence_index: int


from app.core.engine import get_engine, RizalEngine
from sqlalchemy import select

# ... imports ...

@router.get("/characters/appearances", response_model=List[CharacterAppearance])
def get_character_appearances(
    name: str,
    db: Session = Depends(get_db),
    engine: RizalEngine = Depends(get_engine)
):
    """
    Find top 5 semantic matches for the character.
    """
    # 1. Generate query embedding
    query_embedding = engine.model.encode(name, show_progress_bar=False)
    query_list = query_embedding.tolist()
    

    # 2. Vector Search using Cosine Distance
    # Fetch more candidates to allow for deduplication
    sentences = db.scalars(
        select(Sentence)
        .order_by(Sentence.embedding.cosine_distance(query_list))
        .limit(20)
    ).all()
    
    # Deduplicate by sentence_text
    unique_sentences = []
    seen_texts = set()
    
    for s in sentences:
        # Normalize slightly to catch near-duplicates if needed, but exact string match is usually enough
        text = s.sentence_text.strip()
        if text not in seen_texts:
            seen_texts.add(text)
            unique_sentences.append(s)
            if len(unique_sentences) == 5:
                break
    
    return [
        CharacterAppearance(
            book=s.book,
            chapter_number=s.chapter_number,
            chapter_title=s.chapter_title or "",
            sentence_text=s.sentence_text,
            sentence_index=s.sentence_index
        ) for s in unique_sentences
    ]

# ... existing endpoints ...

class ChapterContentResponse(BaseModel):
    sentence_index: int
    sentence_text: str

@router.get("/chapters/{book}/{chapter_number}", response_model=List[ChapterContentResponse])
def get_chapter_content(
    book: str, 
    chapter_number: int, 
    db: Session = Depends(get_db)
):
    """
    Fetch all sentences for a specific chapter, ordered by index.
    """
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
        
    return [
        ChapterContentResponse(
            sentence_index=s.sentence_index,
            sentence_text=s.sentence_text
        ) for s in unique_sentences
    ]
