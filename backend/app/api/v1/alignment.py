"""
alignment.py — FastAPI router for chapter-level buod-to-full-text alignment.

GET /api/v1/align/{book}/{chapter_number}

Returns one AlignedBlock per buod sentence, each containing the set of full-text
sentences that the buod sentence summarises, together with diagnostic scores.

Query params:
  book          : noli | elfili
  chapter_number: integer
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models.database import SessionLocal, Sentence
from app.core.engine import get_engine, RizalEngine
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()


# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class AlignedBlockResponse(BaseModel):
    buod_index: int
    buod_text: str
    full_text_start: int
    full_text_end: int
    full_text_sentences: List[str]
    alignment_anchors: List[str]
    semantic_score: float
    character_score: float
    variance_penalty: float
    length_penalty: float
    total_score: float

class RobustAlignedBlockResponse(BaseModel):
    buod_index: int
    buod_text: str
    best_window_start: int
    best_window_end: int
    best_center_sentence: int
    lexical_score: float
    semantic_score: float
    position_score: float
    tauhan_score: float
    final_score: float
    matched_characters: List[str]

class ChapterAlignmentResponse(BaseModel):
    book: str
    chapter_number: int
    chapter_title: str
    buod_count: int
    full_count: int
    alignment: List[AlignedBlockResponse]

class RobustChapterAlignmentResponse(BaseModel):
    book: str
    chapter_number: int
    chapter_title: str
    buod_count: int
    full_count: int
    alignment: List[RobustAlignedBlockResponse]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get(
    "/align/{book}/{chapter_number}",
    response_model=ChapterAlignmentResponse,
    summary="Align buod sentences to full-text passages for a chapter",
)
def align_chapter(
    book: str,
    chapter_number: int,
    db: Session = Depends(get_db),
    engine: RizalEngine = Depends(get_engine),
):
    """
    Runs the monotonic DP sequence aligner on the specified chapter.

    - **book**: `noli` or `elfili`
    - **chapter_number**: 1-based chapter index
    """
    book_lower = book.lower()
    if book_lower not in ("noli", "elfili", "fili"):
        raise HTTPException(status_code=400, detail="book must be 'noli' or 'elfili'")

    book_key = "noli" if book_lower == "noli" else "elfili"

    # DB names differ between summary and full versions
    summary_book = book_key                          # 'noli' or 'elfili'
    full_book = "Noli Me Tangere" if book_key == "noli" else "El Filibusterismo"

    # ---- Fetch buod sentences ----
    buod_rows = (
        db.query(Sentence)
        .filter(
            Sentence.book == summary_book,
            Sentence.chapter_number == chapter_number,
            Sentence.source_type == "summary",
        )
        .order_by(Sentence.sentence_index)
        .all()
    )

    if not buod_rows:
        raise HTTPException(
            status_code=404,
            detail=f"No buod sentences found for {book} chapter {chapter_number}.",
        )

    # ---- Fetch full text sentences ----
    full_rows = (
        db.query(Sentence)
        .filter(
            Sentence.book == full_book,
            Sentence.chapter_number == chapter_number,
            Sentence.source_type == "full",
        )
        .order_by(Sentence.sentence_index)
        .all()
    )

    if not full_rows:
        raise HTTPException(
            status_code=404,
            detail=f"No full-text sentences found for {book} chapter {chapter_number}.",
        )

    # 1. Prepare embeddings
    import numpy as np
    buod_texts = [r.sentence_text for r in buod_rows]
    full_texts = [r.sentence_text for r in full_rows]
    
    if any(r.embedding is None for r in buod_rows):
        buod_embs = engine.dapt_model.encode(buod_texts)
    else:
        buod_embs = np.array([r.embedding for r in buod_rows], dtype=np.float32)
        
    if any(r.embedding is None for r in full_rows):
        full_embs = engine.dapt_model.encode(full_texts)
    else:
        full_embs = np.array([r.embedding for r in full_rows], dtype=np.float32)

    # 2. Run robust aligner
    aligned_blocks = engine.robust_aligner.align(
        buod_sentences=buod_texts,
        full_sentences=full_texts,
        buod_embeddings=buod_embs,
        full_embeddings=full_embs
    )

    # 3. Map to AlignedBlockResponse (Legacy compatibility)
    response_blocks = []
    for b in aligned_blocks:
        window_sentences = [full_texts[i] for i in range(b.best_window_start, b.best_window_end + 1)]
        
        response_blocks.append(AlignedBlockResponse(
            buod_index=b.buod_index,
            buod_text=b.buod_text,
            full_text_start=b.best_window_start,
            full_text_end=b.best_window_end,
            full_text_sentences=window_sentences,
            alignment_anchors=[full_texts[b.best_center_sentence]],
            semantic_score=b.semantic_score,
            character_score=b.tauhan_score,
            variance_penalty=b.position_score, # Mapping position score to variance
            length_penalty=b.lexical_score,   # Mapping lexical score to length
            total_score=b.final_score,
        ))

    return ChapterAlignmentResponse(
        book=book_key,
        chapter_number=chapter_number,
        chapter_title=chapter_title,
        buod_count=len(buod_texts),
        full_count=len(full_texts),
        alignment=response_blocks
    )

@router.get(
    "/align/robust/{book}/{chapter_number}",
    response_model=RobustChapterAlignmentResponse,
    summary="Align buod sentences using the robust multi-factor system",
)
def robust_align_chapter(
    book: str,
    chapter_number: int,
    db: Session = Depends(get_db),
    engine: RizalEngine = Depends(get_engine),
):
    """
    Runs the RobustAligner on the specified chapter.
    Uses TF-IDF, embeddings, tauhan, and sequential consistency.
    """
    book_lower = book.lower()
    if book_lower not in ("noli", "elfili", "fili"):
        raise HTTPException(status_code=400, detail="book must be 'noli' or 'elfili'")

    book_key = "noli" if book_lower == "noli" else "elfili"
    full_book = "Noli Me Tangere" if book_key == "noli" else "El Filibusterismo"

    # 1. Fetch buod sentences & embeddings
    buod_rows = (
        db.query(Sentence)
        .filter(
            Sentence.book == book_key,
            Sentence.chapter_number == chapter_number,
            Sentence.source_type == "summary",
        )
        .order_by(Sentence.sentence_index)
        .all()
    )
    if not buod_rows:
        raise HTTPException(status_code=404, detail="No buod sentences found.")

    # 2. Fetch full-text sentences & embeddings
    full_rows = (
        db.query(Sentence)
        .filter(
            Sentence.book == full_book,
            Sentence.chapter_number == chapter_number,
            Sentence.source_type == "full",
        )
        .order_by(Sentence.sentence_index)
        .all()
    )
    if not full_rows:
        raise HTTPException(status_code=404, detail="No full-text sentences found.")

    import numpy as np
    buod_texts = [r.sentence_text for r in buod_rows]
    # Check if embeddings exist
    if any(r.embedding is None for r in buod_rows):
        # Trigger mass encode if missing? For now, raise error or placeholder
        buod_embs = engine.dapt_model.encode(buod_texts)
    else:
        buod_embs = np.array([r.embedding for r in buod_rows])

    full_texts = [r.sentence_text for r in full_rows]
    if any(r.embedding is None for r in full_rows):
        full_embs = engine.dapt_model.encode(full_texts)
    else:
        full_embs = np.array([r.embedding for r in full_rows])

    chapter_title = buod_rows[0].chapter_title or ""

    # 3. Run robust aligner
    aligned_blocks = engine.robust_aligner.align(
        buod_sentences=buod_texts,
        full_sentences=full_texts,
        buod_embeddings=buod_embs,
        full_embeddings=full_embs
    )

    # 4. Map to response
    return RobustChapterAlignmentResponse(
        book=book_key,
        chapter_number=chapter_number,
        chapter_title=chapter_title,
        buod_count=len(buod_texts),
        full_count=len(full_texts),
        alignment=[
            RobustAlignedBlockResponse(
                buod_index=b.buod_index,
                buod_text=b.buod_text,
                best_window_start=b.best_window_start,
                best_window_end=b.best_window_end,
                best_center_sentence=b.best_center_sentence,
                lexical_score=b.lexical_score,
                semantic_score=b.semantic_score,
                position_score=b.position_score,
                tauhan_score=b.tauhan_score,
                final_score=b.final_score,
                matched_characters=b.matched_characters
            )
            for b in aligned_blocks
        ]
    )
