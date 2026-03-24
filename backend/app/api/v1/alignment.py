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


class ChapterAlignmentResponse(BaseModel):
    book: str
    chapter_number: int
    chapter_title: str
    buod_count: int
    full_count: int
    alignment: List[AlignedBlockResponse]


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

    buod_texts = [r.sentence_text for r in buod_rows]
    full_texts  = [r.sentence_text for r in full_rows]
    chapter_title = buod_rows[0].chapter_title or ""

    # ---- Run aligner ----
    aligned_blocks = engine.aligner.align(
        buod_sentences=buod_texts,
        full_sentences=full_texts,
        book=book_key,
    )

    # ---- Serialise output ----
    response_blocks = [
        AlignedBlockResponse(
            buod_index=b.buod_index,
            buod_text=b.buod_text,
            full_text_start=b.full_text_start,
            full_text_end=b.full_text_end,
            full_text_sentences=b.full_text_sentences,
            alignment_anchors=b.alignment_anchors,
            semantic_score=b.semantic_score,
            character_score=b.character_score,
            variance_penalty=b.variance_penalty,
            length_penalty=b.length_penalty,
            total_score=b.total_score,
        )
        for b in aligned_blocks
    ]

    return ChapterAlignmentResponse(
        book=book_key,
        chapter_number=chapter_number,
        chapter_title=chapter_title,
        buod_count=len(buod_texts),
        full_count=len(full_texts),
        alignment=response_blocks,
    )
