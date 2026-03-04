from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.models.database import SessionLocal
from app.core.engine import get_engine, RizalEngine

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/search")
async def search_novels(
    q: str = Query(..., min_length=1, description="Search query"),
    source_type: str = Query("summary", description="Search in summary or full version"),
    db: Session = Depends(get_db),
    engine: RizalEngine = Depends(get_engine)
):
    """
    Search across Noli Me Tangere and El Filibusterismo.
    Returns ranked results using hybrid semantic + lexical search.
    """
    try:
        # Map frontend mode to backend source_type
        db_source_type = "summary" if source_type == "buod" else source_type
        
        results = engine.search(db, q, source_type=db_source_type)
        return {"results": results}
    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
