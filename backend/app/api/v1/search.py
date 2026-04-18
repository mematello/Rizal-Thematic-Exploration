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
    novel: str = Query("both", description="Target novel (noli, elfili, or both)"),
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
        
        if "." in q:
            queries = [query.strip() for query in q.split(".") if query.strip()]
        else:
            queries = [q]

        if not queries:
            return {
                "results": {"noli": [], "elfili": []},
                "precise_matches": [],
                "metadata": {"result_mode": "none", "reason": "empty_query"}
            }

        combined_noli = []
        combined_elfili = []
        combined_precise = []
        metadata = None
        seen_noli_ids = set()
        seen_elfili_ids = set()
        seen_precise_ids = set()

        for query in queries:
            search_res = engine.search(db, query, source_type=db_source_type, novel=novel)
            
            if isinstance(search_res, dict) and "metadata" in search_res:
                results = search_res.get("results", {"noli": [], "elfili": []})
                precise_matches = search_res.get("precise_matches", [])
                
                if metadata is None:
                    metadata = search_res.get("metadata")
            else:
                results = search_res if isinstance(search_res, dict) and "noli" in search_res else {"noli": [], "elfili": []}
                precise_matches = []

            for item in results.get("noli", []):
                item_id = item.get("id") if isinstance(item, dict) else getattr(item, 'id', None)
                if item_id is not None:
                    if item_id not in seen_noli_ids:
                        seen_noli_ids.add(item_id)
                        combined_noli.append(item)
                else:
                    combined_noli.append(item)

            for item in results.get("elfili", []):
                item_id = item.get("id") if isinstance(item, dict) else getattr(item, 'id', None)
                if item_id is not None:
                    if item_id not in seen_elfili_ids:
                        seen_elfili_ids.add(item_id)
                        combined_elfili.append(item)
                else:
                    combined_elfili.append(item)
                    
            for item in precise_matches:
                item_id = item.get("id") if isinstance(item, dict) else getattr(item, 'id', None)
                if item_id is not None:
                    if item_id not in seen_precise_ids:
                        seen_precise_ids.add(item_id)
                        combined_precise.append(item)
                else:
                    combined_precise.append(item)

        if len(queries) > 1:
            def get_score(item):
                if isinstance(item, dict):
                    scores = item.get("scores", {})
                    return scores.get("final", 0) + scores.get("precision", 0) * 0.5
                return 0
                
            def get_precise_score(item):
                if isinstance(item, dict):
                    scores = item.get("scores", {})
                    return scores.get("precision", 0)
                return 0
                
            combined_noli = sorted(combined_noli, key=get_score, reverse=True)
            combined_elfili = sorted(combined_elfili, key=get_score, reverse=True)
            combined_precise = sorted(combined_precise, key=get_precise_score, reverse=True)[:5]
            
        final_res = {
            "results": {
                "noli": combined_noli,
                "elfili": combined_elfili
            },
            "precise_matches": combined_precise,
            "metadata": metadata if metadata else {"result_mode": "none", "reason": "matches_found"}
        }
        
        return final_res
    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
