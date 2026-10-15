from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from app.models.database import SessionLocal
from app.core.engine import get_engine, RizalEngine
from app.core.query_counter import reset_query_count, get_query_count
import os
import time

router = APIRouter()
PERF_BENCHMARK = os.getenv("PERF_BENCHMARK") == "1"

def get_db(request: Request = None):
    t_db_start = time.perf_counter()
    db = SessionLocal()
    t_db_end = time.perf_counter()
    if request and PERF_BENCHMARK:
        request.state.db_create_ms = (t_db_end - t_db_start) * 1000
    try:
        yield db
    finally:
        db.close()

@router.get("/search")
async def search_novels(
    request: Request,
    q: str = Query(..., min_length=1, description="Search query"),
    source_type: str = Query("summary", description="Search in summary or full version"),
    novel: str = Query("both", description="Target novel (noli, elfili, or both)"),
    bypass_cache: bool = Query(False, description="Bypass process-level word embedding cache"),
    db: Session = Depends(get_db),
    engine: RizalEngine = Depends(get_engine)
):
    """
    Search across Noli Me Tangere and El Filibusterismo.
    Returns ranked results using hybrid semantic + lexical search.
    """
    t_route_start = time.perf_counter()
    if PERF_BENCHMARK:
        request.state.t_route_start = t_route_start
    try:
        # Reset query counter for this request
        if PERF_BENCHMARK:
            reset_query_count()

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

        t_engine_start = time.perf_counter()
        for query in queries:
            search_res = engine.search(db, query, source_type=db_source_type, novel=novel, bypass_word_vec_cache=bypass_cache)

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
        t_engine_end = time.perf_counter()

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

        # Add DB query count and FastAPI layer timing if in benchmark mode
        t_route_end = time.perf_counter()
        if PERF_BENCHMARK:
            request.state.t_route_end = t_route_end
            if final_res.get("metadata"):
                final_res["metadata"]["db_query_count"] = get_query_count()

                t_middleware_start = getattr(request.state, "t_middleware_start", t_route_start)
                db_create_ms = getattr(request.state, "db_create_ms", 0.0)

                # Test in-route JSON serialization
                ser_t0 = time.perf_counter()
                import json
                _ = json.dumps(final_res, default=str)
                ser_t1 = time.perf_counter()

                final_res["metadata"]["fastapi_layer"] = {
                    "db_session_create_ms": round(db_create_ms, 3),
                    "di_and_routing_ms": round((t_route_start - t_middleware_start) * 1000, 3),
                    "route_pre_search_ms": round((t_engine_start - t_route_start) * 1000, 3),
                    "engine_search_ms": round((t_engine_end - t_engine_start) * 1000, 3),
                    "route_post_search_ms": round((t_route_end - t_engine_end) * 1000, 3),
                    "route_handler_total_ms": round((t_route_end - t_route_start) * 1000, 3),
                    "json_dumps_test_ms": round((ser_t1 - ser_t0) * 1000, 3),
                }

        return final_res
    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/verify_cache")
def verify_word_vec_cache(engine: RizalEngine = Depends(get_engine)):
    """
    Step 2 verification: check that every cached vector equals base_model.encode(w) twice.
    Also verifies all cached vectors are read-only.
    """
    from app.core.engine import _word_vec_cache
    import numpy as np

    total_words = len(_word_vec_cache)
    read_only_count = 0
    mismatches_pass1 = 0
    mismatches_pass2 = 0
    mismatched_words = []

    # Check read-only flags and run Pass 1
    for w, cached_vec in list(_word_vec_cache.items()):
        if not cached_vec.flags.writeable:
            read_only_count += 1
        fresh_vec = engine.base_model.encode(w, show_progress_bar=False)
        if not np.array_equal(cached_vec, fresh_vec):
            mismatches_pass1 += 1
            mismatched_words.append(w)

    # Run Pass 2
    for w, cached_vec in list(_word_vec_cache.items()):
        fresh_vec = engine.base_model.encode(w, show_progress_bar=False)
        if not np.array_equal(cached_vec, fresh_vec):
            mismatches_pass2 += 1

    return {
        "total_cached_words": total_words,
        "read_only_count": read_only_count,
        "mismatches_pass1": mismatches_pass1,
        "mismatches_pass2": mismatches_pass2,
        "mismatched_words": mismatched_words
    }

