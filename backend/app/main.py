import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.models.database import engine, Base

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup: Create tables (for MVP - usually do migrations)
    # Ensure pgvector extension exists
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        import logging
        logging.warning(f"lifespan create_all skipped: {e}")
    
    import os
    import logging
    from pathlib import Path
    
    frontend_ts_path = Path(__file__).parent.parent.parent / "frontend" / "lib" / "characterData.ts"
    backend_json_path = Path(__file__).parent / "data" / "character_aliases.json"
    
    if frontend_ts_path.exists() and backend_json_path.exists():
        if os.path.getmtime(frontend_ts_path) > os.path.getmtime(backend_json_path):
            logging.warning("⚠️ character_aliases.json is older than characterData.ts! Please run scripts/character_index.py.")
    elif frontend_ts_path.exists() and not backend_json_path.exists():
        logging.warning("⚠️ character_aliases.json not found! Please run scripts/character_index.py.")
        
    # Non-blocking warm-up of themes, chapters, and sanggunian cache in background thread
    import threading
    from app.api.v1.content import warm_themes_cache, warm_chapters_cache, warm_sanggunian_cache
    def _warm_all():
        warm_chapters_cache()
        warm_themes_cache()
        warm_sanggunian_cache()
    threading.Thread(target=_warm_all, name="warm-caches", daemon=True).start()

    yield
    # Shutdown logic

settings = get_settings()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# CORS
import os
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://[::1]:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

cors_env = os.getenv("CORS_ORIGINS")
if cors_env:
    for o in cors_env.split(","):
        stripped = o.strip()
        if stripped and stripped not in origins:
            origins.append(stripped)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"^https:\/\/.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
if os.getenv("PERF_BENCHMARK") == "1":
    from starlette.requests import Request
    import time

    @app.middleware("http")
    async def perf_timing_middleware(request: Request, call_next):
        t_start = time.perf_counter()
        request.state.t_middleware_start = t_start
        response = await call_next(request)
        t_end = time.perf_counter()

        t_route_start = getattr(request.state, "t_route_start", t_start)
        t_route_end = getattr(request.state, "t_route_end", t_start)
        db_create_ms = getattr(request.state, "db_create_ms", 0.0)

        total_ms = (t_end - t_start) * 1000
        routing_di_ms = (t_route_start - t_start) * 1000
        route_handler_ms = (t_route_end - t_route_start) * 1000
        serialization_ms = (t_end - t_route_end) * 1000

        response.headers["X-FastAPI-Middleware-Total-Ms"] = f"{total_ms:.2f}"
        response.headers["X-FastAPI-Routing-DI-Ms"] = f"{routing_di_ms:.2f}"
        response.headers["X-FastAPI-DB-Create-Ms"] = f"{db_create_ms:.2f}"
        response.headers["X-FastAPI-Route-Handler-Ms"] = f"{route_handler_ms:.2f}"
        response.headers["X-FastAPI-Serialization-Ms"] = f"{serialization_ms:.2f}"
        return response

@app.get("/health")
def health_check():
    return {"status": "ok", "version": settings.VERSION}

from app.api.v1 import search, content, alignment
app.include_router(search.router, prefix="/api/v1")
app.include_router(content.router, prefix="/api/v1")
app.include_router(alignment.router, prefix="/api/v1")
