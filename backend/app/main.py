from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.models.database import engine, Base

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup: Create tables (for MVP - usually do migrations)
    # Ensure pgvector extension exists
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    
    Base.metadata.create_all(bind=engine)
    yield
    # Shutdown logic

settings = get_settings()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# CORS
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://172.22.160.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "version": settings.VERSION}

from app.api.v1 import search, content
app.include_router(search.router, prefix="/api/v1")
app.include_router(content.router, prefix="/api/v1")
