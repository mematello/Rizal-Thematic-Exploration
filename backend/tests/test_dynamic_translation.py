import os
os.environ["DEBUG_SEARCH"] = "1"
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.engine import RizalEngine
from app.core.config import get_settings

settings = get_settings()
db_engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

rizal = RizalEngine()
db = SessionLocal()
rizal._ensure_themes_loaded(db)

def test_query(query):
    print(f"\n{'='*50}\nTESTING QUERY: {query}\n{'='*50}")
    res = rizal.search(db, query, source_type="summary", novel="both")
    print(f"RESULTS -> Noli: {len(res['results']['noli'])}, Fili: {len(res['results']['elfili'])}")
    print(f"METADATA -> {res['metadata']}")

test_query("betrayal")
test_query("how did Ibarra experience betrayal")
test_query("tiktok is cool")
