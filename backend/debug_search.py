import os
os.environ["DEBUG_SEARCH"] = "1"
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.engine import RizalEngine
from app.core.config import get_settings

settings = get_settings()
db_engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

print("Initializing engine...")
rizal = RizalEngine()
db = SessionLocal()

print("--- SEARCHING 'love' ---")
res = rizal.search(db, "love", source_type="summary", novel="both")
print(res['metadata'])
print("Results:", len(res['results'].get('noli', [])) + len(res['results'].get('elfili', [])))

print("--- SEARCHING 'pagmamahal' ---")
res2 = rizal.search(db, "pagmamahal", source_type="summary", novel="both")
print(res2['metadata'])
print("Results:", len(res2['results'].get('noli', [])) + len(res2['results'].get('elfili', [])))
