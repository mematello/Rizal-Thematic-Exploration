from app.models.database import SessionLocal
from app.core.engine import get_engine

db = SessionLocal()
engine = get_engine()
res = engine.search(db, "edukasyon", top_k=5, source_type="full")
print("FINAL RESULT:", res)
