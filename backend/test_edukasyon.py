import os
os.environ["DEBUG_SEARCH"] = "1"
from app.models.database import SessionLocal
from app.core.engine import RizalEngine

db = SessionLocal()
engine = RizalEngine()
res = engine.search(db, "edukasyon", 10, "summary")

print("NOLI RESULTS COUNT:", len(res["results"]["noli"]))
print("FILI RESULTS COUNT:", len(res["results"]["elfili"]))
