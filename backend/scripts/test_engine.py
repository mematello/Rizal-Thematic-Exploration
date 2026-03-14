import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.engine import RizalEngine
from app.models.database import SessionLocal
import json

def test():
    engine = RizalEngine()
    with SessionLocal() as db:
        res = engine.search(db, "edukasyon", top_k=2)
        print(json.dumps(res, indent=2))

if __name__ == "__main__":
    test()
