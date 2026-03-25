from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.main import app
from app.api.v1.content import get_db, get_engine
from app.core.engine import RizalEngine
from app.models.database import SessionLocal, Sentence

def override_get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

engine_instance = None
def override_get_engine():
    global engine_instance
    if engine_instance is None:
        engine_instance = RizalEngine()
    return engine_instance

app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_engine] = override_get_engine

client = TestClient(app)
db = SessionLocal()

sentence_text = "Dahil sa paggigitgitan, natapakan ng tinyente ang laylayan ng damit ni Donya Victorina dahilan kung bakit nagalit ito."
summary_sentence = db.query(Sentence).filter(Sentence.sentence_text.like(f"%{sentence_text[:50]}%")).first()

if summary_sentence:
    print(f"Testing with sentence ID {summary_sentence.id}: {summary_sentence.sentence_text}")
    response = client.get(f"/api/v1/sentences/{summary_sentence.id}/sanggunian")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Matched Full Text: {data.get('reference_text')}")
        print(f"Char Score: {data.get('char_score')}")
        print(f"Semantic Score: {data.get('semantic_score')}")
        print(f"Lexical Score: {data.get('lexical_score')}")
    else:
        print(f"Failed: {response.status_code} {response.text}")
else:
    print("No summary sentence found.")

db.close()
