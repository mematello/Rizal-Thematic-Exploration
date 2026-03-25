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

# Avoid reloading model multiple times
engine_instance = None
def override_get_engine():
    global engine_instance
    if engine_instance is None:
        engine_instance = RizalEngine()
    return engine_instance

app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_engine] = override_get_engine

client = TestClient(app)

print("Fetching a summary sentence from DB...")
db = SessionLocal()
# Let's get a summary sentence from Noli
summary_sentence = db.query(Sentence).filter(
    Sentence.book == "noli",
    Sentence.source_type == "summary",
    Sentence.chapter_number == 1
).first()

if summary_sentence:
    print(f"Testing with sentence ID {summary_sentence.id}: {summary_sentence.sentence_text}")
    print("Calling /api/v1/sentences/{id}/sanggunian...")
    response = client.get(f"/api/v1/sentences/{summary_sentence.id}/sanggunian")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Response: {data}")
        print(f"Char Score: {data.get('char_score')}")
    else:
        print(f"Failed: {response.status_code} {response.text}")
else:
    print("No summary sentence found.")

db.close()
