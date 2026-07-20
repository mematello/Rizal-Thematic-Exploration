
import sys
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Append backend root to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from app.models.database import Sentence, Base
from app.api.v1.content import get_sentence_sanggunian

def reproduce():
    # Mocking DB and Engine for a quick check is complex, 
    # but we can analyze the code logic.
    # The user says "sequential ratio computed against a bloated total".
    # In content.py:
    # total_buod = max(1, len(buod_sentences))
    # buod_idx = next((i for i, bs in enumerate(buod_sentences) if bs.id == sentence.id), 0)
    # buod_ratio = buod_idx / total_buod
    
    # If buod_sentences has duplicates, total_buod is too large.
    # Example: 10 sentences. If duplicated, total_buod = 20.
    # Sentence at real index 5 (0-based) would have buod_idx = 5.
    # buod_ratio = 5 / 20 = 0.25 (should be 5/10 = 0.5).
    # This matches the user's report of "ratio scores are no longer distributed correctly".
    
    print("Reproduction Analysis:")
    print("1. total_buod is calculated as len(buod_sentences) without deduplication.")
    print("2. If the frontend or some process causes duplicate rows in the DB or if the query returns duplicates, the ratio breaks.")
    print("3. Clicking Sanggunian button in frontend somehow triggers duplication in the view.")

if __name__ == "__main__":
    reproduce()
