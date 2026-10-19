
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve()
while _repo_root.parent != _repo_root and not (_repo_root / "scripts" / "prod_guard.py").exists():
    _repo_root = _repo_root.parent
if str(_repo_root / "scripts") not in sys.path:
    sys.path.insert(0, str(_repo_root / "scripts"))
from prod_guard import verify_not_production_db
verify_not_production_db(caller_name=Path(__file__).name)

import os
import sys
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add backend root to path
sys.path.append(os.path.join(os.getcwd()))

from app.models.database import Sentence
from app.core.config import get_settings

def list_ids():
    settings = get_settings()
    engine = create_engine(settings.DATABASE_URL)
    Session = sessionmaker(bind=engine)
    db = Session()
    
    ids = [
        # Noli (25)
        1, 46, 47, 66, 154, 203, 248, 370, 472, 526, 
        99, 288, 447, 607, 798, 978, 1161, 1162, 3358, 
        171, 187, 228, 238, 267, 871,
        # Fili (15)
        1213, 1222, 1248, 1274, 1298, 1410, 1531, 1611, 
        1733, 4094, 1924, 1941, 1955, 1977, 4226
    ]
    
    print("| ID | Novel | Chapter | Text Snippet |")
    print("|----|-------|---------|--------------|")
    
    for sid in ids:
        s = db.query(Sentence).filter(Sentence.id == sid).first()
        if s:
            novel = "Noli" if s.book.lower() in ["noli", "noli me tangere"] else "Fili"
            text = s.sentence_text[:60].replace("\n", " ") + "..."
            print(f"| {sid} | {novel} | {s.chapter_number} | {text} |")
        else:
            print(f"| {sid} | NOT FOUND | - | - |")

if __name__ == "__main__":
    list_ids()
