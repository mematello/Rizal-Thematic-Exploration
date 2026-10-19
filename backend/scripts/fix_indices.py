
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve()
while _repo_root.parent != _repo_root and not (_repo_root / "scripts" / "prod_guard.py").exists():
    _repo_root = _repo_root.parent
if str(_repo_root / "scripts") not in sys.path:
    sys.path.insert(0, str(_repo_root / "scripts"))
from prod_guard import verify_not_production_db
verify_not_production_db(caller_name=Path(__file__).name)

import sys
import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Append backend root to path so we can import app modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.models.database import Sentence
from app.core.config import get_settings

def fix_indices():
    settings = get_settings()
    engine = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    csv_dir = os.path.join(os.path.dirname(__file__), '../../csvFiles')

    print("Fixing Noli Summary indices...")
    noli_path = os.path.join(csv_dir, 'noli_chapters.csv')
    df_noli = pd.read_csv(noli_path)
    for idx, row in df_noli.iterrows():
        s = session.query(Sentence).filter_by(book='noli', source_type='summary', sentence_index=idx).first()
        if s:
            s.sentence_index = int(row['sentence_number'])
    
    print("Fixing Fili Summary indices...")
    fili_path = os.path.join(csv_dir, 'elfili_chapters.csv')
    df_fili = pd.read_csv(fili_path)
    for idx, row in df_fili.iterrows():
        s = session.query(Sentence).filter_by(book='elfili', source_type='summary', sentence_index=idx).first()
        if s:
            s.sentence_index = int(row['sentence_number'])

    session.commit()
    session.close()
    print("Done fixing indices!")

if __name__ == '__main__':
    fix_indices()
