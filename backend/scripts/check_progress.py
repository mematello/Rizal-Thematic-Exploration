
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
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load .env explicitly
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(env_path)

db_url = os.environ.get('DATABASE_URL')
if not db_url:
    print("DATABASE_URL not found")
    sys.exit(1)

try:
    engine = create_engine(db_url)
    with engine.connect() as conn:
        res = conn.execute(text("SELECT count(1) FROM sentences WHERE source_type = 'full'"))
        count = res.scalar()
        print(f"Full sentences count: {count}")
except Exception as e:
    print(f"Error: {e}")
