
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
from pathlib import Path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))
from app.models.database import SessionLocal
from app.core.engine import get_engine
import cProfile, pstats

db = SessionLocal()
engine = get_engine()
pr = cProfile.Profile()
pr.enable()
res = engine.search(db, 'bayan', source_type='summary', novel='both')
pr.disable()
ps = pstats.Stats(pr)
print("Top 25 functions by cumulative time related to DB / SQLAlchemy / psycopg:")
for k, v in sorted(ps.stats.items(), key=lambda x: x[1][3], reverse=True):
    path, line, func = k
    if any(t in path.lower() or t in func.lower() for t in ['sql', 'psycopg', 'cursor', 'engine.py', 'loading']):
        print(f"  {Path(path).name}:{line:<4} {func:<30} {v[3]*1000:>8.2f} ms ({v[0]} calls)")
db.close()
