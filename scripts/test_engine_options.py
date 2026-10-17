import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from app.models.database import _db_url
import time

e = create_engine(
    _db_url,
    connect_args={'connect_timeout': 5, 'options': '-c statement_timeout=5000'},
    poolclass=NullPool
)
t0 = time.perf_counter()
with e.connect() as conn:
    res = conn.execute(text('SELECT 1')).scalar()
    print('SELECT 1 res:', res)
    res2 = conn.execute(text('SELECT 1 FROM sentences LIMIT 1')).scalar()
    print('SELECT 1 FROM sentences res:', res2)
print(f'Elapsed: {(time.perf_counter() - t0)*1000:.2f} ms')
