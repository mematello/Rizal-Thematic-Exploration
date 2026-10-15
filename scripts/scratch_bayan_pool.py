import os
import sys
import inspect
import textwrap
from dotenv import load_dotenv

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
load_dotenv('backend/.env')
sys.path.insert(0, 'backend')

from app.models.database import SessionLocal, Sentence
from app.core.engine import get_engine, RizalEngine

db = SessionLocal()
engine = get_engine()

# Run search_full for bayan
src = textwrap.dedent(inspect.getsource(RizalEngine.search))
target_lex = "            return unique[:top_k]"
hook_lex = """            global_full_pools[label] = list(unique)
            return unique[:top_k]"""
src_mod = src.replace(target_lex, hook_lex)
target_sem = "            results[book_key] = (lex_reserved + sem_only)[:top_k]"
hook_sem = """            global_full_pools[book_key] = list(lex_reserved + sem_only)
            results[book_key] = (lex_reserved + sem_only)[:top_k]"""
src_mod = src_mod.replace(target_sem, hook_sem)

globals_dict = dict(RizalEngine.search.__globals__)
globals_dict['global_full_pools'] = {}
local_env = {}

exec(src_mod, globals_dict, local_env)
engine.search_full = local_env['search'].__get__(engine, RizalEngine)
engine.global_full_pools = globals_dict['global_full_pools']

res = engine.search_full(db, 'bayan', source_type='summary', novel='both')
elfili_b1_pool = engine.global_full_pools['elfili']

print("B1 Run - Full Evaluated Pool for 'bayan' in El Fili (22 candidates):")
for rank, itm in enumerate(elfili_b1_pool, 1):
    sid = itm['id']
    sort_key = itm['scores']['final'] + itm['scores'].get('precision', 0) * 0.5
    marker = " <--- 38978" if sid == 38978 else (" <--- 39299" if sid == 39299 else "")
    print(f"  Rank {rank:2d}: ID {sid} | sort_key={sort_key} | final={itm['scores']['final']} sem={itm['scores']['semantic']} lex={itm['scores']['lexical']}{marker}")

# Now let's check Golden order (without ORDER BY Sentence.id in lexical query)
# If lexical query has NO ORDER BY Sentence.id:
from sqlalchemy import select
stmt = select(Sentence).filter(Sentence.book == 'elfili').filter(Sentence.source_type == 'summary').filter(Sentence.sentence_text.ilike('%bayan%'))
raw_db_unconstrained = db.scalars(stmt.limit(200)).all()
print("\nPostgreSQL raw return order without ORDER BY Sentence.id:")
print([s.id for s in raw_db_unconstrained])
