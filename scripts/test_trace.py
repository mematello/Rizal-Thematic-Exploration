import os
import sys
import json
from dotenv import load_dotenv

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
load_dotenv('backend/.env')
sys.path.insert(0, 'backend')

from app.models.database import SessionLocal
from app.core.engine import get_engine

db = SessionLocal()
engine = get_engine()

captured_pools = {}

def trace_calls(frame, event, arg):
    if event == 'line':
        code = frame.f_code
        if code.co_name == 'search' and 'engine.py' in code.co_filename:
            # Capture results right before finalize is called or inside finalize
            if frame.f_lineno == 1474:
                res = frame.f_locals.get('results')
                if res:
                    captured_pools['noli'] = [dict(x) for x in res.get('noli', [])]
                    captured_pools['elfili'] = [dict(x) for x in res.get('elfili', [])]
    return trace_calls

sys.settrace(trace_calls)
engine.search(db, 'bayan', source_type='summary', novel='both')
sys.settrace(None)

print(f"Captured Noli pool size: {len(captured_pools.get('noli', []))}")
print(f"Captured El Fili pool size: {len(captured_pools.get('elfili', []))}")
