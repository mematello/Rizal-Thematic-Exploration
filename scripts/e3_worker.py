import os
import sys
import json
import inspect
from pathlib import Path

# Load environment
base_dir = Path(".").resolve()
sys.path.insert(0, str(base_dir / "backend"))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['DATABASE_URL'] = 'postgresql://rizal:rizalpass123@127.0.0.1:5433/rizal_db'
os.environ['REDIS_URL'] = 'redis://localhost:6379'
os.environ['PERF_BENCHMARK'] = '1'

mode = sys.argv[1] # "pre_e3" or "head"
seed = sys.argv[2]
out_path = sys.argv[3]

import app.core.engine as engine_module
from app.core.engine import get_engine
from app.models.database import SessionLocal

# If pre_e3, monkeypatch sorted in engine_module to use set iteration order at line 1290
builtins_sorted = sorted
if mode == "pre_e3":
    def custom_sorted(obj, *args, **kwargs):
        # If called from line 1290 inside search() on sent_words
        frame = inspect.currentframe().f_back
        if frame and frame.f_code.co_name == "search" and frame.f_lineno in (1289, 1290, 1291):
            return list(obj)
        return builtins_sorted(obj, *args, **kwargs)
    engine_module.sorted = custom_sorted

engine = get_engine()
db = SessionLocal()

queries = ["kalayaan", "edukasyon", "pag-ibig", "kamatayan"]
results_data = {}

for q in queries:
    passing_ids = []
    failing_ids = []
    final_tokens = None

    def tracer(frame, event, arg):
        global final_tokens
        if event == 'line':
            co_name = frame.f_code.co_name
            if co_name == 'search':
                line = frame.f_lineno
                if line == 1376:
                    sent = frame.f_locals.get('sent')
                    if sent:
                        passing_ids.append(sent.id)
                elif line == 1371:
                    sent = frame.f_locals.get('sent')
                    if sent:
                        failing_ids.append(sent.id)
                elif line == 1498 and final_tokens is None:
                    final_tokens = set(frame.f_locals.get('valid_tokens', []))
        return tracer

    sys.settrace(tracer)
    res = engine.search(db, q, top_k=10, source_type='summary', novel='both')
    sys.settrace(None)

    timings = res['metadata'].get('engine_timings', {})
    results_data[q] = {
        "final_valid_tokens": sorted(list(final_tokens)) if final_tokens else [],
        "passing_ids": passing_ids,
        "failing_ids": failing_ids,
        "validation_entered": timings.get("validation_entered", 0),
        "validation_skipped": timings.get("validation_skipped", 0),
        "top10_noli": [x['id'] for x in res['results']['noli']],
        "top10_elfili": [x['id'] for x in res['results']['elfili']],
    }

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results_data, f, indent=2)

print(f"Completed {mode} under seed {seed}")
