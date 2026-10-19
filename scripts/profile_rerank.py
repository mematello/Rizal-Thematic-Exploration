#!/usr/bin/env python3
"""
Profile engine.search() directly with cProfile for rerank_loop diagnostics.
No HTTP overhead - direct method call.
"""

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
import cProfile
import pstats
import io
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

os.environ["PERF_BENCHMARK"] = "1"

from app.models.database import SessionLocal
from app.core.engine import get_engine

def profile_query(query_name: str, query_text: str):
    """Profile a single query with cProfile"""
    db = SessionLocal()
    engine = get_engine()

    # Warm up models (first call loads them)
    if query_name == "edukasyon":
        print("Warming up models...")
        engine.search(db, "test", source_type="summary", novel="both")

    print(f"\n{'='*80}")
    print(f"Profiling: {query_name} = '{query_text}'")
    print('='*80)

    profiler = cProfile.Profile()
    profiler.enable()

    result = engine.search(db, query_text, source_type="summary", novel="both")

    profiler.disable()

    # Print timing metadata
    metadata = result.get("metadata", {})
    timings = metadata.get("engine_timings", {})
    print(f"\nEngine timings: {timings}")

    # Generate stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')

    print(f"\nTop 30 functions by cumulative time:")
    print("-" * 80)
    ps.print_stats(30)
    print(s.getvalue())

    # Print call counts for key functions
    print(f"\nKey function call counts:")
    print("-" * 80)
    stats_dict = ps.stats

    encode_calls = 0
    encode_time = 0
    for key, value in stats_dict.items():
        func_name = key[2]
        if 'encode' in func_name.lower() and 'sentence' in str(key[0]).lower():
            cc, nc, tt, ct, callers = value
            encode_calls += cc
            encode_time += ct
            print(f"  {func_name}: {cc} calls, {ct:.3f}s cumulative")

    print(f"\nTotal base_model.encode calls: {encode_calls}")
    print(f"Total time in encode: {encode_time:.3f}s")

    db.close()
    return result

if __name__ == "__main__":
    queries = [
        ("edukasyon", "edukasyon"),
        ("kalayaan", "kalayaan"),
        ("oppression", "oppression"),
        ("bayan", "bayan"),
    ]

    for name, text in queries:
        profile_query(name, text)
        print("\n" + "="*80 + "\n")
