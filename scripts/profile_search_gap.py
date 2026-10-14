"""
Profile engine.search() directly for bayan and oppression with accumulating timers.
Extracts per-stage ms:
- classify_themes (accumulated ms & call count)
- expand_context (accumulated ms & call count)
- DB query execution time (psycopg2 cursor)
- ORM hydration time (SQLAlchemy loading/instances)
- Three untimed encode sites:
    * concept_vector_encode (engine.py:631)
    * query_components_encode (engine.py:597)
    * sig_words_encode (engine.py:919)
    * rerank validation encode / tracked_encode (engine.py:907)
- search_total_ms, stage_sum, untimed_gap
"""
import os
import sys
import time
import cProfile
import pstats
from io import StringIO
from pathlib import Path

# Enforce deterministic hash seed & benchmarking
os.environ["PYTHONHASHSEED"] = "0"
os.environ["PERF_BENCHMARK"] = "1"
os.environ["DISABLE_WORD_VEC_CACHE"] = "0"

backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.models.database import SessionLocal
from app.core.engine import get_engine

def run_profiling():
    db = SessionLocal()
    engine = get_engine()

    # Warmup runs (cache on, models loaded)
    print("Executing warmup searches...")
    for q in ["kalayaan", "bayan", "oppression", "bayan", "oppression"]:
        _ = engine.search(db, q, source_type="summary", novel="both")

    for q in ["bayan", "oppression"]:
        print("\n" + "=" * 80)
        print(f"DETAILED STAGE PROFILE: '{q}' (Accumulating Timers + cProfile)")
        print("=" * 80)

        pr = cProfile.Profile()
        t0 = time.perf_counter()
        pr.enable()
        res = engine.search(db, q, source_type="summary", novel="both")
        pr.disable()
        t1 = time.perf_counter()

        direct_total_ms = (t1 - t0) * 1000
        meta = res.get("metadata", {})
        eng_timings = meta.get("engine_timings", {})

        # Extract accumulating stage timings from PerfTimer
        print("\n1. ACCUMULATING ENGINE STAGE TIMINGS (from PerfTimer):")
        print("-" * 65)
        print(f"{'Stage Name':<30} | {'Total Time (ms)':<15} | {'Call Count':<10}")
        print("-" * 65)

        known_stages = [
            ("preprocessing", "preprocessing_count"),
            ("base_model_encode", "base_model_encode_count"),
            ("query_components_encode", "query_components_encode_count"),
            ("theme_expansion", "theme_expansion_count"),
            ("concept_vector_encode", "concept_vector_encode_count"),
            ("dapt_model_encode", "dapt_model_encode_count"),
            ("lexical_retrieval", "lexical_retrieval_count"),
            ("semantic_fallback", "semantic_fallback_count"),
            ("sig_words_encode", "sig_words_encode_count"),
            ("rerank_loop", "rerank_loop_count"),
            ("context_expansion", "context_expansion_count"),
            ("theme_classification", "theme_classification_count"),
            ("serialization", "serialization_count"),
        ]

        for s_name, c_name in known_stages:
            if s_name in eng_timings:
                s_ms = eng_timings.get(s_name, 0.0)
                s_cnt = eng_timings.get(c_name, 1)
                print(f"{s_name:<30} | {s_ms:>13.2f} ms | {s_cnt:>10}")

        print("-" * 65)
        print(f"{'search_total_ms':<30} | {eng_timings.get('search_total_ms', 0):>13.2f} ms |")
        print(f"{'engine_stage_sum_ms':<30} | {eng_timings.get('engine_stage_sum_ms', 0):>13.2f} ms |")
        print(f"{'untimed_gap_ms':<30} | {eng_timings.get('untimed_gap_ms', 0):>13.2f} ms |")
        print(f"{'direct_wallclock_total_ms':<30} | {direct_total_ms:>13.2f} ms |")

        # Extract DB time, ORM hydration, and functions from cProfile
        s = StringIO()
        ps = pstats.Stats(pr, stream=s)

        # Parse stats
        db_execute_time = 0.0
        db_execute_calls = 0
        db_fetch_time = 0.0
        db_fetch_calls = 0
        orm_hydration_time = 0.0
        orm_hydration_calls = 0

        for func, (cc, nc, tt, ct, callers) in ps.stats.items():
            mod_file, line_no, fn_name = func
            func_str = str(func).lower()

            # psycopg2 cursor execute / fetch (including C extension methods)
            if "psycopg2" in func_str:
                if "execute" in fn_name.lower():
                    db_execute_time += ct
                    db_execute_calls += nc
                elif any(f in fn_name.lower() for f in ["fetchall", "fetchmany", "fetchone"]):
                    db_fetch_time += ct
                    db_fetch_calls += nc

            # SQLAlchemy loading / instances
            if "loading.py" in mod_file and fn_name == "instances":
                orm_hydration_time += ct
                orm_hydration_calls += nc
            elif "result.py" in mod_file and fn_name in ["_allrows", "_fetchall_impl"]:
                orm_hydration_time += tt

        print("\n2. DATABASE & ORM BREAKDOWN (from cProfile):")
        print("-" * 65)
        print(f"{'Component':<30} | {'Time (ms)':<15} | {'Calls':<10}")
        print("-" * 65)
        print(f"{'psycopg2 execute':<30} | {db_execute_time * 1000:>13.2f} ms | {db_execute_calls:>10}")
        print(f"{'psycopg2 fetch':<30} | {db_fetch_time * 1000:>13.2f} ms | {db_fetch_calls:>10}")
        print(f"{'Total DB wire time':<30} | {(db_execute_time + db_fetch_time) * 1000:>13.2f} ms | {db_execute_calls + db_fetch_calls:>10}")
        print(f"{'SQLAlchemy ORM hydration':<30} | {orm_hydration_time * 1000:>13.2f} ms | {orm_hydration_calls:>10}")
        print(f"{'Total DB + ORM overhead':<30} | {(db_execute_time + db_fetch_time + orm_hydration_time) * 1000:>13.2f} ms |")

        print("\n3. ENCODING SITES BREAKDOWN:")
        print("-" * 65)
        print(f"{'Encoding Site':<35} | {'Time (ms)':<15} | {'Calls':<10}")
        print("-" * 65)
        base_enc = eng_timings.get("base_model_encode", 0.0)
        base_cnt = eng_timings.get("base_model_encode_count", 0)
        print(f"{'Initial query (line 589)':<35} | {base_enc:>13.2f} ms | {base_cnt:>10}")

        comp_enc = eng_timings.get("query_components_encode", 0.0)
        comp_cnt = eng_timings.get("query_components_encode_count", 0)
        print(f"{'Query components (line 597)':<35} | {comp_enc:>13.2f} ms | {comp_cnt:>10}")

        concept_enc = eng_timings.get("concept_vector_encode", 0.0)
        concept_cnt = eng_timings.get("concept_vector_encode_count", 0)
        print(f"{'Concept vector (line 631)':<35} | {concept_enc:>13.2f} ms | {concept_cnt:>10}")

        sig_enc = eng_timings.get("sig_words_encode", 0.0)
        sig_cnt = eng_timings.get("sig_words_encode_count", 0)
        print(f"{'Significant words (line 919)':<35} | {sig_enc:>13.2f} ms | {sig_cnt:>10}")

        # In-rerank validation encodings
        rerank_enc_ms = eng_timings.get("encode_time_total_ms", 0.0)
        rerank_enc_cnt = eng_timings.get("encode_calls_total", 0)
        print(f"{'Rerank validation (line 907)':<35} | {rerank_enc_ms:>13.2f} ms | {rerank_enc_cnt:>10}")

    db.close()

if __name__ == "__main__":
    run_profiling()
