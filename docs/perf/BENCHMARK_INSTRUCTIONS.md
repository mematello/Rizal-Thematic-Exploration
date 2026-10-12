# Phase 1 Baseline Measurement Instructions

**Status**: Phase 1 instrumentation complete. Benchmark NOT yet run.

---

## Prerequisites

1. **Docker services running**:
   ```powershell
   docker-compose up -d
   ```

2. **Dependencies installed**:
   ```powershell
   pip install psycopg2-binary requests psutil
   ```

---

## Running the Benchmark

### Terminal 1: Start API Server

**IMPORTANT**: The `PERF_BENCHMARK` environment variable must be set in the **server** process (not the client script). This enables timing instrumentation and DB query counting.

```powershell
cd backend
$env:PERF_BENCHMARK="1"
poetry run uvicorn app.main:app
```

**Without --reload**: For stable benchmarking, run without hot-reload to avoid mid-run restarts.

### Terminal 2: Run Benchmark Script

```powershell
# From project root
poetry run python scripts/benchmark.py
```

**Expected duration**: 10-15 minutes (40 queries × 10 runs)

---

## EXPLAIN ANALYZE Queries

Run database query plan analysis:

```powershell
poetry run python scripts/explain_queries.py
```

**Output**: `docs/perf/query_plans.txt`

---

## Outputs

After benchmark completes:
- `docs/perf/golden_snapshot.json` - Full ranked results for correctness verification
- `docs/perf/baseline_timings.json` - Raw timing data (cold/warm, per-query)
- Console summary - P50/P95 latencies by category

---

## Translator Caching

**Current Implementation**: The benchmark script attempts client-side pre-warming but does NOT freeze server-side translator calls.

**Limitation**: The engine's dynamic translator (in `backend/app/core/engine.py` lines ~496-502) may still make network calls for unmapped English terms during search.

**For fully deterministic runs**: Set `DEBUG_TRANSLATOR_CACHE=1` environment variable in the server terminal to enable server-side caching (if implemented), or temporarily disable dynamic translation during benchmarking.

---

## Next Steps After Benchmark

1. Send the generated JSON files and console output
2. I'll analyze results and fill `BASELINE.md`
3. Review findings before proceeding to Phase 2
