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

**IMPORTANT**: The `PERF_BENCHMARK` environment variable must be set in the **server** process (not the client script). This enables timing instrumentation and DB query counting. Additionally, set `PYTHONHASHSEED="0"` to ensure deterministic set iteration ordering (e.g. in theme expansion text generation) across server runs.

```powershell
cd backend
$env:PYTHONHASHSEED="0"
$env:PERF_BENCHMARK="1"
poetry run uvicorn app.main:app
```

**Without --reload**: For stable benchmarking, run without hot-reload to avoid mid-run restarts.

### Terminal 2: Run Benchmark Script

```powershell
# From project root
poetry run python scripts/benchmark.py
```

**IMPORTANT NOTE ON HOST & CONNECTION REUSE**:
Always target `http://127.0.0.1:<port>` rather than `http://localhost:<port>`. On Windows, resolving `localhost` causes an IPv6 `::1` connection attempt that blocks for ~2.0 seconds before falling back to IPv4 `127.0.0.1` when uvicorn is bound to IPv4. Additionally, `scripts/benchmark.py` uses a `requests.Session()` to reuse HTTP connections across queries. All earlier benchmark latency tables collected with `localhost` and non-reused connections are invalid for absolute HTTP end-to-end latency (which carried an artificial ~2.0s floor), though internal server-side stage timings remain valid.

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
