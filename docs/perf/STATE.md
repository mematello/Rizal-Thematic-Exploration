# Performance Optimization State

**Last Updated**: 2026-10-12  
**Branch**: `perf/speed-optimization`  
**Current Phase**: Phase 0 (Planning)

---

## Current Status
Phase 1 instrumentation complete. Benchmark NOT yet run; awaiting user to run it locally.

---

## Completed Tasks
- ✅ Created branch `perf/speed-optimization`
- ✅ Analyzed codebase and mapped request flow
- ✅ Created memory system (CLAUDE.md, PLAN.md, STATE.md, DECISIONS.md, BASELINE.md, HANDOFF.md)
- ✅ Created golden query set (40 queries across 5 categories)
- ✅ Built benchmark.py with 10-run cold/warm measurement
- ✅ Built compare_golden.py for result verification
- ✅ Added DB query counter middleware
- ✅ Fixed explain_queries.py with proper vector syntax and UTF-8 output
- ✅ Updated PLAN.md with refined hypotheses
- ✅ Created BENCHMARK_INSTRUCTIONS.md with exact commands

---

## Next Actions - USER TO PERFORM

**Terminal 1 - Start API server with instrumentation:**
```powershell
cd backend
$env:PERF_BENCHMARK="1"
poetry run uvicorn app.main:app
```
⚠️ IMPORTANT: Set `PERF_BENCHMARK=1` in the SERVER terminal (process that runs the engine)

**Terminal 2 - Run benchmark:**
```powershell
poetry run python scripts/benchmark.py
```
Expected duration: 10-15 minutes

**Also run EXPLAIN ANALYZE:**
```powershell
poetry run python scripts/explain_queries.py
```

---

## After Benchmark Completes

1. Send generated files: `docs/perf/golden_snapshot.json`, `baseline_timings.json`, console output
2. I'll analyze results and populate BASELINE.md with findings
3. Review BASELINE.md before proceeding to Phase 2 optimizations

---

## Open Questions
- Acceptable P95 latency target for search endpoint?
- Redis eviction policy preference (LRU, LFU)?
- Are there specific query patterns used most by users?
- Budget for infrastructure changes (RAM, CPU cores)?

---

## Blockers
Awaiting user to run benchmark locally (background task timed out).

---

## Open Questions
- Acceptable P95 latency target for search endpoint?
- Redis eviction policy preference (LRU, LFU)?
- Are there specific query patterns used most by users?
- Budget for infrastructure changes (RAM, CPU cores)?

---

## Blockers
None currently.

---

## Notes
- All changes must preserve result rankings (verify with golden query set)
- Document performance impact of each optimization in DECISIONS.md
- Commit memory files with code changes
