# Performance Optimization State

**Last Updated**: 2026-10-12  
**Branch**: `perf/speed-optimization`  
**Current Phase**: Phase 0 (Planning)

---

## Current Status
Phase 1 baseline complete; Phase 1b instrumentation in progress. Stage-level timing and EXPLAIN plans missing.

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
- ✅ **RAN BENCHMARK**: 40/40 queries completed successfully (10 runs each)
- ✅ **ANALYZED RESULTS**: BASELINE.md corrected with accurate data from JSON

---

## Key Findings from Phase 1

**🚨 CRITICAL ISSUES:**
- Single-word Tagalog queries: **10.8s average** (target: <1s)
- "edukasyon" anomaly: **38.9s cold, 32.1s warm** per query
- **2.4s floor** on every query, even with 0 DB work
- Out-of-domain rejection (0 DB queries) takes **3.4s** - slower than valid queries

**Key Observations:**
- DB query counts are identical (38-44) for fast and slow queries → N+1 hypothesis REFUTED
- Slow queries are `semantic_fallback` mode; fast queries are `lexical`
- English fallback queries (education, oppression) are FAST (2.5s), so fallback itself is not the cause
- Warm == cold (no effective caching)

**Phase 1b Requirements:**
1. Add stage-level timing instrumentation to engine.py
2. Fix explain_queries.py (vector syntax error)
3. Targeted re-run: kalayaan, edukasyon, bayan, tiktok, oppression (5 runs each)

---

## Next Actions - IN PROGRESS

**Adding stage-level timing** to capture:
- Query preprocessing, translator, model encoding
- Theme expansion (time + keyword count)
- Lexical/semantic retrieval (candidate counts)
- Re-ranking, theme classification, context expansion
- Serialization

**Fix EXPLAIN queries** with proper vector literal formatting

**Targeted re-run** with instrumentation to identify:
- Which stage causes the 2.4s floor
- Which stage explains kalayaan/edukasyon slowness

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
