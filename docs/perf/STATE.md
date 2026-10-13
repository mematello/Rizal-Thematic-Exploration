# Performance Optimization State

**Last Updated**: 2026-10-13  
**Branch**: `perf/speed-optimization`  
**Current Phase**: Phase 2 (Step 1: Process-Level Word Embedding Cache)

---

## Current Status
Phase 1 complete; Phase 2 Step 1 implemented. Process-level word embedding cache now persists across search() calls, reducing encode calls from 622→0 warm, 166→0 cold for edukasyon. Rankings verified identical via compare_golden across all 40 queries. Remaining optimization phases: HNSW index, query expansion pruning.

---

## Completed Tasks
- ✅ Created branch `perf/speed-optimization`
- ✅ Analyzed codebase and mapped request flow
- ✅ Created memory system (CLAUDE.md, PLAN.md, STATE.md, DECISIONS.md, BASELINE.md, HANDOFF.md)
- ✅ Created golden query set (40 queries across 5 categories)
- ✅ Built benchmark.py with 10-run cold/warm measurement
- ✅ Built compare_golden.py for result verification
- ✅ Phase 1 profiling: identified 622 encodes/edukasyon, 166 encodes/kalayaan in rerank validation
- ✅ Phase 1 diagnostics: oppression fast (1 lexical hit → early exit), edukasyon slow (11 theme keywords → many encodes)
- ✅ Phase 1c: Added PerfTimer; fixed EXPLAIN; created targeted benchmark; added .gitattributes
- ✅ Phase 2 Step 1: Added process-level `_word_vec_cache` dict with threading.Lock in engine.py
- ✅ Verified rankings identical before/after via compare_golden.py (0 changes across all 40 queries)
- ✅ Updated DECISIONS.md with Phase 2 entry
- ✅ Re-run targeted_benchmark.py: reported process-cache hits/misses, encode_calls_total, encode_time_total_ms

---

## Phase 2 Step 1 — Process-Level Word Embedding Cache
- `_word_vec_cache: Dict[str, np.ndarray]` — persists across search() calls within a RizalEngine instance
- Key = exact string passed to `self.base_model.encode(w, show_progress_bar=False)`
- Value = exact numpy array returned (bit-identical vectors, rankings unchanged)
- In validation loop: check cache first; if miss, encode and store with `_word_vec_cache_lock` thread safety
- Per-request `word_sim_cache` semantics unchanged (query-dependent max similarity)
- PERF_BENCHMARK reports: `process_cache_hits`, `process_cache_misses`, `process_cache_size`
- Expected: edukasyon rerank ~31s → ~6-8s warm, ~8s cold → ~1.5-2s warm
- Do NOT precompute at startup; do NOT add HNSW; do NOT skip validation logic
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
- **Need to confirm:** All 5 queries must report timing per stage (bayan 2.5s, edukasyon 32.1s, etc.)

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
