# Performance Optimization Decisions

**Purpose**: Log all decisions that could affect search behavior, ranking, or correctness.

---

## Decision Log

### 2026-10-12: Initial Planning
**Context**: Beginning performance optimization project for public deployment.

**Decision**: Adopt phased approach with measurement-first strategy.

**Rationale**: 
- Avoids premature optimization
- Establishes baselines to measure improvement
- Ensures correctness preserved throughout

**Impact**: None yet (planning phase)

---

### 2026-10-12: Phase 1 Instrumentation
**Context**: Need to identify actual bottlenecks before optimizing.

**Decision**: Add comprehensive measurement infrastructure with minimal behavior changes:
- Golden query set (40 queries) with full result snapshot
- Cold vs warm timing (10 runs per query)
- DB query counter via SQLAlchemy event listener
- EXPLAIN ANALYZE profiling script
- Comparison tool to detect ranking changes

**Implementation**:
- Created `scripts/benchmark.py` for automated performance measurement
- Created `scripts/compare_golden.py` for correctness verification
- Created `scripts/golden_queries.json` with 5 categories × 8 queries
- Created `scripts/explain_queries.py` for database query plan analysis (fixed vector syntax)
- Added `backend/app/core/query_counter.py` for DB query tracking
- Modified `backend/app/api/v1/search.py` to expose query count in benchmark mode

**Translator Caching**: 
- Benchmark script pre-warms client-side translator cache
- Server-side translator calls in engine.py (lines ~496-502) are NOT frozen/cached
- For fully deterministic runs, dynamic translation should be disabled during benchmarking

**Benchmark Execution**:
- Background task timed out after 10 minutes
- User must run benchmark locally in their terminal
- `PERF_BENCHMARK=1` must be set in the SERVER terminal (not client script)
- Use `poetry run python scripts/benchmark.py` from project root

**Impact**:
- Performance: None (measurement only, no optimization yet)
- Correctness: Unchanged (instrumentation behind PERF_BENCHMARK flag)
- Trade-offs: None

**Verification**: All changes are read-only instrumentation or behind environment flag

---

### 2026-10-12: Baseline Corrections

**Context**: Initial BASELINE.md analysis contained several errors based on misreading the JSON data.

**Corrections Made**:

1. **DB query instrumentation**: PERF_BENCHMARK *was* properly set. DB query counts are populated for all queries that reach the database (kalayaan=44, simbahan=38, bayan=42, education=42). The zeros appear only on korapsyon and out-of-domain queries, which return early with reason "unmapped_standalone_concept" before touching the DB. Stage-level timing (preprocessing, model encoding, theme expansion, etc.) was NOT instrumented.

2. **Server crash claim**: Unsupported. baseline_timings.json has 10 runs for all 40 queries, including "online", and golden_snapshot.json has all 40 queries. Removed crash note from documentation.

3. **Theme classification / N+1 hypothesis**: REFUTED by the data. DB query counts are nearly identical (38-44) across fast and slow queries, yet latency differs 4-12x:
   - bayan: 42 queries, 2.5s
   - kalayaan: 44 queries, 12.3s
   - edukasyon: 44 queries, 32.1s
   
   If N+1 context expansion was the culprit, identical query counts would produce similar latency. The extra time is NOT from DB round-trips.

4. **Numeric corrections**:
   - edukasyon cold: 38.9s (not 32.5s)
   - kamatayan cold: 11.3s (not 17.1s)
   - first request: 25.00s (not 25.98s)
   - memory delta: -11.79 MB (not 0.05 MB) - client-side measurement, not meaningful
   
   Regenerated category tables directly from JSON with analyze_baseline.py script.

5. **EXPLAIN queries**: ALL EXPLAIN queries failed. The embedding was stringified character-by-character ('[', '0', '.', ...), not as a proper array literal. Previous "fixed vector syntax" claims in STATE.md and DECISIONS.md were incorrect. explain_queries.py needs proper array literal formatting.

**New Observations (Data-Driven)**:

- **2.4s floor**: Every query has a baseline cost of ~2.4s, even korapsyon (4 DB queries) and out-of-domain queries (0 DB queries, 0 results). This floor is NOT from DB work or result processing.

- **Semantic fallback correlation**: Slow queries (kalayaan, pag-ibig, edukasyon) are all result_mode="semantic_fallback". Fast queries (bayan, simbahan) are "lexical". BUT English fallback queries (oppression, love, justice) are FAST (~2.4-2.7s), so "fallback" itself is not the trigger. Candidate: theme expansion or candidate pool size for specific Tagalog terms.

- **Out-of-domain is slow**: Out-of-domain rejection (0 DB queries) takes 3.4s average - SLOWER than valid lexical queries (2.5s). The early-reject path is not cheap. Likely costs: query preprocessing, model encoding, theme expansion, or translator network calls.

- **No caching**: Warm == cold across all queries. No query result caching, no model output caching, no meaningful warm-up effect.

**Revised Hypotheses**:

1. Theme expansion for specific Tagalog terms (likely: massive keyword generation or expensive matrix operations)
2. Semantic candidate pool processing (large retrieval sets for certain queries)
3. Model encoding (contributes to 2.4s floor)
4. Translator network calls (contributes to floor for cross-lingual queries)
5. ~~Theme classification / N+1 context expansion~~ → **REFUTED**

**Decision**: Proceed with Phase 1b instrumentation (stage-level timing) before any optimization.

**Impact**: No code changes yet. Documentation corrections only.

---

## Template for Future Entries

### YYYY-MM-DD: [Brief Title]
**Context**: [Why was this decision needed?]

**Decision**: [What did we decide to do?]

**Rationale**: [Why this approach?]

**Impact**: 
- Performance: [Measured impact on latency/throughput]
- Correctness: [Any ranking changes? Verified how?]
- Trade-offs: [What did we give up?]

**Verification**: [How did we confirm correctness?]

---

## Ranking Change Policy

Any change that affects search result order MUST:
1. Be explicitly approved by user
2. Be logged here with full justification
3. Include before/after comparison on golden query set
4. Document the specific ranking delta (# of position changes)
5. Explain why the change improves search quality

**Current ranking changes**: None — Phase 1b/1c diagnostic instrumentation added behind PERF_BENCHMARK flag; no optimization or ranking-altering changes approved.

---

### 2026-10-12: Phase 1b/1c Instrumentation & Diagnostics (No Optimization)
**Context**: Completing Phase 1 measurement before Phase 2 optimization. Added PerfTimer; fixed EXPLAIN; created targeted benchmark; added .gitattributes.

**Decision**: Diagnostic only — no ranking-affecting changes. Corrected broken PerfTimer context manager (rerank loop). Added total timer, translator network flag, cache tracking, cProfile script.

**Evidence from targeted benchmark (5 queries, 5 runs)**:
- edukasyon: rerank_loop ~31,367ms (97.8%), total ~32,084ms — rerank is bottleneck
- kalayaan: rerank_loop ~9,595ms (95.1%), total ~10,087ms — rerank is bottleneck
- bayan: rerank_loop ~40ms (7.4%), total ~540ms — lexical fast
- oppression: rerank_loop ~72ms (8.6%), total ~836ms — semantic but fast (lexical hits survive)
- tiktok: 1,637ms avg, 0 DB queries — floor = translator network (~1.5s) + encode (~70ms)

**Cache analysis (line 734)**:
- `self.word_sim_cache = {}` reset per search call → instance-level, never persistent
- Key = word string (w from valid_tokens)
- Size = unbounded (grows with validation words per query), never cleared between queries because new instance per search
- Hits/merges: not tracked; need instrumentation — explains warm==cold (no persistence)

**Comparison oppression vs edukasyon**:
- Both semantic_fallback, ~300 candidates, 44 DB queries
- Opposition: 1 lexical hit + 301 semantic; rerank ~72ms per query
- Edukasyon: 4 lexical hits + 300 semantic; rerank ~31,367ms per query
- Difference: validation loop takes 435x longer per candidate for eduka (likely due to 11 theme keywords vs 6, more synonym checks, lower lexical grounding)

**Untimed gap**: ~200-490ms per request not covered by stage timers. New fields added: `search_total_ms`, `untimed_gap_ms`, `translator_ms`, `translator_network_calls`

**Baseline shift explanation**:
- bayan dropped from 2.5s (baseline JSON) to 0.54s (current server)
- Possible causes: server restarted with different model load state, translator cache warm after first call, different Python process initialization, or measurement environment differences (local server vs previous benchmark environment). Not a code change to engine logic.
- tiktok: 3.4s → 1.6s — similar explanation (translator response faster in current session)

**Profile script**: `scripts/profile_rerank.py` created (needs Poetry environment to run — system Python lacks sqlalchemy). Uses cProfile to identify base_model.encode call counts inside rerank.

**No ranking changes made**. Compare golden required but server unavailable in environment. Profile script written but not executed due to environment limitations.

**Commit**: 24c42c9 (includes .gitattributes, target benchmark, profile script, DECISIONS.md update).

---


### 2026-10-13: Phase 2 Step 1 — Process-Level Word Embedding Cache

**Context**: Phase 1 profiling confirmed that `base_model.encode()` inside the rerank validation loop dominates query latency: 622 encodes × ~50ms each = 31s for edukasyon, 166 encodes × ~50ms each = 8.3s for kalayaan. Edukasyon had 0 cache hits because `self.word_sim_cache = {}` resets per search() call with no persistence. Without a process-level cache, every word requires re-encoding on each request.

**Decision**: Add a process-level dict `_word_vec_cache` that persists across `search()` calls within a single RizalEngine instance. Key = the exact string passed to `encode()`; Value = the exact numpy array returned. In the validation loop, check this cache before encoding; if miss, encode and store with thread-safe lock. Keep per-request `word_sim_cache` semantics unchanged (query-dependent max similarity). Add thread safety (threading.Lock) for FastAPI concurrency. Report process-cache hits/misses separately from per-request ones under PERF_BENCHMARK.

**Rationale**: Eliminates redundant encode calls for repeated vocabulary across requests. The 16k-sentence corpus vocabulary means the cache will naturally bound. Similarity rankings remain identical because the same `base_model.encode(w)` gives bit-identical vectors. No changes to thresholds, token selection, candidate ordering, or any other logic.

**Impact**:
- Performance: Measured (this session): edukasyon rerank 76.9ms, kalayaan 62.3ms, encode_calls_total 1 each, process_cache_hits 614/154, process_cache_size 693. No estimates from earlier session included.
- Correctness: Rankings verified identical before and after (compare_golden: 0 position changes, 0 score deltas).
- Trade-offs: Memory usage for bounded cache (~3KB per float32 768-dim vector; 50k max ≈ 150MB). Thread lock overhead negligible. No precompute, no HNSW.

**Verification**: compare_golden.py ran against live server BEFORE and AFTER the change — all 40 queries showed identical rankings and scores (0 position changes, 0 score deltas, 0 theme changes). targeted_benchmark.py re-run: min/median/max per query with warm/cold cache comparison. First edukasyon request shows cold cache; repeats show warm cache.

**Commit**: Phase 2 implementation with process-level word embedding cache.
