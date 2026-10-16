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

**Approved ranking changes**:
- **Step B Determinism (2026-10-16, approved by user)**: 22 of 40 golden queries changed (21 result/ranking changes + 1 metadata-only). 16 queries attributed to B1 `ORDER BY` tie-breaks at tied score boundaries; 6 additional queries attributed to E1 theme expansion sorting. Commits E2 and E3 produced 0 ranking changes but eliminated cross-seed hash divergence. New permanent reference: `docs/perf/golden_snapshot_deterministic.json`.

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


### 2026-10-13: Phase 2 Step 1 — Process-Level Word Embedding Cache & Hash Randomization Root Cause

**Context**: Phase 1 profiling confirmed that `base_model.encode()` inside the rerank validation loop dominates query latency: 622 encodes × ~50ms each = 31s for edukasyon, 166 encodes × ~50ms each = 8.3s for kalayaan. Edukasyon had 0 cache hits because `self.word_sim_cache = {}` resets per search() call with no persistence. Without a process-level cache, every word requires re-encoding on each request.

**Decision**: 
1. Add a process-level dict `_word_vec_cache` that persists across `search()` calls within a single RizalEngine instance. Key = exact string passed to `encode()`; Value = exact numpy array returned (made read-only via `flags.writeable = False`). In the validation loop, check this cache before encoding; if miss, encode and store with thread-safe lock. Keep per-request `word_sim_cache` semantics unchanged. Add thread safety (`threading.Lock`) for FastAPI concurrency. Report process-cache hits/misses under `PERF_BENCHMARK`.
2. Set `PYTHONHASHSEED=0` in all environment launch scripts/documentation to ensure deterministic set iteration ordering across server restarts.

**Root Cause Analysis for Run-to-Run Variance**:
Initial golden snapshot comparisons between fresh server processes showed 10-12 queries with position changes and score deltas even with the cache disabled. A rigorous 6-step controlled diagnostic isolated the exact mechanism:
- **Same-process cache validation**: Uncached vs cached across all 40 queries in ONE process produced **0 position changes, 0 score deltas (40/40 identical)**.
- **Cache vector immutability**: All 786 cached vectors were verified read-only and tested against fresh `base_model.encode(w)` twice with **0 mismatches (786/786 bit-identical)**.
- **Diagnostic runs across fresh processes**:
  - `4(a) PYTHONHASHSEED=0`: Run 1 vs Run 2 across two fresh uvicorn processes produced **0 position changes, 0 score deltas (40/40 identical)**.
  - `4(b) OMP/MKL threads=1 (default hash seed)`: Run 1 vs Run 2 produced **10 queries with changes, 37 position changes, 113 score deltas**.
  - `4(c) Both PYTHONHASHSEED=0 and threads=1`: Run 1 vs Run 2 produced **0 position changes, 0 score deltas (40/40 identical)**.
- **Point of divergence (Step 6)**: In query `kalayaan` between two fresh runs under default hash seed, the theme expansion text diverged due to Python set iteration order over `candidates`:
  - Run 1 expansion: `"Kasakiman gahaman salapi nabibili pagnanasa nawawala halaga"`
  - Run 2 expansion: `"Kasakiman gahaman salapi nawawala pagnanasa nabibili halaga"`
  - Because word order in the combined concept text `f"{query} {expanded_text}"` differed, `base_model.encode()` produced slightly different query embeddings, causing candidate selection at index 8 (candidate #9) to diverge (ID 37644 vs ID 37600), cascading to 88/87 distinct candidates between runs.
  - Under `PYTHONHASHSEED=0`, expansion text was bit-for-bit identical across processes, yielding 0 candidate divergence across all 304 candidates.
- **Translator impact**: `translator_network_calls` was `False` for all queries in both runs; no translator differences occurred.

**Impact**:
- **Performance**: Process-level cache eliminates redundant encoding in rerank validation loop.
- **Correctness**: 0 position changes, 0 score deltas when running cached vs uncached in same process, and 0 differences across fresh processes with `PYTHONHASHSEED=0`.
- **Trade-offs**: Bounded memory for cache (~3KB per vector, max 50k ≈ 150MB). Requires `PYTHONHASHSEED=0` for cross-process determinism.

**Verification**:
- Same-process 40-query bypass test: 40/40 pass (0 diffs).
- Verify cache endpoint: 786 words, 786 read-only, 0 mismatches over 2 passes.
- Fresh-process test with `PYTHONHASHSEED=0`: 40/40 pass (0 diffs).

---

### 2026-10-14: Diagnostic & Fix for ~2.0s Client Latency Floor, cProfile Untimed Gap, and Corrected Before/After Benchmark

**Context**:
- Benchmark suite previously measured ~2.4s HTTP latency for fast lexical queries like `bayan` when `search_total_ms` was only ~415–600ms, creating a ~2.0s discrepancy outside the search engine.
- In addition, an untimed gap of ~500–600ms existed inside `engine.search()` between total search time and existing stage timers.
- Phase 2 Step 1 Before vs. After needed to be re-measured using the corrected client to obtain valid absolute end-to-end latency numbers.

**Decisions & Findings**:

1. **Client IPv6 `::1` Connect Stall Discovered & Fixed**:
   - Diagnostic socket test confirmed that Windows `socket.getaddrinfo('localhost', port)` returns `AF_INET6` (`::1`) first. Because Uvicorn bound exclusively to IPv4 (`127.0.0.1`), connecting to `localhost` stalled for 2,046 ms on average (Windows TCP connection timeout) before falling back to IPv4 `127.0.0.1`:
     - `socket.create_connection(('localhost', 8002))`: [2066.13, 2027.43, 2055.06, 2049.20, 2032.83] ms
     - `socket.create_connection(('127.0.0.1', 8002))`: [0.92, 0.47, 0.42, 0.42, 0.56] ms
   - Lack of HTTP connection reuse in `benchmark.py` forced a fresh TCP handshake on every single request, multiplying this 2.0s stall across all 400 benchmark queries.
   - FastAPI middleware overhead was measured and ruled out (total FastAPI layer overhead outside engine is ~6.94 ms: routing/DI 1.54 ms, DB session 0.05 ms, serialization 4.67 ms).
   - **Fix Applied**: Updated `scripts/benchmark.py` to default `API_BASE` to `http://127.0.0.1:8000`, route all requests through `requests.Session()`, and update `--port` URL to `127.0.0.1`. Added documentation note in `BENCHMARK_INSTRUCTIONS.md` and marked all past `localhost` latency tables in `BASELINE.md` invalid for absolute HTTP latency while preserving server-side stage timings.

2. **cProfile Profiling of Untimed Search Gap**:
   - Profiled `engine.search()` directly for `bayan` (603.96 ms total) and `oppression` (931.25 ms total) with `cProfile` (warm process, `PYTHONHASHSEED=0`, cache on).
   - **Timer Loop Overwrite Bug**: In `engine.py:1285-1294`, `PerfTimer.stage("theme_classification")` and `PerfTimer.stage("context_expansion")` execute inside a loop over candidates. Because `PerfTimer.stage()` assigns (`self.timings[name] = elapsed`) instead of accumulating, only the 20th candidate's time was retained.
     - In `bayan`: `_classify_themes` took 371.4 ms cumulative (20 calls) but recorded only 18.3 ms (353.1 ms untimed). `_expand_context` took 119.3 ms cumulative (20 calls) but recorded only 6.6 ms (112.7 ms untimed). Together, this explains 465.8 ms of the gap.
     - Inside `_classify_themes`: `_get_passage_context_with_embedding` (`engine.py:190`) took 170.3 ms looking up passages and neighbor embeddings from DB.
   - **Un-timed SentenceTransformer Encodings**: In `oppression`, 6 `SentenceTransformer.encode` calls took 507.2 ms, of which only 75.1 ms was captured in `base_model_encode`. The remaining 432.1 ms was spent in un-timed encodings (concept vector re-encoding line 631, query component encoding line 907, significant words encoding line 919).
   - **Database / ORM Overhead**: 42 queries in `bayan` spent 143.2 ms in cursor execution + 121.8 ms in SQLAlchemy model hydration; 30 queries in `oppression` spent 211.2 ms in cursor execution + 222.1 ms in SQLAlchemy model hydration.

3. **Translator Results Flow & Process Cache Proposal**:
   - Synchronous `GoogleTranslator.translate()` makes an un-cached external HTTP request for unmapped terms (`tiktok` taking 1,453–2,334 ms, 87%–90% of request latency).
   - Proposed a process-level `_translator_cache: Dict[str, Tuple[str, float]]` keyed by exact input string with timeout and `threading.Lock`. Because translator outputs are deterministic for invariant strings, identical ranking and scoring outputs are guaranteed.

4. **Corrected Before vs. After Benchmark Measurements**:
   - Both states measured with the fixed client (`127.0.0.1` + `requests.Session()`) on warmed server (`PYTHONHASHSEED=0`, `PERF_BENCHMARK=1`).
   - "Before" state: Cache disabled (`DISABLE_WORD_VEC_CACHE=1`) recorded in `docs/perf/fixed_client_before_timings.json`.
   - "After" state: Word embedding cache enabled (`DISABLE_WORD_VEC_CACHE=0`) recorded in `docs/perf/fixed_client_after_timings.json`.
   - Discarding Run 1 as warmup, warm statistics over Runs 2-10:
     - `single_word_tagalog` Warm P50: **12,824.4 ms → 1,459.8 ms (-88.6%, 8.78x faster)**
     - `single_word_tagalog` Warm P95: **16,040.1 ms → 1,674.3 ms (-89.6%, 9.58x faster)**
     - `kalayaan` Warm P50: **20,133.2 ms → 2,215.5 ms (-17,917.8 ms, 9.09x faster)**
     - `edukasyon` Warm P50: **48,533.9 ms → 2,749.8 ms (-45,784.1 ms, 17.65x faster)**
     - `pag-ibig` Warm P50: **13,672.1 ms → 1,310.7 ms (-12,361.4 ms, 10.43x faster)**
     - `kamatayan` Warm P50: **15,437.7 ms → 1,055.2 ms (-14,382.5 ms, 14.63x faster)**
     - `bayan` Warm P50: **768.6 ms → 709.3 ms (-59.4 ms)**
     - `multi_word_tagalog` Warm P50: **1,193.1 ms → 1,083.0 ms (-9.2%)**
     - `english_cross_lingual` Warm P50: **725.9 ms → 661.2 ms (-8.9%)**
     - `character_names` Warm P50: **786.1 ms → 702.2 ms (-10.7%)**
     - `out_of_domain` Warm P50: **1,684.7 ms → 2,273.9 ms (+35.0%)** (due to Google Translator API network response variability on unmapped queries)

**Ranking Policy Compliance**: 0 code changes to ranking or search logic; 0 determinism changes (`sorted()` / `ORDER BY`) applied. All measurements strictly empirical.

---

### 2026-10-14: PerfTimer Accumulation & Call Counts Instrumentation

**Context**: In `engine.py`, `with timer.stage(...)` calls inside candidate loops (`theme_classification`, `context_expansion`) were overwriting previous iterations rather than accumulating, and non-loop stages lacked call counts. Three additional encoding sites (`query_components_encode` at line 597, `concept_vector_encode` at line 631, and `sig_words_encode` at line 919) lacked timers.

**Decision**:
- Modified `PerfTimer.stage()` in `app/core/perf_instrumentation.py` to accumulate (`+=`) elapsed ms and increment call counts (`{stage}_count`).
- Added explicit instrumentation wrappers for the three un-timed encode sites.
- Preserved identical timing contract and metadata fields.

**Impact**:
- Correctness: 0/40 queries changed in `golden_snapshot_seed0.json` (verified via `verify_timer_golden.py`).
- Profiling accuracy: Untimed gap eliminated from ~500ms down to near 0ms (e.g. `bayan` untimed gap -37.22 ms, `oppression` 94.49 ms).

---

### 2026-10-14: Clean Benchmarking Protocol (Two Full Warmup Passes & Quiet Machine State)

**Context**: Previous Before vs After measurements exhibited noise due to cold memory pressure and pagefile throttling. A rigorous, controlled protocol was required.

**Decision**:
- Pre-warm the server with two complete 40-query warmup passes before taking any benchmark measurements.
- Track CPU and RAM utilization before, during, and after runs.
- Require non-affected categories (e.g., `character_names`) to differ by <= 15% between Before and After to validate machine stability.
- Exclude the external translator-dependent `out_of_domain` category from any in-domain speedup claims.

**Empirical Results**:
- System load: Before CPU avg = 57.9%, RAM = 565.6 MB free; After CPU avg = 57.5%, RAM = 460.0 MB free.
- Non-affected category check: `character_names` differed by 11.58% (PASS, <= 15%).
- `single_word_tagalog` warm median: dropped from **5,312.9 ms to 646.9 ms (-87.6% speedup)**.
  - `edukasyon`: **31,615.0 ms → 858.1 ms (-97.3% speedup)**.
  - `kalayaan`: **10,459.1 ms → 700.0 ms (-93.3% speedup)**.
  - `pag-ibig`: **9,319.3 ms → 758.6 ms (-91.9% speedup)**.
  - `kamatayan`: **10,637.3 ms → 618.2 ms (-94.2% speedup)**.

---

### 2026-10-14: Process-Level Translator Cache Implementation

**Context**: Unmapped cross-lingual and out-of-domain queries invoked `GoogleTranslator.translate()` synchronously over public HTTP without caching or bounded timeouts, causing 1.2s-2.5s stalls and triggering `TooManyRequests` HTTP 429 errors upon repeated searches.

**Decision**:
- Implemented `TranslatorCache` in `backend/app/core/engine.py`:
  - Lock-protected across worker threads (`threading.Lock`).
  - Keyed by exact stripped lowercase query string (`text.strip().lower()`).
  - Stores exact returned translation string.
  - Positive cache TTL: 24 hours (`86,400s`).
  - Negative cache TTL on error/rate limit: 60 seconds (`60s`).
  - Request timeout: 5.0 seconds via persistent `concurrent.futures.ThreadPoolExecutor`.

**Impact & Verification**:
- Output equivalence: 0/40 queries changed in `golden_snapshot_seed0.json` (100% bit-identical).
- Latency improvement on out-of-domain queries:
  - First call (cold network): median **1,192.98 ms**
  - Repeated call (hot cache): median **123.06 ms**
  - Repeated call speedup: **+89.7%**
- Eliminates repeated Google Translate network traffic and prevents HTTP 429 rate limits.

---

### 2026-10-14: Full 40x10 Clean Benchmark with TranslatorCache ON

**Context**: Need an official 400-query benchmark run (40 queries × 10 runs) on a fully warmed server with `TranslatorCache` active, capturing memory footprint (RSS and free RAM) and isolating out-of-domain first-seen from repeated requests.

**Empirical Results (Runs 2-10 Pooled Warm Statistics, Discarding Run 1)**:
- **Server RSS and System Memory Footprint**:
  - Initial (server start): Server RSS = 76.71 MB, Available RAM = 395.7 MB (95.0% used)
  - Post-warmup (2 full passes): Server RSS = 777.39 MB, Available RAM = 831.9 MB (89.5% used)
  - Post-benchmark (400 queries): Server RSS = 757.07 MB, Available RAM = 478.9 MB (94.0% used)
  - Average CPU load during runs = 60.0%, Total System RAM = 7,955.3 MB
- **Per-Category Warm Latency**:
  - `single_word_tagalog` (72 samples): Warm P50 = **660.67 ms**, Warm P95 = **818.99 ms** (Min: 139.58 ms, Max: 926.17 ms)
  - `multi_word_tagalog` (72 samples): Warm P50 = **504.61 ms**, Warm P95 = **833.18 ms** (Min: 409.60 ms, Max: 899.41 ms)
  - `english_cross_lingual` (72 samples): Warm P50 = **374.90 ms**, Warm P95 = **603.81 ms** (Min: 302.32 ms, Max: 713.82 ms)
  - `character_names` (72 samples): Warm P50 = **431.08 ms**, Warm P95 = **636.11 ms** (Min: 293.09 ms, Max: 711.66 ms)
  - `out_of_domain` (72 samples): Warm P50 = **127.12 ms**, Warm P95 = **159.08 ms** (Min: 106.22 ms, Max: 182.13 ms)
- **Out-of-Domain First-Seen vs. Repeated Breakdown**:
  - First-Seen (Run 1, Cold Network Call): P50 = **1,276.21 ms**, P95 = **1,731.26 ms** (Min: 1,124.24 ms, Max: 1,817.78 ms)
  - Repeated (Runs 2-10, Hot Translator Cache): P50 = **127.12 ms**, P95 = **159.08 ms** (Min: 106.22 ms, Max: 182.13 ms)
  - Latency speedup on repeated out-of-domain queries: **90.0%**

---

### 2026-10-14: Batched Context Fetch for Context Expansion and Theme Classification

**Context**: In Stage 4 post-processing, candidate sentences individually queried PostgreSQL for neighbor sentences: `_classify_themes` executed a SQL query for a $\pm 8$ window for each candidate (20 queries), and `_expand_context` executed a SQL query for a $\pm 3$ window for each candidate (20 queries), totaling 40 roundtrips per search request and contributing ~170–200ms of wire latency.

**Decision**:
- Implemented `_fetch_context_cache` in `backend/app/core/engine.py`:
  - Groups candidates by `(book, chapter_number, source_type)`.
  - Computes $\pm 8$ union bounding windows across candidates within each group.
  - Merges overlapping and contiguous intervals into disjoint ranges.
  - Executes a single SQL query with `where(or_(*clauses)).order_by(Sentence.book, Sentence.chapter_number, Sentence.source_type, Sentence.sentence_index)`.
  - Indexes fetched records in an in-memory dictionary keyed by `(book, chapter_number, source_type, sentence_index) -> Sentence`.
  - Passes `context_cache` to `_expand_context` (extracting $\pm 3$) and `_get_passage_context_with_embedding` (extracting $\pm 8$).
  - Missing neighbors and chapter boundary semantics are identically preserved via key membership tests without relying on list positions or contiguous index assumptions.
  - Query ordering is strictly deterministic.
  - Added `DISABLE_BATCHED_CONTEXT_FETCH` environment variable to support differential testing.

**Verification Results**:
1. **Subtask 3(a) Golden Query Snapshot Equivalence**:
   - Tested all 40 golden queries against `docs/perf/golden_snapshot_seed0.json`.
   - **40/40 queries bit-identical (0 differences)** in rankings, scores, context texts, and theme tags.
2. **Subtask 3(b) Per-Candidate Exhaustive Assertions**:
   - Evaluated all 382 candidate instances across 40 golden queries comparing sequential vs. batched execution:
     - Candidate IDs: 382/382 identical
     - Scores: 382/382 identical
     - `context_text` (expanded text): 382/382 bit-identical
     - `themes` list and scores: 382/382 bit-identical
   - Evaluated an extra set of 180 boundary sentences chosen at chapter starts, second sentences, and chapter ends across 30 chapters in both `summary` and `full` source types:
     - `_expand_context`: 180/180 bit-identical
     - `_get_passage_context_with_embedding` (text and embedding vectors): 180/180 bit-identical
     - `_classify_themes`: 180/180 bit-identical
   - Total assertions passed: 2,068 / 2,068 (0 failures).
3. **Subtask 3(c) DB Query Count Reductions**:
   - `bayan`: 42 queries → 3 queries (**-39 queries, -92.9% reduction**)
   - `oppression`: 34 queries → 5 queries (**-29 queries, -85.3% reduction**)
   - `edukasyon`: 44 queries → 5 queries (**-39 queries, -88.6% reduction**)
   - `Crisostomo Ibarra`: 30 queries → 7 queries (**-23 queries, -76.7% reduction**)
   - `karahasan ng mga prayle`: 52 queries → 13 queries (**-39 queries, -75.0% reduction**)
4. **Subtask 3(d) Latency Profiles (Direct Warm Engine)**:
   - `bayan` (Wall-clock: 394.58 ms → 293.72 ms, **-100.86 ms, +25.6% speedup**):
     - `context_fetch`: 0.00 ms → 66.76 ms (1 batched query)
     - `context_expansion`: 69.62 ms → 4.08 ms (-65.54 ms)
     - `theme_classification`: 201.37 ms → 99.93 ms (-101.44 ms)
     - Net post-processing time: 270.99 ms → 170.77 ms (-100.22 ms)
   - `oppression` (Wall-clock: 597.67 ms → 561.07 ms, **-36.60 ms, +6.1% speedup**):
     - `context_fetch`: 0.00 ms → 72.41 ms (1 batched query)
     - `context_expansion`: 57.12 ms → 3.40 ms (-53.72 ms)
     - `theme_classification`: 126.49 ms → 49.15 ms (-77.34 ms)
     - Net post-processing time: 183.61 ms → 124.96 ms (-58.65 ms)

---

### 2026-10-14: Final Benchmark Passes, Slow First Request Investigation, and Final Report

**Context**: Concluding Phase 2 performance optimization work. Measured the final "Before" (unoptimized baseline) and final "After" (all optimizations ON) across the full 40x10 benchmark suite under identical protocols on a quiet machine. Diagnosed the slow-first-request phenomenon on 5 unseen Tagalog words. Produced `docs/perf/FINAL_REPORT.md`.

**Decisions & Findings**:

1. **Final Before vs. After Benchmark Results (Identical Protocol, Quiet Machine)**:
   - Measured with fixed client (`http://127.0.0.1:8002`, `requests.Session()`) with two full in-domain warmup passes.
   - `single_word_tagalog` Warm P50: **5,069.59 ms → 478.27 ms (-90.6%, 10.6x faster)**
   - `single_word_tagalog` Warm P95: **29,922.43 ms → 691.14 ms (-97.7%, 43.3x faster)**
   - `edukasyon` Warm P50: **~31,615 ms → 624.90 ms (-98.0%, 50.6x faster)**
   - `bayan` Warm P50: **452.70 ms → 298.40 ms (-34.1%)**
   - `multi_word_tagalog` Warm P50: **573.61 ms → 502.93 ms (-12.3%)**
   - `english_cross_lingual` Warm P50: **416.80 ms → 344.67 ms (-17.3%)**
   - `character_names` Warm P50: **510.80 ms → 353.34 ms (-30.8%)**
   - `out_of_domain` Repeated Warm P50: **1,298.91 ms → 129.73 ms (-90.0%, 10.0x faster)**
   - System CPU average: 56.6% (Before) vs. 61.4% (After).
   - Server RSS: 603.58 MB avg (Before) vs. 756.27 MB avg (After).

2. **Root Cause Analysis of Slow First Request**:
   - Tested 5 previously unseen Tagalog words (`kasakiman`, `dangal`, `takot`, `alipin`, `sakripisyo`) on a fresh server boot with empty word vector cache:
     - Lexical queries (`takot`): **322.18 ms on first cold request** (fast from start).
     - Semantic fallback queries (`kasakiman`, `dangal`, `alipin`): First search takes **5.7s to 20.6s** because 100–255 unique tokens in candidate sentences are encoded individually at ~40–50ms each (`rerank_loop` takes 5.1s–12.1s). Once cached, second search drops to **474–509 ms (91%–97% speedup)**.
     - Unmapped concepts (`sakripisyo`): First cold search takes **1,254.53 ms** (Google Translate network call), dropping to **130.43 ms** on subsequent searches.

3. **Scope Constraint Honored**:
   - Zero changes made to rerank logic, encode batching, or thresholds.
   - **No `sorted()` or `ORDER BY` determinism changes applied**; stopped and awaiting user decision.

---

### 2026-10-14: Step A — Startup Vocabulary Precomputation (`word_vec_cache_v1.pkl`)

**Context**: In `semantic_fallback` mode, the first search for a query with unpopulated vocabulary in `_word_vec_cache` was forced to encode 100–255 unique candidate tokens on the fly at ~40–50 ms each, taking 5.7s to 20.6s on a cold server.

**Decision**:
1. Implemented `scripts/build_word_vec_cache.py`:
   - Extracted exact candidate word filter from `engine.py` (`len(w) > 4 and w not in LOW_INFO_WORDS and w in vocabulary`), identifying exactly **24,770 words**.
   - Encoded each word one-at-a-time with `base_model.encode(w, show_progress_bar=False)` to guarantee float32 vector identity against runtime single-word encodes.
   - Serialized vectors to `backend/app/data/word_vec_cache_v1.pkl` and `docs/perf/word_vec_cache_v1.pkl` (73.39 MB) along with model name (`sentence-transformers/paraphrase-xlm-r-multilingual-v1`) and vocabulary SHA-256 hash (`6efe3301743eed6b7552291a9c444e7a4ea426cd5175d89ca9d8b648b1c65569`).
2. Implemented `_load_precomputed_word_vectors()` in `backend/app/core/engine.py`:
   - Validates model name and vocabulary hash.
   - Loads all 24,770 vectors into `_word_vec_cache` at startup in **238.2 ms**.
   - Marks arrays read-only (`vec.flags.writeable = False`).
   - Respects 50,000 cap (`_MAX_WORD_VEC_CACHE_SIZE`).
   - Falls back to on-demand encode on any missing word.

**Verification Results**:
- **Verification 3(a)**: 500 / 500 random sample vectors verified 100% bit-identical (`np.array_equal == True`) against fresh single-word encode.
- **Verification 3(b)**: 40 / 40 golden benchmark queries verified bit-identical vs `golden_snapshot_seed0.json` under `PYTHONHASHSEED=0` (**0 differences**).
- **Verification 3(c) Fresh-Server First-Search Latency (10 Unseen Words)**:
  - `kasakiman`: Cold rerank loop = **78.07 ms** (down from 9,798 ms, 209 hits, **0 misses**).
  - `alipin`: Cold latency = **606.57 ms** (down from 12,583 ms, **20.7x faster**; rerank loop = 79.33 ms, 370 hits, **0 misses**).
  - `dangal`: Cold latency = **598.48 ms** (down from 5,706 ms, **9.5x faster**; rerank loop = 59.52 ms, 164 hits, **0 misses**).
  - `katapangan`: Cold latency = **486.48 ms** (rerank loop = 58.12 ms, 168 hits, **0 misses**).
  - `kabayanihan`: Cold latency = **539.43 ms** (rerank loop = 75.49 ms, 397 hits, **0 misses**).
  - `kalupitan`: Cold latency = **484.56 ms** (rerank loop = 61.96 ms, 172 hits, **0 misses**).
  - `pagdurusa`: Cold latency = **487.48 ms** (rerank loop = 64.41 ms, 179 hits, **0 misses**).
  - `himagsikan`: Cold latency = **508.99 ms** (rerank loop = 63.38 ms, 169 hits, **0 misses**).
  - `takot`: Cold latency = **500.87 ms** (lexical mode).
  - `sakripisyo`: Cold latency = **1,775.13 ms** (first-seen network call) $\rightarrow$ Warm: **153.67 ms**.
- **Verification 3(d) Startup Time & RSS**:
  - Uvicorn startup time to health check: **10.53 seconds**.
  - Cache load time: **238.2 ms** (24,770 entries).
  - Initial process RSS: **460.35 MB**.
  - Post-search process RSS: **1,215.35 MB**.

**Ranking Policy Compliance**: 0 ranking changes across all 40 golden queries. No determinism changes (`sorted()` / `ORDER BY`) applied yet; awaiting user decision for Step B.

---

### 2026-10-16: Step B — Search Pipeline Determinism (Approved)

**Context**: In Python 3.3+, SipHash randomizes hash seeds on process startup unless `PYTHONHASHSEED` is explicitly pinned. In `backend/app/core/engine.py`, four non-deterministic mechanisms caused search results and ranking orders to diverge across independent server processes:
1. SQL queries lacked deterministic secondary ordering keys on `LIMIT top_k * 20` and `ORDER BY` clauses.
2. Theme candidate extraction iterated over unordered Python `set()` objects, and tie-breaking specificity scores lacked secondary alphabetical tie-break keys.
3. Cross-lingual synthesized bridge translations iterated over unordered `bridge_tokens` sets when assembling `bridge_text`.
4. Dynamic semantic validation iterated over unordered `sent_words = set(...)`, causing candidate tokens to mutate the shared `valid_tokens` set in arbitrary order across processes.

**Decision**: Apply minimal, targeted determinism fixes strictly for reproducibility across processes, without altering search scoring models, thresholds, or reranking algorithms:
1. **Commit B1 (`646deb6`)**: Added explicit `ORDER BY Sentence.id` to lexical retrieval statements, `ORDER BY Theme.id` to theme loading, and deterministic primary/secondary keys on SQL queries.
2. **Commit D (`964a2c1`)**: Diagnostic instrumentation only behind `PERF_BENCHMARK=1` (SHA-256 hash of final query vector bytes, bridge token metadata, translator diagnostics).
3. **Commit E1 (`17896f3`)**: Deterministic theme candidate iteration via `sorted(candidates)` and composite sort key `(-specificity, word)` in `_expand_query_with_themes`.
4. **Commit E2 (`6e16b16`)**: Deterministic bridge text synthesis via `sorted([bt for bt in bridge_tokens if len(bt) > 3])`.
5. **Commit E3 (`73f1d1a`)**: Deterministic candidate word iteration in dynamic validation via `cand_words = [w for w in sorted(sent_words) if len(w) > 4 and w not in self.LOW_INFO_WORDS]`.

**Rationale**:
- Public deployment across multiple worker processes (Uvicorn/Gunicorn workers, autoscaled instances, CI test suites) requires bit-identical search results regardless of process startup order or Python hash seeds.
- Zero scoring model changes: no semantic weights, lexical weights, precision boosts, or reranking formulas were modified.

**Empirical Behavior Diffs vs Golden Seed0 Reference**:
- **22 queries changed** out of 40 total (21 result/ranking + 1 metadata-only).
- **16 queries** from B1 (`ORDER BY` tie-breaks) — all at tied score boundaries.
- **6 additional queries** from E1 (theme expansion sorting) — stabilized concept vectors.
- **E2 & E3**: 0 ranking changes on the 40 golden queries at `PYTHONHASHSEED=0`; eliminated cross-seed hash divergence in `bridge_tokens` (`love`, `corruption`) and `valid_tokens` (`edukasyon`).

**Verification (Phase B6 — 9 Evidence Items)**:
- 6-process matrix (40 queries): 100% bit-identical across all 15 pairwise comparisons.
- 206-query held-out suite: 100% bit-identical across all 6 pairwise comparisons.
- Deterministic translator stub reruns: 100% bit-identical (all 15 pairs, all 6 held-out pairs).
- 49 `set()` sites audited across all reachable files — every remaining site verified as pure membership lookup or dedup filter.
- All 9 evidence items documented with SHA-256 hashes in `docs/perf/evidence/b6_*.txt`.

**Known Limitations**:
1. `ibarra` LIMIT-200 truncation: High-frequency content words exceed `LIMIT top_k * 20 = 200` in full corpus search. While `ORDER BY Sentence.id` makes truncation deterministic, occurrences with `id > 200th match` are excluded from evaluation.

**Impact**:
- Performance: No measurable latency impact (sorting small sets is negligible).
- Correctness: 22 queries with position shifts at tied score boundaries; 100% bit-identical determinism across all server processes without `PYTHONHASHSEED=0`.
- Trade-offs: `PYTHONHASHSEED=0` is no longer required for cross-process determinism.

**User Approval**: Approved 2026-10-16.

---

### 2026-10-16: First-Seen Out-of-Domain Translation Latency Policy

**Context**: Unmapped cross-lingual queries incur a ~1.2–1.7s synchronous network roundtrip to Google Translate on their first arrival. Once translated, `TranslatorCache` caches positive hits for 24h and failures for 60s, dropping subsequent response times to ~129 ms (-90.0%). We evaluated whether to introduce aggressive pre-translator heuristics (e.g. blocklists/OOV filters) or shorten timeouts vs. accepting cold translation latency.

**Decision**: Formally accept cold translation network latency (~1.3s) as an acceptable one-time cold cost. Preserve current `TranslatorCache` parameters (5.0s timeout, 24h positive TTL, 60s negative TTL, `max_workers=2`). Reject pre-translator heuristics/blocklists.

**Rationale**:
1. **Academic Cross-Lingual Recall**: The search engine is an academic exploration tool for Jose Rizal's novels. Historical and literary English concepts not found in the small 30-word static dictionary `EN_TL_DICTIONARY` (e.g., "tyranny", "martyrdom", "friar abuses", "rebellion", "indio", "exile", "censorship") depend on dynamic translation to bridge into Rizal's Tagalog vocabulary. Local heuristics or blocklists risk false positives that would permanently block valid research queries.
2. **Effective Amortization**: In steady state, warm queries are served in ~129.7 ms. Negative caching (60s) prevents repeated abuse or thread exhaustion from invalid terms.
3. **Zero Snapshot Drift**: Preserves 100% bit-identical consistency with `golden_snapshot_deterministic.json` and ensures all 9 items of Phase B6 verification evidence remain valid without metadata alterations.

**Impact**:
- Performance: First-seen cold out-of-domain latency remains ~1.3s; warm repeated queries remain ~129 ms.
- Correctness: 0 ranking changes; 0 snapshot diffs; 100% determinism preserved.
- Trade-offs: Novel out-of-domain terms incur a one-time network latency penalty on their initial search.

---

## Pending Decisions

### 1. ~~Determinism Enforcement via `sorted()` and `ORDER BY` (Step B)~~ ✅ RESOLVED
- **Status**: **Approved and committed** (2026-10-16). See decision entry above.
- **Resolution**: All 5 commits (B1, D, E1, E2, E3) applied. 100% bit-identical determinism verified across 6 processes and 206 held-out queries. `PYTHONHASHSEED=0` no longer required. New permanent reference: `golden_snapshot_deterministic.json`.

### 2. ~~First-Seen Out-of-Domain Translation Latency Policy~~ ✅ RESOLVED
- **Status**: **Approved and adopted** (2026-10-16).
- **Decision**: Accept cold external translation network latency (~1.3s) as an acceptable one-time cold cost, amortized by `TranslatorCache` (24h positive TTL, 60s negative TTL, 5.0s request timeout with `max_workers=2`). Reject aggressive pre-translator OOV / dictionary heuristics.
- **Rationale**:
  1. **Academic Cross-Lingual Recall**: The search engine serves research on Jose Rizal's novels. Legitimate historical/academic English terms not present in the small static `EN_TL_DICTIONARY` (e.g., "tyranny", "martyrdom", "friar abuses", "rebellion", "indio") must be translated dynamically into Tagalog to discover novel passages. Pre-translator heuristics risk false-positive rejections.
  2. **Amortization in Steady-State**: `TranslatorCache` slashes warm latency by 90.0% (from ~1,298 ms down to ~129.7 ms). Unmapped terms are negative-cached for 60s to prevent spam/DoS.
  3. **Preservation of Determinism**: Zero changes to search results, metadata reasons, or `golden_snapshot_deterministic.json`. 100% of Phase B6 evidence items remain bit-identical.

---

### 2026-10-16: Production Deployment Sizing & Host Configuration

**Context**: Public and production deployment of the optimized search service requires explicit server memory budgets, worker process limits, and artifact packaging procedures to prevent out-of-memory crashes and guarantee fast first-seen search response times.

**Decision**: Formally establish the host configuration and sizing rules:
1. **Memory Budget**:
   - Initial boot RSS: ~460 MB per worker.
   - Dual-model allocation (Base XLM-R + DAPT): ~2.2 GB commit memory per worker.
   - Steady-state working memory under inference: ~1.2 GB to ~1.5 GB RSS per worker.
   - Dedicated RAM sizing: Minimum 4 GB RAM for 1 worker; 8 GB RAM for 2 workers; 16 GB RAM for 4–6 workers.
2. **Worker Concurrency Formula**:
   - `NUM_WORKERS = max(1, min(CPU_CORES, floor((TOTAL_RAM_GB - 2.5GB_RESERVED) / 1.5GB)))`
   - Production launch: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers $NUM_WORKERS`
3. **Cache Packaging**:
   - `backend/app/data/word_vec_cache_v1.pkl` (73.39 MB, 24,770 words) must be generated via `python scripts/build_word_vec_cache.py` or copied from release storage during CI/CD build.
   - At startup, the engine validates the vocabulary SHA-256 hash (`6efe3301743eed6b7552291a9c444e7a4ea426cd5175d89ca9d8b648b1c65569`) in 238 ms. If missing, it falls back gracefully to on-demand encoding.

### 2026-10-16: Item A — Themes Endpoint Optimization (Process Cache, LATERAL Join, Cache-Control)

**Context**: In production on Hugging Face Spaces (US) connected to Supabase (Singapore), `GET /api/v1/themes` took ~10.5 seconds (`TTFB - time_appconnect` ≈ 10,115 ms). Profiling revealed that `get_themes()` executed 26 sequential SQL queries: 1 query to fetch all 545 theme rows, followed by a sequential Python loop across all 25 deduplicated Tagalog titles executing individual `Sentence.embedding.cosine_distance(theme_obj.embedding)` vector scans. Each remote roundtrip took ~400 ms.

**Decision**:
1. **Set-Based LATERAL SQL Query**: Replaced the 25 sequential vector queries with a single set-based LATERAL join in PostgreSQL:
   ```sql
   WITH unique_t AS (
       SELECT DISTINCT ON (tagalog_title) id, tagalog_title, meaning, embedding
       FROM themes
       ORDER BY tagalog_title, id
   )
   SELECT u.id, u.tagalog_title, u.meaning, s.book, s.chapter_number, s.chapter_title, s.sentence_text, s.sentence_index
   FROM unique_t u
   LEFT JOIN LATERAL (
       SELECT s.book, s.chapter_number, s.chapter_title, s.sentence_text, s.sentence_index
       FROM sentences s
       WHERE u.embedding IS NOT NULL
       ORDER BY s.embedding <=> u.embedding, s.id
       LIMIT 1
   ) s ON true
   ORDER BY u.id
   ```
   Deterministic secondary tie-breakers (`s.id`) are enforced on both unique theme selection and sentence matching.
2. **Process-Level Memory Cache**: Added thread-safe `_themes_cache` with double-checked locking in `backend/app/api/v1/content.py`. Because theme and novel sentence data are static, the query runs at most once per worker process lifecycle.
3. **HTTP Cache-Control**: Added `Cache-Control: public, max-age=3600, stale-while-revalidate=86400` so intermediate CDNs and client browsers cache the response.
4. **Resilient Dual Fallback**: Preserved sequential query execution as Fallback 1, and added static JSON bundle fallback (`backend/app/data/themes_cache.json`) as Fallback 2 if database connectivity is disrupted.

**Impact & Verification**:
- Output Equivalence: Verified against `docs/perf/evidence/themes_before.json`. All 25 themes map to the exact same sentence IDs, books, and chapter numbers.
- SQL Queries: Cold queries slashed from 26 to 1 (-96.2%); warm queries slashed to 0 (-100%).
- Latency: Local cold request executes in 2,285 ms; local warm requests drop to 0.013 ms mean (0.003 ms median).
- Search Invariance: Verified against `docs/perf/golden_snapshot_deterministic.json`: 0 position changes across all 40 golden queries.

---

## Pending Decisions

### 1. ~~Determinism Enforcement via `sorted()` and `ORDER BY` (Step B)~~ ✅ RESOLVED
- **Status**: **Approved and committed** (2026-10-16).

### 2. ~~First-Seen Out-of-Domain Translation Latency Policy~~ ✅ RESOLVED
- **Status**: **Approved and adopted** (2026-10-16).

### 3. ~~Production Deployment Sizing & Host Configuration~~ ✅ RESOLVED
- **Status**: **Approved and finalized** (2026-10-16). See specifications above.

### 4. ~~Themes Endpoint Optimization (Item A)~~ ✅ FULLY RESOLVED
- **Status**: **Fully implemented and verified** (2026-10-16).
- **Backend**: Set-based LATERAL join, process cache, Cache-Control header, graceful dual fallback, non-blocking startup warmer in FastAPI lifespan (94.55 ms yield).
- **Live Supabase Parity**: Byte-identical match against production Supabase DB (SHA-256: `a11ef242257780d624d3ba9924a558a31e971a4034eea64ff683ad91b6f9c5c5`).
- **Frontend**: TanStack React Query (`useQuery` with 24h `staleTime`), homepage mount prefetch, and rich 8-card responsive skeleton state. Clean production build verified.

### 5. Deliberately Unexecuted Optimizations
- **Status**: Held / excluded from scope.
- **Items**:
  - **HNSW / ANN Vector Indexing**: Exact vector similarity search is preserved to prevent recall degradation on academic query sets.
  - **pg_trgm GIN Indexing**: Existing ILIKE query performance was deemed sufficient given that candidate retrieval takes <25 ms.
  - **Query Model Encoding Batching**: Sequential calls to `base_model.encode()` during preprocessing remain unbatched to preserve existing function call boundaries and avoid subtle tensor-padding floating-point divergence.




