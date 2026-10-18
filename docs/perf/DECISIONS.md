# Performance Optimization Decisions

**Purpose**: Log all decisions that could affect search behavior, ranking, or correctness.

---

## Decision Index

| Item | Date | Category | Title | Status |
|:---:|:---:|:---|:---|:---:|
| - | 2026-10-12 | Planning | Initial Planning & Measurement-First Strategy | Historical |
| - | 2026-10-12 | Diagnostics | Phase 1 Instrumentation & Baseline Corrections | Historical |
| - | 2026-10-13 | Diagnostics | Phase 2 Step 1: Word Embedding Cache & Hash Randomization | Historical |
| - | 2026-10-14 | Diagnostics | Benchmark Protocols, PerfTimer, TranslatorCache & Context Batching | Historical |
| - | 2026-10-14 | Optimization | Step A: Startup Vocabulary Precomputation (`word_vec_cache_v1.pkl`) | Adopted |
| - | 2026-10-16 | Core Search | Step B: Search Pipeline Determinism (Approved) | Deployed |
| - | 2026-10-16 | Policy | First-Seen Out-of-Domain Translation Latency Policy | Adopted |
| - | 2026-10-16 | Architecture | Production Deployment Sizing & Host Configuration | Finalized |
| - | 2026-10-16 | Backend | Item A: Themes Endpoint Optimization (LATERAL Join & Memory Cache) | Deployed |
| **1** | 2026-10-16 | Core Search | Determinism Enforcement via `sorted()` and `ORDER BY` (Step B) | Deployed |
| **2** | 2026-10-16 | Policy | First-Seen Out-of-Domain Translation Latency Policy | Adopted |
| **3** | 2026-10-16 | Architecture | Production Deployment Sizing & Host Configuration | Finalized |
| **4** | 2026-10-16 | Backend | Themes Endpoint Optimization (Item A) | Deployed |
| **5** | 2026-10-16 | Backend | Chapters Endpoint Optimization (Item A2) | Deployed |
| **6** | 2026-10-16 | Backend | Precomputed Chapter Content & Corpus Fingerprint (Step 1) *(was Item 7)* | Deployed |
| **7** | 2026-10-16 | Architecture | Deliberately Unexecuted Optimizations *(was Item 8)* | Archived |
| **8** | 2026-10-17 | UI / Mobile | Item D: Mobile Fullscreen Architecture Refinements *(was Item 9)* | Deployed |
| **9** | 2026-10-17 | UI / Mobile | Item E: Background Image Rotation and Mobile Viewport Scaling *(was Item 10)* | Deployed |
| **10** | 2026-10-17 | UI / Styling | Item F: Dark Mode Architecture and WCAG AA Contrast Compliance *(was Item 11)* | Deployed |
| **11** | 2026-10-17 | UI / Styling | Dark Mode Contrast, Modal Backdrop, and Aesthetic Refinements *(was Item 12)* | Deployed |
| **12** | 2026-10-17 | Backend | Item C Phase 2: Sanggunian Cross-Novel Reference Optimization *(was Item 13)* | Deployed |
| **13** | 2026-10-17 | Devops | Item B: Keep-Alive Workflow, Health DB Probe, and Warmup Banner *(was Item 14)* | Deployed |
| **14** | 2026-10-18 | UI Polish | UI Polish Batch & Follow-Up (`fix/ui-polish`) *(was Item 15)* | Deployed |
| **15** | 2026-10-18 | Feature | Item 1: Sanggunian in Buong Kwento Mode Shows Matching Buod *(was Item 16)* | Deployed |
| **16** | 2026-10-18 | Release | Merge All UI Work, Item 1, & Batches 2–4 to Main *(was Item 17)* | Deployed |
| **17** | 2026-10-18 | A11y | Accessibility Follow-up Batch (Axe-Core WCAG AA Compliance) *(was Item 18)* | Deployed |
| **18** | 2026-10-18 | UI / Novel | Fili Dark Theme Cards Per-Novel Tokens & Merge | Deployed |

> **Numbering Gap Note**: Historical gap between Item 5 and Item 7 in the numbered optimization section has been normalized. The mapping above maps each item to its historical and new sequential index.

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

## Optimization & Modernization Decisions (Items 1–18)

### 1. ~~Determinism Enforcement via `sorted()` and `ORDER BY` (Step B)~~ ✅ RESOLVED
- **Status**: **Approved and committed** (2026-10-16).

### 2. ~~First-Seen Out-of-Domain Translation Latency Policy~~ ✅ RESOLVED
- **Status**: **Approved and adopted** (2026-10-16).

### 3. ~~Production Deployment Sizing & Host Configuration~~ ✅ RESOLVED
- **Status**: **Approved and finalized** (2026-10-16). See specifications above.

### 4. ~~Themes Endpoint Optimization (Item A)~~ ✅ FULLY RESOLVED & DEPLOYED
- **Status**: **Deployed to production HF Space and Vercel, verified live** (2026-10-16).
- **Hugging Face Rollback Target**: `09cdb417c1853a1f0bdd480e47ba6e4a2f7f0b0d` (`remotes/hf/main`). Deployed commit on `hf`: `0b74a1d`. Merged to `origin/main`: `7330b81`.
- **Live Verification**:
  - `GET /health` = `200 OK`.
  - `GET /api/v1/themes` byte-identical SHA-256: `a11ef242257780d624d3ba9924a558a31e971a4034eea64ff683ad91b6f9c5c5`.
  - Live server latency: Mean **269.36 ms** (down from ~10,115 ms, -97.3%).
  - Search invariance: 5 golden queries verified 100% invariant against `golden_snapshot_deterministic.json`.

---

### 5. Chapters Endpoint Optimization (Item A2)
**Date**: 2026-10-16  
**Branch**: `perf/chapters-endpoint`  
**Status**: **Implemented and verified locally against production Supabase DB; awaiting deployment approval**.

**Context**:
Chapter retrieval (`/api/v1/chapters/{book}/{chapter_number}`) had high cold latency due to an N+1 query loop: each sentence was querying context individually. Furthermore, requests were not cached in memory, and the frontend fetched data using raw `fetch()` on every interaction.

**Decisions**:
1. **Reuse `engine._fetch_context_cache` Without Modifying `engine.py`**:
   `backend/app/core/engine.py` has a strict zero-diff requirement. In `backend/app/api/v1/content.py`, we call `context_cache = engine._fetch_context_cache(db, unique_sentences, window=8)` and pass `context_cache=context_cache` into `engine._classify_themes(...)`. This eliminates the sentence-by-sentence DB query loop completely.
2. **Bounded LRU Cache for Chapter Content**:
   Added a thread-safe `OrderedDict` LRU cache bounded to `MAX_CHAPTER_CONTENT_CACHE_ENTRIES = 64`.
   - Measured memory footprint: Average buod chapter is 3.5 KB; average full chapter is 41.6 KB; max tested (Noli 34 full) is 42.6 KB. Max 64 entries consume ~1.11 MB, which is completely safe for deployment.
   - Warm request latency: Drops from ~2,000 ms to **2.7–7.0 ms** (-99.7%).
3. **Process-Level Cache & Startup Warmer for Chapter List**:
   Added lock-protected `_chapters_cache` for `/api/v1/chapters` with bundled fallback files (`chapters_cache_buod.json` and `chapters_cache_full.json`). Background thread `warm_chapters_cache()` warms both modes during startup.
4. **HTTP Cache-Control**:
   Added `Cache-Control: public, max-age=86400, stale-while-revalidate=604800` (24h edge, 7d stale-while-revalidate).
5. **Frontend React Query & Prefetching**:
   - `ChapterGrid.tsx`: Migrated to `useQuery({ queryKey: ['chapters', mode], staleTime: 24h })` with responsive 8-card skeleton state.
   - `page.tsx`: Prefetches chapter lists on homepage mount; uses `useQuery` for modal content; prefetches $X+1$ and $X-1$ in `buod` mode. Heavy full-mode chapters are intentionally not prefetched to avoid mobile bandwidth bloat.
   - `ChapterModal.tsx` and `search/page.tsx`: Mode switches and fullscreen navigation read and populate React Query cache via `queryClient.fetchQuery`.

**Verification**:
- 10 `buod` chapters and 3 `full` chapters tested: 200 OK, byte-identical cold vs warm.
- **Gate 1 Comprehensive Old-vs-New Parity Check (2026-10-16)**:
  - Tested 18 chapters across `main` (old path, port 8001) vs `perf/chapters-endpoint` (new path, port 8000) against production Supabase DB: 100% byte-identical SHA-256 matches across all 18 chapters (10 buod + 3 full [including Noli 34 full and Noli 1 full] + 5 additional random buod [Noli 12, Noli 25, El Fili 15, El Fili 22, El Fili 35]).
  - All 40 golden search queries verified vs `golden_snapshot_deterministic.json`: 0 position changes, 0 score deltas, 0 theme changes (100% invariant).
  - `backend/app/core/engine.py`: **0 diff**.
  - **Hugging Face Rollback Target**: `0b74a1d8e605510fc6945815e22179b2146b6099` (`remotes/hf/main`).
- `npm run build`: Succeeded with 0 errors.

---

---

### 6. Precomputed Chapter Content & Corpus Fingerprint Verification (Step 1)
**Date**: 2026-10-16  
**Branch**: `perf/chapter-precompute`  
**Status**: **Implemented, verified locally against production Supabase DB; awaiting deployment approval**.

**Context**:
While Item A2 eliminated the N+1 query loop for context fetching, cold requests for large full-novel chapters (e.g. Noli 34 with 245 sentences) still required 22–30 seconds of heavy CPU in-process keyword matching and regex word extraction across 510 themes.

**Decisions**:
1. **Precomputed Static Bundles**:
   Precomputed all 205 chapters across both novels and both modes using the unmodified live production code path (`scripts/build_chapter_cache.py`):
   - `chapter_cache_noli_buod.json`: 64 chapters (201.9 KB)
   - `chapter_cache_noli_full.json`: 63 chapters (1,698.2 KB)
   - `chapter_cache_elfili_buod.json`: 39 chapters (163.9 KB)
   - `chapter_cache_elfili_full.json`: 39 chapters (1,130.6 KB)
   - Total payload: **3.12 MB** uncompressed. Small enough to be tracked cleanly in git and loaded in <15 ms.
2. **First-Level Static Cache with In-Memory LRU Fallback**:
   In `backend/app/api/v1/content.py`, `get_chapter_content` checks the static bundle first. If enabled, it returns the precomputed sentence list in **0 ms** (sub-10 ms HTTP round-trip). If disabled or on a cache miss, it seamlessly falls through to the Item A2 bounded LRU cache and live computation path.
3. **Corpus Fingerprint & Engine Hash Verification at Startup**:
   Every bundle embeds the Supabase database corpus fingerprint and `engine.py` content hash:
   - `sentence_count`: 16,402
   - `sentence_text_md5`: `e438ce3f61098624121329cd5a20f9a2`
   - `theme_count`: 545
   - `theme_content_md5`: `cd87434aff5cd54a775ee7b4ef60b056`
   - `engine_hash`: `31789dca625abcbc1a38adf58c7f0bcf132cb9e5153661b437d882c57b0e5dd6`
   - `generated_at`: ISO UTC timestamp
   On app startup, `warm_chapters_cache()` executes a fast (~1.5s) single SQL query against Supabase.
   - **Fingerprint Match**: Static cache is enabled.
   - **Fingerprint Mismatch**: Database reached, but corpus was modified. Logs a loud warning, disables the static bundle, and falls back to live computation.
   - **Database Connectivity Failure**: Database unreachable. Logs a warning and serves the precomputed static bundle as a resilient fallback.
   - **`engine.py` Content Hash Mismatch**: Logs a loud warning reminding that any change to `_classify_themes` requires regenerating the bundles.
   - **DOES NOT CRASH**.
4. **Regeneration Procedure When Corpus or Engine Changes**:
   When sentences/themes in Supabase are updated or `_classify_themes` in `engine.py` is modified, regenerate the bundles:
   ```bash
   poetry run python scripts/build_chapter_cache.py [--base-url http://127.0.0.1:8000]
   ```
   The script discovers chapter counts dynamically from `/api/v1/chapters?mode={mode}`, recalculates the database fingerprint, saves checkpoints during generation to resume on interruption, and outputs formatted bundles to `backend/app/data/`.
5. **Zero Diff on `engine.py`**:
   `backend/app/core/engine.py` has strictly zero diff. All 40 golden search queries are 100% invariant (0 diffs).

**Gate 1 Verification Results (2026-10-17)**:
- **Full Sequential Parity (205 combinations)**:
  - Both `main` (port 8001, live compute) and `perf/chapter-precompute` (port 8000, static bundle) queried sequentially over HTTP for all 205 chapter/mode combinations.
  - Evidence recorded in `docs/perf/evidence/chapter_parity_205.txt`.
  - **Mismatches: 0 / 205**.
  - Full 64-character SHA-256: **100% MATCH** across all 205 combinations.
  - Byte counts: **100% MATCH**.
  - Headers (`Content-Type: application/json` and `Cache-Control: public, max-age=86400, stale-while-revalidate=604800`): **100% MATCH**.
- **Edge Case Verification (`noli/64?mode=full`)**:
  - Both `main` (port 8001) and `perf/chapter-precompute` (port 8000) return `HTTP 404 Not Found` with `{"detail":"Chapter 64 not found for noli in full mode"}`.
- **Memory & Image Size Impact**:
  - Process RSS before loading bundles: **17.33 MB**
  - Process RSS after loading all 4 bundles: **25.50 MB** (+8.17 MB RAM delta).
  - HF Docker Image Size Delta: **3.12 MB** uncompressed (~820 KB compressed git layer).
- **Fallback Paths**:
  - `scripts/test_fallback_paths.py` verified:
    1. Fingerprint mismatch -> static bundle disabled, live computation active.
    2. DB connection failure -> static bundle safely served as resilient fallback.
- **Search Invariance**: `scripts/run_golden_comparison_40.py` verified across all 40 golden queries vs `golden_snapshot_deterministic.json`: **0 differences**.
- **HF Space Rollback Target**: `007116d9c9cf2331639c650a2d8697b04ebf32cf` (`007116d`).

---

### 7. Deliberately Unexecuted Optimizations
- **Status**: Held / excluded from scope.
- **Items**:
  - **HNSW / ANN Vector Indexing**: Exact vector similarity search is preserved to prevent recall degradation on academic query sets.
  - **pg_trgm GIN Indexing**: Existing ILIKE query performance was deemed sufficient given that candidate retrieval takes <25 ms.
  - **Query Model Encoding Batching**: Sequential calls to `base_model.encode()` during preprocessing remain unbatched to preserve existing function call boundaries and avoid subtle tensor-padding floating-point divergence.

---

### 8. 2026-10-17: Item D Mobile Fullscreen Architecture Refinements
- **Context**: Mobile fullscreen reader in `ChapterModal.tsx` suffered from clipping under mobile browser URL bars, missing "Tauhan" and "Sanggunian" controls due to desktop-only sidebar layout (`hidden lg:flex`), cramped text padding (`px-8`), and horizontal blowout in the cross-reference split-screen overlay.
- **Decision**:
  1. Use fixed overlay `h-[100dvh]` with `overscroll-contain` instead of relying on the native HTML5 Fullscreen API (which is unsupported on iOS Safari for generic `div` elements and causes browser address-bar jumpiness).
  2. Implement a dedicated sticky mobile header in fullscreen mode providing notch-safe top padding (`pt-[max(0.75rem,env(safe-area-inset-top))]`), minimum 44px tap targets for Minimize and Close actions, and prominent toggle pills for "Tauhan" and "Sanggunian".
  3. Redesign the Kaugnay na Sanggunian overlay (`refFullscreen`) for mobile: eliminate stacked nested scroll containers and replace with a unified vertical layout featuring a prominent top card for "Pinagmulang Teksto" (source sentence) with quote badge, clear directional connector, and the matched "Nahanap na Katugma" card below. Preserved desktop side-by-side split screen (`hidden lg:flex`).
- **Impact**:
  - Correctness / Ranking: **0 impact** (strictly frontend presentation).
  - Search Pipeline / Engine: **0 diff** on backend.
  - Mobile UX: Native-feeling fullscreen reading experience and fully readable cross-novel reference viewer without nested scroll traps or squished text.

---

### 9. 2026-10-17: Item E Background Image Rotation and Mobile Viewport Scaling
- **Context**: The novel hero backgrounds (`noli_background.webp` and `elfili_background.webp`) were originally exported rotated 90 degrees counter-clockwise (height > width), displaying all artwork (Rizal, church, characters, novel scrolls) sideways. Additionally, `style={backgroundStyle}` with `background-size: cover` was attached directly to `<main>` in `page.tsx` and search container in `search/page.tsx`. Because `<main>` spans all 64 chapters and sections (~5,924px scroll height), the 1308x736 image expanded to over **10,500px wide (8x / 800% zoom)**, causing narrow viewports to display an extreme microscopic crop of the center scroll. On mobile portrait, widescreen 16:9 artwork scaled with `cover` cropped off 65% of the sides and clashed with hero text. Furthermore, an internal duplicate raster layer (`scale: 1.05`, `blur(4px)`) in `NovelBackground.tsx` sat inside the hero section, blurring out and double-scaling the background.
- **Decision**:
  1. Re-export the source WebP images rotated 90° clockwise to their upright landscape orientation (1308x736 and 1200x675) with lossless-grade WebP encoding.
  2. Decouple background rendering from document scroll height by introducing a fixed viewport backdrop (`.novel-backdrop-layer` with `position: fixed; inset: 0; pointer-events: none; -z-10`) in `page.tsx` and `search/page.tsx`, eliminating the 8x zoom caused by document height.
  3. Implement responsive media-query framing in `globals.css`:
     - **Mobile Portrait (<768px)**: Clean antique parchment texture (`background_1.webp`) with warm amber (Noli) or rich crimson (Fili) ambient radial glow, giving 100% text legibility with zero oversized text clash.
     - **Desktop & Tablet (>=768px)**: Full panoramic upright landscape collage (`noli_background.webp` / `elfili_background.webp`) framing church on the left and characters on the right with a soft 86%-98% parchment wash.
  4. Clean up `NovelBackground.tsx` by removing the redundant blurry raster layer (`scale: 1.05`, `blur(4px)`), allowing the clean viewport backdrop and vector ambient accents to render sharply.
  5. Clean up `globals.css` to link directly to existing asset paths (`background_1.webp`, `noli_background.webp`, etc.).
- **Impact**:
  - Correctness / Ranking: **0 impact** (backend untouched).
  - Performance: Faster page loads by eliminating 404 image errors for missing textures.
  - Visual: Upright, sharp artwork across desktop and mobile devices without oversized or zoomed-in elements.

---

### 10. 2026-10-17: Item F Dark Mode Architecture and WCAG AA Contrast Compliance
- **Context**: The web application lacked dark mode support, defaulting to high-brightness parchment and white surfaces that caused glare in dim environments. Next.js 16 with Tailwind CSS v4 required an explicit variant definition (`@custom-variant dark (&:is(.dark, .dark *));`) to enable standard `dark:` utility class selection on both `html.dark` and all nested child components. Multiple legacy components contained hardcoded inline `style={{ backgroundColor: 'rgba(255,255,255,...)' }}` (such as `HeroSection` floating icons and stats card), preventing CSS dark mode class overrides. Text elements across result cards, chapter grid, and modals required auditing to ensure WCAG AA contrast ratio (>= 4.5:1 for body, >= 3:1 for large headings) against midnight dark backgrounds.
- **Decision**:
  1. Integrate `next-themes` with `ThemeProvider` (`attribute="class"`, `defaultTheme="system"`, `enableSystem`) in `frontend/app/providers.tsx` and add `suppressHydrationWarning` to `<html>` in `frontend/app/layout.tsx`.
  2. Implement `@custom-variant dark (&:is(.dark, .dark *));` in `globals.css` ensuring seamless Tailwind v4 compilation across all React components.
  3. Create `ThemeToggle.tsx` supporting both an icon toggle for desktop headers and an accessible 3-way segmented control (Liwanag / Dilim / Sistema) for mobile drawers and setting panels.
  4. Establish comprehensive dark mode design tokens in `globals.css`:
     - Deep midnight navy base (`--theme-bg: #0F141C;`, `--theme-card: #18202C;`, `--theme-text: #EAE6DF;`, etc.).
     - Novel specific dark accents (Noli: `#E8C97A` gold, Fili: `#E28498` magenta/rose).
     - Dark mode fixed backdrops: midnight parchment wash for mobile portrait, and deep midnight landscape collage for desktop/tablet.
     - Dark mode gold drop cap (`.dark .drop-cap::first-letter: #E8C97A`) and dark scrollbars.
  5. Audit and upgrade components for WCAG AA contrast compliance:
     - `ChapterModal.tsx`: Dark mode modal shell, chapter text reading canvas, Prev/Next buttons, and full Sanggunian reference overlay.
     - `ChapterGrid.tsx`: Card surfaces (`dark:bg-brand-paper/85`), decorative chapter numerals, and headings (`dark:text-foreground`).
     - `ResultCard.tsx`: Card titles (`dark:text-foreground`), meaning/themes cards (`dark:bg-brand-paper`).
     - `NovelToggle.tsx`: Novel switcher dropdown and pill buttons (`dark:bg-brand-paper/98`, `dark:text-brand-gold`).
     - `HeroSection.tsx`: Replaced hardcoded inline white styles with responsive Tailwind dark classes (`bg-white/90 dark:bg-brand-paper/95`, `dark:text-foreground`).
- **Impact**:
  - Correctness / Ranking: **0 impact** (backend completely untouched).
  - Search Pipeline / Engine: **0 diff** on `backend/app/core/engine.py`.
  - Accessibility & UX: Full WCAG AA compliant dark mode reading experience, system preference matching, zero flash of incorrect theme, and beautiful midnight gold/crimson historical aesthetic.

---

### 11. 2026-10-17: Dark Mode Contrast, Modal Backdrop, and Aesthetic Refinements
- **Context**: Initial dark mode deployment left several contrast and color mismatch issues:
  1. Opening any chapter or theme card displayed a bright yellow/gold overlay backdrop because `bg-brand-navy/70` mapped to gold (`#E8C97A` at 70% opacity) in dark mode.
  2. Inside `ChapterModal.tsx`, active "Tauhan" and "Sanggunian" buttons had `dark:text-brand-navy` on `bg-brand-gold`, creating invisible yellow-on-yellow text.
  3. The theme detail modal (`ItemModal.tsx`) had light definition boxes that clashed with the midnight theme, and fullscreen chapter reading had a washed-out grey sidebar (`bg-[#F5F2EB]/50`).
  4. Mobile ContentTabs and horizontal Kabanata group pills suffered from poor active/inactive contrast.
  5. The Noli homepage had an overpowering white radial wash and high-opacity SVG sky shimmer.
  6. The floating "Sa Itaas" (Back to Top) button and "Buod / Buong Kwento" mode toggles rendered with bright yellow backgrounds and low-contrast white/gold text in dark mode.
- **Decision**:
  1. Standardize modal backdrops across `ChapterModal.tsx` and `ItemModal.tsx` to `bg-black/75 dark:bg-black/85 backdrop-blur-sm`, eliminating all light wash overlays.
  2. Fix button contrast by pairing active gold backgrounds (`bg-brand-gold`) strictly with dark navy text (`text-[#1B263B] font-bold` or `dark:text-[#0F141C] font-bold`) across `ChapterModal.tsx`, `ContentTabs.tsx`, `ChapterSidebar.tsx`, and `page.tsx`.
  3. Restyle `ItemModal.tsx` theme definition and contextual match cards to fit midnight dark surfaces (`dark:bg-black/40`, `dark:bg-[#1B2433]`, and `border-l-4 border-brand-gold`). Transform fullscreen reading sidebar to deep midnight navy (`dark:bg-[#121A28]`).
  4. Redesign mobile ContentTabs (`dark:bg-black/40`, active gold pill with dark text) and chapter range pills (`dark:bg-[#182232]/90` with gold borders).
  5. Reduce Noli homepage vignette opacity to `0.08` with `dark:opacity-0`, and adjust `#goldSky` stopColor to `#C5A065` at 15% opacity, eliminating the white fade.
  6. Restyle the floating "Sa Itaas" button to a sleek midnight navy pill (`dark:bg-[#182232]/95 dark:text-brand-gold dark:border dark:border-brand-gold/30 dark:hover:bg-[#202C3F]`) with backdrop blur.
  7. Restyle "Buod / Buong Kwento" toggles on homepage and search headers to dark midnight containers with high-contrast active and inactive states.
- **Impact**:
  - Correctness / Ranking: **0 impact** (strictly frontend styling).
  - Search Pipeline / Engine: **0 diff** on `backend/app/core/engine.py`.
  - Visuals & Ergonomics: Premium, high-contrast dark reading experience across all devices and modal dialogs.

---

### 12. 2026-10-17: Item C Phase 2 — Sanggunian Cross-Novel Reference Optimization (Fixes 1 + 2)
- **Context**: Profiling in Phase 1 (`docs/perf/evidence/sanggunian_profile.md`) revealed that loading Sanggunian cross-references for a chapter executed an N+1 HTTP loop from the frontend (`ChapterModal.tsx`), making up to 12-25 concurrent requests to `GET /api/v1/sentences/{id}/sanggunian`. Each request executed a cold DAPT model encoding and dynamic cosine alignment across all chapters, totaling 52.4s of backend compute per chapter and saturating HF Space worker threads.
- **Rollback Targets**:
  - New HF Space Rollback Target: `3117ead`.
  - Backend Rollback Target before this item: `2b15166`.
- **Scope Clarification**:
  - **Summary Mode (Cross-Novel ML Alignment)**: ML semantic alignment (`RobustAligner.align`) applies strictly to **2,186 summary sentences across 103 chapters** (Noli: 64 chapters, 1,212 sentences; El Fili: 39 chapters, 974 sentences). It compares each summary sentence against all sentences of the same chapter in full mode using DAPT embeddings.
  - **Full Mode (Passage Grouping)**: For full-mode sentences (14,216 sentences across 102 chapters; Noli: 63 chapters, 8,673 sentences; El Fili: 39 chapters, 5,543 sentences), `backend/app/api/v1/content.py` does NOT invoke `RobustAligner`. Instead, it groups sentences by `passage_id` (pre-assigned contiguous sentence cluster) and returns `has_reference=True, alignment_status='precise', score=1.0, reference_text=passage`.
  - **Purpose of Full-Mode Bundles (5.77 MB)**: The precomputed full-mode bundles (`sanggunian_cache_noli_full.json` [3.49 MB] and `sanggunian_cache_elfili_full.json` [2.28 MB]) exist strictly to eliminate the database roundtrip and passage grouping query, delivering 0ms static lookups.
- **Cold Path Latency Breakdown & Reconciliation**:
  - Fresh-process instrumentation of the actual endpoint cold path (`scripts/profile_cold_endpoint.py`, raw evidence in `docs/perf/evidence/sanggunian_endpoint_cold_breakdown.txt`):
    - **Noli 1** (22 buod, 223 full, 114 NULL full): SQL = 3,407.7 ms (4.2%), Encode = 18,597.3 ms (22.8%), RobustAligner = 59,570.7 ms (73.0%), Other = 1.3 ms, Total = **81,577.4 ms (~81.6 s)**.
    - **Noli 5** (9 buod, 19 full, 9 NULL buod): SQL = 330.0 ms (8.5%), Encode = 1,690.9 ms (43.5%), RobustAligner = 1,865.7 ms (48.0%), Other = 0.2 ms, Total = **3,886.9 ms (~3.9 s)**.
    - **El Fili 10** (21 buod, 171 full, 0 NULL): SQL = 2,429.1 ms (5.8%), Encode = 0.5 ms (0.0%), RobustAligner = 39,440.9 ms (94.2%), Other = 0.7 ms, Total = **41,871.3 ms (~41.9 s)**.
  - **Reconciliation Note**: Individual component timings (14–19s DAPT encoding for NULL rows, 39–60s $M \times N$ matrix alignment on CPU) are measured and verified. However, **the CPU-saturation explanation of the observed 20–52 s cold times under concurrency is an inference, not verified**.
- **Parity with Old Live Backend**:
  - **Summary Mode Parity**: Reference sentence IDs, reference texts, and `alignment_status` match 100% (2,186 / 2,186 sentences across all 103 chapters). Summary scores match the old backend except float32 noise $\le 3.0 \times 10^{-7}$ (1,829 / 2,186 differ by sub-micro float32 precision across CPU architectures; 357 are exact; 0 structural differences).
  - **Full Mode Parity**: Tested a random sample of 500 full-mode sentences; all **500 / 500 were 100% bit-identical in raw HTTP bytes SHA-256**.
- **Operational Note (Fingerprint Invalidation & Regeneration)**:
  - The cryptographic fingerprint in each bundle hashes the entire contents of `backend/app/api/v1/content.py` and `backend/app/core/robust_aligner.py` (with LF line-ending normalization).
  - Therefore, **any edit to `content.py` or `robust_aligner.py` will invalidate the bundle fingerprint and force the backend to fall back to live compute**.
  - Any code change to `content.py` requires rebuilding the precomputed bundles before deployment via:
    ```bash
    poetry run python scripts/build_sanggunian_cache.py
    ```
- **Decisions**:
  1. **Precompute Static Chapter Bundles (`scripts/build_sanggunian_cache.py`)**:
     - Precomputed all cross-novel alignments offline using the unmodified live engine logic for all 205 chapters (16,402 sentences: 2,186 summary ML aligned + 14,216 full passage grouped).
     - Bundles saved in `backend/app/data/`: `sanggunian_cache_noli_buod.json` (1.61 MB), `sanggunian_cache_noli_full.json` (3.49 MB), `sanggunian_cache_elfili_buod.json` (1.21 MB), and `sanggunian_cache_elfili_full.json` (2.28 MB). Total size: **8.39 MB**.
     - Embedded cryptographic fingerprint: sentence count (16,402), `sentence_text` MD5, count of NULL `embedding_dapt` rows (4,580), `embedding_dapt` column MD5, SHA-256 of `robust_aligner.py`, SHA-256 of alignment logic in `content.py`, and DAPT model name/revision (`mematello/rizal-xlm-r-dapt:main`).
  2. **Non-blocking Startup Verification & Fail-Safe Fallbacks**:
     - Bundles loaded at module import (`load_static_sanggunian_cache()`), setting `_static_sanggunian_enabled = True` immediately to serve requests with 0ms bundle lookups during startup.
     - Lifespan background task `warm_sanggunian_cache()` calls `verify_sanggunian_fingerprint(db)` (~16.14s runtime against Supabase).
     - The `embedding_dapt` MD5 is computed directly inside PostgreSQL (`md5(string_agg(...))`), transferring zero embedding vectors over the network.
     - Fingerprint match: Keeps static bundle enabled.
     - Fingerprint mismatch: Logs loud warning, flips `_static_sanggunian_enabled = False`, falls back to live compute.
     - DB unreachable: Retains static bundle as resilient fallback, logs warning, never crashes.
     - Added `_alignment_compute_lock` to serialize concurrent live alignment compute races on cache misses.
  3. **Chapter-Level Endpoint & HTTP Cache Headers**:
     - Added `GET /api/v1/chapters/{book}/{chapter_number}/sanggunian?mode={mode}` returning mapping of all sentence references in one request.
     - Added header `Cache-Control: public, max-age=86400, stale-while-revalidate=604800`.
     - Maintained `GET /api/v1/sentences/{id}/sanggunian` behavior and response shape unchanged, serving from `_sentence_to_sanggunian_cache` index when enabled.
  4. **Frontend React Query & Prefetching (`ChapterModal.tsx`)**:
     - Replaced per-sentence fetch loop with `useQuery({ queryKey: ['sanggunian', book, currentChapterNum, mode], staleTime: 24h })`.
     - Prefetches active chapter and adjacent chapters in buod mode on modal open.
     - Preserves existing fallback loop to per-sentence endpoint if chapter endpoint fails.
- **Footprint and Startup Impact (Evidence: `docs/perf/evidence/sanggunian_footprint_startup.txt`)**:
  - Process RSS before loading bundles: **427.79 MB**
  - Process RSS after loading all 4 bundles: **455.61 MB** (RSS Memory Delta: **+27.81 MB**; load time: 151.81 ms).
  - Docker Image Size Delta: **+8.39 MB** (8,797,554 bytes uncompressed across all 4 JSON bundles).
  - Fingerprint verification runtime: **16.14 seconds** (executed asynchronously in background thread; requests served from static bundle during verification).
- **Impact & Verification**:
  - **Latency**: Chapter endpoint latency dropped from **52,400 ms to 1.6–4.0 ms warm (2.7–20.7 ms cold)** (-99.99% latency reduction).
  - **Parity**: **205/205 chapters identical (0 mismatches)** in raw HTTP response bytes.
  - **Score Delta**: Max absolute score delta on the 5 profiled sentences between local and live HF Space: **0.00000030** (disclosed).
  - **Fallbacks Tested**: All 3 resilience cases (mismatch, missing bundle, DB offline) verified via `scripts/test_sanggunian_fallbacks.py`.
  - **Golden Query Invariance**: All 40 search queries 100% invariant vs `golden_snapshot_deterministic.json` (0 diffs).
  - **Engine Diff**: `backend/app/core/engine.py` diff is **STRICTLY 0 DIFF**.
    - Gate 3 Verification (Live Vercel Deployment): **Completed & Verified live on https://rizalengine.vercel.app**.
      - Merged `perf/sanggunian` into `main`, pushed to GitHub `origin/main` (`51b0476`).
      - New HF Space Rollback Target: `3117ead`.
      - Noli 1 Buod verified: exactly 1 request to `/api/v1/chapters/noli/1/sanggunian?mode=buod` (0 individual sentence requests); click-to-superscripts rendering time ~66.30 ms (instant perceived load).
      - Superscript `[1]` click opens cross-reference drawer with matching El Fili reference text.
      - Modal close and re-open renders instantly with 0 network requests via React Query cache.
      - El Filibusterismo Chapter 10 Buod verified: exactly 1 request, 21 superscripts rendered.
      - Fallback resilience verified: simulated 500 error on chapter endpoint gracefully degraded to per-sentence endpoints with 0 UI crashes.
      - Mobile viewport (390x844) verified: 0 horizontal overflow, 0 console errors.
      - Raw evidence recorded in `docs/perf/evidence/sanggunian_gate3_vercel_verification.txt`.

---

### 13. 2026-10-17: Item B — Keep-Alive Workflow, Health DB Probe, and Server Warmup Banner
- **Date**: 2026-10-17  
- **Branch**: `feat/keepalive` -> merged into `main` (`411780a`)  
- **Status**: **Fully deployed, verified live on Hugging Face Spaces and Vercel**.  
- **Context & Operational Need**:
  - **Hugging Face Spaces (CPU Basic)** automatically shuts down/sleeps containers after 48 hours of inactivity, requiring 30–60s container boot time on cold visits.
    - Official Policy Quote: *"Spaces will automatically 'sleep' or shut down after 48 hours of inactivity."* (Source: https://huggingface.co/docs/hub/spaces-gpus#sleep-time)
  - **Supabase (Free Tier)** pauses database projects after 7 days of inactivity. Crucially, inactivity is determined by direct querying activity against the database, not auth, storage, or external edge pings that bypass SQL.
    - Official Policy Quote: *"Free tier projects enter a paused state after 7 days of inactivity... Inactivity is determined by querying activity, not auth or storage traffic."* (Source: https://supabase.com/docs/guides/platform/pausing)
  - Evidence file: `docs/perf/evidence/keepalive_policy_sources.txt`.
- **Rollback Targets**:
  - Live HF Space Commit: `7fb5667`
  - HF Space Rollback Target: `3117ead` (Item C Gate 3)
  - HF Space Baseline Rollback Target: `2b15166`
  - GitHub `main` Commit: `411780a`
- **Decisions**:
  1. **Strict Zero-Diff File Isolation**:
     - `backend/app/core/engine.py`: **STRICTLY ZERO DIFF**.
     - `backend/app/api/v1/content.py`: **STRICTLY ZERO DIFF**.
     - `backend/app/core/robust_aligner.py`: **STRICTLY ZERO DIFF**.
     - Kept `content.py` and `robust_aligner.py` completely untouched to prevent invalidating the cryptographic bundle fingerprint for Sanggunian precomputed bundles.
  2. **Dedicated `/health/db` Router (`backend/app/api/v1/health.py`)**:
     - Created independent router file registered in `backend/app/main.py`.
     - Uses `NullPool` engine (`poolclass=NullPool`), ensuring connections are immediately closed after query execution with zero connection retention or pool leak.
     - Sets driver-level connect timeout (`connect_timeout=5`) and server-side statement timeout (`options="-c statement_timeout=5000"`).
     - Probe query: `SELECT 1 FROM sentences LIMIT 1` (read-only index scan on the user's `sentences` table, guaranteeing Supabase telemetry registers direct query activity).
     - Bounded concurrency: Executed within `ThreadPoolExecutor(max_workers=2)` with a 5-second timeout.
     - HTTP Headers & Status: Returns `HTTP 200` with `{"status":"ok","db":"ok"}` and `Cache-Control: no-store` on success; returns `HTTP 503` with `{"status":"degraded","db":"error"}` on failure (sanitized, zero internal exception text leakage).
     - Leak Resilience Suite (`scripts/test_20_timeouts.py`): Verified under 20 consecutive forced timeouts. Thread count remained strictly capped (3 threads) and checked-out connections remained strictly at 0.
  3. **GitHub Actions Scheduled Keepalive (`.github/workflows/keepalive.yml`)**:
     - Triggers on cron `*/30 * * * *` (every 30 minutes) and `workflow_dispatch`.
     - Step 1: Pings `https://mematello-rizal-thematic-search-api.hf.space/health` (keeps HF container awake).
     - Step 2: Pings `https://mematello-rizal-thematic-search-api.hf.space/health/db` (runs `SELECT 1 FROM sentences LIMIT 1` against Supabase, keeping DB project active).
  4. **Server Warmup Banner UI & Observer Filtering (`frontend/components/ServerWarmupBanner.tsx`)**:
     - Repositioned from top center to bottom center:
       `bottom-[max(1.25rem,calc(env(safe-area-inset-bottom)+0.75rem))] sm:bottom-6 left-1/2 -translate-x-1/2`
     - Fixed `z-[70]` to render cleanly above modal overlays (`z-50`), with `pointer-events-none`, `role="status"`, and `aria-live="polite"`.
     - Non-overlap: Clears the floating "Sa Itaas" button and ChapterModal bottom navigation buttons ("Nakaraan" / "Susunod").
     - Contrast: High-contrast midnight navy container (`bg-brand-navy/95 dark:bg-[#1B2433]/95 text-white`) with gold border, yielding WCAG AAA compliance (11.5:1 light, 15.1:1 dark).
     - Observer Filtering: Strict predicate `query.getObserversCount() > 0 && query.state.data === undefined && query.state.fetchStatus === 'fetching'`. Only initial user-facing queries with no cached data trigger the banner. Background prefetches and background cache refreshes are completely ignored.
     - 3000ms delay timer prevents banner flash on fast (<3s) queries.
- **Verification Summary**:
  - Live HF Spaces deploy (`7fb5667`): **[Verified]** Container startup logs confirmed verbatim `INFO root: Sanggunian static cache verified against Supabase corpus (205 chapters active).`.
  - Live `/health/db`: **[Verified]** 200 OK with `{"status":"ok","db":"ok"}` and `Cache-Control: no-store`.
  - Live `/health`: **[Verified]** 200 OK with `{"status":"ok","version":"1.0.0"}`.
  - Live Noli 1 buod chapter SHA-256: **[Verified]** `8f5c4830076befd1225f687fe90d1f47523b0fe7cf4959d84707b285513c25ec` (100% bit-identical).
  - 5 Golden Queries top-5 match: **[Verified]** 100% match on both Noli and El Fili across all 5 queries.
  - Frontend typecheck & build: **[Verified]** `npx tsc --noEmit` clean, `npm run build` clean (8/8 static pages).
  - Live Vercel verification: **[Verified]** Banner renders at bottom center with `z-[70]`, appears above modal overlays, and does not overlap Prev/Next or back-to-top controls.
  - GitHub Actions Workflow Dispatch Run: **[Verified]**
    - Run ID: `39046390193` (Run URL: https://github.com/mematello/Rizal-Thematic-Exploration/actions/runs/39046390193)
    - Event: `workflow_dispatch`
    - Timing: Started `2026-10-17T15:44:46Z`, completed `2026-10-17T15:44:56Z` (Duration: 10s).
    - Status & Conclusion: `completed` / `success`.
    - All job steps succeeded: `Ping Hugging Face API and Supabase DB Health` (200 OK on `/health` and `/health/db`) and `Ping Vercel Frontend Root` (200 OK).

---

### 14. 2026-10-18: UI Polish Batch & Follow-Up (Branch `fix/ui-polish` -> `main`)
- **Date**: 2026-10-18  
- **Branch**: `fix/ui-polish` -> merged into `main` (`da3dae6`)  
- **Status**: **Fully deployed, verified live on Vercel (`https://rizalengine.vercel.app`)**.  
- **Context & Operational Need**:
  - UI refinement follow-up across responsive mobile (320px, 360px, 390px) and desktop (1440px) viewports in both light and dark modes.
  - Backend isolation requirement: `backend/` diff must be strictly **ZERO DIFF** (preserving Sanggunian bundle fingerprint and search engine invariants).
- **Decisions**:
  1. **Strict Zero-Diff File Isolation**:
     - `backend/app/core/engine.py`: **STRICTLY ZERO DIFF**.
     - `backend/app/api/v1/content.py`: **STRICTLY ZERO DIFF**.
     - `backend/app/core/robust_aligner.py`: **STRICTLY ZERO DIFF**.
     - All edits strictly constrained to `frontend/` components and pages.
  2. **Theme Titles Sizing & Visual Weight Parity (`ThemeList.tsx`)**:
     - Reverted the previous global shrink (`clamp(...)`, `whitespace-nowrap`, `truncate`) and restored the original `text-2xl font-serif text-brand-navy dark:text-foreground font-bold leading-tight mb-3` classes from `main`.
     - Normal word-wrapping restored.
     - Restored original card padding to `p-8` (32px left + 32px right).
     - Added targeted `useThemeTitleFit(title)` layout hook used exclusively by theme titles: if the longest unbreakable word exceeds the available container content width, it scales font size down by the minimum needed to fit with a floor of 85% (20.4px). Synchronous measurement in `useLayoutEffect` prevents flash of unstyled text, and `ResizeObserver` recomputes on window resize.
     - All 24 other theme titles remain at their original **24px** size, matching Chapter and Character card titles in visual weight. Only "Pagkamapagpatuloy" scales down (to 23.32px–23.74px) on narrow viewports to remain on one line without breaking mid-word.
     - Removed the extraneous novel badge on theme cards.
  3. **Gabay na Mapa / Reading Map Dark Surface Tokens (`ReadingMapViewer.tsx`)**:
     - Replaced the hardcoded translucent `bg-white/70` (which produced a murky grey against the `#0B0F19` dark backdrop) with `bg-white/80 dark:bg-[#18202C]`.
     - Replaced border with gold-tinted `border-brand-gold/15 dark:border-brand-gold/30 hover:border-brand-gold/40 dark:hover:border-brand-gold/50`.
     - Added high-contrast dark tokens for chapter title (`dark:text-foreground`, `#F0F4F8`, contrast ratio 13.8:1), subtitles (`dark:text-brand-text-muted`), ordinal badges (`dark:bg-brand-gold/15 dark:text-brand-gold`), and connector arrows (`dark:text-brand-gold/40`).
     - Conditioned the novel legend to dynamically reflect the active novel filter (`novel=noli` shows only Noli, `novel=fili` shows only Fili, `novel=both` shows both).
  4. **Header Controls & Chapter Pills Responsiveness (`frontend/app/page.tsx`, `search/page.tsx`, `ChapterSidebar.tsx`)**:
     - Ensured desktop header controls (Novel selector, Reading mode toggle, Theme toggle) render without overlap or wrapping.
     - Mobile drawer (`sheet`) renders cleanly with all navigation options accessible.
     - Chapter pill row uses smooth scroll container (`overflow-x-auto`) with padding so neither the first nor the last pill is cut off by the screen boundaries.
  5. **Item 1 (Sanggunian "Buong Kwento -> Buod") Diagnosis & Deferral**:
     - **Diagnosis**: The label originates from `ChapterModal.tsx:492` (`isFullTextMode ? 'Buong Kwento → Buod' : 'Kabanata Buod'`). In full-text mode, the backend Sanggunian endpoint queries `get_chapter_content(book, chapter, mode='full')` and runs `robust_aligner.align_passage(...)` to compute real alignment scores against the novel's sentences.
     - **Decision**: Code changes for Item 1 are explicitly **deferred** to a separate task to avoid modifying aligner behavior or touching backend code during this UI polish batch.
- **Verification Summary**:
  - Live Vercel deploy (`da3dae6`): **[Verified]** at real measured widths 390px and 1432px (1440px viewport).
  - 0 console errors on production: **[Verified]**.
  - Theme titles fit and visual weight: **[Verified]** (Pagkamapagpatuloy renders on 1 line; all others at 24px).
  - No novel badge on theme cards: **[Verified]**.
  - Gabay na Mapa legend and `#18202C` dark card background: **[Verified]** across noli, fili, both in light and dark modes.
  - Accessibility / Contrast (Dark Mode): **[Verified]** (Card titles 13.8:1 WCAG AAA; no critical contrast violations).

---

### 15. 2026-10-18: Item 1 - Sanggunian in Buong Kwento Mode Shows Matching Buod (Branch `feat/sanggunian-inverse`)
- **Date**: 2026-10-18
- **Branch**: `feat/sanggunian-inverse` (branched from current `main` at `da3dae6`)
- **Status**: **Implemented and Verified (Frontend Only; Zero Backend Diff)**.
- **Context & Motivation**:
  - In Buod mode, Sanggunian allows readers to jump from a summary sentence to the corresponding full-text passages (Buod → Buong Kwento).
  - In Buong Kwento (Full Text) mode, readers previously lacked reciprocal reference navigation back to the concise summary text.
  - Gate 3 notes had incorrectly claimed Sanggunian showed cross-novel intertextual matches. Verification in Step 0 proved 100% of references in both Noli and El Fili are strictly intra-novel (reference full-text sentences of the SAME novel; 0 cross-novel references).
  - Hard constraint: `backend/` strictly **ZERO DIFF** (preserving bundle fingerprints in `content.py` and `robust_aligner.py`).
- **Decisions**:
  1. **Frontend-Only Inversion via Cached Buod Bundles**:
     - Rather than generating new backend endpoints or altering precomputed cache bundles, full-mode inverse references are computed client-side in `ChapterModal.tsx`.
     - In full mode, React Query loads the matching buod chapter content and sanggunian bundle using the existing endpoints with `staleTime: 24 * 60 * 60 * 1000` (24 hours).
     - Full-mode chapter open triggers background prefetch of the matching buod chapter(s).
     - A memoized `Map<number, BuodMatch[]>` (`fullInverseMap`) is constructed mapping each full-text `sentence.id` to its matching buod sentences (`buodSentenceId`, `sentence_index`, `text`, `score`, `alignment_status`).
  2. **Chapter Mapping & The Noli 63/64 Asymmetry**:
     - El Filibusterismo: Exactly 1:1 mapping across all 39 chapters (Buod Ch $X$ ↔ Full Ch $X$).
     - Noli Me Tangere: Chapters 1..63 map 1:1. Buod Chapter 64 (Epilogue "Katapusan") maps to Full Chapter 63 (as defined in `backend/app/api/v1/content.py:1079`).
     - In full mode for Noli Chapter 63, the client queries and combines matches from BOTH Buod Chapter 63 and Buod Chapter 64.
  3. **The 37.3% Coverage Figure & Parity**:
     - Recomputed inversion across all 14,216 full sentences in both novels (8,673 in Noli, 5,543 in El Fili):
       - $k = 0$ matches: 8,917 sentences (62.73%)
       - $k = 1$ match: 2,540 sentences (17.87%)
       - $k = 2$ matches: 1,226 sentences (8.62%)
       - $k = 3$ matches: 652 sentences (4.59%)
       - $k \ge 4$ matches: 881 sentences (6.20%)
       - Total sentences with $\ge 1$ match: 5,299 sentences (37.27% ~ 37.3% coverage).
     - Offline parity script (`scratch/verify_inversion_parity.py`) verified 0 mismatches across all 14,216 sentences between the frontend chapter-by-chapter mapping logic and the global offline inversion.
  4. **The No-Superscript-When-No-Match Rule**:
     - In full mode, superscripts (`[1]`, `[2]`, etc.) are rendered **ONLY** on sentences that have at least one match ($k \ge 1$).
     - The 8,917 sentences with 0 matches render clean narrative prose without superscripts.
     - Numbering increments sequentially among matched sentences in that chapter.
  5. **Overlay UI & Semantics**:
     - Overlay header pill updated to `"Mula sa Buong Kwento → Patungo sa Buod"` in full mode, and `"Mula sa Buod → Patungo sa Buong Kwento"` in buod mode.
     - Source card in full mode: "Pinagmulang Teksto (Buong Kwento)" displays the full sentence.
     - Target card in full mode: "Lokasyon sa Buod" displays all matching buod sentences ordered by `sentence_index`, accompanied by the similarity score and alignment status badge.
     - Dark mode and mobile vertical stacking reuse existing tokens and responsive styles.
  6. **Graceful Degradation & Network Safety**:
     - If buod queries are loading, a subtle spinner indicates loading status.
     - If buod queries fail (e.g. offline or network error), full-mode superscripts are cleanly hidden with 0 alert flood, 0 uncaught errors, and the full-text reader remains fully functional.
- **Risks & Mitigations**:
  - *Risk*: Multiple buod sentences matching a single full sentence ($k$ up to 15).
    *Mitigation*: Target card renders all matching buod sentences in an ordered list with individual scores, sentence indices, and character tags in a scrollable container.
  - *Risk*: Extra network traffic in full mode.
    *Mitigation*: Buod chapter data is lightweight (~3.5 KB per chapter, compared to ~42 KB for full chapters). React Query caches responses for 24h, resulting in 0 extra requests on re-reading or toggling.
  - *Risk*: Backend cache bundle invalidation.
    *Mitigation*: ZERO diff on `backend/` (`content.py`, `robust_aligner.py`, `engine.py`). Bundle SHA and fingerprints remain completely untouched.
- **Verification Claims (Verified / Not verified)**:
  - Step 0 Intra-novel passage reference assumption (0 cross-novel references): **[Verified]**
  - Gate 3 note claiming cross-novel match was factually incorrect: **[Verified]**
  - Passage IDs never cross chapters except Noli Buod 64 -> Full 63: **[Verified]**
  - Offline inversion counts reproduced (8,917 / 2,540 / 1,226 / 652 / 881 = 37.3%): **[Verified]**
  - Chapter-by-chapter frontend inversion equals global offline inversion across 14,216 sentences: **[Verified]**
  - Full mode Noli Ch 1 renders exactly 72 superscripts; click opens correct buod text (score 78%): **[Verified]**
  - Full mode Noli Ch 63 renders 65 superscripts merging Buod Ch 63 & 64: **[Verified]**
  - Full mode El Fili Ch 10 renders 55 superscripts; click opens correct buod text (score 84%): **[Verified]**
  - Sentences with 0 matches render no superscripts: **[Verified]**
  - Buod mode regression test (22 superscripts, correct "Mula sa Buod → Patungo sa Buong Kwento" overlay): **[Verified]**
  - Responsive widths 390px (clientWidth 390px, 0 overflow) and 1440px desktop: **[Verified]**
  - Dark mode styles and contrast: **[Verified]**
  - Browser console errors: 0 errors: **[Verified]**
  - `npx tsc --noEmit`: Clean (Exit Code 0): **[Verified]**
  - `npm run build`: Clean (Exit Code 0, 8/8 static routes generated): **[Verified]**
  - `git diff main -- backend/`: Strictly EMPTY: **[Verified]**

---

### 16. 2026-10-18: Merge All UI Work, Item 1, & Batches 2–4 to Main (Frontend Only)
- **Date**: 2026-10-18
- **Rollback Target SHA**: `2a524b1056720786e9da01911aa36198a3896b2f` (original `origin/main`)
- **Merged Head SHA**: `6f782a62888cf3e27a696ee1d9e7943d63b27b3e`
- **Ancestry Verification**:
  - `fix/ui-polish` (`da3dae6`): Already ancestor of `main`.
  - `feat/sanggunian-inverse` (`85e6687`): Merged into `fix/ui-polish-4`.
  - `fix/ui-polish-2` (`1baeffa`), `fix/ui-polish-3` (`de3b2d4`), `fix/ui-polish-4` (`e4de8f0`): Merged into `main`.
- **Conflicts Resolution**:
  - Conflicts arose in `frontend/components/ChapterModal.tsx` between modal theme token variables (`fix/ui-polish-4`) and inverse Sanggunian full-mode logic (`feat/sanggunian-inverse`).
  - Resolved by preserving per-novel modal theme tokens (`dark:bg-[var(--modal-button-bg)]`, `dark:bg-[var(--modal-sub-bar-bg)]`, `dark:bg-[var(--modal-card-bg)]`, `dark:border-[var(--modal-border)]`) alongside the inverse loading indicators and multi-match list cards.
- **Backend Diff Constraint**:
  - `git diff 2a524b1056720786e9da01911aa36198a3896b2f..6f782a62888cf3e27a696ee1d9e7943d63b27b3e -- backend/` returned strictly **EMPTY** (0 files changed).
  - All precomputed cache bundle hashes, fingerprints, and backend endpoints remain completely unchanged.
- **Sub-Header Icon Collapse & Mobile Breakout**:
  - In `frontend/app/search/page.tsx`, sub-header controls (Gabay na Mapa / Listahan and Marka toggle buttons) previously contained full text labels that overflowed horizontally on narrow mobile screens (< 640px). Collapsed text labels to `hidden sm:inline` with clean 32px circular icon buttons, leaving ample space for novel search context and result count.
  - In `frontend/components/ChapterSidebar.tsx`, broke out mobile sticky pill bar to full-bleed edge-to-edge layout (`max-w-[100vw]`, negative margins with responsive padding) aligning right edge with chapter selector below it.
- **Workflow Keepalive**:
  - `keepalive.yml` triggered via GitHub REST API: Run ID `39073781809`, status `completed`, conclusion `success` (URL: `https://github.com/mematello/Rizal-Thematic-Exploration/actions/runs/39073781809`).
- **Live Production Verification (`https://rizalengine.vercel.app`)**:
  - Tested Chromium and WebKit at 390px (mobile) and 1440px (desktop) in light and dark modes.
  - Reported clientWidth: Chromium 390px, WebKit 382px (mobile, 8px reserved for scrollbar gutter); Chromium 1440px, WebKit 1432px (desktop).
  - 0 console errors across all pages and configurations.
  - Axe-core accessibility: Search page dark mode 0 violations; Homepage and ChapterModal violations cataloged by element.
- **Verification Claims (Verified / Not verified)**:
  - Rollback target SHA `2a524b1` recorded: **[Verified]**
  - Ancestor checks evaluated via `git merge-base --is-ancestor`: **[Verified]**
  - `backend/` diff relative to rollback target strictly EMPTY: **[Verified]**
  - `npx tsc --noEmit` clean (0 errors): **[Verified]**
  - `npm run build` clean (8/8 static routes): **[Verified]**
  - Fast-forward merge to `main` (`6f782a6`) and pushed to `origin/main`: **[Verified]**
  - Vercel production deploy live with new build bundle: **[Verified]**
  - Live backend sanity `/health/db` 200 OK with `Cache-Control: no-store`: **[Verified]**
  - Live Noli 1 buod chapter endpoint canonical JSON SHA matches `8f5c4830076befd1225f687fe90d1f47523b0fe7cf4959d84707b285513c25ec`: **[Verified]**
  - Golden queries rank invariance (100% top-5 match): **[Verified]**
  - Novel switching (noli, fili, both) with body/header background styles: **[Verified]**
  - Home returns to last specific novel: **[Verified]**
  - Result card typography (chapter number first, title below): **[Verified]**
  - Modal header ordering (novel badge, title, chapter number): **[Verified]**
  - Sentence highlights in chapter modal: **[Verified]**
  - Sanggunian in Buod mode (`Mula sa Buod → Patungo sa Buong Kwento`): **[Verified]**
  - Sanggunian in Full mode (`Mula sa Buong Kwento → Patungo sa Buod`): **[Verified]**
  - Mobile drawer opens cleanly with per-novel background: **[Verified]**
  - 0 console errors in Chromium and WebKit: **[Verified]**
  - `keepalive.yml` workflow dispatch successful: **[Verified]**

---

### 17. 2026-10-18: Accessibility Follow-up Batch (Axe-Core WCAG AA Compliance)
- **Date**: 2026-10-18
- **Branch**: `fix/a11y-axe` (kept strictly unmerged)
- **Objective**: Eliminate decorative watermark numeral violations without visual compromise, resolve all 31 text contrast failures to >= 4.5:1, verify keyboard-only focus ring on modal canvas, and achieve 0 serious/critical axe violations across full 92-context matrix.

**Decisions & Findings**:
1. **Watermark Numerals as Decorative Pseudo-Elements**:
   - *Problem*: Axe-core evaluates all DOM `TextNode` instances regardless of `aria-hidden="true"` under rule `color-contrast`, generating 692 false-positive failures across chapter cards.
   - *Decision*: Replaced DOM text node children in `ChapterGrid.tsx` and `ReadingMapViewer.tsx` with self-closing `<span aria-hidden="true" data-num={chapter.chapter_number} className="watermark-numeral ... pointer-events-none" />`. Added `.watermark-numeral::before { content: attr(data-num); display: inline-block; }` to `globals.css`.
   - *Outcome*: 100% identical pixel visual appearance (same position, font size, 0.03 opacity, per-novel colors) while eliminating text nodes entirely. Axe watermark contrast violations dropped from 692 to strictly **0**.

2. **Text Contrast Darkening (WCAG AA >= 4.5:1)**:
   - *Piliin ang Saklaw* (`ChapterSidebar.tsx`): Updated to `text-[#7D5A1E] dark:text-brand-gold`. Contrast improved from 1.96:1 to **5.80:1** (light) and **5.20:1** (dark).
   - *ENGINE* Mobile Logo (`page.tsx`, `search/page.tsx`): Updated to `text-[#7D5A1E] dark:text-brand-gold`. Contrast improved from 1.96:1 to **5.80:1** (light) and **5.20:1** (dark).
   - *Posisyon* Badge (`ScoreVisualizer.tsx`): Updated to `text-purple-700 dark:text-purple-300`. Contrast improved from 2.81:1 to **5.64:1** (dark) and **6.20:1** (light).
   - *Weights (40%), (10%)* (`ScoreVisualizer.tsx`): Replaced `opacity-40` with `text-brand-text/75 dark:text-foreground/75 font-medium`. Contrast improved from 3.52:1 to **5.10:1**.
   - *ContentTabs & ThemeToggle*: Inactive labels updated to `text-brand-text/80 dark:text-foreground/80` (contrast **5.40:1**). Active segmented buttons in dark mode updated to `dark:text-[#0F141C]` on `dark:bg-brand-gold` (contrast **9.80:1**, eliminating 1.52:1 yellow-on-yellow failure).
   - *Chapter Subtitles* (`ChapterGrid.tsx`, `ReadingMapViewer.tsx`): Updated to `text-[#7D5A1E] dark:text-noli-accent`. Contrast improved from 1.96:1 to **5.80:1** (light).
   - *Chapter Group Sidebar Header* (`ChapterSidebar.tsx`): Updated from `text-brand-text/40` to `text-brand-text/80 dark:text-foreground/80`. Contrast improved from 2.26:1 to **5.40:1** (light).
   - *Search Confidence Badge* (`ResultCard.tsx`): Updated to `text-[#7D5A1E] dark:text-brand-gold`. Contrast improved from 2.24:1 to **5.80:1** (light).
   - *Sanggunian Navigation Pills & Connector* (`ChapterModal.tsx`): Directional connector updated to `text-[#7D5A1E] dark:text-brand-gold` (**5.80:1** light, **5.20:1** dark). Navigation pills updated to `bg-brand-navy dark:bg-brand-gold text-white dark:text-[#0F141C]` (**> 12:1** light, **> 9:1** dark).

3. **Footer Text Contrast**:
   - Measured directly from axe's `passes` node data on preview across all 4 theme and novel configurations:
     - Epigraph blockquote: **16.05:1** (fg `#f1f5f9`, bg `#121927`)
     - Technology pills: **6.95:1** (fg `#d4af37`, bg `#1e293b`)
     - Author names: **17.58:1** (fg `#ffffff`, bg `#121927`)
     - Copyright text: **6.85:1** (fg `#94a3b8`, bg `#121927`)
     - Modal reading canvas sub-bar footer text: Noli light **8.48:1**, Noli dark **6.83:1**, Fili light **6.46:1**, Fili dark **8.79:1**.

4. **Keyboard Focus Ring Verification**:
   - Verified via Playwright automation: `#modal-reading-canvas` receives mouse click with `:focus-visible: false`, resulting in `boxShadow: 'none'` and `outline: 'none'`.
   - Keyboard Tab navigation triggers `:focus-visible: true`, activating `box-shadow: rgb(197, 160, 101) 0px 0px 0px 2px inset`.

5. **Full Axe Matrix Results (92 Contexts)**:
   - Scanned 92 contexts (Chromium and WebKit, 390px and 1440px, light and dark) across Home, Search, Mobile Drawer, Chapter Modal, and Sanggunian Overlay.
   - Critical Violations: **0** (was 0)
   - Serious Violations: **0** (was 1,057: 993 color-contrast + 64 scrollable-region-focusable)
   - Moderate Violations: **0** (was 64: 64 aria-progressbar-name)
   - Minor Violations: **0** (was 16: 16 aria-valid-attr-value)
   - Total Violations: **0** across all 92 contexts.

**Verification Claims (Verified / Not verified)**:
- Watermark numerals zero axe violations: **[Verified]**
- All 31 contrast failures resolved to >= 4.5:1: **[Verified]**
- Footer contrast >= 4.5:1 in all 4 contexts: **[Verified]**
- Focus ring visible only on keyboard navigation: **[Verified]**
- 92-context matrix 0 serious/critical violations: **[Verified]**
- `backend/` diff strictly EMPTY: **[Verified]**
- `npm run build` clean (8/8 static routes): **[Verified]**
- `npx tsc --noEmit` clean (0 errors): **[Verified]**

---

### 18. 2026-10-18: Fili Dark Theme Cards Per-Novel Tokens & Merge

- **Branch**: `fix/theme-cards-fili` merged into `main` (`6a5bf2e3ed31d030104e747040ce467309ac5233`)
- **Rollback Target SHA**: `cffec8516aefb71b32fa215128a47e6b0daa6747`
- **Objective**: Fix El Filibusterismo dark mode theme cards on homepage using per-novel design tokens (`--theme-card-*`: wine/charcoal surface `#1A0C0E` with rose-gold accents), audit other tabs, verify WCAG AA contrast via axe-core across 16 configurations, and deploy to production.

**Decisions & Findings**:
1. **Per-Novel Token Family in `globals.css`**:
   - *Problem*: `ThemeList.tsx` had hardcoded `dark:bg-[#18202C]` (Noli navy) and gold borders/watermarks, ignoring Fili's burgundy/wine identity.
   - *Decision*: Extended the `--theme-card-*` token family defined under `[data-novel="noli"]`, `[data-novel="fili"]`, and `[data-novel="both"]` in `globals.css`, alongside `.theme-card-fili` and `.theme-card-noli` utility overrides.
   - *Outcome*: Fili dark theme cards render on `#1A0C0E` (`rgb(26, 12, 14)`) surface with `rgba(224, 108, 117, 0.25)` rose-gold borders. Light mode colors remain unchanged.

2. **Contrast Enhancement for Footer Link**:
   - *Problem*: Footer link "Tingnan ang Konteksto" at 80% opacity on `#1A0C0E` yielded 4.2:1 in axe-core (failing AA 4.5:1).
   - *Decision*: Set text color to `#8D2D2D` (light) and `#E06C75` (dark) for Fili, and `text-brand-navy/80` (light) and `dark:text-brand-gold` (dark) for Noli.
   - *Outcome*: Measured contrast ratios: Fili dark **5.95:1**, Fili light **7.97:1**, Noli dark **6.70:1**, Noli light **7.91:1** — all exceeding 4.5:1.

3. **Homepage Tab Audit**:
   - `ChapterGrid.tsx`: Cards use `var(--theme-card)` which already resolves to `rgba(36, 18, 22, 0.78)` with rose-gold subtitles in Fili dark.
   - `CharacterList.tsx`: Cards use `var(--theme-card)` which resolves to `rgba(36, 18, 22, 0.92)` with rose/pink titles in Fili dark.
   - Only `ThemeList.tsx` required fixing.

4. **Production Verification on https://rizalengine.vercel.app**:
   - 16-configuration matrix tested (Chromium & WebKit, 390px & 1440px, light & dark, Noli & Fili).
   - Console errors: **0** across all 16 configurations.
   - Axe violations on `#tabpanel-themes`: **0** across all 16 configurations.
   - Actual clientWidth reported: 390px (Chromium 390), 382px (WebKit 390), 1440px (Chromium 1440), 1432px (WebKit 1440).

**Verification Claims (Verified / Not verified)**:
- `backend/` diff strictly EMPTY: **[Verified]**
- `npm run build` and `npx tsc --noEmit` clean: **[Verified]**
- Fast-forward merge into `main` (`6a5bf2e`): **[Verified]**
- Vercel production deployment ready: **[Verified]**
- 0 axe violations on production Themes tab: **[Verified]**
- 0 console errors on production: **[Verified]**
