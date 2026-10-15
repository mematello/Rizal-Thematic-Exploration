# Performance Optimization Plan & Milestone Tracking

**Goal**: Deliver sub-second search latencies for public deployment while strictly guaranteeing ranking correctness and determinism.

---

## Completed Milestones (Verified 0/40 Against Golden Snapshot)

### Phase 0: Baseline & Measurement Infrastructure
- [x] Create golden query set (40 queries across 5 distinct categories: Tagalog single/multi, English, characters, out-of-domain)
- [x] Build automated benchmark client (`scripts/benchmark.py`) with 10-run cold/warm statistics
- [x] Implement golden snapshot verification tool (`scripts/compare_golden.py`)
- [x] Fix client IPv6 `::1` connect stall by targeting `http://127.0.0.1:8000` with persistent `requests.Session()`
- [x] Implement accumulating `PerfTimer` with call counts and explicit stage timers for all model encoding sites
- [x] Isolate run-to-run non-determinism root cause (Python hash randomization on unsorted set iterations)

### Phase 1 & 2: Backend Search Pipeline Optimizations
- [x] **Process-Level Word Vector Cache (`_word_vec_cache`)**:
  - Thread-safe dictionary caching SentenceTransformer word vectors across requests
  - Reduced `edukasyon` rerank loop from ~32s to ~65–77 ms
- [x] **Process-Level Translator Cache (`TranslatorCache`)**:
  - Thread-safe cache with 24h positive TTL, 60s negative TTL, and 5.0s timeout
  - Reduced repeated out-of-domain latency from ~1.3s to ~129 ms (-90.0%)
- [x] **Batched Context Retrieval (`_fetch_context_cache`)**:
  - Replaced 40 sequential candidate SQL queries with a single union-bounding SQL query
  - Reduced database roundtrips by 75.0% to 92.9% per search request
  - Verified 382/382 candidate instances and 180/180 boundary sentences 100% bit-identical
- [x] **Startup Vocabulary Precomputation (Step A)**:
  - Precomputed all 24,770 candidate vocabulary words one-at-a-time into `backend/app/data/word_vec_cache_v1.pkl`
  - Loads in 238.2 ms at server startup with read-only arrays
  - Verified 500/500 sample vectors 100% bit-identical (`np.array_equal == True`)
  - Eliminated slow first request on unseen Tagalog words (rerank loops dropped from 5.1s–12.1s to 58–79 ms with 0 misses)
- [x] **Comprehensive Reporting**:
  - Published complete methodology and empirical tables in `docs/perf/FINAL_REPORT.md`

---

## Pending Milestones

### Step B: Determinism Enforcement (Approved 2026-10-16)
- [x] Apply Task 2 Item 8 (`ORDER BY` tie-breaks in SQL queries) first and report comparison vs golden snapshot
- [x] Apply Task 2 Items 1, 2, 3, 4, 5, 6, 7 (`sorted()` across set iterations in theme expansion and tie-breaks)
- [x] Run two fresh-process full 40-query passes without `PYTHONHASHSEED` and verify 100% identity
- [x] Generate `docs/perf/golden_snapshot_deterministic.json` as the new permanent reference
- [x] Log exact ranking delta (# of position changes) in `DECISIONS.md` under the Ranking Change Policy
- [x] Stop and request user review before merging — **Approved 2026-10-16**

### 2. First-Seen Out-of-Domain Translation Policy (Resolved 2026-10-16)
- [x] Review ~1.3s cold network latency on unmapped modern terms — **Resolved: Accepted as one-time cold cost amortized by 24h positive / 60s negative TranslatorCache (~129 ms warm)**
- [x] Evaluate fast local out-of-vocabulary (OOV) heuristic or dictionary filter vs. external translation timeout tuning — **Resolved: Pre-translator heuristics rejected to preserve academic cross-lingual recall for unmapped historical concepts; existing 5.0s timeout maintained**


### 3. Production Deployment Sizing & Host Configuration (Resolved 2026-10-16)
- [x] Finalize server memory budget (account for ~460 MB initial RSS and ~1.2 GB working memory under inference) — **Resolved: Budgeted ~460 MB initial RSS, ~2.2 GB commit memory for dual models, and ~1.2–1.5 GB RSS under active inference per worker**
- [x] Configure Uvicorn worker count based on deployment machine CPU cores and RAM limits — **Resolved: Established formula NUM_WORKERS = max(1, min(CPU_CORES, floor((RAM_GB - 2.5) / 1.5)))**
- [x] Ensure `word_vec_cache_v1.pkl` is packaged or built during deployment CI/CD — **Resolved: Build step `python scripts/build_word_vec_cache.py` or release artifact copy integrated into deployment workflow**


---

## Deliberately Excluded / Deferred (Scope Constraints Honored)

- [-] **HNSW / ANN Vector Indexing**: Preserved exact vector similarity search to protect search quality and academic recall.
- [-] **Query Model Encoding Batching**: Sequential calls during query preprocessing remain unbatched to prevent cross-call tensor padding variations.
- [-] **pg_trgm Full-Text GIN Indexing**: Existing ILIKE execution times are <25 ms, rendering additional index maintenance unnecessary.
