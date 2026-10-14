# Performance Optimization State

**Last Updated**: 2026-10-14  
**Branch**: `perf/speed-optimization`  
**Current Phase**: Phase 2 Step A Completed (Startup Vocabulary Precompute); Step B (Determinism) Awaiting Approval

---

## Current Status

1. **Step A Completed — Startup Vocabulary Precomputation (`word_vec_cache_v1.pkl`)**:
   - Built `scripts/build_word_vec_cache.py`: Extracted exact candidate vocabulary matching engine validation logic (`len(w) > 4 and w not in LOW_INFO_WORDS and w in vocabulary`), yielding exactly **24,770 words**.
   - Encoded each word one-at-a-time with `base_model.encode(w, show_progress_bar=False)` and serialized to `backend/app/data/word_vec_cache_v1.pkl` (73.39 MB) with model name (`sentence-transformers/paraphrase-xlm-r-multilingual-v1`) and vocabulary SHA-256 hash (`6efe3301743eed6b7552291a9c444e7a4ea426cd5175d89ca9d8b648b1c65569`).
   - Integrated `_load_precomputed_word_vectors()` in `RizalEngine.__init__`: Loads all 24,770 float32 arrays into `_word_vec_cache` in **238.2 ms** with `writeable = False`, respecting the 50,000 cap and falling back to on-demand encode on miss.
   - **Verification 3(a)**: 500 / 500 random sample vectors verified 100% bit-identical (`np.array_equal == True`) against fresh single-word encode.
   - **Verification 3(b)**: 40 / 40 golden queries verified bit-identical vs `golden_snapshot_seed0.json` under `PYTHONHASHSEED=0` (**0 differences**).
   - **Verification 3(c) Fresh-Server First-Search Latency for 10 Unseen Words**:
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

2. **Final Benchmark Suite Executed (Identical Protocols, Quiet System)**:
   - Final "Before" (unoptimized baseline: all caches & batching disabled) vs. Final "After" (all optimizations active: word-vec cache ON, translator cache ON, batched context fetch ON):
   - **`single_word_tagalog`**: Warm P50 dropped from **5,069.59 ms to 478.27 ms (-90.6%, 10.6x faster)**; Warm P95 dropped from **29,922.43 ms to 691.14 ms (-97.7%, 43.3x faster)**.
   - **`multi_word_tagalog`**: Warm P50 dropped from **573.61 ms to 502.93 ms (-12.3%)**; Warm P95 dropped from **2,087.86 ms to 754.86 ms (-63.8%)**.
   - **`english_cross_lingual`**: Warm P50 dropped from **416.80 ms to 344.67 ms (-17.3%)**; Warm P95 dropped from **649.97 ms to 526.54 ms (-19.0%)**.
   - **`character_names`**: Warm P50 dropped from **510.80 ms to 353.34 ms (-30.8%)**; Warm P95 dropped from **757.49 ms to 522.37 ms (-31.0%)**.
   - **`out_of_domain` (Repeated)**: Warm P50 dropped from **1,298.91 ms to 129.73 ms (-90.0%, 10.0x faster)**; Warm P95 dropped from **1,495.15 ms to 171.45 ms (-88.5%, 8.7x faster)**.
   - Out-of-domain First-Seen (Run 1): 1,312.99 ms (Before) vs. 1,291.12 ms (After).

2. **Slow First Request Issue Diagnosed & Measured (5 Unseen Tagalog Words)**:
   - Measured on a fresh server boot with empty word vector cache:
     - Lexical queries (`takot`): **322.18 ms on first cold request** (fast from start).
     - Semantic fallback queries (`kasakiman`, `dangal`, `alipin`): First cold search takes **5.7s to 20.6s** because 100–255 unique tokens in candidate sentences are encoded individually at ~40–50ms each (`rerank_loop` takes 5.1s–12.1s).
     - Second search drops to **474–509 ms (91%–97% speedup)** once cached in `_word_vec_cache`.
     - Unmapped concepts (`sakripisyo`): First cold search takes **1,254.53 ms** (Google Translate network call), dropping to **130.43 ms** on subsequent searches.

3. **Batched Context Fetch Implemented & Verified**:
   - Replaced individual N+1 SQL queries in `_get_passage_context_with_embedding` ($\pm 8$) and `_expand_context` ($\pm 3$) with `_fetch_context_cache`.
   - Single SQL query per search request reduced database query volume by **75.0% to 92.9%** (e.g., `bayan` dropped from 42 to 3 queries; `oppression` dropped from 34 to 5 queries).
   - Saved ~100 ms of wire time on candidate post-processing.
   - 40/40 golden queries verified bit-identical; 382/382 candidate instances bit-identical; 180/180 boundary sentences bit-identical.

4. **Comprehensive Final Report Generated**:
   - Published to `docs/perf/FINAL_REPORT.md` containing methodology, complete before/after tables, memory footprint, verification evidence, slow first request analysis, known limitations, and unexecuted opportunities.

5. **Strict Scope Constraints & Determinism Policy**:
   - Zero changes to rerank logic, encode batching, or thresholds.
   - **No `sorted()` or `ORDER BY` changes applied** (stopped and awaiting user decision).
   - Preserves exact chapter boundary conditions and missing neighbor sentences.
   - Retains `DISABLE_BATCHED_CONTEXT_FETCH` flag for regression testing.

4. **Batched Context Fetch Verification Results**:
   - **Subtask 3(a)**: 40/40 queries verified identical against `golden_snapshot_seed0.json` (**0 differences**).
   - **Subtask 3(b)**:
     - 382/382 candidate instances across 40 queries verified against sequential: **100% bit-identical** (IDs, scores, `context_text`, and themes).
     - 180/180 boundary sentences across 30 chapters in both `summary` and `full` source types: **100% bit-identical**.
     - Total assertions passed: 2,068 / 2,068 (0 failures).
   - **Subtask 3(c) DB Query Reductions**:
     - `bayan`: 42 queries → 3 queries (**-39 queries, -92.9% reduction**)
     - `oppression`: 34 queries → 5 queries (**-29 queries, -85.3% reduction**)
     - `edukasyon`: 44 queries → 5 queries (**-39 queries, -88.6% reduction**)
     - `Crisostomo Ibarra`: 30 queries → 7 queries (**-23 queries, -76.7% reduction**)
     - `karahasan ng mga prayle`: 52 queries → 13 queries (**-39 queries, -75.0% reduction**)
   - **Subtask 3(d) Latency Profiles (Direct Warm Engine)**:
     - `bayan` (Wall-clock: 394.58 ms → 293.72 ms, **-100.86 ms, +25.6% speedup**):
       - `context_expansion`: 69.62 ms → 4.08 ms (-65.54 ms)
       - `theme_classification`: 201.37 ms → 99.93 ms (-101.44 ms)
       - `context_fetch`: 66.76 ms (1 batched query)
     - `oppression` (Wall-clock: 597.67 ms → 561.07 ms, **-36.60 ms, +6.1% speedup**):
       - `context_expansion`: 57.12 ms → 3.40 ms (-53.72 ms)
       - `theme_classification`: 126.49 ms → 49.15 ms (-77.34 ms)
       - `context_fetch`: 72.41 ms (1 batched query)

5. **Additional Optimization Opportunities Identified (Scope Constraint Honored)**:
   - **Batching Multiple Query Encodings**: `base_model.encode()` is called 3-4 separate times in sequence during `engine.search()` (`base_model_encode`, `query_components_encode`, `concept_vector_encode`, `sig_words_encode`), each taking 30-65 ms. Combining these into a single batch call `base_model.encode([q1, q2, ...])` would leverage PyTorch tensor parallelism and save ~50–80 ms per search.
   - **Pre-normalized Embeddings Cache**: Runtime computes `np.linalg.norm()` repeatedly across candidate sentence vectors in `_classify_themes` and `_compute_neighbor_score`. Storing pre-normalized $L_2$ vectors in PostgreSQL or in an in-memory normalized cache would eliminate repeated norm calculations.
   - **Database Index Optimization for Multi-Word Lexical Queries**: Multi-word queries like `karahasan ng mga prayle` run multi-term `tsquery` searches across chapters. A unified GIN index on `to_tsvector('simple', sentence_text)` would reduce lexical retrieval time.

6. **Strict Constraints Honored**:
   - No encode batching, rerank logic, or threshold modifications made.
   - Still no `sorted()` or `ORDER BY` determinism changes applied.


---

## Clean Benchmark Summary Table (40 Queries × 10 Runs, Warmed with 2 Full Passes)

| Category | Query | Before Med | After Med | Delta | Speedup | Min (B/A) | Max (B/A) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **single_word_tagalog** | kalayaan | 10,459.1 ms | 700.0 ms | -9,759.2 ms | +93.3% | 9,098 / 624 | 11,015 / 829 |
| | edukasyon | 31,615.0 ms | 858.1 ms | -30,756.8 ms | +97.3% | 29,440 / 726 | 34,088 / 1,151 |
| | simbahan | 622.3 ms | 594.6 ms | -27.7 ms | +4.5% | 531 / 577 | 736 / 648 |
| | korapsyon | 1,353.2 ms | 1,267.7 ms | -85.6 ms | +6.3% | 1,188 / 1,064 | 1,557 / 1,351 |
| | pag-ibig | 9,319.3 ms | 758.6 ms | -8,560.7 ms | +91.9% | 8,634 / 663 | 12,620 / 870 |
| | kamatayan | 10,637.3 ms | 618.2 ms | -10,019.1 ms | +94.2% | 10,157 / 592 | 11,737 / 879 |
| | bayan | 452.7 ms | 463.0 ms | +10.3 ms | -2.3% | 409 / 391 | 523 / 529 |
| | paaralan | 574.8 ms | 528.2 ms | -46.6 ms | +8.1% | 503 / 502 | 672 / 602 |
| **multi_word_tagalog** | pag-ibig ni Maria Clara | 913.7 ms | 813.3 ms | -100.5 ms | +11.0% | 752 / 663 | 1,228 / 1,113 |
| | karahasan ng mga prayle | 2,085.2 ms | 2,170.6 ms | +85.4 ms | -4.1% | 1,876 / 1,848 | 2,810 / 2,443 |
| | paghihirap ng bayan | 547.6 ms | 479.1 ms | -68.6 ms | +12.5% | 488 / 404 | 632 / 589 |
| | edukasyon para sa kalayaan | 623.0 ms | 587.0 ms | -36.0 ms | +5.8% | 479 / 525 | 1,007 / 733 |
| | kamatayan ni Elias | 513.4 ms | 564.7 ms | +51.2 ms | -10.0% | 500 / 483 | 583 / 646 |
| | kapangyarihan ng simbahan | 660.0 ms | 558.5 ms | -101.6 ms | +15.4% | 489 / 538 | 757 / 756 |
| | kalayaan ng Pilipinas | 494.9 ms | 482.6 ms | -12.3 ms | +2.5% | 427 / 434 | 579 / 580 |
| | katarungan sa lipunan | 505.5 ms | 564.8 ms | +59.3 ms | -11.7% | 413 / 474 | 571 / 788 |
| **english_cross_lingual** | corruption | 432.6 ms | 381.8 ms | -50.8 ms | +11.7% | 390 / 344 | 530 / 446 |
| | education | 539.0 ms | 528.2 ms | -10.8 ms | +2.0% | 427 / 414 | 707 / 827 |
| | freedom | 370.2 ms | 371.9 ms | +1.7 ms | -0.5% | 316 / 314 | 425 / 621 |
| | oppression | 636.0 ms | 741.1 ms | +105.1 ms | -16.5% | 569 / 640 | 817 / 974 |
| | love | 473.2 ms | 425.2 ms | -48.0 ms | +10.1% | 376 / 364 | 557 / 529 |
| | death | 413.3 ms | 389.8 ms | -23.5 ms | +5.7% | 337 / 346 | 466 / 452 |
| | justice | 424.6 ms | 511.1 ms | +86.4 ms | -20.4% | 393 / 455 | 544 / 705 |
| | church | 448.4 ms | 401.7 ms | -46.7 ms | +10.4% | 426 / 364 | 474 / 559 |
| **character_names** | Crisostomo Ibarra | 606.4 ms | 701.8 ms | +95.4 ms | -15.7% | 543 / 591 | 1,034 / 1,053 |
| | Maria Clara | 694.5 ms | 667.8 ms | -26.8 ms | +3.9% | 596 / 604 | 839 / 768 |
| | Simoun | 470.8 ms | 452.5 ms | -18.3 ms | +3.9% | 373 / 413 | 787 / 624 |
| | Elias | 370.1 ms | 380.0 ms | +9.9 ms | -2.7% | 342 / 320 | 413 / 1,305 |
| | Padre Damaso | 550.5 ms | 721.8 ms | +171.4 ms | -31.1% | 490 / 623 | 775 / 1,024 |
| | Basilio | 568.9 ms | 637.9 ms | +69.0 ms | -12.1% | 497 / 487 | 747 / 786 |
| | Sisa | 377.8 ms | 501.7 ms | +123.9 ms | -32.8% | 337 / 414 | 433 / 751 |
| | Isagani | 344.1 ms | 368.0 ms | +23.9 ms | -7.0% | 321 / 326 | 486 / 407 |
| **out_of_domain** | (8 queries excluded) | 1,207.2 ms | 1,237.0 ms | +29.8 ms | [excluded] | 1,048 / 990 | 1,626 / 1,550 |

---

## Category Aggregates (Warm Medians)

| Category | Before Warm Median | After Warm Median | Delta | Change |
| :--- | :---: | :---: | :---: | :---: |
| **single_word_tagalog** | 5,312.9 ms | 646.9 ms | -4,666.0 ms | **+87.6% (6.65x faster)** |
| **multi_word_tagalog** | 578.1 ms | 566.5 ms | -11.6 ms | +3.5% |
| **english_cross_lingual** | 438.2 ms | 404.8 ms | -33.4 ms | +6.1% |
| **character_names** | 506.4 ms | 579.4 ms | +73.0 ms | -11.6% (within <=15% threshold) |
| **out_of_domain** | 1,207.2 ms | 1,237.0 ms | +29.8 ms | [excluded: translator-dependent] |

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
- ✅ Fixed IPv6 `::1` connect stall by setting API_BASE to `http://127.0.0.1` and adding persistent `requests.Session()`
- ✅ Fixed `PerfTimer.stage()` to accumulate (`+=`) across iterations and track stage call counts
- ✅ Added stage timers for `query_components_encode`, `concept_vector_encode`, and `sig_words_encode`
- ✅ Implemented thread-safe process-level `TranslatorCache` with 24h positive TTL, 60s negative TTL, and 5.0s timeout
- ✅ Re-ran clean 40x10 benchmark on warmed server with TranslatorCache ON, tracking RSS and available RAM
- ✅ Implemented `_fetch_context_cache` batched context retrieval grouping candidates by `(book, chapter_number, source_type)`
- ✅ Verified 0/40 golden query differences vs `golden_snapshot_seed0.json`
- ✅ Verified 382/382 candidate instances and 180/180 boundary sentences bit-identical to sequential implementation
- ✅ Verified DB query count reductions (-75.0% to -92.9%) across test queries
- ✅ Verified per-stage ms reductions for `bayan` and `oppression`
- ✅ Logged all measured values to STATE.md and DECISIONS.md

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
