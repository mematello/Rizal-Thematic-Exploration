# Performance Baselines - Phase 1 Results

**Measurement Date**: 2026-10-12  
**Branch**: `perf/speed-optimization`  
**API Version**: 1.0.0  
**Queries Tested**: 40 (all completed successfully)

> **⚠️ INVALIDATION NOTICE (2026-10-14)**:
> All HTTP end-to-end latency numbers in this baseline table are **INVALID for absolute HTTP latency**.
> Diagnostic profiling discovered that `scripts/benchmark.py` targeted `http://localhost:8000` without a reused `requests.Session()`. On Windows, resolving `localhost` triggered an IPv6 (`::1`) connection attempt that stalled for ~2.0 seconds per request before falling back to IPv4 (`127.0.0.1`), artificially inflating all HTTP measurements by ~2,000 ms.
> **Server-side stage timings and query execution logic recorded here remain valid.** Corrected end-to-end baselines must be collected using `http://127.0.0.1:<port>` with `requests.Session()`.
> 
> **For the valid, rigorously controlled Before vs. After performance tables, please refer to [FINAL_REPORT.md](FINAL_REPORT.md) (Section 3).**

---

## Executive Summary

**🚨 CRITICAL FINDINGS:**
- Single-word Tagalog queries average **10.8 seconds** (unacceptable for production)
- "edukasyon" query is **38.9 seconds cold, 32.1s warm** (severe outlier)
- **~2.4s floor** on every query, even with minimal DB work
- **Out-of-domain rejection (0 DB queries) takes 3.4s** - slower than valid queries
- **No effective caching**: warm == cold across all queries
- **Slow queries are semantic_fallback mode**; fast queries are lexical mode

**Target for public deployment**: P95 latency < 1000ms  
**Current performance**: 10-12x too slow for critical queries

---

## Model Load & Startup

| Metric | Value |
|--------|-------|
| First request (cold start) | 25.00s |
| Memory delta | -11.79 MB (client-side measurement, not meaningful) |
| Peak memory | 28.73 MB (client-side measurement, not meaningful) |

**Note**: Memory measurements were captured in the benchmark client script, not the server process, so they do not reflect actual server memory usage.

---

## Search Endpoint (`/api/v1/search`)

### Latency by Category (HTTP end-to-end)

| Category | Cold (ms) | Warm P50 (ms) | Warm P95 (ms) | Status |
|----------|-----------|---------------|---------------|--------|
| **Single-word Tagalog** | 10,814 | 9,772 | 11,833 | 🚨 **CRITICAL** |
| Multi-word Tagalog | 2,943 | 2,771 | 2,913 | ⚠️ Slow |
| English cross-lingual | 2,482 | 2,447 | 2,520 | ⚠️ Slow |
| Character names | 2,595 | 2,476 | 2,560 | ⚠️ Slow |
| Out-of-domain (reject) | 3,388 | 3,411 | 3,624 | ⚠️ Should be fast |

### Key Individual Queries

| Query | Cold (ms) | Warm P50 (ms) | DB Queries | Result Mode | Observation |
|-------|-----------|---------------|------------|-------------|-------------|
| **edukasyon** | 38,886 | 32,101 | 44 | semantic_fallback | 🚨 Severe outlier (12x slower than similar queries) |
| **kalayaan** | 10,693 | 12,310 | 44 | semantic_fallback | 🚨 Critical slow |
| **pag-ibig** | 10,542 | 11,506 | 44 | semantic_fallback | 🚨 Critical slow |
| **kamatayan** | 11,278 | 10,862 | 44 | semantic_fallback | 🚨 Critical slow |
| simbahan | 2,704 | 2,588 | 38 | lexical | ✅ Fast (lexical mode) |
| bayan | 2,497 | 2,493 | 42 | lexical | ✅ Fast (lexical mode) |
| paaralan | 2,659 | 2,576 | 42 | lexical | ✅ Fast (lexical mode) |
| education | 2,447 | 2,482 | 42 | lexical | ✅ Fast (lexical + cross-lingual) |
| oppression | 2,706 | 2,663 | 34 | lexical | ✅ Fast (English fallback) |
| **korapsyon** | 4,465 | 3,454 | **0** | early reject | ⚠️ Rejected before DB |
| **tiktok** | 3,694 | 3,432 | **0** | early reject | ⚠️ Out-of-domain slower than valid queries |
| **online** | 3,161 | 3,454 | **0** | early reject | ⚠️ Out-of-domain slower than valid queries |

---

## Critical Observations

### 1. DB Query Count vs Latency (Refutes N+1 Hypothesis)

DB query counts are nearly identical across fast and slow queries, but latency differs by 4-12x:

| Query | DB Queries | Latency (warm P50) | Ratio |
|-------|------------|--------------------|-------|
| bayan | 42 | 2.5s | 1x (baseline) |
| simbahan | 38 | 2.6s | 1x |
| education | 42 | 2.5s | 1x |
| **kalayaan** | 44 | **12.3s** | **4.9x** |
| **pag-ibig** | 44 | **11.5s** | **4.6x** |
| **edukasyon** | 44 | **32.1s** | **12.8x** |

**Conclusion**: The extra time is NOT from DB round-trips or N+1 queries. Theme classification / context expansion N+1 hypothesis is **REFUTED**.

### 2. The 2.4-Second Floor

Every query has a baseline cost of ~2.4-2.6 seconds, regardless of:
- DB work (korapsyon: 4 DB queries, still 3.5s)
- Result count (out-of-domain: 0 results, 3.4s)
- Query complexity (single-word vs multi-word)

**Candidates for the floor**:
- Model encoding (base + DAPT)
- Theme expansion / matrix operations
- Query preprocessing / analysis
- Translator network calls (for cross-lingual queries)
- Response serialization

### 3. Semantic Fallback Correlation

**Slow queries** (10+ seconds): kalayaan, pag-ibig, edukasyon, kamatayan  
**Fast queries** (2-3 seconds): bayan, simbahan, paaralan, education, oppression

From `golden_snapshot.json` metadata:
- Slow queries have `result_mode: "semantic_fallback"`
- Fast queries have `result_mode: "lexical"`

**But**: English cross-lingual queries (education, oppression) also trigger fallback yet remain fast (2.4-2.7s).

**Hypothesis**: The bottleneck is NOT "semantic fallback" itself, but something specific to certain Tagalog terms during fallback:
- Theme expansion keyword generation
- Semantic candidate pool size
- Validation/filtering passes

### 4. No Effective Caching

Warm queries show NO improvement over cold:
- kalayaan: 10.7s cold → 12.3s warm (actually slower!)
- edukasyon: 38.9s cold → 32.1s warm (marginal improvement, but still slow)
- bayan: 2.5s cold → 2.5s warm (identical)

**Conclusion**: No query result caching, no model output caching, no meaningful warm-up effect.

### 5. Out-of-Domain Queries Are Slow

Queries that reject early (korapsyon, tiktok, bitcoin, etc.) with 0 DB queries still take **3.4 seconds** on average - SLOWER than valid lexical queries (2.5s).

**Implication**: The early-reject path is NOT cheap. Likely costs:
- Full query preprocessing
- Model encoding
- Theme expansion (to check domain relevance)
- Translation API calls

---

## Database Performance

### Statistics

- Total sentences: 16,402
- Summary sentences: 2,186 (primary search target)
- Full text sentences: 14,216
- Total themes: 545
- Sentences with embeddings: 16,402 (100%)
- Sentences with DAPT embeddings: 11,822 (72%)

### Existing Indexes

- `ix_sentences_book` (B-tree)
- `ix_sentences_chapter_number` (B-tree)
- `ix_sentences_id` (B-tree, primary key)
- `ix_sentences_passage_id` (B-tree)

### Missing Indexes

- ❌ No `source_type` index
- ❌ No composite index on `(book, source_type, chapter_number)`
- ❌ No GIN index on `sentence_text` for ILIKE queries (pg_trgm)
- ❌ No vector index (HNSW or IVFFlat) on `embedding` column

### Query Plans

**❌ ALL EXPLAIN QUERIES FAILED**: Vector embedding syntax error. The embedding was being stringified character-by-character (`'[', '0', '.', ...`) instead of as a proper array literal.

**Status**: `explain_queries.py` needs to be fixed and re-run (see Phase 1b tasks below).

---

## Stage-Level Timing Breakdown

**❌ NOT CAPTURED**: Only `db_query_count` was returned in `engine_timings`. Stage-level timing (preprocessing, model encoding, theme expansion, retrieval, re-ranking, etc.) was not instrumented.

**Why**: The instrumentation was not added to `backend/app/core/engine.py`. Only the DB query counter was active.

**Impact**: Cannot determine which specific stage (theme expansion vs model encoding vs candidate pool processing) is the bottleneck for kalayaan/edukasyon without stage-level timing.

---

## Phase 1b: Required Corrections

### Tasks Before Phase 2

1. **Add stage-level timing instrumentation** (behind `PERF_BENCHMARK` flag):
   - Query preprocessing / analysis
   - Translator call (network vs cache)
   - Base model encode
   - DAPT model encode  
   - Theme expansion (time + keyword count)
   - Lexical retrieval (candidate count)
   - Semantic fallback retrieval (requested vs returned count)
   - Re-ranking loop
   - `_classify_themes` total
   - `_expand_context` total
   - Precision scoring
   - Response serialization

2. **Fix `explain_queries.py`**:
   - Format embedding as proper array literal for PostgreSQL
   - Run each EXPLAIN in its own transaction to avoid abort cascade
   - Regenerate `docs/perf/query_plans.txt`

3. **Targeted re-run** (5 queries × 5 runs each):
   - kalayaan (slow semantic fallback)
   - edukasyon (severe outlier)
   - bayan (fast lexical)
   - tiktok (out-of-domain early reject)
   - oppression (fast English fallback)

4. **Deliverable**: Short report with per-stage timing, explaining:
   - Which stage causes the 2.4s floor
   - Which stage explains kalayaan/edukasyon slowness
   - Database query plans (EXPLAIN ANALYZE)
   - Revised ranked bottleneck list

---

## Revised Bottleneck Hypotheses (Pre-Instrumentation)

Based on the corrected data analysis:

### 🔴 Hypothesis #1: Theme Expansion for Specific Terms (NEW)
**Evidence**:
- Slow queries are semantic_fallback with identical DB counts (44) as fast queries
- English fallback queries (education, oppression) are fast, so "fallback" itself is not the cause
- Likely: certain Tagalog terms generate massive theme expansion keyword sets or expensive matrix operations

**Test**: Capture theme expansion timing + keyword count for kalayaan vs bayan

### 🔴 Hypothesis #2: Semantic Candidate Pool Processing (NEW)
**Evidence**:
- Semantic fallback queries retrieve `top_k * 15` candidates for re-ranking
- If kalayaan retrieves a huge candidate pool (e.g., 3000 rows) vs bayan (200 rows), the re-ranking loop would be expensive

**Test**: Log semantic candidate pool size (requested vs returned)

### 🔴 Hypothesis #3: Model Encoding (Floor Contributor)
**Evidence**:
- 2.4s floor on every query, including 0-DB-query rejects
- Out-of-domain queries still encode the query with base + DAPT models

**Test**: Measure base_model.encode() and dapt_model.encode() timing

### 🟡 Hypothesis #4: Translator Network Calls (Floor Contributor)
**Evidence**:
- 2.4s floor includes cross-lingual queries
- Translator cache was client-side only; server-side translation not frozen

**Test**: Log translator timing + cache hit/miss

### 🟢 Hypothesis #5: Theme Classification / Context Expansion (REFUTED)
**Evidence**:
- DB query counts are identical (38-44) for fast and slow queries
- If N+1 context expansion was the culprit, bayan (42 queries) would be as slow as kalayaan (44 queries)
- Slow queries spend extra time elsewhere, NOT in DB round-trips

**Status**: **REFUTED** by the data

---

## Next Steps

1. ✅ **COMPLETED**: Baseline measurements captured, corrected analysis
2. ⚠️ **IN PROGRESS**: Add stage-level timing instrumentation
3. ⚠️ **IN PROGRESS**: Fix EXPLAIN queries
4. ⚠️ **TODO**: Targeted re-run (5 queries × 5 runs)
5. **DECISION POINT**: Review instrumented results before Phase 2 optimizations

---

## Notes

- All 40 queries completed successfully (10 runs each)
- Golden snapshot contains full ranked results for all queries
- DB query counting worked correctly (PERF_BENCHMARK was set properly)
- Stage-level timing is the missing piece for root cause analysis
- Memory measurements from client script are not meaningful for server analysis

---

**Status**: Phase 1 baseline complete; Phase 1b instrumentation required before Phase 2.
