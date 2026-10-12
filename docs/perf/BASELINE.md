# Performance Baselines - Phase 1 Results

**Measurement Date**: 2026-10-12  
**Branch**: `perf/speed-optimization`  
**API Version**: 1.0.0  
**Queries Tested**: 40 (39 completed, 1 failed due to server crash)

---

## Executive Summary

**🚨 CRITICAL FINDINGS:**
- Single-word Tagalog queries are **10+ seconds** (unacceptable for production)
- "edukasyon" query is **32-34 seconds** (severe outlier)
- First request cold start: **26 seconds**
- DB query instrumentation returned 0 counts (PERF_BENCHMARK flag issue)

**Target for public deployment**: P95 latency < 1000ms

---

## Model Load & Startup

| Metric | Value |
|--------|-------|
| First request (cold start) | 25.98s |
| Memory footprint (delta) | 0.05 MB |
| Peak memory | 33.4 MB |

**Analysis**: 26-second cold start is acceptable for server initialization, but NOT for per-request behavior. Memory footprint is surprisingly low (suggests measurement captured script memory, not server process).

---

## Search Endpoint (`/api/v1/search`)

### Latency by Category (HTTP end-to-end)

| Category | Cold (ms) | Warm P50 (ms) | Warm P95 (ms) | Status |
|----------|-----------|---------------|---------------|--------|
| **Single-word Tagalog** | 10,448 | 9,701 | 13,493 | 🚨 **CRITICAL** |
| Multi-word Tagalog | 3,048 | 2,833 | 3,744 | ⚠️ Slow |
| English cross-lingual | 2,650 | 2,512 | 3,025 | ⚠️ Slow |
| Character names | 2,657 | 2,520 | 2,699 | ⚠️ Slow |
| Out-of-domain (reject) | 3,401 | 3,433 | 3,689 | ⚠️ Should be fast |

### Individual Query Analysis

**🔴 Severe Outliers (>30s):**
- `edukasyon`: 32.5s cold, 33.2s warm P50 (**REGRESSION or ANOMALY**)

**🔴 Critical Slow Queries (>10s):**
- `kalayaan`: 10.5s cold, 10.4s warm P50
- `pag-ibig`: 10.4s cold, 11.3s warm P50
- `kamatayan`: 17.1s cold, 11.7s warm P50

**🟢 Acceptable Queries (<3s):**
- `bayan`: 2.5s
- `simbahan`: 2.7s
- `paaralan`: 2.7s
- Most character names: 2.4-2.9s
- English cross-lingual: 2.4-3.2s

---

## Database Performance

### Query Plans (from EXPLAIN ANALYZE)

**⚠️ WARNING**: EXPLAIN queries failed due to vector syntax error in initial run.

**Database Statistics:**
- Total sentences: 16,402
- Summary sentences: 2,186 (primary search target)
- Full text sentences: 14,216
- Total themes: 545
- Sentences with embeddings: 16,402 (100%)
- Sentences with DAPT embeddings: 11,822 (72%)

**Existing Indexes:**
- `ix_sentences_book` (B-tree)
- `ix_sentences_chapter_number` (B-tree)
- `ix_sentences_id` (B-tree, primary key)
- `ix_sentences_passage_id` (B-tree)

**Missing Indexes:**
- ❌ No `source_type` index
- ❌ No composite index on `(book, source_type, chapter_number)`
- ❌ No GIN index on `sentence_text` for ILIKE queries (pg_trgm)
- ❌ No HNSW or IVFFlat index on `embedding` vector column

---

## Stage-Level Timing Breakdown

**❌ NOT CAPTURED**: DB query counts and engine-level timings returned 0.

**Root Cause**: `PERF_BENCHMARK=1` environment variable was likely not set in the server process, only in the benchmark script. The instrumentation in `backend/app/api/v1/search.py` requires the server to have this flag set.

**Impact**: Cannot determine which stage (preprocessing, model encoding, lexical retrieval, semantic fallback, theme classification, context expansion) is the bottleneck without re-running with proper instrumentation.

---

## Confirmed Bottlenecks (Pre-Measurement Hypotheses)

Based on HTTP end-to-end latency patterns:

### 🔴 Hypothesis #1: Theme Classification + Context Expansion (LIKELY CONFIRMED)
**Evidence**: 
- Single-word queries (which return 10-20 results) are 4x slower than multi-word queries (which return 0-7 results)
- Latency correlates with result count, not query complexity
- Each result triggers `_classify_themes()` + `_expand_context()` with neighbor lookups

**Calculation**: 
- `kalayaan` returns 20 results × ~500ms per result = ~10s ✅ matches observed latency

### 🔴 Hypothesis #2: Model Inference (PARTIALLY CONFIRMED)
**Evidence**:
- Cold vs warm timing shows minimal difference (10.4s cold vs 10.4s warm P50 for `kalayaan`)
- Suggests model is not the primary bottleneck once loaded
- But 26s first request indicates heavy model initialization

### 🔴 Hypothesis #3: Semantic Fallback Retrieval (NEEDS VERIFICATION)
**Evidence**:
- Out-of-domain queries (which trigger fallback but return 0 results) are 3.4s
- If semantic fallback is expensive, we'd expect longer latency even with 0 results
- Unclear without stage-level timing

### ⚠️ "edukasyon" Anomaly (INVESTIGATE)
**Observation**: This query is 10x slower than similar queries (32s vs 2-3s)
**Possible causes**:
- Theme expansion returns massive keyword set
- Fallback retrieves huge candidate pool
- Database query plan degrades for this specific term
- Network timeout or external API call (translator?)

---

## Preliminary Optimization Priorities

### Phase 2 Immediate Actions:

**Priority 1: Fix Instrumentation and Re-Run**
- Set `PERF_BENCHMARK=1` in server terminal
- Re-run benchmark with stage-level timing
- Confirm theme classification is the bottleneck

**Priority 2: Optimize Theme Classification**
1. Batch theme classification (one query for all results, not N queries)
2. Remove or optimize context expansion (8-sentence window × 20 results = 160 DB queries)
3. Add composite indexes: `(book, chapter_number, source_type, sentence_index)`
4. Consider caching theme classifications per sentence_id

**Priority 3: Add Result Caching**
- Redis cache for full search results (key: query hash)
- TTL: 1 hour
- Expected impact: 90%+ cache hit rate for common queries

**Priority 4: Investigate "edukasyon" Anomaly**
- Run with `DEBUG_SEARCH=1` to see execution path
- Check theme expansion output
- Profile database query for this term
- May reveal systemic issue affecting other queries

---

## Golden Snapshot Status

✅ **Golden snapshot captured**: `docs/perf/golden_snapshot.json` (181 KB)
- 39 queries with full ranked output
- Result IDs, scores (semantic, lexical, final, precision)
- Theme labels per result
- Ready for correctness verification with `compare_golden.py`

---

## Next Steps

1. ✅ **COMPLETED**: Baseline measurements captured
2. ⚠️ **TODO**: Re-run with proper `PERF_BENCHMARK=1` in server to capture stage timings
3. ⚠️ **TODO**: Fix and re-run `explain_queries.py` for database query plans
4. **DECISION POINT**: Review this baseline before proceeding to Phase 2 optimizations

---

## Notes

- Server crashed during "online" query (run 9/10) - likely memory/resource exhaustion after 39 consecutive searches
- Deep_translator not available during benchmark (but engine has it)
- Translator caching is client-side only; server-side translation calls still occur
- Warm query performance shows no improvement over cold (suggests no effective caching)

---

**Conclusion**: Search is **10-30x slower** than production targets. Theme classification per result is the likely culprit. Must re-run with full instrumentation to confirm before optimizing.
