# Performance Optimization Plan

**Goal**: Make search and data-loading endpoints feel fast for public users with NO change to result rankings or correctness.

## Phases

### Phase 0: Baseline & Measurement Setup
- [ ] Create golden query set (~40 queries: Tagalog, English, out-of-domain)
- [ ] Add stage-level timing instrumentation to search pipeline
- [ ] Document current performance baselines (P50, P95, P99)
- [ ] Identify EXPLAIN ANALYZE targets for slow DB queries
- [ ] Profile model inference times (base XLM-R vs DAPT)

### Phase 1: Backend Quick Wins (Low-Hanging Fruit)
- [ ] Add Redis caching for frequent queries and theme lookups
- [ ] Optimize N+1 query patterns in content endpoints
- [ ] Add connection pooling tuning
- [ ] Lazy-load theme matrix (defer until first search)
- [ ] Cache model embeddings for static theme data

### Phase 2: Database & pgvector Optimization
- [ ] Add appropriate indexes (sentence_text, book+chapter+source_type)
- [ ] Create covering indexes for common query patterns
- [ ] Optimize vector search with HNSW index parameters
- [ ] Tune postgres.conf for read-heavy workload
- [ ] Consider materialized views for theme aggregations

### Phase 3: Model & Embedding Optimization
- [ ] Batch encode operations where possible
- [ ] Pre-compute and cache component vectors for multi-word queries
- [ ] Evaluate ONNX runtime for faster inference
- [ ] Profile memory usage and optimize model loading
- [ ] Consider model quantization (if correctness preserved)

### Phase 4: Frontend Optimization
- [ ] Add request debouncing for search
- [ ] Implement optimistic UI updates
- [ ] Add loading skeletons and progressive rendering
- [ ] Optimize bundle size (code splitting, lazy loading)
- [ ] Add service worker for static asset caching

### Phase 5: Pre-Deployment Load Testing
- [ ] Run load tests with realistic query distribution
- [ ] Identify bottlenecks under concurrent load
- [ ] Stress test Redis and Postgres
- [ ] Validate caching behavior
- [ ] Document resource requirements (RAM, CPU)

### Phase 6: Review & PR
- [ ] Document all performance improvements with before/after metrics
- [ ] Verify no ranking changes using golden query set
- [ ] Create comprehensive PR with performance report
- [ ] Get approval before merge

---

## Ranked Hypotheses for Slowness

### High Confidence (Likely Bottlenecks)
1. **Model inference on every request** — No caching, encoding happens per query
2. **Expensive semantic fallback** — Fetches top_k * 15 candidates for semantic search
3. **Theme classification per result** — Calls `_classify_themes()` for every returned sentence with full context expansion
4. **N+1 context expansion queries** — Each result triggers separate DB queries for neighbor sentences
5. **No query result caching** — Identical queries recompute everything

### Medium Confidence
6. **Large theme matrix dot products** — Computed multiple times per search
7. **Character regex matching** — Runs pattern matching against all character aliases
8. **Unindexed filtering** — ILIKE queries on sentence_text without proper indexes
9. **Cross-lingual translation overhead** — Google Translator API calls for unmapped terms
10. **Passage context embedding** — Averages embeddings from 8-sentence windows per result

### Low Confidence (Needs Profiling)
11. **Frontend bundle size** — May delay initial page load
12. **Sanggunian alignment** — RobustAligner runs DP algorithm on full chapters
13. **Database connection overhead** — No connection pooling configured
14. **JSON serialization** — Large embedding vectors in responses
15. **Stopword/vocabulary checks** — Multiple set membership tests per query token

---

**Next**: Phase 0 measurement setup (see STATE.md for current progress)
