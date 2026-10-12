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

## Ranked Hypotheses for Slowness (Pre-Measurement)

### 🔴 High Confidence (Likely Top 3 Bottlenecks)
1. **Theme classification + context expansion** — Runs `_classify_themes()` with 8-sentence context window per result (10-20 results × multiple DB queries each)
2. **Model inference on every request** — No caching; base + DAPT models encode query every time (cold start + per-query overhead)
3. **Semantic fallback retrieval** — When lexical insufficient, fetches `top_k × 15` candidates with pgvector cosine_distance

### 🟡 Medium Confidence
4. **Theme expansion preprocessing** — Computes dot product with theme_matrix (~100s of themes) before candidate retrieval
5. **Character regex matching** — Runs compiled patterns against all aliases for 4-step theme gating
6. **Cross-lingual translation API** — Google Translator calls for unmapped English terms (network latency)
7. **N+1 query patterns** — Context expansion triggers separate DB calls per result for neighbor sentences
8. **No result caching** — Identical queries recompute everything from scratch

### 🟢 Low Confidence (Probably Not Primary Bottlenecks)
9. **ILIKE queries** — Note: `WHERE sentence_text ILIKE '%x%'` cannot use B-tree index; pg_trgm GIN index candidate
10. **HNSW vector index with WHERE filters** — Potential recall risk if filters applied after vector search
11. **Query analysis tokenization** — Pure Python but minimal work
12. **JSON serialization overhead** — Embeddings are large but typically not the blocker
13. **Stopword/vocabulary set membership** — Multiple checks but in-memory and fast
14. **Connection pooling** — SQLAlchemy default may not be tuned
15. **Sanggunian alignment** — RobustAligner DP algorithm (cached per chapter)

### Phase 1 Measurement Goals
- Validate top 3 hypotheses with timing data
- Identify if bottlenecks are consistent across query types
- Quantify cold vs warm start impact
- Count DB queries per request (N+1 patterns)
- Profile memory usage and model load time
- Check EXPLAIN plans for index usage and row counts

---

**Next**: Phase 0 measurement setup (see STATE.md for current progress)
