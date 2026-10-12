# Performance Baselines

**Purpose**: Document current performance metrics before optimization begins.

---

## Search Endpoint (`/api/v1/search`)

### Timing Breakdown
To be measured in Phase 0:
- [ ] Query preprocessing time
- [ ] Model encoding time (base XLM-R)
- [ ] Theme expansion time
- [ ] Lexical candidate retrieval time
- [ ] Semantic fallback time (when triggered)
- [ ] Hybrid re-ranking time
- [ ] Theme classification time (per result)
- [ ] Context expansion time (per result)
- [ ] Total end-to-end latency

### Query Categories
To be measured across golden query set:
- [ ] Single-word Tagalog queries
- [ ] Multi-word Tagalog queries
- [ ] English cross-lingual queries
- [ ] Character name queries
- [ ] Theme keyword queries
- [ ] Out-of-domain queries (should reject quickly)

### Metrics
- [ ] P50 latency: ___ ms
- [ ] P95 latency: ___ ms
- [ ] P99 latency: ___ ms
- [ ] Mean latency: ___ ms
- [ ] Cache hit rate: ___ % (after caching implemented)

---

## Content Endpoints

### `/api/v1/chapters` (list chapters)
- [ ] Current latency: ___ ms
- [ ] Database query time: ___ ms

### `/api/v1/chapters/{book}/{chapter}` (chapter content)
- [ ] Current latency: ___ ms
- [ ] Theme classification overhead: ___ ms per sentence
- [ ] Typical chapter size: ___ sentences

### `/api/v1/sentences/{id}/sanggunian` (buod-to-full alignment)
- [ ] Current latency: ___ ms
- [ ] Alignment cache hit rate: ___ %
- [ ] RobustAligner runtime: ___ ms per chapter

### `/api/v1/characters/chapters` (character appearances)
- [ ] Lexical mode latency: ___ ms
- [ ] Semantic mode latency: ___ ms

---

## Database Performance

### Query Plans (EXPLAIN ANALYZE)
To be captured:
- [ ] Semantic search with pgvector cosine_distance
- [ ] Lexical ILIKE filters on sentence_text
- [ ] Multi-filter queries (book + chapter + source_type)
- [ ] Theme classification neighbor lookups

### Index Usage
Current indexes (to be documented):
- [ ] List all existing indexes
- [ ] Identify missing indexes
- [ ] Measure index scan vs seq scan ratios

---

## Model Inference

### Encoding Performance
- [ ] Base XLM-R single sentence: ___ ms
- [ ] Base XLM-R batch (10 sentences): ___ ms
- [ ] DAPT model single sentence: ___ ms
- [ ] DAPT model batch (10 sentences): ___ ms

### Memory Usage
- [ ] Model load time: ___ s
- [ ] Model memory footprint: ___ MB
- [ ] Theme matrix size: ___ MB

---

## Frontend Performance

### Initial Load
- [ ] Time to First Byte (TTFB): ___ ms
- [ ] First Contentful Paint (FCP): ___ ms
- [ ] Largest Contentful Paint (LCP): ___ ms
- [ ] Bundle size: ___ KB (gzipped)

### Search Interaction
- [ ] Time from keystroke to API request: ___ ms
- [ ] Time from API response to UI update: ___ ms

---

**Status**: Awaiting Phase 0 measurement instrumentation.
