# Performance Optimization Handoff

**Purpose**: Context for continuing this work across sessions.

---

## Quick Start

1. **Read this first**: Always check `docs/perf/STATE.md` for current status
2. **Never commit to main**: Work on `perf/speed-optimization` branch
3. **Update memory**: After completing any task, update STATE.md and DECISIONS.md
4. **Verify correctness**: Use golden query set to ensure no ranking changes

---

## Project Context

This is a thesis project — an academic search engine for José Rizal's novels. The system uses:
- **Hybrid search**: Lexical (ILIKE) + Semantic (pgvector with XLM-RoBERTa)
- **Custom DAPT model**: Domain-adapted XLM-R for Tagalog literary text
- **Multi-stage pipeline**: Query analysis → Candidate retrieval → Re-ranking → Theme classification → Context expansion

**Critical constraint**: Result rankings and search quality MUST NOT change. Any ranking change requires explicit user approval and detailed logging in DECISIONS.md.

---

## Key Files to Know

### Backend Search Pipeline
- **[backend/app/core/engine.py](../../backend/app/core/engine.py)**: Main RizalEngine (search logic, scoring, theme classification)
- **[backend/app/api/v1/search.py](../../backend/app/api/v1/search.py)**: Search endpoint
- **[backend/app/api/v1/content.py](../../backend/app/api/v1/content.py)**: Chapter content, themes, character endpoints
- **[backend/app/api/v1/alignment.py](../../backend/app/api/v1/alignment.py)**: Buod-to-full text alignment
- **[backend/app/core/robust_aligner.py](../../backend/app/core/robust_aligner.py)**: DP sequence aligner
- **[backend/app/models/database.py](../../backend/app/models/database.py)**: SQLAlchemy models

### Infrastructure
- **[docker-compose.yml](../../docker-compose.yml)**: Postgres + Redis services
- **[backend/app/core/config.py](../../backend/app/core/config.py)**: Settings and configuration

### Data
- Embeddings stored in Postgres `sentences` table (768-dim vectors)
- Two embedding columns: `embedding` (base XLM-R), `embedding_dapt` (DAPT model)
- Themes stored in `themes` table with pre-computed embeddings

---

## Request Flow Summary

### Search Flow (`/api/v1/search`)
1. **Query preprocessing**: Tokenization, stopword analysis, OOV detection
2. **Cross-lingual mapping**: English → Tagalog via dictionary + dynamic translator
3. **Theme expansion**: Match query to theme keywords using base model
4. **Candidate retrieval**:
   - **Lexical first**: ILIKE filters on sentence_text
   - **Semantic fallback**: pgvector cosine_distance if lexical insufficient
5. **Hybrid re-ranking**: Weighted combo of semantic + lexical scores
6. **Post-processing**: Theme classification + context expansion per result
7. **Precision scoring**: Multi-component matching for complex queries

### Content Flow (`/api/v1/chapters/{book}/{chapter}`)
1. Query sentences for book+chapter+source_type
2. For each sentence: Call `_classify_themes()`
   - Uses 4-step character-gated matching
   - Requires neighbor context (8-sentence window)
3. Return sentences with theme labels

### Sanggunian Flow (`/api/v1/sentences/{id}/sanggunian`)
1. Fetch buod sentence
2. Check per-chapter alignment cache
3. If cache miss: Run RobustAligner on full chapter
4. Return aligned full-text passage with scores

---

## Known Performance Bottlenecks (Hypotheses)

### High Priority
1. **No result caching** — Every search recomputes from scratch
2. **Model inference per request** — Base + DAPT models encode on every call
3. **Theme classification overhead** — Runs on every result with context expansion
4. **Expensive semantic fallback** — Retrieves 15x more candidates than needed
5. **N+1 context queries** — Separate DB calls for neighbor sentences

### Needs Measurement
- Theme matrix operations (dot products, argmax)
- Character regex matching performance
- Unindexed ILIKE queries
- Cross-lingual translation API calls
- JSON serialization overhead

---

## Measurement Strategy (Phase 0)

### Golden Query Set (~40 queries)
Categories to cover:
- Single-word Tagalog (e.g., "kalayaan", "simbahan")
- Multi-word Tagalog (e.g., "pag-ibig ni Maria Clara")
- English cross-lingual (e.g., "corruption", "education")
- Character names (e.g., "Crisostomo Ibarra", "Simoun")
- Out-of-domain (should reject: e.g., "tiktok", "bitcoin")

### Instrumentation Points
Add timing to:
1. Query preprocessing
2. Model encoding (separate for base vs DAPT)
3. Theme expansion
4. Lexical retrieval
5. Semantic fallback (when triggered)
6. Re-ranking loop
7. Theme classification per result
8. Context expansion per result
9. Total end-to-end

### EXPLAIN ANALYZE Targets
- Vector similarity search: `ORDER BY embedding.cosine_distance(query_vec)`
- Lexical filtering: `WHERE sentence_text ILIKE '%word%'`
- Multi-filter: `WHERE book = X AND chapter = Y AND source_type = Z`
- Neighbor lookups: Range queries on sentence_index

---

## Common Pitfalls

1. **Don't assume bottlenecks** — Measure first, optimize second
2. **Preserve rankings** — Any change affecting result order needs approval
3. **Cache invalidation is hard** — Design cache keys carefully
4. **Batching isn't always faster** — Model inference has overhead per batch
5. **Indexes have write cost** — Consider read/write ratio
6. **pgvector HNSW tuning is non-trivial** — Default params may not be optimal

---

## Next Session Checklist

- [ ] Read STATE.md
- [ ] Check for updates since last session
- [ ] Verify branch is `perf/speed-optimization`
- [ ] Review recent commits in DECISIONS.md
- [ ] Check for open questions or blockers

---

## Resources

- **pgvector docs**: https://github.com/pgvector/pgvector
- **SentenceTransformers**: https://www.sbert.net/
- **FastAPI performance**: https://fastapi.tiangolo.com/deployment/

---

**Remember**: Speed is important, but correctness is non-negotiable.
