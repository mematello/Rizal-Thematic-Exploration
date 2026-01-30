# System Architecture

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Browser (Mobile/Desktop)                                  │  │
│  │  - React Components (Next.js 14)                           │  │
│  │  - React Query (API state management)                      │  │
│  │  - Zustand (UI state: filters, tabs, pagination)           │  │
│  │  - Service Worker (offline caching)                        │  │
│  └───────────────────┬───────────────────────────────────────┘  │
│                      │ HTTPS                                     │
└──────────────────────┼───────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CDN / EDGE LAYER                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Cloudflare                                                │  │
│  │  - Static asset caching (CSS, JS, fonts, icons)           │  │
│  │  - Image optimization (theme card icons)                  │  │
│  │  - DDoS protection                                         │  │
│  │  - SSL/TLS termination                                     │  │
│  └───────────────────┬───────────────────────────────────────┘  │
└──────────────────────┼───────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND HOSTING                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Vercel (Next.js)                                          │  │
│  │  - Server-Side Rendering (SSR) for /explore               │  │
│  │  - Client-Side Rendering (CSR) for /search                │  │
│  │  - Edge Functions (geolocation-aware)                     │  │
│  │  - Automatic deployments from GitHub                      │  │
│  └───────────────────┬───────────────────────────────────────┘  │
└──────────────────────┼───────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     API GATEWAY LAYER                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  FastAPI (Python 3.11+)                                    │  │
│  │  - POST /api/v1/search                                     │  │
│  │  - GET  /api/v1/themes                                     │  │
│  │  - GET  /api/v1/suggestions                                │  │
│  │  - Pydantic request/response validation                   │  │
│  │  - CORS configuration (allow Vercel origin)               │  │
│  │  - Rate limiting (via Redis)                               │  │
│  └───────────────────┬───────────────────────────────────────┘  │
└──────────────────────┼───────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│ CACHE LAYER  │ │  ENGINE  │ │  DATA LAYER  │
├──────────────┤ ├──────────┤ ├──────────────┤
│ Redis        │ │ Rizal    │ │ PostgreSQL   │
│              │ │ Engine   │ │ + pgvector   │
│ - Query      │ │          │ │              │
│   cache      │ │ - BERT   │ │ - Sentences  │
│ - Rate limit │ │   embeddi│ │   table      │
│   counters   │ │   ngs    │ │ - Embeddings │
│ - Session    │ │ - Hybrid │ │   (768-dim)  │
│   data       │ │   scoring│ │ - Themes     │
│              │ │ - Context│ │   metadata   │
└──────────────┘ └──────────┘ └──────────────┘
```

---

## Data Flow: Search Query Lifecycle

### Step-by-Step Flow

```
1. USER ACTION
   User types "Edukasyon ni Ibarra" in SearchBar
   ↓

2. FRONTEND VALIDATION
   - Zod schema validates query length (min 3 chars)
   - Debounce 300ms (prevent excessive API calls)
   ↓

3. REACT QUERY CACHE CHECK
   - Check cache: queryKey: ['search', 'edukasyon ni ibarra', filters]
   - If HIT: Return cached results (skip steps 4-9)
   - If MISS: Proceed to API call
   ↓

4. API REQUEST
   POST https://api.rizal-explorer.com/v1/search
   Headers: { 'Content-Type': 'application/json' }
   Body: {
     "query": "edukasyon ni ibarra",
     "top_k": 10,
     "filters": { "match_type": "all", "min_score": 0 }
   }
   ↓

5. API GATEWAY (FastAPI)
   a. CORS check (origin: rizal-explorer.com)
   b. Rate limit check (Redis: max 10 req/min per IP)
   c. Pydantic validation (query, top_k types)
   ↓

6. REDIS CACHE CHECK (Backend)
   - Generate cache key: MD5("edukasyon ni ibarra")
   - Check Redis: GET cache_key
   - If HIT: Return cached JSON (skip steps 7-8)
   - If MISS: Proceed to engine
   ↓

7. RIZAL ENGINE PROCESSING
   a. Query Validation
      - domain_adaptive_semantic_validation(query)
      - If invalid: Return {"status": "blocked", "suggestions": [...]}
   
   b. Generate Query Embedding
      - model.encode("edukasyon ni ibarra") → 768-dim vector
      - Time: ~200ms (cached model in memory)
   
   c. Vector Search (PostgreSQL pgvector)
      - SELECT * FROM sentences
        ORDER BY embedding <=> query_embedding
        LIMIT 20
      - Cosine similarity search
      - Time: ~100ms (indexed)
   
   d. Lexical Search (PostgreSQL full-text)
      - SELECT * FROM sentences
        WHERE to_tsvector(sentence_text) @@ to_tsquery('edukasyon & ibarra')
      - Time: ~50ms
   
   e. Hybrid Scoring
      - Combine semantic + lexical scores
      - Apply CLEAR formula weighting
      - Rank top 10 per novel
   
   f. Context Retrieval
      - Fetch prev_sentences, next_sentences for each result
      - JOIN on chapter_id
   
   g. Generate Followup Suggestions
      - extract_key_themes(query) → ["pag-asa", "reporma"]
      - generate_followup_suggestions() → ["Pag-asa sa kabataan", ...]
   ↓

8. CACHE RESULTS (Backend)
   - Redis: SET cache_key, JSON.stringify(results), EX 3600 (1hr TTL)
   ↓

9. API RESPONSE
   Response (200 OK):
   {
     "status": "success",
     "results": {
       "noli": { "items": [...], "metadata": {...} },
       "elfili": { "items": [...], "metadata": {...} }
     },
     "next_queries": ["Pag-asa sa kabataan", "Edukasyon vs kolonya"],
     "query_time_ms": 285
   }
   ↓

10. FRONTEND STATE UPDATE
    - React Query caches response (staleTime: 5 min)
    - Zustand updates UI state (visibleCount, activeTab)
    - Re-render triggered
    ↓

11. UI RENDERING
    - Transform backend data → ResultCard props
    - Apply highlighting (lexical/semantic)
    - Render ResultCards with animations
    ↓

12. USER INTERACTION
    User clicks "Show Context" on a result
    → Expand animation (CSS Grid transition)
    → Display prev/next sentences (already loaded)
```

---

## Component Architecture

### Page-Level Components

```
app/
├── layout.tsx                    # Root layout (fonts, providers)
├── page.tsx                      # Home page (hero search)
├── search/
│   └── page.tsx                  # SearchResultsPage
├── explore/
│   └── page.tsx                  # ThemeExplorationPage
├── about/
│   └── page.tsx                  # MethodologyPage
└── api/                          # Next.js API routes (if needed)
    └── health/
        └── route.ts              # Health check endpoint
```

### Component Hierarchy (Search Results Page)

```
SearchResultsPage
├── SearchBar (persistent, sticky)
├── FilterBar (sticky)
├── (Desktop) SplitView
│   ├── NovelColumn (Noli)
│   │   ├── ResultCard[]
│   │   └── EmptyNovelState (if no results)
│   └── NovelColumn (Fili)
│       └── ResultCard[]
├── (Mobile) TabSwitcher
│   └── AnimatedView
│       └── ResultCard[]
└── LoadMoreButton (pagination)

ResultCard
├── Header
│   ├── ChapterTitle
│   └── ConfidenceBadge
├── PassageBody (highlighted HTML)
├── ContextExpansion (collapsible)
└── ScoreVisualizer
    ├── SemanticBar
    └── LexicalBar
```

---

## State Management Strategy

### React Query (Server State)

**Purpose**: Manage API data fetching, caching, and synchronization.

**Queries**:
```typescript
// Search results
useQuery({
  queryKey: ['search', query, filters],
  queryFn: () => searchAPI(query, filters),
  staleTime: 5 * 60 * 1000,      // 5 min
  cacheTime: 10 * 60 * 1000,     // 10 min
  retry: 2,
});

// Theme list (static data, long cache)
useQuery({
  queryKey: ['themes'],
  queryFn: fetchThemes,
  staleTime: Infinity,           // Never refetch
});
```

**Benefits**:
- Automatic background refetching
- Deduplication (multiple components can use same query)
- Loading/error states built-in
- Optimistic updates support

---

### Zustand (Client State)

**Purpose**: Manage UI state (filters, tabs, pagination).

**Store Example**:
```typescript
// stores/searchStore.ts
import create from 'zustand';

interface SearchState {
  // Filters
  activeFilter: 'all' | 'exact' | 'semantic';
  minScore: number;
  
  // UI State
  activeTab: 'noli' | 'fili';
  visibleCount: number;
  
  // Actions
  setFilter: (filter: 'all' | 'exact' | 'semantic') => void;
  setMinScore: (score: number) => void;
  switchTab: (tab: 'noli' | 'fili') => void;
  loadMore: () => void;
}

export const useSearchStore = create<SearchState>((set) => ({
  activeFilter: 'all',
  minScore: 0,
  activeTab: 'noli',
  visibleCount: 10,
  
  setFilter: (filter) => set({ activeFilter: filter }),
  setMinScore: (score) => set({ minScore: score }),
  switchTab: (tab) => set({ activeTab: tab }),
  loadMore: () => set((state) => ({ visibleCount: state.visibleCount + 10 })),
}));
```

**Why Zustand over Redux?**
- 10x less boilerplate
- No Provider wrapper needed
- Tiny bundle size (1KB vs Redux's 15KB)
- Perfect for simple UI state

---

## Backend Architecture (FastAPI)

### Project Structure

```
backend/
├── app/
│   ├── main.py                   # FastAPI app entry point
│   ├── api/
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── search.py         # POST /search
│   │       ├── themes.py         # GET /themes
│   │       └── suggestions.py    # GET /suggestions
│   ├── core/
│   │   ├── config.py             # Settings (env vars)
│   │   ├── engine.py             # RizalEngine class
│   │   └── dependencies.py       # FastAPI dependencies
│   ├── models/
│   │   ├── schemas.py            # Pydantic DTOs
│   │   └── database.py           # SQLAlchemy models
│   ├── services/
│   │   ├── embedding.py          # BERT model wrapper
│   │   ├── cache.py              # Redis utilities
│   │   └── validator.py          # Query validation
│   └── middleware/
│       ├── cors.py               # CORS configuration
│       └── rate_limit.py         # Rate limiting
├── tests/
│   ├── test_search.py
│   └── test_engine.py
├── alembic/                      # Database migrations
├── Dockerfile
├── requirements.txt
└── .env.example
```

### API Endpoint Specifications

#### POST /api/v1/search

**Request**:
```json
{
  "query": "string (min 3 chars)",
  "top_k": "integer (default: 10, max: 20)",
  "filters": {
    "match_type": "'all' | 'exact' | 'semantic'",
    "min_score": "integer (0-100)"
  },
  "include_context": "boolean (default: true)"
}
```

**Response (Success)**:
```json
{
  "status": "success",
  "results": {
    "noli": {
      "items": [
        {
          "id": "string",
          "chapter_number": "integer",
          "chapter_title": "string",
          "sentence_text": "string",
          "scores": {
            "semantic": "integer (0-100)",
            "lexical": "integer (0-100)",
            "final": "integer (0-100)"
          },
          "match_type": "'exact' | 'partial_lexical' | 'semantic'",
          "context": {
            "prev": ["string[]"],
            "next": ["string[]"]
          },
          "themes": [
            {
              "id": "string",
              "label": "string",
              "confidence": "float (0-1)"
            }
          ]
        }
      ],
      "metadata": {
        "count": "integer",
        "avg_score": "float"
      }
    },
    "elfili": { /* same structure */ }
  },
  "next_queries": ["string[]"],
  "query_time_ms": "integer"
}
```

**Response (Validation Failure)**:
```json
{
  "status": "blocked",
  "reason": "'domain_incoherent' | 'too_short' | 'stops_only'",
  "message": "string",
  "suggestions": ["string[]"]
}
```

---

## Database Schema (PostgreSQL + pgvector)

### Tables

#### `sentences`
```sql
CREATE TABLE sentences (
  id SERIAL PRIMARY KEY,
  book VARCHAR(10) NOT NULL,              -- 'noli' | 'elfili'
  chapter_number INTEGER NOT NULL,
  chapter_title TEXT NOT NULL,
  sentence_index INTEGER NOT NULL,        -- Position within chapter
  sentence_text TEXT NOT NULL,
  embedding vector(768),                  -- pgvector type
  created_at TIMESTAMP DEFAULT NOW(),
  
  UNIQUE(book, chapter_number, sentence_index)
);

-- Indexes
CREATE INDEX idx_sentences_book ON sentences(book);
CREATE INDEX idx_sentences_chapter ON sentences(book, chapter_number);

-- Vector similarity index (IVFFlat for speed)
CREATE INDEX idx_sentences_embedding ON sentences 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Full-text search index
CREATE INDEX idx_sentences_fts ON sentences 
USING GIN (to_tsvector('filipino', sentence_text));
```

#### `themes`
```sql
CREATE TABLE themes (
  id VARCHAR(50) PRIMARY KEY,             -- 'education', 'justice', etc.
  label_filipino TEXT NOT NULL,
  label_english TEXT NOT NULL,
  description TEXT,
  icon_name VARCHAR(50),                  -- 'book', 'scale', etc.
  keywords TEXT[],                        -- Array of related terms
  created_at TIMESTAMP DEFAULT NOW()
);
```

#### `sentence_themes` (Many-to-Many)
```sql
CREATE TABLE sentence_themes (
  sentence_id INTEGER REFERENCES sentences(id),
  theme_id VARCHAR(50) REFERENCES themes(id),
  confidence FLOAT NOT NULL,              -- 0.0 to 1.0
  PRIMARY KEY (sentence_id, theme_id)
);

CREATE INDEX idx_sentence_themes_theme ON sentence_themes(theme_id);
```

---

## Caching Strategy

### Three-Layer Cache

```
┌─────────────────────────────────────────────────┐
│  Layer 1: Browser Cache (Service Worker)       │
│  - Static assets (CSS, JS, fonts): 1 week      │
│  - App shell HTML: 1 day                        │
│  - Previous search results: 5 entries (IndexedDB)
└─────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Layer 2: React Query Cache (Memory)           │
│  - Search results: 5 min stale, 10 min cache   │
│  - Theme list: Infinity (static data)          │
└─────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Layer 3: Redis Cache (Server)                 │
│  - Query results: 1 hour TTL                    │
│  - Rate limit counters: 1 min TTL              │
│  - BERT embeddings (hot queries): 24 hr TTL    │
└─────────────────────────────────────────────────┘
```

### Cache Invalidation Rules

**Never invalidate**:
- Theme list (changes rarely, deployed via migration)
- Sentence embeddings (pre-computed, immutable)

**Invalidate on deploy**:
- Service Worker cache (new version detected)
- React Query cache (page refresh)

**Time-based expiry**:
- Search results: 1 hour (balance freshness vs. performance)
- Rate limits: 1 minute window (sliding)

---

## Security Considerations

### API Security

1. **CORS Configuration**
   ```python
   # backend/app/middleware/cors.py
   from fastapi.middleware.cors import CORSMiddleware
   
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://rizal-explorer.com"],  # Prod only
       allow_credentials=True,
       allow_methods=["GET", "POST"],
       allow_headers=["Content-Type"],
   )
   ```

2. **Rate Limiting**
   - 10 requests/minute per IP address
   - 100 requests/hour per IP address
   - Implemented via Redis + middleware

3. **Input Validation**
   - Pydantic schemas reject malformed requests
   - SQL injection protection (parameterized queries via SQLAlchemy)
   - XSS protection (React auto-escapes, but double-check `dangerouslySetInnerHTML`)

### Data Privacy

**No personal data collected**:
- No user accounts (no PII)
- No tracking cookies
- Anonymous usage analytics only (optional: Vercel Analytics)

**Query logging**:
- Backend logs queries for debugging
- Logs rotated daily, deleted after 7 days
- No IP addresses stored long-term

---

## Scalability Considerations

### Current Architecture Limits

**Expected Load (First 6 Months)**:
- 100-500 daily active users
- ~1,000 searches/day
- 10-50 concurrent users (peak)

**This architecture handles**:
- 10,000 searches/day easily
- 500 concurrent users (with caching)

### Future Scaling Paths

**If traffic grows 10x** (1,000+ concurrent):
1. Add CDN edge caching for search results (Cloudflare Cache API)
2. Horizontal scaling: Multiple FastAPI instances behind load balancer
3. Read replicas for PostgreSQL (pgvector supports read-only replicas)

**If traffic grows 100x** (10,000+ concurrent):
1. Dedicated vector database (Pinecone, Weaviate)
2. Kubernetes deployment (auto-scaling)
3. Separate BERT inference service (GPU instances)

**For MVP**: Current architecture is sufficient. Optimize when needed.

---

## Monitoring & Observability

### Frontend Monitoring

**Vercel Analytics** (built-in):
- Core Web Vitals (LCP, FID, CLS)
- Real User Monitoring (RUM)
- Page load times by geography

**Sentry** (error tracking):
```typescript
// app/layout.tsx
import * as Sentry from "@sentry/nextjs";

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  tracesSampleRate: 0.1,  // 10% of transactions
  environment: process.env.NODE_ENV,
});
```

### Backend Monitoring

**FastAPI Logging**:
```python
# app/main.py
import logging
from fastapi import FastAPI, Request
import time

logger = logging.getLogger("uvicorn")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        f"{request.method} {request.url.path} "
        f"- {response.status_code} - {process_time:.2f}s"
    )
    return response
```

**Key Metrics to Track**:
- Search query latency (p50, p95, p99)
- BERT embedding generation time
- Cache hit rate (Redis)
- Database query performance
- Error rate by endpoint

---

## Disaster Recovery

### Backup Strategy

**Database Backups**:
- Automated daily backups (Supabase/Railway built-in)
- Retention: 7 days
- Manual snapshot before major migrations

**Code Backups**:
- GitHub as source of truth
- Tagged releases for each deployment
- Protected main branch (no force push)

### Rollback Plan

**Frontend** (Vercel):
- One-click rollback to previous deployment
- Preview deployments for testing

**Backend** (Railway/Fly.io):
- Docker image tagging (rollback to previous tag)
- Database migrations use Alembic (reversible)

**Time to Recover**:
- Frontend: < 5 minutes (Vercel rollback)
- Backend: < 15 minutes (redeploy previous image)
- Database: 1-2 hours (restore from backup if needed)

---

## Next Steps

After understanding the architecture, proceed to:
1. **03_TECH_STACK.md**: Detailed technology choices
2. **04_COMPONENT_SPECS.md**: Component implementation details
3. **05_API_INTEGRATION.md**: Backend integration guide