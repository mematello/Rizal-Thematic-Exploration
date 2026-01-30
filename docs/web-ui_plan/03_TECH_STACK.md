# Technology Stack & Rationale

## Stack Overview

| Layer | Technology | Version | Why This Choice |
|-------|-----------|---------|-----------------|
| **Frontend Framework** | Next.js | 14+ | SSR/SSG, performance, React ecosystem |
| **UI Library** | React | 18+ | Industry standard, large ecosystem |
| **Language** | TypeScript | 5+ | Type safety, better DX, fewer runtime errors |
| **Styling** | Tailwind CSS | 3.4+ | Utility-first, small bundle, mobile-first |
| **Component Library** | shadcn/ui | Latest | Accessible, customizable, copy-paste |
| **State Management** | Zustand | 4+ | Lightweight, simple API |
| **Server State** | React Query | 5+ | Caching, fetching, synchronization built-in |
| **Animations** | Framer Motion | 11+ | Declarative, performant |
| **Icons** | Lucide React | Latest | Tree-shakeable, consistent style |
| **Backend** | FastAPI | 0.109+ | Python ML ecosystem, async, auto-docs |
| **Database** | PostgreSQL | 15+ | Reliable, pgvector extension |
| **Vector Search** | pgvector | Latest | Native Postgres, no separate DB needed |
| **Caching** | Redis | 7+ | Fast, standard for caching |
| **ML Model** | sentence-transformers | 2.3+ | Pre-trained multilingual BERT |
| **Frontend Hosting** | Vercel | N/A | Next.js-native, edge network, zero-config |
| **Backend Hosting** | Railway / Fly.io | N/A | Docker support, Postgres included, PH edge nodes |
| **CDN** | Cloudflare | N/A | Free tier, global edge, image optimization |

---

## Frontend Stack Deep Dive

### Next.js 14 (App Router)

**What it is**: React framework with server-side rendering, file-based routing, and API routes.

**Why we chose it**:
1. **Performance**: SSR for /explore (SEO), CSR for /search (dynamic)
2. **Image Optimization**: Automatic WebP conversion, lazy loading
3. **Code Splitting**: Automatic per-route, reduces bundle size
4. **Edge Functions**: Can run API routes close to users (Philippines edge)
5. **Developer Experience**: Hot reload, TypeScript native, great docs

**Alternatives considered**:
- **Remix**: Better data loading patterns, but smaller ecosystem
- **SvelteKit**: Lighter bundle, but smaller talent pool
- **Vite + React Router**: More manual setup, no SSR out-of-box

**Decision**: Next.js wins for maturity + Vercel deployment synergy.

---

### TypeScript 5+

**What it is**: JavaScript with static typing.

**Why we chose it**:
1. **Type Safety**: Catch errors at compile-time (e.g., wrong API response shape)
2. **IntelliSense**: Better autocomplete in VS Code
3. **Refactoring**: Rename variables across files safely
4. **Documentation**: Types serve as inline documentation

**Example benefit**:
```typescript
// Without TypeScript
const result = await searchAPI(query);
console.log(result.ressults.noli); // Typo! Runtime error

// With TypeScript
const result: SearchResponse = await searchAPI(query);
console.log(result.results.noli); // Typo caught at compile-time ✓
```

**Alternatives considered**:
- **Plain JavaScript**: Faster to write, but error-prone at scale
- **JSDoc**: Type hints without TS, but less robust

**Decision**: TypeScript is industry standard for serious React projects.

---

### Tailwind CSS 3.4+

**What it is**: Utility-first CSS framework.

**Why we chose it**:
1. **Bundle Size**: Purges unused styles → ~12KB final CSS (vs Bootstrap's 150KB)
2. **Design System**: Enforces spacing/color consistency via config
3. **Mobile-First**: Breakpoints (sm:, md:, lg:) built-in
4. **Developer Speed**: No context switching between HTML/CSS files
5. **Customization**: Easy to add Paper & Ink colors

**Configuration**:
```javascript
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      colors: {
        'brand-cream': '#FFF8E1',
        'noli-gold': '#F57F17',
        // ...
      },
      fontFamily: {
        crimson: ['var(--font-crimson)', 'serif'],
        roboto: ['var(--font-roboto)', 'sans-serif'],
      }
    }
  }
}
```

**Alternatives considered**:
- **Styled Components**: CSS-in-JS, but larger bundle + runtime cost
- **CSS Modules**: Good isolation, but more boilerplate
- **Vanilla CSS**: Maximum control, but no design system enforcement

**Decision**: Tailwind aligns with performance budget + mobile-first needs.

---

### shadcn/ui

**What it is**: Collection of accessible, unstyled React components you copy into your project.

**Why we chose it**:
1. **Accessibility**: Built on Radix UI (WCAG 2.1 AA compliant)
2. **No Vendor Lock-in**: You own the code (not an npm dependency)
3. **Customizable**: Tailwind-styled, easy to match Paper & Ink theme
4. **Components we need**: Accordion (context expansion), Tabs (Noli/Fili switcher), Dropdown

**Example usage**:
```bash
npx shadcn-ui@latest add accordion
# Copies component to components/ui/accordion.tsx
# You can edit it directly
```

**Alternatives considered**:
- **Material UI**: Too heavy (200KB+), doesn't match aesthetic
- **Ant Design**: Enterprise feel, not academic
- **Headless UI**: Good, but shadcn adds Tailwind styling

**Decision**: shadcn/ui is perfect for accessible, customizable components.

---

### Zustand (UI State)

**What it is**: Lightweight state management (1KB).

**Why we chose it**:
1. **Simple API**: No boilerplate (unlike Redux)
2. **No Provider**: Direct imports, cleaner code
3. **DevTools**: Time-travel debugging support
4. **Perfect for UI state**: Filters, tabs, pagination

**Example store**:
```typescript
import create from 'zustand';

export const useSearchStore = create((set) => ({
  activeTab: 'noli',
  switchTab: (tab) => set({ activeTab: tab }),
}));

// Usage in component:
const { activeTab, switchTab } = useSearchStore();
```

**Alternatives considered**:
- **Redux Toolkit**: Overkill for simple UI state
- **Context API**: Re-renders all consumers (performance issue)
- **Jotai**: Similar to Zustand, slightly different API

**Decision**: Zustand is simplest for our needs.

---

### React Query (Server State)

**What it is**: Data fetching + caching library.

**Why we chose it**:
1. **Automatic Caching**: Don't re-fetch same query immediately
2. **Background Refetch**: Keep data fresh without manual logic
3. **Loading/Error States**: Built-in (no manual useState)
4. **Deduplication**: Multiple components can use same query
5. **Optimistic Updates**: Update UI before server responds

**Example**:
```typescript
const { data, isLoading, error } = useQuery({
  queryKey: ['search', query],
  queryFn: () => searchAPI(query),
  staleTime: 5 * 60 * 1000, // Don't refetch for 5 min
});
```

**Alternatives considered**:
- **SWR**: Similar, but React Query has better DevTools
- **Apollo Client**: Overkill (we're not using GraphQL)
- **Manual fetch + useState**: Too much boilerplate

**Decision**: React Query is industry standard for REST APIs.

---

### Framer Motion (Animations)

**What it is**: Declarative animation library for React.

**Why we chose it**:
1. **Performant**: Uses CSS transforms (GPU-accelerated)
2. **Declarative**: Animations described in JSX, not imperative code
3. **Gestures**: Drag, swipe support (for mobile tab switcher)
4. **Layout Animations**: Automatically animates layout changes

**Example**:
```typescript
<motion.div
  initial={{ opacity: 0, x: -20 }}
  animate={{ opacity: 1, x: 0 }}
  exit={{ opacity: 0, x: 20 }}
  transition={{ duration: 0.2 }}
>
  {content}
</motion.div>
```

**Alternatives considered**:
- **CSS Animations**: Manual, harder to sync with React state
- **React Spring**: Physics-based, but steeper learning curve
- **GSAP**: Powerful, but not React-native (imperative API)

**Decision**: Framer Motion is most React-idiomatic.

---

### Lucide React (Icons)

**What it is**: Icon library with 1,000+ icons, optimized for React.

**Why we chose it**:
1. **Tree-Shakeable**: Only bundle icons you use (~200 bytes each)
2. **Consistent Style**: Line-art design matches Paper & Ink aesthetic
3. **Accessible**: Proper `aria-hidden` attributes
4. **Customizable**: Size, color, stroke width props

**Usage**:
```typescript
import { Search, BookOpen, Scale } from 'lucide-react';

<Search size={20} className="text-brand-brown" />
```

**Alternatives considered**:
- **Hero Icons**: Good, but Lucide has more variety
- **React Icons**: Large bundle if not careful
- **Font Awesome**: Outdated icon font approach

**Decision**: Lucide is modern, lightweight, and complete.

---

## Backend Stack Deep Dive

### FastAPI (Python 3.11+)

**What it is**: Modern Python web framework, async-first.

**Why we chose it**:
1. **ML Integration**: sentence-transformers is Python, no language barrier
2. **Async Support**: Handle slow BERT embeddings without blocking
3. **Auto Validation**: Pydantic DTOs validate requests automatically
4. **Auto Docs**: Swagger UI at /docs for free
5. **Type Hints**: Python 3.10+ type checking (mypy compatible)
6. **Performance**: Comparable to Node.js (Starlette under the hood)

**Example endpoint**:
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

@app.post("/api/v1/search")
async def search(req: SearchRequest):
    # Pydantic validates req.query exists and is string
    results = await engine.search(req.query, req.top_k)
    return results
```

**Alternatives considered**:
- **Flask**: Simpler, but no async, no auto-validation
- **Django**: Too heavy for API-only project
- **Node.js/Express**: Would require Python subprocess for BERT (messy)

**Decision**: FastAPI is perfect for Python ML + modern API design.

---

### PostgreSQL 15+ with pgvector

**What it is**: Relational database with vector similarity extension.

**Why we chose it**:
1. **Vector Search**: pgvector supports cosine similarity on 768-dim embeddings
2. **Full-Text Search**: Built-in `to_tsvector` for lexical matching
3. **Relational Data**: Proper foreign keys for sentences ↔ themes
4. **Mature**: Battle-tested, good tooling (pgAdmin, Postico)
5. **Free Hosting**: Supabase/Railway include Postgres

**Vector search example**:
```sql
SELECT sentence_text,
       1 - (embedding <=> query_embedding) AS similarity
FROM sentences
ORDER BY embedding <=> query_embedding
LIMIT 10;
```

**Alternatives considered**:
- **Pinecone/Weaviate**: Dedicated vector DBs, but extra cost + complexity
- **MongoDB**: No native vector search (would need Atlas Vector Search)
- **SQLite**: No pgvector support, not scalable

**Decision**: Postgres + pgvector is simplest for MVP.

---

### Redis 7+ (Caching)

**What it is**: In-memory key-value store.

**Why we chose it**:
1. **Speed**: Sub-millisecond reads (vs Postgres ~5ms)
2. **Expiry**: Built-in TTL (Time To Live) for cache invalidation
3. **Rate Limiting**: Track request counts with INCR + EXPIRE
4. **Session Storage**: Can store user preferences later (if needed)

**Cache strategy**:
```python
import hashlib
import redis
import json

r = redis.Redis()

def search_cached(query: str):
    cache_key = f"search:{hashlib.md5(query.encode()).hexdigest()}"
    
    # Try cache
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Compute
    results = engine.search(query)
    
    # Cache for 1 hour
    r.setex(cache_key, 3600, json.dumps(results))
    return results
```

**Alternatives considered**:
- **Memcached**: Similar, but Redis has more features (sorted sets, pub/sub)
- **In-memory Python dict**: Lost on server restart, not shared across instances

**Decision**: Redis is standard for API caching.

---

### sentence-transformers (BERT)

**What it is**: Python library for sentence embeddings using Transformer models.

**Why we chose it**:
1. **Pre-trained Models**: `xlm-roberta-base` supports 100+ languages (Filipino included)
2. **Easy API**: `model.encode(text)` → 768-dim vector
3. **Fine-tuning**: Can train on domain-specific data later
4. **Efficient**: Optimized for inference (quantization support)

**Model selection**:
```python
from sentence_transformers import SentenceTransformer

# Multilingual BERT variant
model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

# Filipino text works:
embedding = model.encode("Ang hindi magmahal sa sariling wika")
# Returns: array([0.123, -0.456, ...]) shape=(768,)
```

**Alternatives considered**:
- **OpenAI Embeddings**: Too expensive ($0.0001/token), needs API key
- **Cohere**: Similar cost issue
- **Custom BERT training**: Too time-consuming for MVP

**Decision**: sentence-transformers is free, proven, and Filipino-ready.

---

## Hosting & Infrastructure

### Vercel (Frontend)

**What it is**: Platform-as-a-Service for Next.js apps.

**Why we chose it**:
1. **Zero Config**: `git push` → automatic deployment
2. **Edge Network**: 70+ global locations (including Asia-Pacific)
3. **Preview Deployments**: Every PR gets a unique URL
4. **Free Tier**: Generous for student projects (100GB bandwidth/month)
5. **Next.js Native**: Built by Next.js creators (Vercel)

**Deployment flow**:
```bash
# 1. Connect GitHub repo to Vercel
# 2. Every push to main triggers:
   - Build (npm run build)
   - Deploy to production
   - Invalidate CDN cache
# 3. PRs get preview URLs (e.g., pr-42.vercel.app)
```

**Alternatives considered**:
- **Netlify**: Similar, but Vercel has better Next.js integration
- **Railway**: Can host frontend, but Vercel specializes in it
- **AWS Amplify**: More complex setup

**Decision**: Vercel is obvious choice for Next.js.

---

### Railway / Fly.io (Backend)

**What it is**: Platform for deploying Docker containers.

**Why we chose them**:
1. **Docker Support**: Package FastAPI + BERT model in one image
2. **PostgreSQL Included**: No separate DB hosting needed
3. **Redis Included**: Railway offers add-on
4. **Logs & Metrics**: Built-in dashboards
5. **Philippines Edge**: Fly.io has Manila region (low latency)

**Deployment**:
```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Railway deployment
railway login
railway init
railway up  # Deploys from Dockerfile
```

**Alternatives considered**:
- **Heroku**: Deprecated free tier
- **AWS EC2**: Too manual (security groups, load balancers)
- **Google Cloud Run**: Good, but Railway simpler

**Decision**: Railway for ease, Fly.io if need Philippines edge.

---

### Cloudflare (CDN)

**What it is**: Content Delivery Network with free tier.

**Why we chose it**:
1. **Free**: Unlimited bandwidth on free plan
2. **Global Edge**: Caches static assets close to users
3. **Image Optimization**: Automatic WebP conversion, resizing
4. **DDoS Protection**: Included by default
5. **Analytics**: Basic traffic insights

**What it caches**:
- Static assets: CSS, JS, fonts (1 week cache)
- Images: Theme card icons, logos (1 month cache)
- HTML: Not cached (dynamic search results)

**Setup**:
```bash
# 1. Point domain DNS to Cloudflare nameservers
# 2. Enable "Auto Minify" for CSS/JS
# 3. Set cache TTL rules in dashboard
```

**Alternatives considered**:
- **AWS CloudFront**: More powerful, but complex setup
- **Fastly**: Enterprise-focused, no free tier

**Decision**: Cloudflare free tier is perfect for MVP.

---

## Development Tools

### Package Managers

**Frontend**: `pnpm` (faster than npm, disk-efficient)
**Backend**: `poetry` (better dependency resolution than pip)

### Linting & Formatting

**Frontend**:
- ESLint (catch bugs, enforce code style)
- Prettier (auto-format on save)

**Backend**:
- Ruff (Python linter, replaces Flake8 + Black)
- mypy (type checking)

### Testing

**Frontend**:
- Vitest (faster than Jest, Vite-native)
- React Testing Library (component tests)

**Backend**:
- pytest (Python standard)
- httpx (async HTTP client for testing FastAPI)

---

## Dependency Management

### Frontend (package.json)

```json
{
  "dependencies": {
    "next": "^14.0.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "zustand": "^4.4.0",
    "@tanstack/react-query": "^5.0.0",
    "framer-motion": "^11.0.0",
    "lucide-react": "^0.300.0",
    "zod": "^3.22.0",
    "clsx": "^2.0.0",
    "tailwind-merge": "^2.0.0"
  },
  "devDependencies": {
    "typescript": "^5.3.0",
    "tailwindcss": "^3.4.0",
    "@types/react": "^18.2.0",
    "eslint": "^8.56.0",
    "prettier": "^3.1.0",
    "vitest": "^1.0.0"
  }
}
```

**Total bundle size** (gzipped): ~180KB

---

### Backend (pyproject.toml)

```toml
[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.109.0"
uvicorn = {extras = ["standard"], version = "^0.25.0"}
sentence-transformers = "^2.3.0"
psycopg2-binary = "^2.9.9"
pgvector = "^0.2.4"
redis = "^5.0.0"
pydantic = "^2.5.0"
python-dotenv = "^1.0.0"

[tool.poetry.dev-dependencies]
pytest = "^7.4.0"
pytest-asyncio = "^0.21.0"
httpx = "^0.25.0"
ruff = "^0.1.0"
mypy = "^1.7.0"
```

---

## Environment Variables

### Frontend (.env.local)

```bash
# API endpoint
NEXT_PUBLIC_API_URL=https://api.rizal-explorer.com

# Analytics (optional)
NEXT_PUBLIC_SENTRY_DSN=https://...

# Environment
NEXT_PUBLIC_ENV=development
```

### Backend (.env)

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/rizal_db

# Redis
REDIS_URL=redis://localhost:6379

# ML Model
BERT_MODEL_PATH=/models/xlm-roberta-base

# API Keys (if needed later)
# OPENAI_API_KEY=sk-...

# Environment
ENVIRONMENT=development
DEBUG=true
```

---

## Why This Stack Works Together

### The Full Flow

1. **User types query** → React component (TypeScript)
2. **Zustand** manages filter state (client-side)
3. **React Query** caches and fetches from API
4. **Next.js** makes HTTP request → FastAPI backend
5. **FastAPI** checks **Redis** cache → if miss:
   - **sentence-transformers** (BERT) generates embedding
   - **PostgreSQL + pgvector** searches vectors
   - **PostgreSQL full-text** searches keywords
6. Results returned → **React Query** caches → UI updates
7. **Framer Motion** animates transition
8. **Tailwind CSS** styles everything
9. **Vercel** serves frontend, **Cloudflare** caches assets
10. **Railway/Fly.io** runs backend

**Every piece has a clear job. No redundancy.**

---

## Cost Analysis (Monthly)

| Service | Free Tier | Expected Usage | Cost |
|---------|-----------|----------------|------|
| Vercel | 100GB bandwidth | ~20GB | $0 |
| Railway | 500 hrs compute | 1 instance 24/7 | $5 |
| Postgres (Railway) | 1GB storage | 500MB | $0 |
| Redis (Railway) | 25MB | 10MB | $0 |
| Cloudflare | Unlimited | N/A | $0 |
| Domain (optional) | N/A | 1 domain | $12/year |
| **Total** | | | **~$5-6/month** |

**For a thesis project, this is exceptional value.**

---

## Summary: Why This Stack?

1. **Performance**: Meets 500KB budget, <3s load time
2. **Scalability**: Can handle 10x traffic with minimal changes
3. **Developer Experience**: TypeScript, hot reload, great docs
4. **Cost**: ~$5/month for production-ready hosting
5. **Maintenance**: Managed services (no server management)
6. **Academic Credibility**: Modern, professional stack (good for portfolio)

**This stack balances pragmatism (ship fast) with quality (production-ready).**

---

## Next Steps

1. Read **04_COMPONENT_SPECS.md** for implementation details
2. Read **05_API_INTEGRATION.md** for backend connection
3. Start coding! See **06_IMPLEMENTATION_PLAN.md**