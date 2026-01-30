# Implementation Plan

## 10-Week Development Roadmap

This document provides a week-by-week plan for building the Rizal Thematic Exploration System from scratch to production deployment.

---

## Prerequisites

Before starting Week 1, ensure you have:

- [ ] Node.js 20+ installed
- [ ] Python 3.11+ installed
- [ ] Docker Desktop installed
- [ ] VS Code with recommended extensions
- [ ] GitHub account (repo created)
- [ ] Vercel account (linked to GitHub)
- [ ] Railway/Fly.io account

---

## Week 1-2: Foundation & Setup

### Week 1: Project Scaffolding

**Goal**: Set up development environment and basic project structure.

#### Day 1: Frontend Setup
```bash
# Create Next.js project
npx create-next-app@latest rizal-explorer \
  --typescript \
  --tailwind \
  --app \
  --eslint

cd rizal-explorer

# Install core dependencies
npm install \
  zustand \
  @tanstack/react-query \
  framer-motion \
  lucide-react \
  zod \
  clsx \
  tailwind-merge

# Install dev dependencies
npm install -D \
  @types/node \
  prettier \
  prettier-plugin-tailwindcss \
  vitest \
  @testing-library/react \
  @testing-library/user-event
```

**Checklist**:
- [ ] Next.js project created
- [ ] Dependencies installed
- [ ] Git initialized, first commit
- [ ] GitHub repo connected

#### Day 2: Backend Setup (Parallel Task)

```bash
# Create backend directory
mkdir backend
cd backend

# Initialize Poetry
poetry init
poetry add fastapi uvicorn[standard] sentence-transformers \
  psycopg2-binary pgvector redis pydantic python-dotenv

poetry add --dev pytest pytest-asyncio httpx ruff mypy
```

**Project structure**:
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   └── v1/
│   │       └── search.py
│   ├── core/
│   │   ├── config.py
│   │   └── engine.py
│   └── models/
│       └── schemas.py
├── tests/
├── pyproject.toml
└── .env.example
```

**Checklist**:
- [ ] Poetry project initialized
- [ ] FastAPI app structure created
- [ ] Basic health check endpoint (`/health`)
- [ ] Can run: `uvicorn app.main:app --reload`

#### Day 3: Design System Implementation

Create `tailwind.config.ts`:
```typescript
import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'brand-cream': '#FFF8E1',
        'brand-brown': '#3E2723',
        'brand-blue': '#1A237E',
        'brand-text': '#261612',
        'noli-gold': '#F57F17',
        'fili-magenta': '#AD1457',
        'semantic-teal': '#00695C',
        'lexical-amber': '#FFF59D',
        'lexical-text': '#E65100',
      },
      fontFamily: {
        roboto: ['var(--font-roboto)', 'sans-serif'],
        crimson: ['var(--font-crimson)', 'serif'],
      },
    },
  },
  plugins: [],
};

export default config;
```

Add fonts to `app/layout.tsx`:
```typescript
import { Crimson_Text, Roboto } from 'next/font/google';

const crimsonText = Crimson_Text({
  weight: ['400', '600'],
  subsets: ['latin'],
  variable: '--font-crimson',
  display: 'swap',
});

const roboto = Roboto({
  weight: ['400', '500', '700'],
  subsets: ['latin'],
  variable: '--font-roboto',
  display: 'swap',
});
```

**Checklist**:
- [ ] Tailwind config with custom colors
- [ ] Fonts loaded (Crimson Text, Roboto)
- [ ] Global CSS with highlight styles
- [ ] Test page renders with correct styling

#### Day 4-5: Database Setup

**Docker Compose** for local development:
```yaml
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: ankane/pgvector:latest
    environment:
      POSTGRES_USER: rizal
      POSTGRES_PASSWORD: dev123
      POSTGRES_DB: rizal_db
    ports:
      - "5432:5432"
    volumes:
      - pg_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  pg_data:
```

**Database migrations** (Alembic):
```bash
cd backend
poetry add alembic
alembic init alembic

# Create first migration
alembic revision -m "create sentences table"
```

**Migration file**:
```python
# alembic/versions/001_create_sentences.py
def upgrade():
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    op.create_table(
        'sentences',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('book', sa.String(10), nullable=False),
        sa.Column('chapter_number', sa.Integer, nullable=False),
        sa.Column('chapter_title', sa.Text, nullable=False),
        sa.Column('sentence_index', sa.Integer, nullable=False),
        sa.Column('sentence_text', sa.Text, nullable=False),
        sa.Column('embedding', Vector(768)),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )
    
    # Indexes
    op.create_index('idx_sentences_book', 'sentences', ['book'])
    op.execute("""
        CREATE INDEX idx_sentences_embedding ON sentences 
        USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = 100)
    """)
```

**Checklist**:
- [ ] Docker Compose running PostgreSQL + Redis
- [ ] Alembic migrations set up
- [ ] Sentences table created
- [ ] Can connect from backend: `psycopg2.connect()`

---

### Week 2: Core Components

**Goal**: Build foundational UI components.

#### Day 6-7: ResultCard Component

Follow specifications in `04_COMPONENT_SPECS.md`.

**Implementation steps**:
1. Create `types/search.ts` with interfaces
2. Create `components/ResultCard.tsx`
3. Create `components/ScoreVisualizer.tsx`
4. Add highlight styles to `globals.css`
5. Create test file `__tests__/ResultCard.test.tsx`

**Test data** (`lib/mockData.ts`):
```typescript
export const mockResults: ResultCardProps[] = [
  {
    id: 'test-1',
    novel: 'noli',
    chapter: 4,
    chapterTitle: 'Erehe at Filibustero',
    passageHtml: 'Ang <span class="lexical-match">edukasyon</span> ay susi...',
    contextHtml: '<strong>Before:</strong> Context here...',
    semanticScore: 85,
    lexicalScore: 92,
    confidenceBadge: true,
  },
];
```

**Checklist**:
- [ ] ResultCard component complete
- [ ] ScoreVisualizer component complete
- [ ] Unit tests passing
- [ ] Storybook stories (optional)
- [ ] Visual regression tested on mobile

#### Day 8-9: SearchBar Component

**Implementation**:
1. Create `components/SearchBar.tsx`
2. Implement type-ahead dropdown
3. Add debouncing (300ms)
4. Test keyboard navigation

**Checklist**:
- [ ] Hero variant works
- [ ] Persistent variant works
- [ ] Type-ahead suggestions display
- [ ] Keyboard navigation (Arrow keys, Enter)
- [ ] Mobile keyboard shows correct type

#### Day 10: SkeletonLoader Component

**Simple implementation**:
```typescript
// components/SkeletonLoader.tsx
import { useState } from 'react';

const LOADING_TIPS = [
  'Did you know? El Filibusterismo was finished in Ghent, Belgium.',
  'Analyzing semantic connections...',
  // ... more tips
];

export function SkeletonLoader() {
  const [tip] = useState(() =>
    LOADING_TIPS[Math.floor(Math.random() * LOADING_TIPS.length)]
  );

  return (
    <div role="status" aria-busy="true">
      <p className="text-center italic animate-pulse">{tip}</p>
      {[1, 2, 3].map((i) => (
        <div key={i} className="animate-pulse">
          {/* Skeleton structure */}
        </div>
      ))}
    </div>
  );
}
```

**Checklist**:
- [ ] Loading tips rotate randomly
- [ ] Skeleton cards look realistic
- [ ] Pulse animation smooth
- [ ] Accessible (aria-busy, sr-only text)

---

## Week 3-4: Backend Integration

### Week 3: API Development

**Goal**: Build working search endpoint.

#### Day 11-12: RizalEngine Integration

**Tasks**:
1. Copy RizalEngine from your existing `vbest.py`
2. Refactor into `app/core/engine.py`
3. Load BERT model on startup
4. Implement search method

**Example**:
```python
# app/core/engine.py
from sentence_transformers import SentenceTransformer
import numpy as np

class RizalEngine:
    def __init__(self, model_name: str = "xlm-roberta-base"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None  # Load from DB
    
    def search(self, query: str, top_k: int = 10):
        # Generate query embedding
        query_emb = self.model.encode(query)
        
        # Vector search (pgvector)
        results = self.db.query(f"""
            SELECT * FROM sentences
            ORDER BY embedding <=> '{query_emb}'
            LIMIT {top_k}
        """)
        
        return results
```

**Checklist**:
- [ ] BERT model loads successfully
- [ ] Can generate embeddings
- [ ] Search returns results
- [ ] Lexical + semantic scoring works

#### Day 13-14: Search Endpoint

**Create** `app/api/v1/search.py`:
```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.engine import RizalEngine

router = APIRouter()
engine = RizalEngine()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

@router.post("/search")
async def search(req: SearchRequest):
    if len(req.query) < 3:
        raise HTTPException(400, "Query too short")
    
    results = engine.search(req.query, req.top_k)
    
    return {
        "status": "success",
        "results": {
            "noli": {"items": results["noli"]},
            "elfili": {"items": results["elfili"]},
        },
        "next_queries": [],
    }
```

**Checklist**:
- [ ] `/api/v1/search` endpoint works
- [ ] Pydantic validation
- [ ] Returns correct JSON structure
- [ ] CORS enabled for localhost:3000

#### Day 15: Data Seeding

**Task**: Load Noli & Fili text into database.

**Script** `scripts/seed_data.py`:
```python
import pandas as pd
from app.core.engine import RizalEngine

def seed_database():
    # Load CSVs
    noli_df = pd.read_csv('data/noli_sentences.csv')
    fili_df = pd.read_csv('data/elfili_sentences.csv')
    
    engine = RizalEngine()
    
    # Generate embeddings
    for _, row in noli_df.iterrows():
        embedding = engine.model.encode(row['sentence_text'])
        # Insert into database
        db.execute("""
            INSERT INTO sentences (book, chapter_number, sentence_text, embedding)
            VALUES ('noli', %s, %s, %s)
        """, (row['chapter_num'], row['sentence_text'], embedding))
    
    print("Seeded database successfully")

if __name__ == "__main__":
    seed_database()
```

**Checklist**:
- [ ] CSV files prepared
- [ ] Embeddings generated (may take 1-2 hours)
- [ ] All sentences in database
- [ ] Test search returns real results

---

### Week 4: Frontend-Backend Connection

**Goal**: Connect Next.js to FastAPI.

#### Day 16-17: API Client

**Create** `lib/api/client.ts`:
```typescript
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }

  return response.json();
}

export async function searchRizal(query: string) {
  return fetchAPI('/api/v1/search', {
    method: 'POST',
    body: JSON.stringify({ query, top_k: 10 }),
  });
}
```

**Checklist**:
- [ ] API client created
- [ ] Test fetch from localhost:8000
- [ ] Error handling works
- [ ] TypeScript types match backend

#### Day 18-19: Data Transformation

**Implement** `lib/transformers/searchResults.ts` from `05_API_INTEGRATION.md`.

**Key functions**:
- `applyHighlights()` - Add HTML highlighting
- `formatContext()` - Format prev/next sentences
- `transformBackendToResultCard()` - Main transformer

**Test**:
```typescript
// __tests__/transformers.test.ts
import { transformBackendToResultCard } from '@/lib/transformers/searchResults';

test('transforms backend data correctly', () => {
  const backendItem = {
    chapter_number: 4,
    sentence_text: 'Edukasyon ang susi',
    scores: { semantic: 85, lexical: 92, final: 88 },
  };

  const result = transformBackendToResultCard(backendItem, 'edukasyon', 'noli');

  expect(result.chapter).toBe(4);
  expect(result.passageHtml).toContain('lexical-match');
  expect(result.confidenceBadge).toBe(true); // final > 85
});
```

**Checklist**:
- [ ] Transformation functions complete
- [ ] Unit tests passing
- [ ] Highlighting works (yellow + teal)
- [ ] Context formatted correctly

#### Day 20: React Query Integration

**Setup** `app/providers.tsx`:
```typescript
'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useState } from 'react';

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() => new QueryClient());

  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
}
```

**Create hook** `hooks/useRizalSearch.ts`:
```typescript
import { useQuery } from '@tanstack/react-query';
import { searchRizal } from '@/lib/api/search';
import { transformSearchResponse } from '@/lib/transformers/searchResults';

export function useRizalSearch(query: string) {
  return useQuery({
    queryKey: ['search', query],
    queryFn: async () => {
      const response = await searchRizal(query);
      return transformSearchResponse(response, query);
    },
    enabled: query.length >= 3,
    staleTime: 5 * 60 * 1000,
  });
}
```

**Checklist**:
- [ ] React Query provider set up
- [ ] useRizalSearch hook works
- [ ] Caching verified (duplicate requests don't hit backend)
- [ ] Loading states work

---

## Week 5-6: Core Features

### Week 5: Search Results Page

**Goal**: Build complete search experience.

#### Day 21-22: Desktop Layout

**Create** `app/search/page.tsx`:
```typescript
'use client';

import { useSearchParams } from 'next/navigation';
import { useRizalSearch } from '@/hooks/useRizalSearch';
import { ResultCard } from '@/components/ResultCard';
import { SkeletonLoader } from '@/components/SkeletonLoader';

export default function SearchPage() {
  const searchParams = useSearchParams();
  const query = searchParams.get('q') || '';

  const { data, isLoading } = useRizalSearch(query);

  if (isLoading) return <SkeletonLoader />;

  return (
    <div className="grid lg:grid-cols-2 gap-6 p-6">
      {/* Noli Column */}
      <div>
        <h2 className="text-xl font-bold mb-4">Noli Me Tangere</h2>
        {data?.noli.map((result) => (
          <ResultCard key={result.id} {...result} />
        ))}
      </div>

      {/* Fili Column */}
      <div>
        <h2 className="text-xl font-bold mb-4">El Filibusterismo</h2>
        {data?.fili.map((result) => (
          <ResultCard key={result.id} {...result} />
        ))}
      </div>
    </div>
  );
}
```

**Checklist**:
- [ ] Split-screen layout works
- [ ] Results render correctly
- [ ] Highlighting visible
- [ ] Scrolling smooth

#### Day 23-24: Mobile Tab Switcher

**Add** tab state management:
```typescript
const [activeTab, setActiveTab] = useState<'noli' | 'fili'>('noli');

// Mobile view
<div className="lg:hidden">
  <TabSwitcher activeTab={activeTab} onChange={setActiveTab} />
  <AnimatePresence mode="wait">
    <motion.div key={activeTab} initial={{ x: 20 }} animate={{ x: 0 }}>
      {activeTab === 'noli' ? noliResults : filiResults}
    </motion.div>
  </AnimatePresence>
</div>
```

**Checklist**:
- [ ] Tabs switch smoothly
- [ ] Swipe gestures work (optional)
- [ ] Scroll position preserved
- [ ] Test on real mobile device

#### Day 25: FilterBar Component

**Create** `components/FilterBar.tsx` per specs.

**Integrate** with Zustand:
```typescript
// stores/searchStore.ts
import create from 'zustand';

export const useSearchStore = create((set) => ({
  activeFilter: 'all',
  setFilter: (filter) => set({ activeFilter: filter }),
}));
```

**Checklist**:
- [ ] FilterBar sticky on scroll
- [ ] Filters apply to results
- [ ] Active state styling correct
- [ ] Horizontal scroll on mobile

---

### Week 6: Additional Features

#### Day 26-27: Empty States

**Create** `components/EmptyNovelState.tsx`:
```typescript
export function EmptyNovelState({ novel, query }: Props) {
  return (
    <div className="bg-gray-50 border-2 border-dashed p-8 text-center">
      <BookX className="mx-auto mb-4" size={48} />
      <h3 className="font-bold mb-2">
        Not found in {novel === 'noli' ? 'Noli Me Tangere' : 'El Filibusterismo'}
      </h3>
      <p>The term "{query}" does not appear in this volume.</p>
    </div>
  );
}
```

**Checklist**:
- [ ] Empty state shows when no results
- [ ] Contextual hints (e.g., "Simoun only in Fili")
- [ ] Styling matches design system

#### Day 28-29: Pagination

**Add** to search page:
```typescript
const [visibleCount, setVisibleCount] = useState(10);
const visibleResults = results.slice(0, visibleCount);

<button onClick={() => setVisibleCount(prev => prev + 10)}>
  Load More Passages (~15KB)
</button>
```

**Checklist**:
- [ ] "Load More" button works
- [ ] Shows remaining count
- [ ] No performance issues with 50+ cards

#### Day 30: Error Handling

**Create** `components/ErrorState.tsx`:
```typescript
export function ErrorState({ error }: { error: AppError }) {
  return (
    <div className="bg-red-50 p-6 rounded-lg">
      <h3 className="font-bold text-red-900">{error.message}</h3>
      {error.suggestions?.map((s) => (
        <button onClick={() => search(s)}>Try: {s}</button>
      ))}
    </div>
  );
}
```

**Checklist**:
- [ ] Displays API errors gracefully
- [ ] Shows suggestions when available
- [ ] Retry button works
- [ ] Network errors handled

---

## Week 7-8: Secondary Pages

### Week 7: Explore Page

#### Day 31-32: Theme Cards

**Create** `app/explore/page.tsx`:
```typescript
const THEMES = [
  { id: 'justice', titleFilipino: 'Katarungan', iconName: 'justice' },
  { id: 'education', titleFilipino: 'Edukasyon', iconName: 'education' },
  // ...
];

return (
  <div className="grid grid-cols-2 md:grid-cols-4 gap-6 p-6">
    {THEMES.map((theme) => (
      <ThemeCard
        key={theme.id}
        {...theme}
        onClick={() => router.push(`/search?theme=${theme.id}`)}
      />
    ))}
  </div>
);
```

**Checklist**:
- [ ] Theme grid responsive
- [ ] Cards clickable
- [ ] Navigate to pre-filled search
- [ ] Icons match themes

#### Day 33-34: Methodology Page

**Create** `app/about/page.tsx`:
```typescript
const [activeTab, setActiveTab] = useState<'student' | 'researcher'>('student');

return (
  <div>
    <TabSwitcher />
    {activeTab === 'student' ? <StudentView /> : <ResearcherView />}
  </div>
);
```

**Student view**: Visual funnel diagram
**Researcher view**: Technical specs in monospace

**Checklist**:
- [ ] Tab switcher works
- [ ] Student view clear and simple
- [ ] Researcher view has formulas
- [ ] Mobile-friendly

#### Day 35: Home Page Polish

**Update** `app/page.tsx`:
```typescript
export default function HomePage() {
  return (
    <main className="min-h-screen flex items-center justify-center bg-brand-cream">
      <div className="text-center">
        <h1 className="text-4xl font-bold mb-8">Rizal Thematic Search</h1>
        <SearchBar variant="hero" onSearch={(q) => router.push(`/search?q=${q}`)} />
        
        {/* Suggestion chips */}
        <div className="flex gap-2 mt-4 justify-center">
          {['Edukasyon', 'Katarungan', 'Pag-ibig'].map((s) => (
            <button key={s} className="chip">{s}</button>
          ))}
        </div>
      </div>
    </main>
  );
}
```

**Checklist**:
- [ ] Hero search bar centered
- [ ] Suggestion chips functional
- [ ] Watermark/background subtle
- [ ] Mobile responsive

---

### Week 8: Polish & Optimization

#### Day 36-37: Performance Optimization

**Tasks**:
1. **Code splitting**: Dynamic imports for heavy components
2. **Image optimization**: Use Next.js `<Image>` for theme icons
3. **Font subsetting**: Verify only used characters loaded
4. **Bundle analysis**: `npm run build && npx @next/bundle-analyzer`

**Target metrics**:
- [ ] Lighthouse score >90 (mobile)
- [ ] First Contentful Paint <1.5s
- [ ] Total bundle <500KB

#### Day 38-39: Accessibility Audit

**Tools**:
- [ ] WAVE Chrome extension (0 errors)
- [ ] Lighthouse accessibility score >95
- [ ] Screen reader test (NVDA/VoiceOver)
- [ ] Keyboard navigation (no mouse)

**Common fixes**:
- Add missing `aria-label`
- Ensure focus indicators visible
- Check color contrast (WCAG AA)
- Semantic HTML everywhere

#### Day 40: Mobile Testing

**Devices to test**:
- [ ] iPhone SE (small screen)
- [ ] iPhone 14 Pro (notch)
- [ ] Android (Samsung Galaxy A-series)
- [ ] Tablet (iPad)

**Checks**:
- [ ] Search bar usable
- [ ] Tabs swipeable
- [ ] Text readable (18px minimum)
- [ ] Touch targets ≥48px
- [ ] No horizontal scroll

---

## Week 9: Testing & QA

### Day 41-43: Comprehensive Testing

#### Unit Tests
```bash
npm run test

# Target coverage:
# - Components: 80%+
# - Utilities: 90%+
# - API client: 85%+
```

#### Integration Tests
```typescript
// __tests__/integration/search.test.tsx
test('full search flow', async () => {
  render(<SearchPage />);
  
  const input = screen.getByRole('searchbox');
  await userEvent.type(input, 'edukasyon');
  await userEvent.keyboard('{Enter}');
  
  await waitFor(() => {
    expect(screen.getByText(/Noli Me Tangere/)).toBeInTheDocument();
  });
});
```

#### End-to-End Tests (Playwright)
```bash
npm install -D @playwright/test

# tests/e2e/search.spec.ts
test('user can search and view results', async ({ page }) => {
  await page.goto('/');
  await page.fill('input[type="search"]', 'edukasyon');
  await page.press('input[type="search"]', 'Enter');
  
  await expect(page.locator('article').first()).toBeVisible();
});
```

**Checklist**:
- [ ] All unit tests passing
- [ ] Integration tests cover critical flows
- [ ] E2E tests for search, explore, methodology
- [ ] Edge cases tested (empty query, no results, errors)

### Day 44-45: User Testing

**Recruit 5-10 Filipino students**:
1. Give them tasks (e.g., "Find passages about education")
2. Observe (don't help)
3. Interview after (what was confusing?)

**Common issues to watch**:
- Can they find the search bar?
- Do they understand semantic vs lexical scores?
- Are results relevant?
- Any UI bugs on their devices?

**Checklist**:
- [ ] User testing sessions completed
- [ ] Feedback documented
- [ ] Critical issues fixed
- [ ] Nice-to-haves added to backlog

---

## Week 10: Deployment & Launch

### Day 46-47: Production Deployment

#### Frontend (Vercel)
```bash
# 1. Connect GitHub repo to Vercel
# 2. Configure environment variables:
NEXT_PUBLIC_API_URL=https://api.rizal-explorer.com

# 3. Deploy
git push origin main
# Vercel auto-deploys
```

#### Backend (Railway/Fly.io)
```bash
# Railway
railway login
railway init
railway add # Add PostgreSQL, Redis
railway up

# Set environment variables in Railway dashboard:
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
BERT_MODEL_PATH=/app/models/xlm-roberta-base
```

**Checklist**:
- [ ] Frontend deployed to Vercel
- [ ] Backend deployed to Railway/Fly.io
- [ ] Database migrated (Alembic)
- [ ] Redis connected
- [ ] Environment variables set
- [ ] CORS configured for production domain

#### Day 48: Domain & SSL

**Tasks**:
1. Buy domain (e.g., `rizal-explorer.com`) - $12/year
2. Configure DNS:
   - `rizal-explorer.com` → Vercel
   - `api.rizal-explorer.com` → Railway
3. SSL certificates (automatic on Vercel/Railway)

**Checklist**:
- [ ] Domain purchased
- [ ] DNS configured
- [ ] HTTPS working
- [ ] www redirect set up

#### Day 49: Monitoring Setup

**Sentry** (error tracking):
```bash
npm install @sentry/nextjs

# sentry.client.config.ts
Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  environment: 'production',
});
```

**Vercel Analytics**: Enable in dashboard (free)

**Backend logging**:
```python
# app/main.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

**Checklist**:
- [ ] Sentry capturing frontend errors
- [ ] Vercel Analytics enabled
- [ ] Backend logs to Railway/Fly.io dashboard
- [ ] Uptime monitoring (UptimeRobot, free)

#### Day 50: Launch!

**Pre-launch checklist**:
- [ ] All tests passing
- [ ] Performance metrics met (Lighthouse >90)
- [ ] Accessibility verified (WCAG AA)
- [ ] Mobile tested on real devices
- [ ] Error tracking working
- [ ] Database backups enabled
- [ ] README updated with usage instructions

**Launch tasks**:
1. Announce on social media (if applicable)
2. Share with thesis advisor
3. Get feedback from initial users
4. Monitor error logs closely

**Post-launch**:
- [ ] Fix any critical bugs within 24 hours
- [ ] Gather user feedback
- [ ] Plan iteration 2 features

---

## Post-Launch: Iteration & Maintenance

### Week 11+: Ongoing Tasks

**Daily**:
- [ ] Check error logs (Sentry)
- [ ] Monitor uptime (should be >99%)
- [ ] Respond to user feedback

**Weekly**:
- [ ] Review analytics (most searched queries)
- [ ] Check performance metrics
- [ ] Update dependencies (security patches)

**Monthly**:
- [ ] Review and prioritize feature requests
- [ ] Database maintenance (vacuum, reindex)
- [ ] Backup verification

---

## Troubleshooting Guide

### Common Issues

**Issue**: BERT model takes 10+ seconds to load
**Solution**: Pre-load on server startup, cache embeddings in Redis

**Issue**: pgvector search slow (>5s)
**Solution**: 
- Ensure IVFFlat index created
- Increase `lists` parameter (100 → 200)
- Use smaller top_k value

**Issue**: Frontend can't reach backend (CORS error)
**Solution**: Check CORS middleware allows frontend domain

**Issue**: Out of memory on backend
**Solution**: 
- Use smaller BERT model (distilbert)
- Limit concurrent requests
- Increase server RAM (Railway: upgrade plan)

**Issue**: Images/fonts not loading
**Solution**: Check Cloudflare cache settings, purge if needed

---

## Success Criteria

By Week 10, you should have:

✅ **Functional Features**
- Search works (semantic + lexical)
- Results display correctly
- Mobile-responsive
- All pages accessible

✅ **Performance**
- Lighthouse score >90
- Search response <2s (p95)
- Total bundle <500KB

✅ **Quality**
- Test coverage >80%
- WCAG 2.1 AA compliant
- No critical bugs

✅ **Deployment**
- Live on custom domain
- HTTPS enabled
- Monitoring active
- Backups configured

---

## Resources

- **Next.js Docs**: https://nextjs.org/docs
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **Tailwind Docs**: https://tailwindcss.com/docs
- **React Query Docs**: https://tanstack.com/query
- **pgvector Guide**: https://github.com/pgvector/pgvector

---

## Next Steps

After completing this plan:
1. Gather user feedback (surveys, interviews)
2. Prioritize iteration 2 features
3. Consider adding: user accounts, saved searches, PDF export
4. Write case study / thesis documentation
5. Present at conferences (if academic project)

**You've built a production-ready semantic search engine!** 🎉