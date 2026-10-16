# Performance Optimization State

**Last Updated**: 2026-10-16  
**Branch**: `perf/themes-endpoint`  
**Current Phase**: Post-launch Item A complete (backend LATERAL query + process cache + non-blocking startup warmer + live Supabase byte-identical SHA-256 verification + frontend React Query prefetch & skeleton). Diagnosis phase for Item A2 underway.

---

## Done and Verified (With Evidence)

- **Phase 0 Diagnostics (Completed 2026-10-16)**:
  - Backend deployment verified: Live Hugging Face deployment is Gradio `space.py` with ZeroGPU (`zero-a10g`, `gcTimeout: 172800` / 48h sleep timeout).
  - Health check verified: Existing `GET /health` in `main.py` is lightweight and does NOT touch the database.
  - Exact query count identified: `get_themes()` executes **exactly 26 SQL queries** (1 query for all themes + 25 sequential pgvector cosine distance queries for each unique Tagalog title).
  - Real live baseline latency: ~10.5s total (`TTFB - time_appconnect` ≈ 10,115 ms server processing time across 26 remote round-trips from HF in US to Supabase in Singapore).
- **Item A Backend Optimization (Steps 1–3 + Startup Warmer, 2026-10-16)**:
  - **Set-based LATERAL Query**: Replaced 26 sequential queries with a single set-based LATERAL join (`Sentence.embedding <=> u.embedding, s.id`), reducing cold SQL queries from 26 to 1 (-96.2%).
  - **Process-Level Memory Cache**: Added lock-protected `_themes_cache` in `content.py`; warm runs execute **0 SQL queries** and return in **0.013 ms mean (0.003 ms median)** (-99.99% latency).
  - **Cache-Control Header**: Added `Cache-Control: public, max-age=3600, stale-while-revalidate=86400` for CDN/browser edge caching.
  - **Dual Graceful Fallbacks**: Preserved sequential query fallback and bundled static fallback (`backend/app/data/themes_cache.json`) generated from live production data.
  - **Startup Cache Warmer**: Added `warm_themes_cache()` executed in a background daemon thread during FastAPI `lifespan`. Startup yield completes in **94.55 ms** (non-blocking). First visitor never incurs cold DB query latency.
  - **Live Supabase Byte-Identical Match**: Executed new endpoint logic directly against live production Supabase PostgreSQL DB. Serialized output SHA-256 matches live snapshot `a11ef242257780d624d3ba9924a558a31e971a4034eea64ff683ad91b6f9c5c5` byte-for-byte.
  - **Search Invariance**: Verified against `docs/perf/golden_snapshot_deterministic.json`: **0 position changes** across all 40 golden search queries. `backend/app/core/engine.py` was untouched.
- **Item A Frontend Optimization (Step 4, 2026-10-16)**:
  - **React Query Migration**: `frontend/components/ThemeList.tsx` migrated from raw `fetch()` in `useEffect` to `useQuery({ queryKey: ['themes'], staleTime: 24h })`.
  - **Homepage Prefetch**: `frontend/app/page.tsx` triggers `queryClient.prefetchQuery(['themes'])` on initial mount so tab switching is instantaneous.
  - **Rich Skeleton UI**: Implemented an 8-card responsive skeleton grid (`grid-cols-1 sm:grid-cols-2 lg:grid-cols-4`) matching card layout and eliminating layout shift.
  - **Production Build Clean**: `npm run build` succeeds in 5.5s with zero TypeScript/lint errors.

---

## Live vs. Local Themes Timing Summary

| State | SQL Query Count | Latency (Server / App) | Cache-Control |
| :--- | :---: | :---: | :--- |
| **Live Unoptimized (Before)** | 26 | ~10,115 ms (TTFB - appconnect) | None |
| **Direct Supabase Cold (1st Query)** | 1 | ~2,100 ms | `public, max-age=3600, stale-while-revalidate=86400` |
| **Direct Supabase Warm (Process Cache)**| **0** | **0.013 ms** (median: 0.003 ms) | `public, max-age=3600, stale-while-revalidate=86400` |
| **App Startup Warmer (Non-blocking)**| N/A | **94.55 ms** startup yield | N/A |

---

## Next Action: User Review & Item A2 Diagnosis

1. Report Item A verification results to user for deployment approval (no push to `hf` or merge to `main` until instructed).
2. Report Item A2 diagnostic findings for slow chapter loading (NO code changes).

