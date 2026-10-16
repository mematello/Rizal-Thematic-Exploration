# Performance Optimization State

**Last Updated**: 2026-10-16  
**Branch**: `perf/chapters-endpoint`  
**Current Phase**: Item A deployed to production HF Space and Vercel, verified live. Item A2 (Chapters endpoint optimization) fully implemented and locally verified against production Supabase DB; ready for user review.

---

## Done and Verified (With Evidence)

- **Item A Production Deployment (Completed 2026-10-16)**:
  - Hugging Face Space deployed with zero leak of frontend/docs files (commit `0b74a1d`).
  - Live `GET /health` = `200 OK`.
  - Live `GET /api/v1/themes` byte-identical SHA-256: `a11ef242257780d624d3ba9924a558a31e971a4034eea64ff683ad91b6f9c5c5`.
  - Live response time: Mean **269.36 ms** (down from 10,115 ms, a -97.3% latency reduction).
  - Live search invariance: 5 diverse golden queries tested, 100% rank invariant.
  - Merged to `main` (`7330b81`) and live Vercel frontend verified with browser subagent (0 errors, instant themes tab).
- **Item A2 Backend Optimization (Chapters Endpoint, 2026-10-16)**:
  - **Eliminated N+1 Query Loop**: In `backend/app/api/v1/content.py`, `get_chapter_content` now calls `engine._fetch_context_cache(db, unique_sentences, window=8)` and passes `context_cache` into `engine._classify_themes(...)`. Reused existing helper without touching `backend/app/core/engine.py` (0 diff).
  - **Bounded LRU Cache**: Implemented thread-safe `OrderedDict` bounded to `MAX_CHAPTER_CONTENT_CACHE_ENTRIES = 64` for chapter content. Memory footprint measured at **~1.11 MB** for 64 chapters (average buod chapter: 3.5 KB, average full chapter: 41.6 KB, max tested Noli 34: 42.6 KB).
  - **Process-Level Chapters List Cache**: Added lock-protected `_chapters_cache` with dual static fallback bundles (`backend/app/data/chapters_cache_buod.json` and `backend/app/data/chapters_cache_full.json`).
  - **Non-blocking Startup Warmer**: Added `warm_chapters_cache()` executed in background thread during FastAPI lifespan.
  - **HTTP Cache-Control**: Added `Cache-Control: public, max-age=86400, stale-while-revalidate=604800` to both `/api/v1/chapters` and `/api/v1/chapters/{book}/{chapter_number}`.
  - **Broader Set Parity Verified**: Tested 10 `buod` chapters (Noli 1-5, El Fili 1-5) and 3 `full` chapters (Noli 1, Noli 34, El Fili 1) against Supabase. All returned 200 OK, payload structures intact, cold vs warm byte-identical.
  - **Cold vs Warm Latency**: Cold requests ~1,700–2,300 ms (buod) and ~17–23s (full); warm requests **2.7–7.0 ms** across all chapters (-99.7% latency reduction).
  - **Search Invariance**: Verified 5 golden queries across both Noli and El Fili (`kalayaan`, `edukasyon`, `simbahan`, `korapsyon`, `pag-ibig`) against `golden_snapshot_deterministic.json`: **100% ID and rank match**.
- **Item A2 Frontend Optimization (React Query & Prefetching, 2026-10-16)**:
  - **ChapterGrid**: Migrated from `useEffect` fetch to `@tanstack/react-query` `useQuery({ queryKey: ['chapters', mode], staleTime: 24h })`. Added responsive 8-card skeleton loading grid.
  - **Homepage Prefetch**: `frontend/app/page.tsx` prefetches chapters list (`buod` & `full`) on mount.
  - **Chapter Content Cache**: `useQuery` with 24h `staleTime` manages modal chapter content.
  - **Adjacent Chapter Prefetching**: When a chapter is opened in `buod` mode, $X+1$ and $X-1$ are prefetched in the background. Heavy full-mode chapters are intentionally not prefetched to conserve bandwidth.
  - **Modal & Search Integration**: `ChapterModal.tsx` and `search/page.tsx` use `queryClient.fetchQuery` for metadata, real-time mode switching, and fullscreen infinite scroll.
  - **Build Verification**: `npm run build` succeeds cleanly in 6.4s; `npx tsc --noEmit` reports 0 errors.

---

## Latency Summary: Item A2 Chapters

| Endpoint / Action | Cold Latency (Supabase) | Warm Latency (Process LRU) | Cache-Control |
| :--- | :---: | :---: | :--- |
| **GET /api/v1/chapters (buod)** | ~105 ms | **3.6 ms** | `public, max-age=86400, stale-while-revalidate=604800` |
| **GET /api/v1/chapters (full)** | ~70 ms | **3.6 ms** | `public, max-age=86400, stale-while-revalidate=604800` |
| **GET /api/v1/chapters/... (buod)** | ~1,700–2,300 ms | **2.7–6.7 ms** | `public, max-age=86400, stale-while-revalidate=604800` |
| **GET /api/v1/chapters/... (full)** | ~17,000–23,000 ms | **4.6–7.0 ms** | `public, max-age=86400, stale-while-revalidate=604800` |

---

## Next Action: User Review & Deployment Approval (Item A2)

1. Report Item A2 verification results to user for review.
2. Await explicit approval before pushing to `hf` or merging `perf/chapters-endpoint` to `main`.

