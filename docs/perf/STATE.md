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

- **Item A2 Deployment & Production Verification (Completed 2026-10-16)**:
  - **Gate 1 (Old vs New Parity)**: All 18 chapters (10 buod + 3 full + 5 random buod) verified 100% byte-identical between old `main` and new endpoint against Supabase DB. All 40 golden search queries verified 100% invariant vs `golden_snapshot_deterministic.json` (0 differences). `engine.py` diff completely empty.
  - **Gate 2 (Hugging Face Backend Deploy)**: Recorded rollback SHA (`0b74a1d`), pushed commit `007116d` (only 4 backend files, 0 frontend/docs leak). Verified live: `/health` (200), `/api/v1/themes` SHA (match), `/api/v1/chapters` SHA (match), `/api/v1/chapters/noli/1?mode=buod` SHA (match), `/api/v1/chapters/elfili/1?mode=buod` SHA (match), Noli 34 full mode single request (match, 22.4s), Cache-Control present on all endpoints, warm latencies ~240–260 ms, and 5 golden search queries 100% invariant.
  - **Gate 3 (Frontend & Vercel Deploy)**: Merged `perf/chapters-endpoint` to `main` (`a0cf2be`), pushed to GitHub. Verified live on `https://rizalengine.vercel.app`: chapter grid loads fast, opening chapter modal in buod & full mode works, mode toggle works, 0 console errors, mobile viewport (375x812) verified with zero horizontal overflow.
  - **Follow-up Profiling**: Profiled cold full-mode chapter (Noli 34 full, 245 sentences). Breakdown: SQL DB transfer: 1.17s (3.7%), JSON serialization: 1.56 ms (0.0%), `_classify_themes` in-process compute: 30.19s (96.3%) due to 125,000 keyword matches and 2.5 million regex word extractions across 510 themes.

- **Step 1 Precomputed Chapter Content (Completed 2026-10-16, branch `perf/chapter-precompute`)**:
  - **Static Bundles Built**: All 205 chapters generated across both novels and modes (`noli_buod`: 64, `noli_full`: 63, `elfili_buod`: 39, `elfili_full`: 39). Total size: **3.12 MB**.
  - **Zero Diff on `engine.py`**: Untouched. All 40 golden search queries 100% invariant vs `golden_snapshot_deterministic.json`.
  - **Corpus Fingerprint Embedded & Verified**: Embeds 16,402 sentence count, 545 theme count, deterministic MD5 hashes. Checked non-blockingly at startup via fast SQL query (~1.5s).
  - **Fallback Safety**: Missing bundle or fingerprint mismatch logs warning and falls back to live path. DOES NOT CRASH.
  - **Latency Improvement**: Cold full chapter (e.g. Noli 34 full) dropped from **22,000 ms to 6.8 ms** (3,200x speedup). Average across 20 random chapters: **11.4 ms**.
  - **Byte Parity**: 100% SHA-256 match on raw HTTP response bytes across all tested chapters.

---

## Next Action: User Review & Deployment Approval for Step 1
Once Step 1 is reviewed and approved for deploy, proceed to Item B (Keep-alive + DB health check + waking UI).

