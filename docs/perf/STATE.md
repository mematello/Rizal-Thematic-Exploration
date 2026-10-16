# Performance Optimization State

**Last Updated**: 2026-10-16  
**Branch**: `perf/themes-endpoint`  
**Current Phase**: Post-launch Item A — Themes endpoint optimization (Steps 1–3 complete, awaiting review before Step 4 / deploy).

---

## Done and Verified (With Evidence)

- **Phase 0 Diagnostics (Completed 2026-10-16)**:
  - Backend deployment verified: Live Hugging Face deployment is Gradio `space.py` with ZeroGPU (`zero-a10g`, `gcTimeout: 172800` / 48h sleep timeout).
  - Health check verified: Existing `GET /health` in `main.py` is lightweight and does NOT touch the database.
  - Exact query count identified: `get_themes()` executes **exactly 26 SQL queries** (1 query for all themes + 25 sequential pgvector cosine distance queries for each unique Tagalog title).
  - Real live baseline latency: ~10.5s total (`TTFB - time_appconnect` ≈ 10,115 ms server processing time across 26 remote round-trips from HF in US to Supabase in Singapore).
- **Item A Backend Optimization (Steps 1–3, 2026-10-16)**:
  - **Set-based LATERAL Query**: Replaced 26 sequential queries with a single set-based LATERAL join (`Sentence.embedding <=> u.embedding, s.id`), reducing cold SQL queries from 26 to 1 (-96.2%).
  - **Process-Level Memory Cache**: Added lock-protected `_themes_cache` in `content.py`; warm runs execute **0 SQL queries** and return in **0.013 ms mean (0.003 ms median)** (-99.99% latency).
  - **Cache-Control Header**: Added `Cache-Control: public, max-age=3600, stale-while-revalidate=86400` for CDN/browser edge caching.
  - **Dual Graceful Fallbacks**: Preserved sequential query fallback and added bundled static fallback (`backend/app/data/themes_cache.json`) if database connection is ever interrupted.
  - **Search Invariance**: Verified against `docs/perf/golden_snapshot_deterministic.json`: **0 position changes** across all 40 golden search queries.

---

## Live vs. Local Themes Timing Summary

| State | SQL Query Count | Latency (Server / App) | Cache-Control |
| :--- | :---: | :---: | :--- |
| **Live Unoptimized (Before)** | 26 | ~10,115 ms (TTFB - appconnect) | None |
| **Local Cold (1st Request)** | 1 | ~2,285 ms | `public, max-age=3600, stale-while-revalidate=86400` |
| **Local Warm (Runs 2–5 Mean)**| **0** | **0.013 ms** (median: 0.003 ms) | `public, max-age=3600, stale-while-revalidate=86400` |

---

## Next Action: User Review & Step 4

1. Await user review of Item A Steps 1–3.
2. Proceed to Step 4 (Frontend React Query + prefetch + skeleton in `ThemeList.tsx`).
3. Obtain explicit approval before pushing to `hf` or merging into `main`.

