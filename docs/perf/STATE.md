# Performance Optimization State

**Last Updated**: 2026-10-18  
**Branch**: `docs/sync` (post-merge `6a5bf2e` on `main`)  
**Current Phase**: Item G Documentation & Repository Sync. Production Deployed & Verified.

---

## 1. Production Deployment Coordinates

- **Frontend UI**: [https://rizalengine.vercel.app](https://rizalengine.vercel.app)
  - Production Commit: `6a5bf2e` (Rollback Target: `cffec85`)
  - Stack: Next.js 16.1.6, React 19.2.3, TailwindCSS v4, React Query v5.90.20
- **Backend API**: [https://mematello-rizal-thematic-search-api.hf.space](https://mematello-rizal-thematic-search-api.hf.space)
  - Production Commit: `7fb5667` (Rollback Target: `3117ead`)
  - Runtime: Hugging Face Spaces (`sdk: gradio`, `app_file: space.py`, hardware tier `zero-a10g`, 1 worker)
  - Stack: FastAPI 0.109+, PyTorch, sentence-transformers 3.3.0, Uvicorn 0.27+
- **Database**: Supabase Singapore AWS Pooler (PostgreSQL 15+ with `pgvector`)
  - Measured DB Size: **130 MB** binary database (**[Verified]** via `pg_database_size`)

---

## 2. Done and Verified (With Evidence & Pointers)

*Detailed architectural rationale, implementation diffs, and verification metrics are logged in [DECISIONS.md](./DECISIONS.md).*

- **Search Pipeline Determinism (Step B)**:
  - Added deterministic SQL `ORDER BY Sentence.id` and in-memory `sorted()` tie-breakers across 5 commits (`646deb6`–`73f1d1a`).
  - Bit-identical reproducibility across independent server processes without `PYTHONHASHSEED=0`.
  - Permanent Reference: `docs/perf/golden_snapshot_deterministic.json` (**[Verified]**, see `DECISIONS.md` Item 1).
- **Themes & Chapters Endpoints (Items A & A2)**:
  - Themes: Set-based PostgreSQL LATERAL join + thread-safe process cache. Cold latency slashed from 10.5s to 269 ms (**[Verified]**, `DECISIONS.md` Item 4).
  - Chapters: Bounded thread-safe LRU cache (64 entries, ~1.11 MB). Cold latency dropped from ~22s to 2.7–7.0 ms (**[Verified]**, `DECISIONS.md` Item 5).
- **Precomputed Static Bundles (Steps 1 & Item C Phase 2)**:
  - 205 chapter bundles (3.12 MB) and 205 Sanggunian cross-reference bundles (6.06 MB) precomputed with cryptographic corpus fingerprint validation (`sentence_count`: 16,402, `theme_count`: 545).
  - Sub-10ms response times with automatic live DB fallback on miss (**[Verified]**, `DECISIONS.md` Item 6 [hist. 7] & Item 12 [hist. 13]).
  - **Corpus Fingerprint Rebuild Rule**: Static caches embed Supabase corpus metrics (`sentence_count`: 16,402, `theme_count`: 545, MD5 hashes). If database corpus data or `engine._classify_themes` changes, the startup fingerprint check fails, automatically disabling the static bundle and reverting to live SQL computation. Bundles must be regenerated using:
    ```bash
    poetry run python scripts/build_chapter_cache.py
    poetry run python scripts/build_sanggunian_cache.py
    ```
- **DevOps & Availability (Item B)**:
  - Added `/health/db` connectivity probe endpoint.
  - Automated GitHub Actions `keepalive.yml` workflow executing hourly to ping `/health/db` and prevent Supabase auto-pausing (**[Verified]**, `DECISIONS.md` Item 13 [hist. 14]).
- **UI Modernization & Responsive Viewports**:
  - Fullscreen reading canvas, mobile drawer outside header blur, edge-to-edge sticky pill breakout, sub-header icon collapse, and Sanggunian Inverse view (`feat/sanggunian-inverse`) (**[Verified]**, `DECISIONS.md` Items 8, 9, 14, 15, 16).
- **Design System, Dark Mode & Accessibility (Item F, A11y Batch, Fili Dark)**:
  - Full WCAG 2.0/2.1 AA compliance: Decorative watermark numerals converted to CSS pseudo-elements (`.watermark-numeral::before`), dropping axe violations from 692 to 0.
  - Color contrast tightened to $\ge 4.5:1$ across all theme elements.
  - Per-novel token family (`--theme-card-*`, `--map-card-*`, `--modal-*`) implemented in `globals.css` for Noli, Fili (`#1A0C0E` wine/charcoal surface + rose-gold accents), and Both.
  - Production matrix verification: **0 axe violations** and **0 console errors** across all 16 configurations (**[Verified]**, `DECISIONS.md` Items 10, 11, 17 [hist. 18], 18 [new]).

---

## 3. Current Integrity & Health Verification

- `backend/` diff relative to `main`: Strictly EMPTY (0 diff) **[Verified]**
- `frontend/` diff relative to `main`: Strictly EMPTY (0 diff) **[Verified]**
- `npx tsc --noEmit` and `npm run build`: Clean (0 errors, 8/8 static routes) **[Verified]**
- Live Backend `/health/db`: `200 OK` `{"status":"ok","db":"ok"}` **[Verified]**
- Live Noli 1 Buod Chapter Canonical SHA-256: `8f5c4830076befd1225f687fe90d1f47523b0fe7cf4959d84707b285513c25ec` **[Verified]**
- Live Golden Search Invariance: 40/40 queries 100% rank invariant against `golden_snapshot_deterministic.json` **[Verified]**

---

## 4. Open Follow-ups & Next Actions

1. **User Review on `docs/sync`**: Await user review of documentation synchronization, old-to-new decision mapping, and verified facts.
2. **Post-Review Merge**: When approved, merge `docs/sync` into `main` via PR / fast-forward merge without modifying code paths.
3. **Future Code Cleanups (Optional Post-Thesis)**:
   - Synchronize `frontend/components/Footer.tsx` badge text from `Next.js 15` to `Next.js 16` and clarify `ZeroGPU` label to reflect CPU-primary architecture. (Kept unchanged in this task under strict zero code diff constraint).
   - Clean up legacy `--reload` comment in `backend/README.md`.
4. **Hugging Face Space Synchronization**: No code push required unless server-side runtime files change. All precomputed bundles in Space match current live HEAD (`7fb5667`).
