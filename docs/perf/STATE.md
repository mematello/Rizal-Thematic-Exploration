# Performance Optimization State

**Last Updated**: 2026-10-18  
**Branch**: `docs/sync` (post-merge `6a5bf2e` on `main`)  
**Current Phase**: Item G Documentation & Repository Sync. Production Deployed & Verified.

---

## 1. Production Deployment Coordinates

- **Frontend UI**: [https://rizalengine.vercel.app](https://rizalengine.vercel.app)
  - Production Commit (`origin/main`): `6a5bf2e3ed31d030104e747040ce467309ac5233`
  - Rollback Target SHA: `cffec8516aefb71b32fa215128a47e6b0daa6747`
  - Stack: Next.js 16.1.6, React 19.2.3, TailwindCSS v4, React Query v5.90.20
- **Backend API**: [https://mematello-rizal-thematic-search-api.hf.space](https://mematello-rizal-thematic-search-api.hf.space)
  - Production Commit (`remotes/hf/main`): `7fb5667d4b622f98dc9095eb0370c2d488f5af27`
  - Rollback Target SHA: `3117ead`
  - Runtime: Hugging Face Spaces (`sdk: gradio`, `app_file: space.py`, hardware tier `zero-a10g`, 1 worker)
  - Stack: FastAPI 0.109+, PyTorch, sentence-transformers 3.3.0, Uvicorn 0.27+
- **Database**: Supabase Singapore AWS Pooler (PostgreSQL 15+ with `pgvector`)
  - Measured DB Size: **130 MB** binary database (**[Verified]** via `SELECT pg_size_pretty(pg_database_size(current_database()))`)

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
  - 205 chapter bundles (3.12 MB) and 205 Sanggunian cross-reference bundles (8.39 MB uncompressed) precomputed with cryptographic corpus fingerprint validation (`sentence_count`: 16,402, `theme_count`: 545).
  - Sub-10ms response times with automatic live DB fallback on miss (**[Verified]**, `DECISIONS.md` Item 7 & Item 13).
  - **Corpus Fingerprint Rebuild Rule**: Static caches embed Supabase corpus metrics (`sentence_count`: 16,402, `theme_count`: 545, MD5 hashes). If database corpus data or `engine._classify_themes` changes, the startup fingerprint check fails, automatically disabling the static bundle and reverting to live SQL computation. Bundles must be regenerated using:
    ```bash
    poetry -C backend run python scripts/build_chapter_cache.py
    poetry -C backend run python scripts/build_sanggunian_cache.py
    ```
- **DevOps & Availability (Item B)**:
  - Added dedicated `/health/db` connectivity probe endpoint using `NullPool` and 5 s timeouts (`DECISIONS.md` Item 14).
  - Automated GitHub Actions `keepalive.yml` workflow configured to run every 30 minutes (`cron: "*/30 * * * *"`). Two manual `workflow_dispatch` runs verified (`2026-10-17T15:44:46Z`, `2026-10-17T22:07:43Z`). Actions tab confirms 9 scheduled runs executed successfully between `2026-10-17T17:53:30Z` and `2026-10-18T18:44:42Z` (**[Verified]**, `DECISIONS.md` Item 14).
- **UI Modernization & Responsive Viewports**:
  - Fullscreen reading canvas, mobile drawer outside header blur, edge-to-edge sticky pill breakout, sub-header icon collapse, and Sanggunian Inverse view (`feat/sanggunian-inverse`) (**[Verified]**, `DECISIONS.md` Items 9, 10, 15, 16, 17).
- **Design System, Dark Mode & Accessibility (Item F, A11y Batch, Fili Dark)**:
  - Decorative watermark numerals converted to CSS pseudo-elements (`.watermark-numeral::before`), dropping axe violations from 692 to 0. Color contrast tightened to $\ge 4.5:1$ across all theme elements (`DECISIONS.md` Item 11 & Item 18).
  - **92-Context Accessibility Matrix**: Scanned across Home, Search, Drawer, Modal, and Sanggunian on `fix/a11y-axe` preview: **0 axe violations** (Evidence: `docs/perf/evidence/a11y_preview_raw.json`).
  - **16-Configuration Themes Tab Matrix**: Scanned live production across Chromium/WebKit, 390px/1440px, light/dark, Noli/Fili: **0 console errors**, **0 axe violations** (Evidence: `scratch/production_theme_verification.json`, `DECISIONS.md` Item 19).
  - Per-novel token family (`--theme-card-*`, `--map-card-*`, `--modal-*`) implemented in `globals.css` for Noli, Fili (`#1A0C0E` wine/charcoal surface + rose-gold accents), and Both.

---

## 3. Current Live Production Integrity Verification

- `npx tsc --noEmit` and `npm run build`: Clean (0 errors, 8/8 static routes) **[Verified]**
- Live Backend `/health/db`: `200 OK` `{"status":"ok","db":"ok"}` **[Verified]**
- Live Noli 1 Buod Chapter Canonical SHA-256: `8f5c4830076befd1225f687fe90d1f47523b0fe7cf4959d84707b285513c25ec` **[Verified]**
- Golden Search Invariance: 5 live queries verified against production HF Space (**[Verified]**, Evidence: `scripts/verify_backend_sanity.py`); 40 queries verified locally (**[Verified]**, Evidence: `scripts/run_golden_comparison_40.py` vs `docs/perf/golden_snapshot_deterministic.json`).

---

## 4. Open Follow-ups & Next Actions

1. **User Review on `docs/sync`**: Await user review of documentation synchronization, restored decision numbers, and verified operational facts.
2. **Post-Review Merge**: When approved, merge `docs/sync` into `main` without modifying code paths.
3. **Known Code & Label Mismatches**: Review `docs/perf/KNOWN_MISMATCHES.md` for cosmetic text updates (Footer "Next.js 15", Footer "ZeroGPU", `backend/README.md` `--reload`/`PYTHONHASHSEED`). (Kept untouched in this task under strict zero code diff constraint).
4. **Hugging Face Space Synchronization**: No code push required. All precomputed bundles in Space match current live HEAD (`7fb5667`).
