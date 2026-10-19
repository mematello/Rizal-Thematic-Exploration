# Performance Optimization State

**Last Updated**: 2026-10-19  
**Branch**: `main` (merged `perf/supabase-egress`)  
**Current Phase**: Supabase Free-Tier Egress Audit & Developer Isolation Safeguards (Option C Merged into Main). Zero diff on `backend/app` and `frontend`.

---

## 1. Production Deployment Coordinates

- **Frontend UI**: [https://rizalengine.vercel.app](https://rizalengine.vercel.app)
  - `git rev-parse origin/main`: `072eea0ee06f5f8f911817b1f4310b68ccb46d8b` (Rollback Target SHA before egress merge)
  - Current `main` commit after egress merge: will reflect merge commit `00c2fa2c00f0188ec1bd95112856e79a1d7487e8`
  - Stack: Next.js 16.1.6, React 19.2.3, TailwindCSS v4, React Query v5.90.20
- **Backend API**: [https://mematello-rizal-thematic-search-api.hf.space](https://mematello-rizal-thematic-search-api.hf.space)
  - `git log -1 --oneline remotes/hf/main`:
    ```text
    7fb5667 feat(keepalive): add /health/db endpoint and register health router
    ```
  - Production Commit (`remotes/hf/main`): `7fb5667d4b622f98dc9095eb0370c2d488f5af27`
  - Rollback Target SHA: `3117ead` (Item C / Gate 3)
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
  - Automated GitHub Actions `keepalive.yml` workflow configured with `cron: "*/30 * * * *"`. GitHub throttled `*/30` scheduling; measured gaps between the 9 successful scheduled runs (`2026-10-17T17:53:30Z` to `2026-10-18T18:44:42Z`) are:
    - **Min gap**: 105.8 min (1.76 h)
    - **Median gap**: 160.5 min (2.68 h)
    - **Max gap**: 302.4 min (5.04 h)
    (Sufficient to prevent 48h HF sleep and 7-day Supabase pause; 2 manual runs also verified at `2026-10-17T15:44:46Z` and `2026-10-17T22:07:43Z`, **[Verified]**, `DECISIONS.md` Item 14).
- **UI Modernization & Responsive Viewports**:
  - Fullscreen reading canvas, mobile drawer outside header blur, edge-to-edge sticky pill breakout, sub-header icon collapse, and Sanggunian Inverse view (`feat/sanggunian-inverse`) (**[Verified]**, `DECISIONS.md` Items 9, 10, 15, 16, 17).
- **Design System, Dark Mode & Accessibility (Item F, A11y Batch, Fili Dark)**:
  - Decorative watermark numerals converted to CSS pseudo-elements (`.watermark-numeral::before`), dropping axe violations from 692 to 0. Color contrast tightened to $\ge 4.5:1$ across all theme elements (`DECISIONS.md` Item 11 & Item 18).
  - **92-Context Accessibility Matrix**: Scanned across Home, Search, Drawer, Modal, and Sanggunian on `fix/a11y-axe` preview: **0 axe violations** (Evidence: `docs/perf/evidence/a11y_preview_raw.json`).
  - **16-Configuration Themes Tab Matrix**: Scanned live production across Chromium/WebKit, 390px/1440px, light/dark, Noli/Fili: **0 console errors**, **0 axe violations** (Evidence: `docs/perf/evidence/production_theme_verification.json`, `DECISIONS.md` Item 19).
  - Per-novel token family (`--theme-card-*`, `--map-card-*`, `--modal-*`) implemented in `globals.css` for Noli, Fili (`#1A0C0E` wine/charcoal surface + rose-gold accents), and Both.
- **Supabase Egress Audit & Developer Isolation Safeguards (Item 20)**:
  - Audited 6.58 GB free-tier egress overrun. Measured via Supabase SQL editor: `embedding` = **8,853.75 bytes**, `embedding_dapt` = **8,488.25 bytes**, `sentence_text` = **117.12 bytes**. 98.0% of wire egress per row was raw vector floats.
  - Implemented and wired `scripts/prod_guard.py` across **all 109+ DB-connecting scripts** in the repository to prevent accidental developer runs against production Supabase.
  - Implemented `scripts/restore_and_validate_local_db.py` to restore `rizal_db_backup.sql` and validate local corpus fingerprint (`sentence_count`: 16,402, `theme_count`: 545, `dapt_null_count`: 4,580) on port 5433 (**[Verified]**).
  - Investigated MD5 difference between local backup (`b534a1...`) and production bundle (`e438ce...`): identical SQL computation verified; confirmed `rizal_db_backup.sql` snapshot itself produces `b534a1...` **[Measured]**. Documented that local DB cannot generate production-deployable bundles without a fresh dump or `--allow-prod` **[Inferred]**.
  - Documented local workflow in `docs/perf/HANDOFF.md` Section 2 and created `backend/.env.example`.
  - Comprehensive audit report in `docs/perf/EGRESS_AUDIT.md` with explicit **[ESTIMATED]** labels, 3–5 day monitoring plan, and non-script traffic inventory (**[Verified]**, `DECISIONS.md` Item 20).
  - **Decision Executed**: **Option C adopted exclusively**. Options A (search LRU cache) and B (defer vector columns) **deferred**, to be reconsidered only if egress stays near the 5.00 GB quota after dev traffic stops.
  - Zero diff on `backend/app/**` strictly preserved.

---

## 3. Current Live Production Integrity Verification

- `npx tsc --noEmit` and `npm run build`: Clean (0 errors, 8/8 static routes) **[Verified]**
- Live Backend `/health/db`: `200 OK` `{"status":"ok","db":"ok"}` **[Verified]**
- Live Noli 1 Buod Chapter Canonical SHA-256: `8f5c4830076befd1225f687fe90d1f47523b0fe7cf4959d84707b285513c25ec` **[Verified]**
- Golden Search Invariance: 5 live queries verified against production HF Space (**[Verified]**, Evidence: `scripts/verify_backend_sanity.py`); 40 queries verified locally (**[Verified]**, Evidence: `scripts/run_golden_comparison_40.py` vs `docs/perf/golden_snapshot_deterministic.json`).
- `backend/app` zero-diff rule: `git diff main -- backend/app` is strictly EMPTY **[Verified]**

---

## 4. Open Follow-ups & Next Actions

1. **Production Dev Freeze for 3–5 Days**:
   - Zero benchmarks (`benchmark.py`), cache bundle rebuilds (`build_*_cache.py`), or axe/golden sweeps against the live Space or Supabase.
   - All developer verification, evaluation, and test suites must run exclusively against the local Postgres instance (`localhost:5433`).
2. **Dashboard Egress Comparison**:
   - Compare the Supabase Usage dashboard (egress per day) against the 15–19 Oct billing spike baseline (~1.0–1.5 GB/day).
   - Reconsider deferred Options A and B only if organic user egress alone keeps usage near 5 GB after dev traffic ceases.
3. **Local Cache Bundle Rebuild Constraint**:
   - Never build or rebuild chapter or Sanggunian static bundles against the local database until the MD5 mismatch is reconciled (local backup dated 2026-10-16 02:43 UTC differs from the production corpus the bundles were fingerprinted against).
   - If bundle rebuilding is ever required, execute against production using `--allow-prod` or pull a fresh production dump first.
4. **No Push to remotes/hf/main**:
   - HF Space backend remains on commit `7fb5667d4b622f98dc9095eb0370c2d488f5af27` (strictly zero diff under `backend/app/`, no deployment required).

