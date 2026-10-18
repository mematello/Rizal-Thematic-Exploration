# Performance Optimization Handoff & Developer Guide

**Project**: Rizal Thematic Explorer  
**Branch**: `docs/sync` (post-merge `6a5bf2e` on `main`)  
**Current Status**: Production deployed, verified live on Vercel (`https://rizalengine.vercel.app`) and Hugging Face Spaces (`https://mematello-rizal-thematic-search-api.hf.space`).  
**Reference Snapshot**: `docs/perf/golden_snapshot_deterministic.json` (Commit `73f1d1a` / HEAD; permanent deterministic reference for all 40 golden queries).

---

## 1. How to Start the Local Servers

### Backend (FastAPI)
Always launch the Uvicorn server explicitly bound to IPv4 `127.0.0.1`. **Do NOT use `--reload`** during performance benchmarking or profiling (file watchers and reload wrappers introduce jitter). `PYTHONHASHSEED=0` is no longer required after Step B determinism fixes.

**Windows PowerShell:**
```powershell
cd backend
$env:PERF_BENCHMARK="1"
..\backend\.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000
```

**Linux / macOS:**
```bash
cd backend
export PERF_BENCHMARK=1
poetry run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Frontend (Next.js 16.1.6)
```bash
cd frontend
npm run dev     # Development mode (http://localhost:3000)
npm run build   # Production build verification
npm run start   # Run production bundle locally
```

---

## 2. Environment Flags & Key Endpoints

| Variable / Param | Default | Purpose |
| :--- | :---: | :--- |
| `PERF_BENCHMARK` | `0` | When `1`, returns internal engine micro-timings, call counts, and cache statistics in search response metadata. |
| `DISABLE_WORD_VEC_CACHE` | `0` | When `1`, skips precomputed vocabulary cache loading and forces on-the-fly candidate token encoding. |
| `DISABLE_TRANSLATOR_CACHE` | `0` | When `1`, bypasses in-memory `TranslatorCache`, making synchronous external Google Translate network requests. |
| `DISABLE_BATCHED_CONTEXT_FETCH`| `0` | When `1`, bypasses `_fetch_context_cache`, falling back to sequential SQL queries for neighbor contexts. |
| `bypass_cache=true` (HTTP Param)| `false`| URL query parameter on `/api/v1/search` to bypass the word vector cache for an isolated request. |

### Key API Endpoints
- **`GET /health`**: Health probe returning `{"status": "ok"}`.
- **`GET /health/db`**: Database connectivity probe returning `{"status": "ok", "db": "ok"}` with `Cache-Control: no-store`. Used by GitHub Actions `keepalive.yml` workflow to prevent Supabase auto-pausing.
- **`GET /api/v1/search`**: Core semantic and lexical hybrid search.
- **`GET /api/v1/chapters`**: List of chapters with `buod` and `full` modes.
- **`GET /api/v1/chapters/{book}/{chapter_number}`**: Chapter content and contextual theme classifications.
- **`GET /api/v1/themes`**: Deduplicated list of 25 novel themes with best-match chapter passage citations.

---

## 3. Precomputed Cache Bundles & Regeneration Procedures

The system uses three sets of static precomputed caches to ensure sub-10ms response times:

1. **Vocabulary Embedding Cache (`backend/app/data/word_vec_cache_v1.pkl`)**:
   - Size: 73.39 MB (24,770 words).
   - Regeneration script: `backend/scripts/build_word_vec_cache.py`.
2. **Chapter Content Bundles (`backend/app/data/chapter_cache_*.json`)**:
   - 4 bundles covering all 205 chapters across Noli and El Fili (`buod` and `full` modes). Total size: 3.12 MB.
   - Regeneration script: `backend/scripts/build_chapter_cache.py`.
3. **Sanggunian Alignment Bundles (`backend/app/data/sanggunian_cache_*.json`)**:
   - 4 bundles precomputing cross-novel sentence alignments between Buod and Buong Kwento. Total size: 6.06 MB.
   - Regeneration script: `backend/scripts/build_sanggunian_cache.py`.

### ⚠️ Critical Invalidation Rules
- **Rule 1 (Sanggunian Bundles)**: Any change to `backend/app/api/v1/content.py` or `backend/app/core/robust_aligner.py` invalidates the Sanggunian fingerprint and **strictly requires rebuilding the Sanggunian bundles** via `build_sanggunian_cache.py`.
- **Rule 2 (Chapter Bundles)**: Any change to `backend/app/core/engine.py` (theme classification or context expansion) **strictly requires rebuilding the chapter bundles** via `build_chapter_cache.py`.
- **Rule 3 (Corpus Fingerprint)**: All bundles embed a cryptographic corpus fingerprint (`sentence_count`: 16,402, `theme_count`: 545, MD5 hash). At startup, the server verifies this fingerprint against the database. If there is a mismatch, the server logs a warning and falls back safely to live database computation without crashing.

### How to Regenerate Bundles
```bash
cd backend

# 1. Regenerate Chapter Bundles
poetry run python scripts/build_chapter_cache.py

# 2. Regenerate Sanggunian Bundles
poetry run python scripts/build_sanggunian_cache.py
```

---

## 4. Frontend `data-novel` Architectural Conventions

To support multi-novel styling, backdrop rotation, and dark mode theming, the application uses normalized `data-novel` attributes across the DOM and CSS:

- **Identifiers**: `'noli'`, `'fili'`, and `'both'`. (Note: `'elfili'` is normalized to `'fili'` across all frontend stores and selectors).
- **CSS Tokens Defined in `frontend/app/globals.css`**:
  - `[data-novel="noli"]`: Navy/gold theme (`--map-card-*`, `--modal-*`, `--theme-card-*`).
  - `[data-novel="fili"]`: Wine/rose-gold theme (surface `#1A0C0E`, accent `#E06C75`).
  - `[data-novel="both"]`: Adaptive dark theme supporting mixed Noli/Fili result cards.
- **Card Overrides**: Individual cards support `.theme-card-fili`, `.theme-card-noli`, `.map-card-fili`, and `.map-card-noli` to render correct novel colors when parent is set to `'both'`.
- **Modal Scoping**: Modal roots (`ChapterModal.tsx`, `SanggunianOverlay.tsx`) set `data-novel={book}` on their root containers so that novel-specific CSS variables resolve correctly even when search page filter is `both`.

---

## 5. Scripts Inventory & Tooling

All utility and verification scripts reside in `backend/scripts/` or `scratch/`:

- **`backend/scripts/build_chapter_cache.py`**: Precomputes all 205 chapter JSON files with embedded corpus fingerprint.
- **`backend/scripts/build_sanggunian_cache.py`**: Precomputes all 205 Sanggunian cross-reference alignment JSON files.
- **`backend/scripts/build_word_vec_cache.py`**: Precomputes float32 embeddings for 24,770 Tagalog candidate words.
- **`scripts/benchmark.py`**: Automated benchmark runner executing 10 runs across all 40 golden queries (`python scripts/benchmark.py --help`).
- **`scripts/compare_golden.py`**: Deep comparator checking snapshot output against reference snapshot (`python scripts/compare_golden.py <new_snapshot.json> <golden_snapshot_deterministic.json>`).
- **`scripts/fetch_snapshot.py`**: Fetches fresh search results for all 40 golden queries from running server (`python scripts/fetch_snapshot.py <output_file.json> [port]`).
- **`scratch/verify_production_live.py`**: Full live production verification suite testing API endpoints and browser UI.
- **`scratch/verify_production_themes.py`**: Automated Playwright test verifying the 16-configuration matrix on the live homepage Themes tab.
- **`scratch/run_production_axe.py`**: Headless axe-core scanner evaluating WCAG 2.0/2.1 AA accessibility.

---

## 6. Deployment Verification Checklist

Before and after deploying any change to production, execute the following verification steps:

1. **Verify Backend Sanity**:
   - `GET /health/db` returns `{"status":"ok","db":"ok"}` with `200 OK`.
   - `GET /api/v1/chapters/noli/1?mode=buod` returns canonical SHA-256:  
     `8f5c4830076befd1225f687fe90d1f47523b0fe7cf4959d84707b285513c25ec`.
   - Confirm startup logs display:
     ```text
     [CACHE] Chapter static cache verified
     [CACHE] Sanggunian static cache verified
     ```
2. **Verify Golden Search Queries**:
   - Compare live search results against `docs/perf/golden_snapshot_deterministic.json`.
   - 40/40 queries must produce 100% invariant ranking order.
3. **Verify Frontend UI & Accessibility**:
   - Run `npx tsc --noEmit` and `npm run build` cleanly with 0 errors.
   - Run axe-core across mobile (390px / 382px clientWidth) and desktop (1440px / 1432px clientWidth) in Chromium and WebKit.
   - 0 console errors and 0 critical/serious accessibility violations.
