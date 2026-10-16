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

- **Reconciliation & Step 0 Output Check (Completed 2026-10-17)**:
  - **Reconciled Byte Counts**: Confirmed that `chapter_parity_205.txt` on disk always recorded exact baseline bytes (Noli 1 buod = 4,310 B, Elfili 1 buod = 1,748 B, Noli 34 full = 43,596 B). The previous report text included an illustrative snippet with hallucinated byte counts (`10609B`, `9745B`, `131238B`), while the actual generated file on disk was 100% matched to baselines.
  - **Identical Live vs Local Fetch**: Fetched `noli/1?mode=buod`, `elfili/1?mode=buod`, and `noli/34?mode=full` from both live Hugging Face Space and local server with identical headers. All returned 100% byte-identical payloads and matching SHA-256 hashes (`70383658eb...`, `120491e145...`, `e5f88889fa...`). Saved raw bodies in `docs/perf/evidence/`.
  - **Model Schema Corrected**: First sentence object verified: `id`, `is_short`, `sentence_index`, `sentence_text`, `themes`.
  - **Bundle Disk Sizes**: `noli_buod` = 206,795 B (201.9 KB), `noli_full` = 1,739,033 B (1,698 KB), `elfili_buod` = 167,905 B (164 KB), `elfili_full` = 1,157,817 B (1,130 KB). Total = 3.12 MB.
  - **Cache Miss Path Tested**: Forced miss with `_static_cache_enabled = False` and cleared LRU cache. Verified lazy `engine = get_engine()` runs and produces 100% byte-identical output.

- **Gate 2 Deployment & Live Verification (Completed 2026-10-17)**:
  - **HF Space Deployment**: Pushed commit `2b15166` containing strictly 7 backend files (0 frontend/docs leaks). All 4 chapter cache bundles tracked and verified.
  - **Live Endpoints Verified**:
    - `GET /health` = `200 OK`
    - `/api/v1/themes` (5 runs): **SHA match**, Mean server time: **241.7 ms**
    - `/api/v1/chapters` (5 runs): **SHA match**, Mean server time: **238.7 ms**
    - `/api/v1/chapters/noli/1?mode=buod` (5 runs): **SHA match**, Mean server time: **254.7 ms**
    - `/api/v1/chapters/elfili/1?mode=buod` (5 runs): **SHA match**, Mean server time: **249.1 ms**
    - `/api/v1/chapters/noli/34?mode=full` (single request): **SHA match**, Server time: **253.0 ms** *(down from 22,400 ms! -98.9% latency reduction)*
  - **5 Live Golden Search Queries Verified (`novel=both`)**:
    - `kalayaan` (noli): Top-5 IDs match = True (`[38300, 37644, 38386, 38258, 38249]`)
    - `edukasyon` (noli): Top-5 IDs match = True (`[37539, 37544, 37541, 37538, 37540]`)
    - `simbahan` (elfili): Top-5 IDs match = True (`[39026, 38425, 39028, 38415, 38533]`)
    - `pag-ibig` (noli): Top-5 IDs match = True (`[37860, 37231, 37191, 37193, 38222]`)
    - `bayan` (elfili): Top-5 IDs match = True (`[39084, 39108, 38574, 38913, 39016]`)

- **Gate 3 Merge to Main (Completed 2026-10-17)**:
  - Merged `perf/chapter-precompute` into `main` (`b151f48`).
  - Pushed to `origin/main` on GitHub.
  - Zero changes to search ranking or engine behavior (`backend/app/core/engine.py` diff: **strictly 0 diff**).

- **Item D Mobile Fullscreen Fix (Completed 2026-10-17, branch `fix/mobile-fullscreen`)**:
  - **Identified Mobile Issues**:
    1. Fixed-height and viewport units (`100vh`) caused clipping under iOS Safari dynamic bars and notch/Dynamic Island cutouts.
    2. Desktop fullscreen sidebar was hidden (`hidden lg:flex`) on mobile screens (<1024px), completely locking mobile users out of the "Tauhan" (Characters) and "Sanggunian" (Cross-Novel References) toggles in fullscreen reader mode.
    3. The mobile header lacked safe-area top insets (`env(safe-area-inset-top)`) and had tap targets smaller than 44x44px (`p-2` instead of minimum 44px).
    4. Reading text container in fullscreen had excessive fixed margins (`px-8` / 64px stolen), squeezing text width on narrow mobile viewports.
    5. In Sanggunian overlay (`refFullscreen`), the split view had a fixed `w-[500px]` width in a flex-row container, blowing out mobile screens horizontally.
  - **Implemented Fixes in `frontend/components/ChapterModal.tsx`**:
    1. Replaced modal height with `h-[100dvh]` and added `overscroll-contain` to prevent background scroll-chaining on iOS and Android.
    2. Added dedicated sticky mobile fullscreen header with `pt-[max(0.75rem,env(safe-area-inset-top))]`, 44x44px minimum tap targets, chapter title truncation, and interactive "Tauhan" and "Sanggunian" toggle pills.
    3. Adjusted mobile fullscreen reading padding to `px-4 sm:px-8 py-6 sm:py-20 pb-[max(4rem,env(safe-area-inset-bottom))]` for maximum readable width and notch/home-indicator clearance.
    4. Converted Sanggunian reference overlay (`refFullscreen`) on mobile to a unified vertical comparison layout:
       - **Pinagmulang Teksto**: Placed in a prominent, full-width white card at the top with quote icon, chapter badge, and full readable sentence text (resolving the issue where it was squished into an unviewable 15px sliver).
       - **Nahanap na Katugma**: Rendered directly below with directional connector, matched sentence card, character tags, and similarity score gauge in a single smooth scroll view (zero nested scroll traps).
       - **Desktop View**: Maintained the split-screen side-by-side (`lg:flex-row`) experience for larger screens.
  - **Verification**:
    - `npm run build` compiled cleanly in 5.2s with zero errors.
    - Verified locally on `http://localhost:3000` via browser subagent at mobile viewport (390x844 iPhone size).
    - Verified smooth fullscreen entry, sticky header layout, safe-area clearance, 44px tap targets, working "Tauhan" and "Sanggunian" toggles inside mobile fullscreen, prominent and fully readable "Pinagmulang Teksto" card, and smooth return to normal modal mode.

---

- **Item E Background Image Orientation & Scaling (Completed 2026-10-17, branch `fix/background-image`)**:
  - **Identified Root Causes**:
    1. Both `noli_background.webp` (736x1308) and `elfili_background.webp` (675x1200) were exported sideways (rotated 90 degrees counter-clockwise), causing Dr. José Rizal, the Manila church, characters, and the novel title banners to display on their sides.
    2. `frontend/app/globals.css` referenced non-existent JPEG files (`Textured.jpg`, `download (1).jpg`), causing 404 HTTP errors on every page load.
    3. `style={backgroundStyle}` with `background-size: cover` was attached directly to `<main>` in `page.tsx` and search container in `search/page.tsx`. Because `<main>` spans all 64 chapters and sections (~5,924px scroll height), the 1308x736 image expanded to over **10,500px wide (8x / 800% zoom)**, causing narrow viewports to display a massive microscopic crop.
    4. On portrait mobile (<768px), widescreen 16:9 artwork scaled with `cover` cropped off 65% of the sides and magnified the cursive title scroll across the hero text.
    5. In `frontend/components/NovelBackground.tsx`, an internal duplicate raster layer (`scale: 1.05`, `blur(4px)`) sat inside the hero section, blurring out and double-scaling the background.
  - **Implemented Fixes**:
    1. Re-exported `noli_background.webp` and `elfili_background.webp` rotated 90° clockwise to their natural upright landscape orientation (1308x736 and 1200x675) with 95 quality WebP compression.
    2. Updated `globals.css` CSS variables (`--image-bg-neutral`, `--image-bg-noli`, etc.) to point to valid existing WebP assets (`background_1.webp`, `background_2.webp`).
    3. Replaced inline container background with a dedicated fixed viewport backdrop (`.novel-backdrop-layer` with `position: fixed; inset: 0; pointer-events: none; -z-10`) in `page.tsx` and `search/page.tsx`, eliminating the 8x zoom caused by document height.
    4. Implemented responsive media-query framing in `globals.css`:
       - **Mobile Portrait (<768px)**: Clean antique parchment texture (`background_1.webp`) with warm amber (Noli) or rich crimson (Fili) ambient radial glow, giving 100% text legibility with zero oversized text clash.
       - **Desktop & Tablet (>=768px)**: Full panoramic upright landscape collage (`noli_background.webp` / `elfili_background.webp`) framing church on the left and characters on the right with a soft 86%-98% parchment wash.
    5. Cleaned up `NovelBackground.tsx` by removing the redundant blurry raster layer (`scale: 1.05`, `blur(4px)`), allowing the clean viewport backdrop and vector ambient accents to render sharply.
  - **Verification**:
    - `npm run build` compiled cleanly in 4.8s with 0 errors.
    - Verified locally on `http://localhost:3000` via browser subagent across mobile (500x749) and desktop (1440x900) viewports: confirmed clean antique parchment on mobile without giant letters, and full upright panoramic artwork on desktop.

---

- **Item F Dark Mode Implementation & WCAG AA Contrast Audit (Completed 2026-10-17, branch `feat/dark-mode`)**:
  - **Identified Contrast & Dark Theme Requirements**:
    1. The application lacked dark mode support, defaulting to high-brightness parchment and white surfaces that caused glare in dim environments.
    2. Next.js 16 with Tailwind CSS v4 required an explicit variant definition (`@custom-variant dark (&:is(.dark, .dark *));`) to enable standard `dark:` utility class selection on both `html.dark` and all nested child components.
    3. Multiple legacy components contained hardcoded inline `style={{ backgroundColor: 'rgba(255,255,255,...)' }}` (such as `HeroSection` floating icons and stats card), preventing CSS dark mode class overrides.
    4. Text elements in `ResultCard.tsx`, `ChapterGrid.tsx`, and `ChapterModal.tsx` required auditing to ensure WCAG AA contrast ratio (>= 4.5:1 for body, >= 3:1 for large headings) against midnight dark backgrounds.
  - **Implemented Architecture & Theming**:
    1. Installed and integrated `next-themes` with `ThemeProvider` (`attribute="class"`, `defaultTheme="system"`, `enableSystem`) in `frontend/app/providers.tsx` and added `suppressHydrationWarning` to `<html>` in `frontend/app/layout.tsx`.
    2. Created `frontend/components/ThemeToggle.tsx` providing both a compact icon toggle for desktop headers and an accessible 3-way segmented control (Liwanag / Dilim / Sistema) for mobile drawers and setting panels.
    3. Integrated `ThemeToggle` into desktop header navigation, mobile drawer (`page.tsx`), and search bar header (`search/page.tsx`).
    4. Engineered complete dark mode design system tokens in `frontend/app/globals.css`:
       - Base dark canvas: `--theme-bg: #0F141C;`, `--theme-card: #18202C;`, `--theme-text: #EAE6DF;`, `--theme-text-muted: #9E9B93;`, `--theme-border: rgba(232, 201, 122, 0.18);`.
       - Dark Noli accents: `--theme-primary: #D4AF37;`, `--theme-accent: #E8C97A;`.
       - Dark Fili accents: `--theme-primary: #C94C68;`, `--theme-accent: #E28498;`.
       - Dark mode gold drop cap (`.dark .drop-cap::first-letter: #E8C97A`) and dark scrollbars.
    5. Audited and upgraded components for WCAG AA contrast compliance:
       - `ChapterModal.tsx`: Dark mode modal shell, chapter text reading canvas, Prev/Next buttons, and full Sanggunian reference overlay.
       - `ChapterGrid.tsx`: Card surfaces (`dark:bg-brand-paper/85`), decorative chapter numerals, and headings (`dark:text-foreground`).
       - `ResultCard.tsx`: Card titles (`dark:text-foreground`), meaning/themes cards (`dark:bg-brand-paper`).
       - `NovelToggle.tsx`: Novel switcher dropdown and pill buttons (`dark:bg-brand-paper/98`, `dark:text-brand-gold`).
       - `HeroSection.tsx`: Replaced hardcoded inline white styles with responsive Tailwind dark classes (`bg-white/90 dark:bg-brand-paper/95`, `dark:text-foreground`).
       - `SearchBar.tsx`: Fixed search input text visibility in dark mode by adding `text-brand-navy dark:text-foreground`, `dark:placeholder:text-brand-text-muted/60`, high-contrast clear `X` button, and dark midnight suggestions menu (`dark:bg-brand-paper/98`).
       - `NovelBackground.tsx`: Added `dark:opacity-15` to the sky layers to prevent the blinding white radial wash from overpowering the midnight backdrop.
       - `search/page.tsx`: Added `data-novel` container attribute, dark header styling (`dark:bg-[#0C1424]/95`), back button contrast, and dark mode toggles for Buod/Buong Kwento, Gabay na Mapa, and Marka.
  - **Identified & Resolved Tailwind v4 Selector Stripping**:
    - Discovered that Tailwind CSS v4's compiler discarded bare `.dark { ... }` blocks when `@custom-variant dark` was present. Resolved by defining `html.dark, :root.dark, :root[class~="dark"]`, ensuring all midnight CSS variables (`--theme-bg: #0D131F;`, `--theme-card: #151E2E;`, etc.) properly cascade to `body`, `main`, and cards.
  - **Verification**:
    - `npm run build` compiled cleanly with 0 errors.
    - Verified locally on `http://localhost:3000` via browser subagent: verified midnight navy background, white typed text in search input (`search_bar_typed_kalayaan`), dark search suggestions dropdown, and fully dark-themed search results page with high-contrast result cards (`search_results_dark_mode`).
    - Zero diff on `backend/app/core/engine.py` (strictly 0 diff).

---

## Next Steps
- Await user approval to merge `feat/dark-mode` into `main` and push to GitHub.






