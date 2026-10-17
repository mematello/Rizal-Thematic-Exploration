# Performance Optimization State

**Last Updated**: 2026-10-18  
**Branch**: `feat/sanggunian-inverse`  
**Current Phase**: Item 1 (Sanggunian Inverse in Buong Kwento mode) Implemented & Verified. Strict Backend ZERO Diff Preserved.

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
  - **Dark Mode Aesthetic & Contrast Refinements (2026-10-17)**:
    1. **Modal Backdrop Fix**: Replaced `bg-brand-navy/70` (which computed to bright gold at 70% opacity in dark mode because `--theme-primary` is gold) with `bg-black/75 dark:bg-black/85 backdrop-blur-sm` in `ChapterModal.tsx` and `ItemModal.tsx`, eliminating all light wash overlays when opening modals.
    2. **Tauhan & Sanggunian Button Contrast**: In `ChapterModal.tsx`, active buttons now use `bg-brand-gold text-[#1B263B] font-bold` (dark navy on gold, eliminating the invisible gold-on-gold text bug). Inactive buttons use `bg-white/70 dark:bg-[#1B2433] text-brand-text dark:text-foreground/80`.
    3. **Fullscreen Sidebar in ChapterModal**: Replaced washed-out light grey `bg-[#F5F2EB]/50` with `dark:bg-[#121A28] border-r border-brand-gold/10 dark:border-brand-gold/20`, dark headings, and gold active toggle indicators.
    4. **Theme Detail Card / ItemModal Styling**: Overhauled `ItemModal.tsx` theme definition box (`bg-brand-gold/10 dark:bg-black/40 border border-brand-gold/20 dark:border-brand-gold/30`) and contextual match card (`bg-white dark:bg-[#1B2433] border-l-4 border-brand-gold`) to match the midnight gold palette. Chapter and paksa cards in character view updated to dark surfaces.
    5. **Mobile View Tabs & Kabanata Pills Contrast**:
       - `ContentTabs.tsx`: Container uses `dark:bg-black/40 border-brand-gold/30`. Active tab uses `dark:bg-brand-gold dark:text-[#0F141C] font-bold`. Inactive tabs use `dark:text-foreground/70`. Desktop tabs use `sm:dark:bg-transparent` with gold underline indicator.
       - `ChapterSidebar.tsx`: Active mobile pill uses `dark:bg-brand-gold dark:text-[#0F141C] dark:border-brand-gold`. Inactive pills use `dark:bg-[#182232]/90 dark:text-foreground/80 dark:border-brand-gold/30`. Desktop sidebar links use `dark:text-brand-gold dark:bg-brand-gold/20`.
    6. **Noli Homepage White Fade Elimination**: In `HeroSection.tsx`, reduced white vignette opacity to `0.08` and added `dark:opacity-0`. In `NovelBackground.tsx`, replaced `#FFFFFF` 60% opacity stop on `#goldSky` with warm `#C5A065` at 15% opacity, and replaced intense white background radial gradients with subtle gold accents, removing the bright white haze across Noli.
    7. **Floating "Sa Itaas" Button**: Styled as midnight navy pill (`dark:bg-[#182232]/95 dark:text-brand-gold dark:border dark:border-brand-gold/30 dark:hover:bg-[#202C3F]`) with backdrop blur, replacing the bright gold pill with low-contrast white text.
    8. **Buod / Buong Kwento Mode Toggle**: Updated desktop header (`page.tsx`), mobile drawer (`page.tsx`), and search header (`search/page.tsx`) to render active mode with `dark:bg-brand-gold dark:text-[#0F141C] font-bold` on dark midnight pill containers (`dark:bg-[#182232] border-brand-gold/30`), completely fixing the yellow-on-yellow unreadable text trap.
  - **Verification & Deployment**:
    - `npm run build` compiled cleanly in Turbopack with 0 errors.
    - Verified all fixes across desktop and mobile (500x800) via browser subagent with recorded video artifacts and screenshots.
    - Merged branch `feat/dark-mode` into `main` and pushed to GitHub (`b18ddf0`).
    - Strictly **0 diff** on `backend/app/core/engine.py`.

- **Item C Phase 2 Sanggunian Cross-Novel Reference Optimization (Completed 2026-10-17, branch `perf/sanggunian`)**:
  - **Step 0 Baseline Profiling & Measurements (Corrected Scope)**:
    - Raw output documented in `docs/perf/evidence/sanggunian_step0_measurements.txt` and `docs/perf/evidence/sanggunian_endpoint_cold_breakdown.txt`.
    - **Scope Clarification**: ML semantic alignment (`RobustAligner.align`) applies strictly to **2,186 summary sentences across 103 chapters** (Noli: 64 chapters, 1,212 sentences; El Fili: 39 chapters, 974 sentences). For full mode (14,216 sentences across 102 chapters; Noli: 63 chapters, 8,673 sentences; El Fili: 39 chapters, 5,543 sentences), `backend/app/api/v1/content.py` does NOT run `RobustAligner`; it groups sentences by `passage_id` with `score=1.0`. The full-mode static bundles (5.77 MB) exist strictly to eliminate the database roundtrip and passage grouping query.
    - **Actual Endpoint Cold Path Breakdown (Fresh Process)**:
      - Noli 1 (22 buod, 223 full, 114 NULL full): SQL = 3,407.7 ms (4.2%), Encode = 18,597.3 ms (22.8%), RobustAligner = 59,570.7 ms (73.0%), Total = **81,577.4 ms (~81.6 s)**.
      - Noli 5 (9 buod, 19 full, 9 NULL buod): SQL = 330.0 ms (8.5%), Encode = 1,690.9 ms (43.5%), RobustAligner = 1,865.7 ms (48.0%), Total = **3,886.9 ms (~3.9 s)**.
      - El Fili 10 (21 buod, 171 full, 0 NULL): SQL = 2,429.1 ms (5.8%), Encode = 0.5 ms (0.0%), RobustAligner = 39,440.9 ms (94.2%), Total = **41,871.3 ms (~41.9 s)**.
    - **Reconciliation with 20–52s Cold Times**: Fully reconciled. NULL `embedding_dapt` rows require on-the-fly DAPT CPU encoding (14–19s), while `RobustAligner.align()` scales with matrix dimensions ($M \times N$), requiring 39.4s (21x171) to 59.6s (22x223) on CPU. Under concurrent requests in Phase 1, CPU saturation produced 20–52s+ latency.
  - **Step 1 Backend Precomputed Bundles & Endpoints**:
    - Created `scripts/build_sanggunian_cache.py` with checkpointing and resume. Precomputed all 205 chapters into 4 JSON bundles under `backend/app/data/`:
      - `sanggunian_cache_noli_buod.json`: 1,606.1 KB (64 ch, 1,212 sents)
      - `sanggunian_cache_noli_full.json`: 3,493.2 KB (63 ch, 8,673 sents)
      - `sanggunian_cache_elfili_buod.json`: 1,214.3 KB (39 ch, 974 sents)
      - `sanggunian_cache_elfili_full.json`: 2,277.9 KB (39 ch, 5,543 sents)
      - Total bundle size: **8.39 MB** across 205 chapters (16,402 sentences).
    - Bundle cryptographic fingerprint: Embeds `sentence_count` (16,402), `sentence_text` MD5, NULL `embedding_dapt` count (4,580), `embedding_dapt` column MD5, SHA-256 of `robust_aligner.py`, SHA-256 of alignment logic in `content.py`, and DAPT model name/revision (`mematello/rizal-xlm-r-dapt:main`).
    - Startup verification: Non-blocking `verify_sanggunian_fingerprint(db)` runs during lifespan (~16.14s runtime against Supabase). Matches: serves bundle (0ms). Mismatch: logs loud warning, disables static cache, falls back to live compute. DB unreachable: serves static bundle, logs warning, never crashes.
    - SQL Computation: `embedding_dapt` MD5 is computed inside Supabase PostgreSQL (`md5(string_agg(...))`). Zero vector data transferred over the wire. Requests served from static bundle during verification.
    - Concurrency lock: Added `_alignment_compute_lock` to serialize concurrent live alignment compute races on cache misses.
    - Endpoints:
      - `GET /api/v1/sentences/{id}/sanggunian`: Preserved behavior and response shape unchanged; served from in-memory index when bundle enabled.
      - `GET /api/v1/chapters/{book}/{chapter_number}/sanggunian?mode={mode}`: Returns whole-chapter mapping in one request with `Cache-Control: public, max-age=86400, stale-while-revalidate=604800`.
  - **Footprint & Startup Impact (Evidence: `docs/perf/evidence/sanggunian_footprint_startup.txt`)**:
    - Process RSS before loading bundles: **427.79 MB**
    - Process RSS after loading all 4 bundles: **455.61 MB** (RSS Memory Delta: **+27.81 MB**; load time: 151.81 ms).
    - Docker Image Size Delta: **+8.39 MB** (8,797,554 bytes uncompressed across all 4 JSON bundles).
    - Fingerprint verification runtime: **16.14 seconds** (executed asynchronously in background thread).
  - **Step 2 Parity & Resilience Verification**:
    - **Raw Parity Test**: Compared SHA-256 hashes of raw HTTP bytes across all 205 chapters against live path logic on the same machine. Result: **205/205 chapters identical (0 mismatches)**.
    - **Score Delta on Profiled Sentences**: Queried the 5 profiled sentences against live HF Space and local bundle:
      - Sentence 37177: Local=0.781451 | HF=0.781451 | Delta=0.00000003
      - Sentence 37275: Local=0.822976 | HF=0.822976 | Delta=0.00000003
      - Sentence 37768: Local=0.621622 | HF=0.621622 | Delta=0.00000030
      - Sentence 38389: Local=0.919948 | HF=0.919948 | Delta=0.00000006
      - Sentence 38586: Local=0.838446 | HF=0.838446 | Delta=0.00000003
      - **Max absolute score delta: 0.00000030** (disclosed).
    - **Fallback Resilience**: Verified all 3 cases via `scripts/test_sanggunian_fallbacks.py` (fingerprint mismatch, missing bundle, DB connection down). All passed.
    - **Latency**: Chapter endpoint cold 2.7–20.7 ms, warm avg 2.4–4.0 ms, warm min 1.6–3.1 ms (down from 52,400 ms baseline per chapter).
    - **Golden Invariance**: All 40 golden search queries vs `golden_snapshot_deterministic.json`: **0 diffs (100% invariant)**.
    - **Engine Diff**: `backend/app/core/engine.py` diff: **STRICTLY 0 DIFF**.
  - **Step 3 Frontend Integration & Viewport Verification**:
    - Updated `ChapterModal.tsx` and `ChapterGrid.tsx` with `useQuery({ queryKey: ['sanggunian', book, currentChapterNum, mode], staleTime: 24h })`.
    - Prefetches active chapter and adjacent chapters in buod mode on modal open.
    - Maintained fallback loop to per-sentence endpoint if chapter endpoint fails.
    - Verification: `npx tsc --noEmit` clean, `npm run build` clean.
    - End-to-end browser subagent verified desktop (1920x1080) and mobile (390x844) viewport layouts: Sanggunian toggle enables reference superscripts `[1]`, clicking opens detail comparison modal, responsive layout stacks cleanly without horizontal overflow.

  - **Step 4 Pre-Gate-2 Checks & Parity Verification (Evidence: `docs/perf/evidence/`)**:
    - **Scope Clarification**: Corrected in Gate 1 report, `STATE.md`, and `DECISIONS.md`: ML alignment (`RobustAligner`) applies strictly to 2,186 summary sentences across 103 chapters. Full mode (14,216 sentences across 102 chapters) uses passage ID grouping with `score=1.0`. Full-mode bundles (5.77 MB) exist strictly to eliminate DB roundtrips.
    - **Cold Path Breakdown Reconciled**: Noli 1 = 81.6s (SQL 3.4s, Encode 18.6s, Aligner 59.6s); Noli 5 = 3.9s; El Fili 10 = 41.9s. Individual component timings are verified; however, the CPU-saturation explanation of the observed 20–52 s cold times under concurrency is an inference, not verified.
    - **Live HF Parity (All 2,186 Summary Sentences)**: 2,186 / 2,186 reference sentence IDs matched (100%), 2,186 / 2,186 reference texts matched (100%), 2,186 / 2,186 `alignment_status` matched (100%). Non-score field mismatches: 0. Summary scores match the old backend except float32 noise $\le 3.0 \times 10^{-7}$ (1,829 / 2,186 differ by sub-micro float32 precision across CPU architectures; 357 are exact; 0 structural differences).
    - **500 Random Full Sentences HTTP Byte SHA-256**: 500 / 500 matched 100% bit-identical SHA-256 against live HF Space.
    - **Fingerprint Enhanced**: Added `content.py` alignment code hash and DAPT model revision (`mematello/rizal-xlm-r-dapt:main`). Re-verified 205-chapter parity (205/205 identical). Normalized CRLF/LF line endings for platform-independent fingerprint hashing.
    - **Operational Note (Fingerprint Regeneration)**: The cryptographic fingerprint hashes all of `content.py` and `robust_aligner.py`. Any edit to `content.py` invalidates the bundle fingerprint and requires rebuilding bundles via `poetry run python scripts/build_sanggunian_cache.py`.
  - **Step 5 Gate 2 Backend Deployment to Hugging Face Spaces (Completed & Verified)**:
    - **Rollback Targets**: New HF Space Rollback Target: `3117ead`. Backend rollback target before this item: `2b15166`.
    - **Backend Files Deployed (0 leak)**: Exactly 6 backend files deployed to HF Space commit `4bbc31e`:
      1. `backend/app/api/v1/content.py`
      2. `backend/app/main.py`
      3. `backend/app/data/sanggunian_cache_noli_buod.json`
      4. `backend/app/data/sanggunian_cache_noli_full.json`
      5. `backend/app/data/sanggunian_cache_elfili_buod.json`
      6. `backend/app/data/sanggunian_cache_elfili_full.json`
      - Confirmed strictly 0 frontend files, 0 docs files, and 0 diff on `backend/app/core/engine.py`.
    - **Live HF Verification (Evidence: `docs/perf/evidence/sanggunian_gate2_live_verification.txt`)**:
      - `/health`: **Verified** (200 OK, `{"status":"ok","version":"1.0.0"}`).
      - Old per-sentence endpoint: **Verified** (Sentence IDs 37177, 37275, 38586, 39363 all match reference IDs, texts, and scores).
      - New chapter endpoint for 5 chapters (`noli 1 buod`, `noli 5 buod`, `elfili 10 buod`, `noli 25 full`, `elfili 39 full`): **Verified** with `Cache-Control: public, max-age=86400, stale-while-revalidate=604800` present on all 5 responses.
      - Canonical JSON SHA-256 vs local bundles: **Verified** (100% bit-identical on all 5 chapters).
      - 5-run warm latency across 5 chapters: **Verified** (Avg latencies: 292.2 ms, 240.3 ms, 237.1 ms, 293.1 ms, 422.0 ms over WAN).
      - 5 golden search queries (`kalayaan`, `edukasyon`, `simbahan`, `korapsyon`, `pag-ibig`): **Verified** (Top-5 IDs in Noli and El Fili match 100% with `golden_snapshot_deterministic.json`).
  - **Step 6 Gate 3 Pre-Merge Validation & Early Logging (Completed & Verified)**:
    - Added early `logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True)` to `backend/app/main.py`.
    - Pushed strictly `main.py` to HF Space commit `3117ead`.
    - Pulled live startup logs via `HfApi().fetch_space_logs()`. Verbatim line confirmed:
      `INFO root: Sanggunian static cache verified against Supabase corpus (205 chapters active).` (0 warnings, 0 mismatches, 0 fallbacks).
    - Live `/health` verified (200 OK), Noli 1 buod chapter endpoint verified unchanged (`8f5c4830076befd1225f687fe90d1f47523b0fe7cf4959d84707b285513c25ec`), 5 golden search queries verified 100% invariant.
    - New Rollback Target recorded: `3117ead` (previous rollback target `4bbc31e`, baseline rollback target `2b15166`).
    - Frontend build & typecheck: `npm run build` and `npx tsc --noEmit` verified **clean (0 errors)**.
  - **Step 7 Gate 3 Production Verification on Vercel (Completed & Verified live on https://rizalengine.vercel.app)**:
    - Merged `perf/sanggunian` into `main`, pushed to GitHub `origin/main` (`51b0476`).
    - **Step 5 (Noli 1 Buod 1-request verification & timing)**: **[Verified]** Exactly 1 request to `/api/v1/chapters/noli/1/sanggunian?mode=buod` (0 individual sentence requests). Real `performance.now()` click-to-DOM superscripts rendering time: **66.30 ms** (instant perceived load). 22 superscripts rendered.
    - **Step 6 (Superscript Click & React Query Cache)**: **[Verified]** Superscript `[1]` click opened cross-reference drawer with matching El Fili reference text. Closed drawer and modal; re-opening Noli 1 and toggling Sanggunian made **0 network requests** (instant hydration from React Query cache).
    - **Step 7 (El Filibusterismo Testing)**: **[Verified]** Switched novel to El Filibusterismo, opened Chapter 10 Buod: exactly 1 request to `/api/v1/chapters/elfili/10/sanggunian?mode=buod`, 21 superscripts rendered, drawer opened with cross-reference.
    - **Step 8 (Fallback Resilience)**: **[Verified]** Intercepted fetch to simulate 500 error on chapter endpoint. Chapter 4 opened and gracefully fell back to individual per-sentence endpoints (`/api/v1/sentences/{id}/sanggunian`), rendering 30 superscripts with 0 UI crashes.
    - **Step 9 (Mobile Viewport 390x844)**: **[Verified]** Resized viewport to 390x844. Chapter modal and Sanggunian drawer layout rendered cleanly with `hasHorizontalOverflow: false` (`scrollWidth === clientWidth`) and 0 console errors.
    - **Diff & Safety Constraints**: `backend/app/core/engine.py` diff: **STRICTLY ZERO DIFF**. Supabase DB writes: **STRICTLY ZERO WRITES**.
    - Raw evidence saved in `docs/perf/evidence/sanggunian_gate3_vercel_verification.txt`.

- **Item B Keep-Alive Workflow, Health DB Probe & Server Warmup Banner (Completed 2026-10-17, branch `feat/keepalive`)**:
  - **Policy Research & Quoted Standards (`docs/perf/evidence/keepalive_policy_sources.txt`)**:
    - **HF Spaces Sleep Policy**: *"Spaces will automatically 'sleep' or shut down after 48 hours of inactivity."* (Source: https://huggingface.co/docs/hub/spaces-gpus#sleep-time)
    - **Supabase Pausing Policy**: *"Free tier projects enter a paused state after 7 days of inactivity... Inactivity is determined by querying activity, not auth or storage traffic."* (Source: https://supabase.com/docs/guides/platform/pausing)
  - **Backend Health Probe Router (`backend/app/api/v1/health.py`)**:
    - Strictly **0 diff** on `backend/app/core/engine.py`, `backend/app/api/v1/content.py`, and `backend/app/core/robust_aligner.py`.
    - Dedicated engine using `NullPool` to guarantee immediate socket closure with 0 connection pool retention.
    - Driver-level connect timeout (5s) and statement timeout (`options="-c statement_timeout=5000"`).
    - Query: `SELECT 1 FROM sentences LIMIT 1` (read-only index scan on the user's `sentences` table to guarantee Supabase activity telemetry registers real database activity).
    - Returns `HTTP 200` with `{"status":"ok","db":"ok"}` and `Cache-Control: no-store` on success; returns `HTTP 503` with `{"status":"degraded","db":"error"}` on failure (sanitized, zero internal exception text leakage).
    - Verified under 20 consecutive forced timeouts: thread count remained strictly capped (3 threads) and checked-out connections remained strictly at 0.
  - **GitHub Actions Scheduled Keepalive (`.github/workflows/keepalive.yml`)**:
    - Triggers on cron `*/30 * * * *` (every 30 minutes) and `workflow_dispatch`.
    - Step 1: Pings `https://mematello-rizal-thematic-search-api.hf.space/health` (keeps HF container awake).
    - Step 2: Pings `https://mematello-rizal-thematic-search-api.hf.space/health/db` (runs `SELECT 1 FROM sentences LIMIT 1` against Supabase, keeping DB project active).
    - Both curl steps tested and verified returning HTTP 200.
  - **Server Warmup Banner UI & Observer Filtering (`frontend/components/ServerWarmupBanner.tsx`)**:
    - Repositioned from top center to bottom center:
      `bottom-[max(1.25rem,calc(env(safe-area-inset-bottom)+0.75rem))] sm:bottom-6 left-1/2 -translate-x-1/2`
    - High-contrast midnight navy container (`bg-brand-navy/95 dark:bg-[#1B2433]/95 text-white`) with gold border, yielding WCAG AAA compliance (11.5:1 light, 15.1:1 dark).
    - Fixed `z-[70]` to render cleanly above modal overlays (`z-50`), with `pointer-events-none`, `role="status"`, and `aria-live="polite"`.
    - Observer Filtering: Strict predicate `query.getObserversCount() > 0 && query.state.data === undefined && query.state.fetchStatus === 'fetching'`. Only initial user-facing queries with no cached data trigger the banner. Background prefetches and background cache refreshes are completely ignored.
    - Verified with throttled requests: appears after ~3s, disappears on success, absent on warm loads.
    - Non-overlap verified: clears the floating "Sa Itaas" button and ChapterModal bottom navigation buttons ("Nakaraan" / "Susunod").
  - **Live Deployments & Rollback Targets**:
    - Hugging Face Space current commit: `7fb5667` (contains strictly `health.py` and `main.py`).
    - HF Space rollback target: `3117ead` (previous Item C Gate 3).
    - HF Space baseline: `2b15166`.
    - GitHub `main` commit: `411780a` (merged `feat/keepalive`).
  - **Verification Claims (Verified / Not verified)**:
    - Live container startup log quotes verbatim `Sanggunian static cache verified against Supabase corpus (205 chapters active).`: **[Verified]**
    - Live `GET /health/db` returns 200 OK with `{"status":"ok","db":"ok"}` and `Cache-Control: no-store`: **[Verified]**
    - Live `GET /health` returns 200 OK with `{"status":"ok","version":"1.0.0"}`: **[Verified]**
    - Live Noli 1 buod chapter endpoint canonical JSON SHA-256 is `8f5c4830076befd1225f687fe90d1f47523b0fe7cf4959d84707b285513c25ec`: **[Verified (100% bit-identical)]**
    - 5 Golden Queries top-5 IDs match 100% vs `golden_snapshot_deterministic.json`: **[Verified]**
    - Zero diff on `engine.py`, `content.py`, `robust_aligner.py`: **[Verified]**
    - Supabase database writes: **[STRICTLY ZERO WRITES - Verified]**
    - Live Vercel banner position, z-index (70), and non-overlap: **[Verified]**
    - Keepalive workflow dispatch / execution: **[Verified]**
      - Run ID: `39046390193` (https://github.com/mematello/Rizal-Thematic-Exploration/actions/runs/39046390193)
      - Event: `workflow_dispatch`
      - Timing: Started `2026-10-17T15:44:46Z`, completed `2026-10-17T15:44:56Z` (Duration: 10s).
      - Conclusion: `success` (All steps succeeded: setup, ping HF/Supabase `/health` & `/health/db`, ping Vercel, complete).
      - Scheduled runs continue every 30 minutes (`*/30 * * * *`).

- **UI Polish Batch & Follow-Up (Completed 2026-10-18, branch `fix/ui-polish` -> `main`, commit `da3dae6`)**:
  - **Backend Isolation**: Strictly **ZERO DIFF** on `backend/` (`engine.py`, `content.py`, `robust_aligner.py`).
  - **Header Controls & Drawer**: Clean responsiveness across 320px, 360px, 390px, and 1440px desktop. Desktop novel selector, reading mode pill, and theme toggle render without wrapping. Mobile drawer dialog opens smoothly with complete navigation hierarchy.
  - **Chapter Pills**: Rendered in smooth horizontal scroll container with left/right padding so neither first nor last pill is clipped by the screen edges.
  - **Theme Titles**: Reverted global shrink (`clamp(...)`, `whitespace-nowrap`, `truncate`) and restored original `text-2xl font-serif text-brand-navy dark:text-foreground font-bold leading-tight mb-3` classes from `main`. Implemented `useThemeTitleFit(title)` hook with 85% floor (20.4px). All 24 other theme titles remain at their original **24px** size, matching Chapter and Character card titles in visual weight. Only "Pagkamapagpatuloy" scales down slightly to remain unbroken on 1 line. Removed extraneous novel badges from theme cards.
  - **Gabay na Mapa / Reading Map Cards**: Replaced translucent `bg-white/70` with `bg-white/80 dark:bg-[#18202C]`. Borders use gold-tinted tokens (`border-brand-gold/15 dark:border-brand-gold/30`), titles use `dark:text-foreground` (`#F0F4F8`, contrast ratio 13.8:1 WCAG AAA), and novel legend dynamically adapts to active novel filter.
  - **Item 1 Diagnosis & Deferral**: Diagnosed "Buong Kwento → Buod" label origin in `ChapterModal.tsx:492`. Backend Sanggunian aligner queries full text sentences and scores them as intended. Code modifications for Item 1 are **deferred** to a dedicated task to preserve backend zero diff.
  - **Live Verification Claims (Verified / Not verified)**:
    - Merged `fix/ui-polish` into `main` (`da3dae6`) and pushed to `origin/main`: **[Verified]**
    - Vercel live production deploy: **[Verified]** (served from `https://rizalengine.vercel.app`)
    - 0 console errors on production: **[Verified]**
    - Header controls on desktop and mobile drawer: **[Verified]**
    - Chapter pills not cut off at screen edges: **[Verified]**
    - Theme titles (Pagkamapagpatuloy on 1 line, all 24 others at 24px): **[Verified]**
    - No novel badge on theme cards: **[Verified]**
    - Gabay na Mapa legend for novel=noli/fili/both: **[Verified]**
    - Gabay na Mapa dark card background `#18202C`: **[Verified]**
    - Accessibility & dark mode contrast: **[Verified]** (Card titles 13.8:1, no critical contrast violations)

- **Item 1: Sanggunian in Buong Kwento Mode Shows Matching Buod (Completed 2026-10-18, branch `feat/sanggunian-inverse`)**:
  - **Hard Constraints Honored**:
    - `backend/` strictly **ZERO DIFF** (`content.py`, `robust_aligner.py`, `engine.py` untouched).
    - Precomputed cache bundles and bundle fingerprints preserved.
    - Zero changes to Buod mode behavior, API responses, or existing search rankings.
  - **Step 0 Findings & Data Parity**:
    - Confirmed 100% of buod `passage_ids` reference full-text sentence IDs of the SAME novel (0 cross-novel references).
    - Corrected Gate 3 note claiming cross-novel match: it was factually incorrect; Sanggunian is strictly intra-novel.
    - Confirmed `passage_ids` never cross chapters, with the sole exception of Noli Buod Ch 64 mapping to Full Ch 63 (Epilogue "Katapusan").
    - Chapter Mapping: El Fili is 1:1 (39:39). Noli Ch 1..63 map 1:1, while Noli Full Ch 63 receives matches from BOTH Buod Ch 63 and Ch 64.
    - Recomputed full-text sentence match distribution across all 14,216 full sentences (8,673 Noli, 5,543 Fili):
      - 0 matches: 8,917 (62.73%)
      - 1 match: 2,540 (17.87%)
      - 2 matches: 1,226 (8.62%)
      - 3 matches: 652 (4.59%)
      - 4+ matches: 881 (6.20%)
      - Total with >= 1 match: 5,299 (37.27% ~ 37.3% coverage). Exact reproduction of baseline figures.
    - Scripted unit check (`scratch/verify_inversion_parity.py`) verified 0 mismatches across all 14,216 sentences between the frontend chapter-by-chapter mapping logic and the global offline inversion.
  - **Frontend Implementation (`frontend/components/ChapterModal.tsx`)**:
    - In full mode, React Query fetches matching buod content and sanggunian bundle (`staleTime: 24h`).
    - Added background prefetch on chapter open.
    - Memoized inverse map (`fullInverseMap`) maps full `sentence.id` to matching `BuodMatch[]` records.
    - No-superscript-when-no-match rule: `[n]` superscripts render exclusively on full sentences with k >= 1 matches. Sentences with 0 matches render clean prose.
    - Overlay label updated to `"Mula sa Buong Kwento → Patungo sa Buod"` in full mode and `"Mula sa Buod → Patungo sa Buong Kwento"` in buod mode.
    - Target card in full mode renders all matching buod sentences ordered by `sentence_index` with alignment score and status.
    - Graceful degradation: if buod query fails, full-mode superscripts are hidden cleanly with 0 alert flood.
  - **Verification Claims (Verified / Not verified)**:
    - `backend/` diff relative to `main` strictly EMPTY: **[Verified]**
    - Step 0 intra-novel passage reference assumption (0 cross-novel references): **[Verified]** (Evidence: `docs/perf/evidence/sanggunian_inverse_step0.txt`)
    - Offline inversion parity across 14,216 sentences (0 mismatches): **[Verified]** (Evidence: `docs/perf/evidence/sanggunian_inverse_unit_check.txt`)
    - Browser verification (Noli Ch 1, Noli Ch 63/64 edge, El Fili Ch 10): **[Verified]** (Evidence: `docs/perf/evidence/sanggunian_inverse_browser.txt`)
    - Buod mode regression test (22 superscripts, correct overlay, 100% identical to main): **[Verified]**
    - Responsive layout (390px mobile vertical layout with 0 horizontal overflow, 1440px desktop): **[Verified]**
    - Dark mode styling and contrast: **[Verified]**
    - Browser console errors: 0 errors: **[Verified]**
    - `npx tsc --noEmit`: Clean (Exit Code 0): **[Verified]**
    - `npm run build`: Clean (Exit Code 0, 8/8 static routes generated): **[Verified]**

---

## Next Steps
- Item 1 on branch `feat/sanggunian-inverse` is **100% complete, tested, and documented**.
- Code remains on branch `feat/sanggunian-inverse`. Do NOT merge or push to `main` until explicitly reviewed.


