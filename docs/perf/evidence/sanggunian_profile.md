# Profile Report: Sanggunian Cross-Reference Speed (Phase 1)

**Date**: 2026-10-17  
**Branch**: `perf/sanggunian`  
**Status**: Completed Phase 1 (Profile Only — No Code Changes to App Logic)  
**Constraints Verified**:
- `backend/app/core/engine.py` diff: **STRICTLY ZERO DIFF** (Verified).
- Ranking/Scoring: **NO CHANGES** (Verified).
- Protocols: `127.0.0.1` and persistent `requests.Session()` used for all timings without `--reload` (Verified).

---

## 1. Executive Summary

When a user opens a chapter in `ChapterModal.tsx` and enables the **"Sanggunian"** cross-novel view, the application attempts to display superscript references (`[1]`, `[2]`, etc.) aligning summary sentences to corresponding full-novel passages.

Empirical profiling reveals that **clicking a superscript itself is instantaneous (0.80 ms render)**. The severe latency perceived by users (ranging from **3.8 seconds** on warm cache up to **52.4 seconds** on cold cache) is caused by:
1. **Unbounded Cold Request-Time CPU Compute (98.8% time share, 1.5s – 52.4s)**: On a cold chapter, the backend runs `RobustAligner.align()` over the entire chapter (generating 3–6 sentence sliding windows, Tagalog stemming, TF-IDF matrices, and Needleman-Wunsch DP alignment). If any sentence has `embedding_dapt IS NULL` (which applies to **4,580 out of 16,402 sentences, 27.9% of corpus**), it additionally executes `dapt_model.encode()` on CPU at request time.
2. **Extreme Client-Side N+1 HTTP Roundtrips (N requests per chapter)**: When toggling Sanggunian, `ChapterModal.tsx` does not fetch chapter-level alignment. Instead, it dispatches an individual HTTP `fetch` to `/api/v1/sentences/{id}/sanggunian` for **every single sentence** in batches of 5 (e.g., 22 HTTP requests for Kabanata 1).
3. **Backend Database Roundtrip Floor on Every Warm Request (97% - 98.5% warm share, ~40–80 ms local, ~870–1,080 ms live)**: Even when `_alignment_cache` holds the chapter alignment in memory, each sentence endpoint call still executes `db.query(Sentence).filter(Sentence.id == id).first()` across the wire to Supabase. Across 22 requests from the US Hugging Face Space to Supabase in Singapore, this consumes ~3,800 ms of serialized wire wait.
4. **Complete Absence of Client-Side Cache or Prefetching**: `ChapterModal.tsx` uses raw `window.fetch()` and local component `useState` (`referenceResults`). There is zero React Query integration, zero prefetching on chapter open, and closing the modal completely discards all fetched references.

---

## 2. Code Path Trace (End-to-End)

### Frontend Trace
- **File**: `frontend/components/ChapterModal.tsx`
- **Toggle Triggers**:
  - Desktop Modal Header: Line 539 (`<button type="button" onClick={() => setShowReference(!showReference)} ...>`)
  - Fullscreen Sidebar: Line 560 (`<button onClick={() => setShowReference(!showReference)} ...>`)
  - Mobile Fullscreen Header: Line 582 (`<button type="button" onClick={() => setShowReference(!showReference)} ...>`)
- **Request Dispatch**:
  - Lines 156–192 (`useEffect` monitoring `[showReference, isFullscreen, loadedChapters, content]`):
    ```typescript
    const sentences = isFullscreen ? loadedChapters.flatMap(c => c.content) : content;
    const results: Record<number, any> = { ...referenceResults };
    const toFetch = sentences.filter(s => !results[s.id]);
    
    const batchSize = 5;
    for (let i = 0; i < toFetch.length; i += batchSize) {
        const batch = toFetch.slice(i, i + batchSize);
        await Promise.all(batch.map(async (s) => {
            const res = await fetch(`${apiUrl}/api/v1/sentences/${s.id}/sanggunian`);
            ...
    ```
- **Superscript Rendering & Click**:
  - Lines 627–656:
    When `showReference && referenceResults[sentence.id]?.has_reference`:
    Renders `<sup onClick={(e) => { e.stopPropagation(); setSelectedSentenceForRef(sentence.sentence_index); setRefFullscreen(true); setSelectedReference({ ... }); }} className={superscriptColor}>[{sentence.sentence_index}]</sup>`.
  - Lines 688–770: `refFullscreen && selectedSentenceForRef !== null` renders the split/stacked "Kaugnay na Sanggunian" view.
- **Client Cache / Prefetch Status**:
  - **React Query (`useQuery` / `queryClient`)**: **NONE** (Verified). Raw `fetch` inside `useEffect`.
  - **Prefetching**: **NONE** (Verified). No references are prefetched when the chapter modal opens or when chapters are navigated.
  - **Client Persistence**: **NONE** (Verified). Stored in local component state `const [referenceResults, setReferenceResults] = useState<Record<number, any>>({})` (Line 74). Closing the modal purges state.

### Backend Trace
- **File**: `backend/app/api/v1/content.py`
- **Endpoint**: Line 831 (`@router.get("/sentences/{id}/sanggunian", response_model=SanggunianResponse)`)
- **Handler**: Line 832 (`def get_sentence_sanggunian(id: int, db: Session = Depends(get_db), engine: RizalEngine = Depends(get_engine)):`)
- **Executed Operations**:
  1. `db.query(Sentence).filter(Sentence.id == id).first()` (Line 833).
  2. If `sentence.source_type == 'full'`:
     - Queries `db.query(Sentence).filter(... Sentence.passage_id == sentence.passage_id).order_by(Sentence.sentence_index).all()` (Line 840).
  3. If `sentence.source_type == 'summary'`:
     - Checks `cache_key in _alignment_cache` (Line 871), where `cache_key = (full_book, search_chapter, sentence.book, sentence.chapter_number)`.
     - **On Cache Miss**:
       - Query 2: `buod_rows = db.query(Sentence).filter(Sentence.book == sentence.book, Sentence.chapter_number == sentence.chapter_number, Sentence.source_type == 'summary').order_by(Sentence.sentence_index).all()` (Lines 873–877).
       - Query 3: `full_rows = db.query(Sentence).filter(Sentence.book == full_book, Sentence.chapter_number == search_chapter, Sentence.source_type == 'full').order_by(Sentence.sentence_index).all()` (Lines 879–883).
       - Missing embedding check:
         - `if any(r.embedding_dapt is None for r in buod_rows): buod_embs = engine.dapt_model.encode(buod_texts)` (Lines 893–896).
         - `if any(r.embedding_dapt is None for r in full_rows): full_embs = engine.dapt_model.encode(full_texts)` (Lines 898–901).
       - Alignment compute: `engine.robust_aligner.align(buod_sentences=buod_texts, full_sentences=full_texts, buod_embeddings=buod_embs, full_embeddings=full_embs, ...)` in `backend/app/core/robust_aligner.py:55` (Lines 906–912).
       - Stores alignment in process-level dictionary: `_alignment_cache[cache_key] = ...` (Line 904).
     - **Result Extraction**:
       - Finds index `buod_idx` of current sentence in `buod_rows` (Line 923).
       - Slices full chapter sentences: `full_chapter_sentences[match.best_window_start : match.best_window_end + 1]` (Line 930).
       - Returns `SanggunianResponse` (Lines 934–948).

### DAPT Embedding Storage on `Sentence` Model
- **File**: `backend/app/models/database.py:37`
- **Definition**: `embedding_dapt = Column(Vector(768), nullable=True)  # DAPT — used by Sanggunian`
- **Audit Findings**:
  - Total sentences: **16,402**
  - Summary sentences: **2,186** total (**1,084 NULL**, 1,102 populated)
  - Full sentences: **14,216** total (**3,496 NULL**, 10,720 populated)
  - **Total NULL DAPT embeddings in DB**: **4,580 sentences (27.9%)**
  - Root Cause: `backend/scripts/seed_dapt_db.py` used `session.query(Sentence).offset(offset).limit(batch_size).all()` without an `ORDER BY` clause. In SQL, un-ordered `OFFSET` pagination causes non-deterministic row skipping when rows are updated, leaving 4,580 sentences completely un-seeded.

### Request-Time Encoding Assessment
- **Does the request path call any model `.encode()` at request time?**  
  **YES (Verified)**. Whenever `any(r.embedding_dapt is None)` evaluates to True for a chapter's summary or full rows, the backend invokes `engine.dapt_model.encode(texts)` synchronously on the CPU during the HTTP request. For a chapter with 200–300 full sentences, this adds 20–40 seconds of un-cached compute time.

### Server-Side Caching & Headers
- **Server Cache**: Process-level dictionary `_alignment_cache: dict = {}` in `backend/app/api/v1/content.py:23`.
  - **Thread-Safety**: **UNPROTECTED** (no `threading.Lock`). When 5 concurrent requests hit a cold chapter, all 5 race and recompute alignment simultaneously.
  - **Persistence**: **IN-MEMORY ONLY**. Cleared on server restart or worker restart.
  - **Precomputed Bundles**: **NONE** (unlike `/api/v1/chapters` which has static fallback bundles).
- **HTTP Cache-Control**: **NONE** (Verified). No HTTP caching headers (`Cache-Control`) exist on `/api/v1/sentences/{id}/sanggunian`.

---

## 3. Empirical Measurements

All measurements below are quoted directly from saved raw evidence files in `docs/perf/evidence/`.

### Table 1: Endpoint Latency (Cold vs. Warm Across 10 Runs)
*Raw output: `docs/perf/evidence/sanggunian_latency_local.txt` and `docs/perf/evidence/sanggunian_latency_live.txt`.*

| Sentence Description | Sentence ID | Local Cold (Run 1) | Local Warm Median (Runs 2–10) | Live HF Cold (Run 1) | Live HF Warm Median (Runs 2–10) |
|---|---|---|---|---|---|
| **Noli Ch 1 Sent 1** | 37177 | 52,435.46 ms | 130.74 ms | 1,804.60 ms | 1,065.49 ms |
| **Noli Ch 5 Sent 1** | 37275 | 1,930.54 ms | 127.70 ms | 882.20 ms | 868.89 ms |
| **Noli Ch 34 Sent 1** | 37768 | 26,742.29 ms | 134.77 ms | 1,949.18 ms | 874.96 ms |
| **El Fili Ch 1 Sent 1** | 38389 | 20,937.84 ms | 126.79 ms | 878.11 ms | 870.94 ms |
| **El Fili Ch 10 Sent 1** | 38586 | 25,853.77 ms | 130.59 ms | 1,088.10 ms | 1,079.17 ms |

*(Note on Live HF Cold: The live HF Space had received earlier probe requests which pre-warmed `_alignment_cache` on that worker; live runs reflect steady-state remote roundtrip latency).*

### Table 2: Single Request Execution Breakdown (SQL vs. Python Compute vs. Serialization)
*Raw output: `docs/perf/evidence/sanggunian_timing_breakdown.txt`.*

| Sentence Tested | Run State | Total Server Time | SQL Count | Total SQL Time | Python / Compute Time | Serialization Time |
|---|---|---|---|---|---|---|
| **Noli Ch 1 Sent 1** (ID 37177) | **Cold** | 45,950.28 ms | 3 | 573.85 ms (1.2%) | **45,376.43 ms (98.8%)** | 0.33 ms |
| | **Warm** | 81.83 ms | 1 | **80.64 ms (98.5%)** | 1.20 ms (1.5%) | 0.02 ms |
| **Noli Ch 5 Sent 1** (ID 37275) | **Cold** | 1,585.33 ms | 3 | 206.56 ms (13.0%) | **1,378.78 ms (87.0%)** | 0.04 ms |
| | **Warm** | 39.29 ms | 1 | **38.45 ms (97.9%)** | 0.84 ms (2.1%) | 0.02 ms |
| **Noli Ch 34 Sent 1** (ID 37768) | **Cold** | 25,127.52 ms | 3 | 439.39 ms (1.7%) | **24,688.13 ms (98.3%)** | 0.06 ms |
| | **Warm** | 44.10 ms | 1 | **42.83 ms (97.1%)** | 1.28 ms (2.9%) | 0.03 ms |
| **El Fili Ch 1 Sent 1** (ID 38389) | **Cold** | 20,449.75 ms | 3 | 464.45 ms (2.3%) | **19,985.30 ms (97.7%)** | 0.05 ms |
| | **Warm** | 39.41 ms | 1 | **38.28 ms (97.1%)** | 1.13 ms (2.9%) | 0.03 ms |
| **El Fili Ch 10 Sent 1** (ID 38586) | **Cold** | 24,461.86 ms | 3 | 453.14 ms (1.9%) | **24,008.72 ms (98.1%)** | 0.09 ms |
| | **Warm** | 79.40 ms | 1 | **78.06 ms (98.3%)** | 1.34 ms (1.7%) | 0.02 ms |

### Table 3: Database Query Plans (EXPLAIN ANALYZE)
*Raw output: `docs/perf/evidence/sanggunian_explain.txt`.*

| Query Purpose | SQL Pattern | Scan Type | Rows Scanned | Planning Time | Execution Time |
|---|---|---|---|---|---|
| **Sentence Lookup** | `SELECT * FROM sentences WHERE id = :id` | **Index Scan** using `ix_sentences_id` | 1 row | 0.085 ms | 0.047 ms |
| **Chapter Summary** | `SELECT * FROM sentences WHERE book='noli' AND chapter_number=1 AND source_type='summary' ORDER BY sentence_index` | **Bitmap Index Scan** on `ix_sentences_chapter_number` + `ix_sentences_book` | 22 rows (1 block) | 0.129 ms | 0.170 ms |
| **Chapter Full (Ch 1)** | `SELECT * FROM sentences WHERE book='Noli Me Tangere' AND chapter_number=1 AND source_type='full' ORDER BY sentence_index` | **Bitmap Index Scan** on `ix_sentences_chapter_number` + `ix_sentences_book` | 223 rows (11 blocks) | 0.126 ms | 0.505 ms |
| **Chapter Full (Ch 34)** | `SELECT * FROM sentences WHERE book='Noli Me Tangere' AND chapter_number=34 AND source_type='full' ORDER BY sentence_index` | **Bitmap Index Scan** on `ix_sentences_chapter_number` + `ix_sentences_book` | 245 rows (8 blocks) | 0.126 ms | 0.494 ms |

- **Vector Similarity Query (`<=>`)**: **NONE EXECUTED IN DATABASE** (Verified). The Sanggunian pipeline does not run pgvector queries. Cosine similarity is computed in memory via NumPy.

### Table 4: N+1 Simulation Audit (Chapter 1, 22 Sentences)
*Raw output: `docs/perf/evidence/sanggunian_nplus1_trace.txt`.*

| Metric | Measured Value |
|---|---|
| **Client HTTP Calls Fired** | **22 separate requests** (in 5 batches) |
| **Total Database Queries** | **24 SQL queries** (3 on cold Call 1, 1 on each subsequent call) |
| **Sequential Cumulative Server Time** | **48,843.38 ms** |
| **N+1 Verdict** | **CONFIRMED SEVERE N+1**: 1 HTTP roundtrip + 1 DB query per sentence |

### Table 5: Client-Side Browser Timing
*Raw output: `docs/perf/evidence/sanggunian_client_timing.txt`.*

| Action | Measured Latency | Analysis |
|---|---|---|
| **Click Superscript `[1]` $\rightarrow$ Modal Overlay Mount** | **0.80 ms** | Operates purely on in-memory React state (`selectedReference`). Zero network calls. |
| **Click "Sanggunian" Toggle $\rightarrow$ Superscript Render** | **2.10 ms** (hot cache) / **3,800–52,000 ms** (network/compute) | Client waits for batch HTTP fetches to complete before rendering superscripts. |
| **Network vs. Client Render Share** | Render: **< 0.1%** / Network Wait: **> 99.9%** | Slowness is 100% backend compute, N+1 wire wait, and cache absence. |

---

## 4. Ranked Root Causes by Measured Time Share

| Rank | Root Cause | Time Share | Measured Delay Impact | Classification |
|---|---|---|---|---|
| **1** | **Cold In-Process Alignment & Request-Time Encoding** | **98.8%** of cold requests | **+20,000 ms to +52,000 ms** per cold chapter | **Verified** |
| | `RobustAligner.align()` and `dapt_model.encode()` execute on CPU synchronously during the first request to any un-cached chapter. | | | |
| **2** | **Remote Database Roundtrips on Every Warm Request** | **97.0%–98.5%** of warm requests | **+40–80 ms** (local) / **+870–1,080 ms** (HF Space) per sentence | **Verified** |
| | Even with hot in-memory alignment, `get_sentence_sanggunian` queries `SELECT * FROM sentences WHERE id = ?` across the Singapore wire on every request. | | | |
| **3** | **Client-Side N+1 Request Flooding** | Multiplies latency by **$N/5$ batches** | **+22 HTTP requests** per chapter (e.g. 3.8s–5.0s steady-state on live Space) | **Verified** |
| | `ChapterModal.tsx` requests references sentence-by-sentence instead of fetching a single chapter bundle. | | | |
| **4** | **Ephemeral Client State & Lack of React Query** | Re-triggers full fetch on every modal open | **100% redundant fetches** upon reopening modal or switching chapters | **Verified** |
| | State is held only in local `useState`; no React Query prefetching or caching exists. | | | |

---

## 5. Proposed Fixes & Implementation Options

Following the guidelines and constraints, here are the proposed solutions for user review:

### Proposed Fix 1: Static Precomputed Sanggunian Bundles (Corpus Fingerprint Pattern)
*Follows the proven pattern from `perf/chapter-precompute`.*

- **Mechanism**:
  1. Build an offline precomputation script (`scripts/precompute_sanggunian.py`) that runs the existing `RobustAligner.align()` and `get_sentence_sanggunian()` logic across all 103 chapters (64 Noli + 39 Fili) in both summary and full modes.
  2. Embed corpus fingerprint (exact sentence counts, theme count, MD5 checksum) to ensure static bundles are invalidated if text changes.
  3. Save static JSON bundles: `backend/app/data/sanggunian_cache_noli.json` and `backend/app/data/sanggunian_cache_elfili.json` (~450 KB total).
  4. Serve chapter references either through a new chapter-level endpoint (`GET /api/v1/chapters/{book}/{chapter_number}/sanggunian`) or via fast dictionary lookup on the existing endpoint, falling back to live compute if the bundle is absent.
- **Expected Latency Gain**:
  - Cold requests: **45,000 ms $\rightarrow$ ~1.2 ms (-99.9% reduction)**.
  - Eliminates all database queries on warm hits.
- **Risk to Ranking or Byte Parity**:
  - **Zero Risk**: Offline precompute executes the exact live code path and generates byte-for-byte identical payloads.
- **Verification Plan**:
  - Automated differential parity script testing all 2,186 summary sentences against the live Python implementation to ensure 100% SHA-256 byte parity.

---

### Proposed Fix 2: Chapter-Level Endpoint & React Query Integration
*Addresses Client-Side N+1 and Ephemeral State.*

- **Mechanism**:
  1. Add a chapter-level Sanggunian endpoint: `GET /api/v1/chapters/{book}/{chapter_number}/sanggunian` returning a map of `{ [sentence_id]: SanggunianResponse }` for the entire chapter in a single HTTP response.
  2. In `frontend/components/ChapterModal.tsx`:
     - Replace the raw `useEffect` batch loop with a `@tanstack/react-query` hook:
       ```typescript
       useQuery({
           queryKey: ['sanggunian', book, chapterNumber, mode],
           queryFn: () => fetchChapterSanggunian(book, chapterNumber, mode),
           staleTime: 1000 * 60 * 60 * 24, // 24 hours
           enabled: showReference,
       })
       ```
     - Prefetch visible chapter references when the chapter modal opens using `queryClient.prefetchQuery`.
- **Expected Latency Gain**:
  - Eliminates 22 separate HTTP roundtrips down to **1 single request**.
  - On repeat opens of the same chapter: **0 ms network wait (instantaneous render from React Query cache)**.
- **Risk to Ranking or Byte Parity**:
  - **Zero Risk**: Frontend refactor only; payload structure and data fields remain 100% identical.
- **Verification Plan**:
  - Browser subagent verification measuring 1 network call instead of 22, and instantaneous render upon opening references.

---

### Proposed Fix 3: Request-Time Encoding Remediation (Database Column Seeding)
*FLAGGED: Immediate attention required.*

- **Finding**:
  - **4,580 sentences (27.9% of corpus)** currently have `embedding_dapt IS NULL` in Supabase due to non-deterministic `OFFSET` pagination in `seed_dapt_db.py`.
- **Proposed Action**:
  - Create a deterministic seeding script `scripts/reseed_dapt_missing.py` that queries `WHERE embedding_dapt IS NULL ORDER BY id`, encodes them with `mematello/rizal-xlm-r-dapt`, and commits them to the database.
- **Risk Assessment**:
  - **Potential Impact on Alignment**: Because missing sentences are currently encoded at runtime by `dapt_model.encode()`, populating them in the database does NOT alter their embedding values, but guarantees they are read from disk/DB rather than re-encoded.
  - **Status**: Per guidelines, **DO NOT FIX WITHOUT EXPLICIT APPROVAL**.

---

## 6. Stop Point & Recommendation

### Recommended Action Plan for Phase 2:
1. **Approve Fix 1 + Fix 2**:
   - Precompute Sanggunian alignments into static bundles (`perf/chapter-precompute` pattern).
   - Add chapter-level endpoint and migrate frontend to React Query with 24h `staleTime`.
   - This delivers **instant (< 10 ms)** reference display with **zero diff on `backend/app/core/engine.py`** and **100% byte parity**.
2. **Review Request-Time Encoding (Fix 3)**:
   - If static bundles (Fix 1) are generated, runtime database seeding (Fix 3) is bypassed completely because alignments are served instantly from the precomputed artifact.

**Awaiting user review and approval before proceeding.**
