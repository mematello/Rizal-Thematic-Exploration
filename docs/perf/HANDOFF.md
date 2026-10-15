# Performance Optimization Handoff & Developer Guide

**Project**: Rizal Thematic Explorer  
**Branch**: `perf/step-b-determinism` (pending merge into `perf/speed-optimization`)  
**Current Status**: Phase 2 Steps 1–A complete; Step B (determinism) **approved** (2026-10-16). Pending merge and remaining policy decisions.  
**Reference Snapshot**: `docs/perf/golden_snapshot_deterministic.json` (Step B deterministic reference, replaces `golden_snapshot_seed0.json`)  

---

## 1. How to Start the Server

Always launch the Uvicorn server explicitly bound to IPv4 `127.0.0.1`. **Do NOT use `--reload`** during performance profiling or benchmarking (the reload reloader process adds file-watching threads, IPC overhead, and process jitter). `PYTHONHASHSEED=0` is no longer required after Step B determinism changes, but remains recommended for reproducibility with older snapshots.

### Windows PowerShell
```powershell
cd backend
$env:PYTHONHASHSEED="0"
$env:PERF_BENCHMARK="1"
.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Linux / macOS / Bash
```bash
cd backend
export PYTHONHASHSEED=0
export PERF_BENCHMARK=1
poetry run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

---

## 2. Environment Flags & Toggles

The codebase provides runtime toggles to enable differential testing and regression analysis:

| Variable | Default | Purpose |
| :--- | :---: | :--- |
| `PYTHONHASHSEED` | *(any)* | No longer required after Step B determinism. Previously needed `0` to eliminate Python hash randomization. Set to `0` for compatibility with pre-Step-B snapshots. |
| `PERF_BENCHMARK` | `0` | When `1`, returns internal engine micro-timings, call counts, and cache statistics in the search response metadata. |
| `DISABLE_WORD_VEC_CACHE` | `0` | When `1`, skips precomputed cache loading and in-memory process cache, forcing on-the-fly candidate token encoding. |
| `DISABLE_TRANSLATOR_CACHE` | `0` | When `1`, bypasses the in-memory `TranslatorCache`, making synchronous external Google Translate network requests. |
| `DISABLE_BATCHED_CONTEXT_FETCH`| `0` | When `1`, bypasses `_fetch_context_cache`, falling back to sequential individual SQL queries for neighbor contexts. |
| `bypass_cache=true` (HTTP Param)| `false`| URL query parameter on `/api/v1/search` to bypass the word vector cache for an isolated request. |

---

## 3. Key Files & Architecture Map

### Core Search Pipeline (`backend/app/`)
- **`core/engine.py`**: The central `RizalEngine`. Contains query preprocessing, theme expansion, lexical retrieval, semantic fallback, reranking validation, batched context retrieval (`_fetch_context_cache`), and word vector cache loading (`_load_precomputed_word_vectors`).
- **`core/perf_instrumentation.py`**: Thread-safe accumulating `PerfTimer` context manager tracking stage durations and call counts.
- **`api/v1/search.py`**: FastAPI search endpoint (`/api/v1/search`). Parses query parameters and returns search payload and benchmark metadata.
- **`models/database.py`**: SQLAlchemy ORM models (`Sentence`, `Theme`, `NovelContent`).
- **`data/word_vec_cache_v1.pkl`**: Precomputed float32 word embeddings (24,770 words, 73.39 MB) loaded at boot.

### Performance Documentation (`docs/perf/`)
- **`STATE.md`**: Compact source of truth (<120 lines). Current phase, verified empirical findings, and immediate next action.
- **`DECISIONS.md`**: Complete log of all architectural decisions, evidence, and pending policy items.
- **`FINAL_REPORT.md`**: Comprehensive thesis-ready report with before/after tables, resource footprints, and verification evidence.
- **`golden_snapshot_deterministic.json`**: Permanent deterministic reference snapshot (Step B, 40 golden queries, all server processes produce bit-identical results).
- **`golden_snapshot_seed0.json`**: Historical pre-Step-B reference snapshot (requires `PYTHONHASHSEED=0`).
- **`BASELINE.md`**: Historical record of early Phase 1 baseline measurements (annotated with IPv6 invalidation note).
- **`PLAN.md`**: Roadmap of completed vs. pending optimization milestones.

---

## 4. Scripts Inventory & Tooling

All utility scripts reside in `scripts/`:

- **`build_word_vec_cache.py`**: Precomputes embeddings for all 24,770 candidate vocabulary words one-at-a-time using SentenceTransformer and serializes to `backend/app/data/word_vec_cache_v1.pkl`.
- **`benchmark.py`**: Automated test client executing 10 runs across all 40 golden queries. Uses `http://127.0.0.1:8000` and persistent `requests.Session()` to avoid IPv6 connection stalls.
- **`fetch_snapshot.py`**: Fetches fresh search results for all 40 golden queries from a running server and outputs a structured JSON snapshot.
- **`compare_golden.py`**: Deeply compares a target snapshot against a reference snapshot (`golden_snapshot_seed0.json`), reporting any differences in candidate ranking order, IDs, scores, context strings, or theme tags.
- **`measure_fresh_server.py`**: Boots a dedicated server instance, measures startup duration to `/health`, captures initial RSS, evaluates first-search latency for 10 unseen Tagalog words, logs post-search RSS, and cleanly terminates the server.
- **`verify_batched_context.py`**: Exhaustive candidate-level and boundary-sentence comparison verifying sequential vs. batched context fetching across 382 candidates and 180 boundary sentences.

---

## 5. How to Run Golden Comparisons

Whenever making changes to engine logic or database queries, verify ranking invariance against the golden snapshot:

1. **Start the server with benchmark settings**:
   ```powershell
   $env:PYTHONHASHSEED="0"; $env:PERF_BENCHMARK="1"; .venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
2. **Fetch the current snapshot**:
   ```powershell
   .venv\Scripts\python scripts/fetch_snapshot.py --port 8000 --output docs/perf/new_snapshot.json
   ```
3. **Compare against the reference snapshot**:
   ```powershell
   .venv\Scripts\python scripts/compare_golden.py --ref docs/perf/golden_snapshot_seed0.json --target docs/perf/new_snapshot.json
   ```
   *Expected outcome*: `40/40 queries matched (0 differences)`.

---

## 6. Hard-Learned Rules & Methodological Principles

1. **Never trust summaries without raw output**: Always verify raw JSON numbers directly rather than relying on unverified terminal summaries or client-side timings.
2. **Control experiments before blaming a change**: Run A/B tests inside the exact same Python process (e.g. bypassing a cache via a query parameter) to isolate application logic from OS scheduling, pagefile throttling, and garbage collection noise.
3. **State what was not verified**: Never extrapolate unmeasured claims. If a metric was not directly captured under controlled conditions, explicitly classify it as "Not verified".
4. **Beware of OS-level network defaults**: On Windows, `localhost` resolves to IPv6 (`::1`) first. If the server is bound to IPv4, connection attempts stall for ~2.0 seconds before falling back. Always use explicit IPv4 `127.0.0.1` and persistent HTTP sessions.
5. **Python hash randomization no longer breaks cross-process determinism**: After Step B, all set iterations in the search pipeline are explicitly sorted. `PYTHONHASHSEED=0` is no longer required, but the historical lesson stands: always sort set iterations before combining text for transformer model encoding.
