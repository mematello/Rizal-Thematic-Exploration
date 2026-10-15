# UNVERIFIED
# Decision Draft: Phase 2 Step B — Search Pipeline Determinism

**Status**: DRAFT (Pending User Review & Approval; NOT Committed)
**Branch**: `perf/step-b-determinism`
**Baseline Reference**: `docs/perf/golden_snapshot_seed0.json` (Commit `3e58e32`)
**New Reference**: `docs/perf/golden_snapshot_deterministic.json` (Commit `73f1d1a` / HEAD)

---

### Context
In Python 3.3+, SipHash randomizes hash seeds on process startup unless `PYTHONHASHSEED` is explicitly pinned. In our search pipeline (`backend/app/core/engine.py`), four non-deterministic mechanisms caused search results and ranking orders to diverge across independent server processes:
1. SQL queries lacked deterministic secondary ordering keys on `LIMIT top_k * 20` and `ORDER BY` clauses.
2. Theme candidate extraction iterated over unordered Python `set()` objects (`candidates = set()`), and tie-breaking specificity scores (`count_in_target / df`) lacked secondary alphabetical tie-break keys.
3. Cross-lingual synthesized bridge translations iterated over unordered `bridge_tokens` sets when assembling `bridge_text`.
4. Dynamic semantic validation iterated over unordered `sent_words = set(...)`, causing candidate tokens to mutate the shared `valid_tokens` set in arbitrary order across processes.

### Decision
Apply minimal, targeted determinism fixes strictly for **reproducibility across processes**, without altering search scoring models, thresholds, or reranking algorithms:
1. **Commit B1 (`646deb6`)**: Added explicit `ORDER BY Sentence.id` to lexical retrieval statements, `ORDER BY Theme.id` to theme loading, and deterministic primary/secondary keys on SQL queries.
2. **Commit D (`964a2c1`)**: Diagnostic instrumentation only behind `PERF_BENCHMARK=1` (SHA-256 hash of final query vector bytes, bridge token metadata, translator diagnostics).
3. **Commit E1 (`17896f3`)**: Deterministic theme candidate iteration via `sorted(candidates)` and composite sort key `(-specificity, word)` in `_expand_query_with_themes`.
4. **Commit E2 (`6e16b16`)**: Deterministic bridge text synthesis via `sorted([bt for bt in bridge_tokens if len(bt) > 3])`.
5. **Commit E3 (`73f1d1a`)**: Deterministic candidate word iteration in dynamic validation via `cand_words = [w for w in sorted(sent_words) if len(w) > 4 and w not in self.LOW_INFO_WORDS]`.

### Rationale
- **Reproducibility Only**: Public deployment across multiple worker processes (Uvicorn / Gunicorn workers, autoscaled instances, CI test suites) requires that identical query inputs produce bit-identical search results, ranking positions, and score metadata regardless of process startup order or Python hash seeds.
- **Zero Scoring Model Changes**: No semantic weights, lexical weights, precision boosts, or reranking formulas were modified.

### Empirical Behavior Diffs vs Golden Seed0 Reference
Across the 40 golden queries between `golden_snapshot_seed0.json` and HEAD (`golden_snapshot_deterministic.json`):
- **Queries Changed**: 19 queries changed ranking positions (164 position changes in old comparator; 21 queries in strict comparator including metadata-only expansion changes).
- **Attribution**:
  - **Commit B1 (ORDER BY tie-breaks)**: Attributed 16 of the 19 queries (133 position changes).
    - 8 queries were pure same-ID-set reorderings among tied candidates (`Basilio` Noli, `Crisostomo Ibarra` Noli, `Elias` Noli, `Isagani` El Fili, `Sisa` Noli, `bayan` Noli, `freedom` Noli, `paaralan` Noli/El Fili, `simbahan` Noli).
    - 8 queries experienced candidate set changes where tied candidates at the 10th slot were resolved deterministically by `Sentence.id` (`Basilio` El Fili, `Maria Clara` Noli, `Padre Damaso` Noli, `Simoun` El Fili, `bayan` El Fili, `church` Noli, `death` Noli/El Fili, `education` Noli/El Fili, `freedom` El Fili).
  - **Commit E1 (Theme Expansion Determinism)**: Attributed 3 additional ranking changes (`kalayaan`, `edukasyon`, `oppression`):
    - **E1 Semantic-Fallback Effect**: In multi-word Tagalog and abstract queries that trigger theme expansion, sorting candidates alphabetically before specificity tie-breaking stabilized `expanded_text`. In `kalayaan` and `edukasyon`, this produced a consistent concept vector that altered candidate selection in the 10-candidate pool while eliminating cross-process divergence.
  - **Commits E2 & E3**: Produced 0 ranking changes on the 40 golden queries at `PYTHONHASHSEED=0`, but eliminated cross-seed hash divergence in `bridge_tokens` (`love`, `corruption`) and `valid_tokens` (`edukasyon`).

### Known Limitations
1. **`ibarra` LIMIT-200 Truncation**:
   - In full corpus search (`source_type = 'full'`), high-frequency content words exceed `LIMIT top_k * 20 = 200` (e.g. `ibarra` has 246 occurrences in Noli). In `summary` mode (`source_type = 'summary'`), only `ibarra` in Noli (28 occurrences) and common terms occur within 30-50 rows.
   - The hard-coded SQL `limit(top_k * 20)` (200 rows) truncates the lexical candidate pool before semantic scoring can evaluate later occurrences. While `ORDER BY Sentence.id` makes this truncation strictly deterministic, occurrences with `id > 200th match` are excluded from evaluation.
2. **Proof Scope**:
   - Bit-identical determinism was proven across 6 server processes under `unset`, `unset reverse`, `seed 1`, `seed 2`, `seed 12345`, `seed 0` on the 40 golden queries.
   - Broader proof was confirmed across 206 held-out queries (all 31 `EN_TL_DICTIONARY` keys, 100 Tagalog vocabulary words across frequency bands, 50 phrases, 15 names, and 10 multi-sentence period queries) with 0 differences across all pairwise comparisons.
   - Bit-identical determinism was also confirmed with `PERF_BENCHMARK` unset (0 differences).
