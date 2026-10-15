# Performance Optimization State

**Last Updated**: 2026-10-16  
**Branch**: `perf/speed-optimization`  
**Current Phase**: Steps 1–A optimizations, Step B determinism, Translation Policy (#2), and Deployment Sizing (#3) **approved and finalized**. Ready for final integration merge into `main`.

---

## Done and Verified (With Evidence)

- **IPv6 localhost stall**: `scripts/benchmark.py` updated to `http://127.0.0.1:8000` + persistent `requests.Session()`; socket tests proved `::1` connect caused ~2,046 ms TCP stall; all prior "localhost" latency tables in `BASELINE.md` marked invalid for absolute latency.
- **Process-level word-vector cache**: `_word_vec_cache` eliminated rerank token encoding (`edukasyon` rerank loop dropped from ~32s to ~65–77 ms); same-process cached vs uncached 0/40; 786/786 vectors verified bit-identical.
- **Root cause of run-to-run variance**: Unsorted set iteration in theme expansion caused non-deterministic word order in combined concept text under Python hash randomization; setting `PYTHONHASHSEED=0` yielded 100% bit-identical rankings and scores across fresh server processes.
- **Accumulating PerfTimer fix**: Fixed `PerfTimer.stage()` to accumulate (`+=`) across candidate loops and track call counts; added stage timers for 3 previously untimed encode sites; verified 0/40 vs `golden_snapshot_seed0.json`.
- **Process-level TranslatorCache**: Added thread-safe cache with 24h positive TTL, 60s negative TTL, and 5s timeout; verified out-of-domain repeated queries drop from ~1.3s first-seen to ~127 ms warm with 0/40 golden snapshot differences.
- **Batched context fetch**: Replaced 40 sequential candidate queries with `_fetch_context_cache` union bounding SQL fetch; DB queries slashed from 42 to 3 for `bayan` (-92.9%) and 44 to 5 for `edukasyon` (-88.6%); verified 382/382 candidate instances and 180/180 boundary sentences 100% bit-identical.
- **Startup precomputed vocabulary cache**: Precomputed 24,770 words one-at-a-time into `backend/app/data/word_vec_cache_v1.pkl` (73.39 MB); loads in 238.2 ms at boot; 500/500 sample vectors bit-identical (`np.array_equal == True`); 0/40 vs `golden_snapshot_seed0.json`; first search of unseen Tagalog words dropped from 5.7s–20.6s to 486–606 ms with 0 cache misses.
- **Step B Determinism (Approved 2026-10-16)**: Applied 5 targeted commits (B1, D, E1, E2, E3) to eliminate all sources of cross-process non-determinism. 22/40 golden queries changed at tied score boundaries. 100% bit-identical across 6 server processes (40 queries) and 206 held-out queries. `PYTHONHASHSEED=0` no longer required. New permanent reference: `golden_snapshot_deterministic.json`.

---

## Final Before vs. After Benchmark (Runs 2–10 Pooled Warm Statistics)

Measured on a quiet system with identical protocols (`http://127.0.0.1:8002`, `requests.Session()`, 2 warmup passes):

| Category | Before Warm P50 | After Warm P50 | Delta (P50) | Before Warm P95 | After Warm P95 | Delta (P95) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `single_word_tagalog` | 5,069.59 ms | **478.27 ms** | -4,591.32 ms (-90.6%) | 29,922.43 ms | **691.14 ms** | -29,231.29 ms (-97.7%) |
| `multi_word_tagalog` | 573.61 ms | **502.93 ms** | -70.68 ms (-12.3%) | 2,087.86 ms | **754.86 ms** | -1,333.00 ms (-63.8%) |
| `english_cross_lingual` | 416.80 ms | **344.67 ms** | -72.13 ms (-17.3%) | 649.97 ms | **526.54 ms** | -123.43 ms (-19.0%) |
| `character_names` | 510.80 ms | **353.34 ms** | -157.46 ms (-30.8%) | 757.49 ms | **522.37 ms** | -235.12 ms (-31.0%) |
| `out_of_domain` (repeated) | 1,298.91 ms | **129.73 ms** | -1,169.18 ms (-90.0%) | 1,495.15 ms | **171.45 ms** | -1,323.70 ms (-88.5%) |

*Out-of-Domain First-Seen (Run 1, Live Network Call): 1,312.99 ms (Before) vs. 1,291.12 ms (After).*

---

## Known Limitations

1. ~~**PYTHONHASHSEED Requirement**: Must launch server with `PYTHONHASHSEED=0` to guarantee cross-process determinism until Step B `sorted()` changes land.~~ **RESOLVED** by Step B.
2. **First-Seen Translator Latency**: First search of an unmapped out-of-domain term takes ~1.3s due to synchronous external Google Translate network call.
3. **Memory Footprint**: Process RSS is 460.35 MB on boot, reaches ~950 MB during 400-query benchmark, and settles at ~1.2 GB during intensive un-cached inference.
4. **Per-Worker In-Memory Caches**: `_word_vec_cache` and `TranslatorCache` live in Python process memory; multi-worker deployments do not share caches across workers.
5. **Static Vocabulary Cache Binding**: `word_vec_cache_v1.pkl` is bound to model name and vocabulary hash; must be re-generated if novel text or model changes.
6. **ibarra LIMIT-200 Truncation**: High-frequency content words exceed `LIMIT top_k * 20 = 200` in full corpus search. `ORDER BY Sentence.id` makes truncation deterministic but excludes later occurrences.

---

## Not Verified / Open Questions

- **Precomputed Cache Miss Count**: Cache miss count of the precomputed cache over a full fresh-boot 40x10 benchmark has not been explicitly logged.
- **Multi-Word Latency Outlier**: Single run maximum outlier of 2,460 ms on `karahasan ng mga prayle` remains unexplained (likely OS scheduling/DB query spike).
- **Measurement Noise Floor**: Per-query latency deltas under ~100 ms between runs are noise on this system.

---

## Next Action: Final Integration
 
1. **Final Integration**: Merge `perf/speed-optimization` → `main`.
2. Push `main` to remote repository if desired.

