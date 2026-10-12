# Performance Optimization Decisions

**Purpose**: Log all decisions that could affect search behavior, ranking, or correctness.

---

## Decision Log

### 2026-10-12: Initial Planning
**Context**: Beginning performance optimization project for public deployment.

**Decision**: Adopt phased approach with measurement-first strategy.

**Rationale**: 
- Avoids premature optimization
- Establishes baselines to measure improvement
- Ensures correctness preserved throughout

**Impact**: None yet (planning phase)

---

### 2026-10-12: Phase 1 Instrumentation
**Context**: Need to identify actual bottlenecks before optimizing.

**Decision**: Add comprehensive measurement infrastructure with minimal behavior changes:
- Golden query set (40 queries) with full result snapshot
- Cold vs warm timing (10 runs per query)
- DB query counter via SQLAlchemy event listener
- EXPLAIN ANALYZE profiling script
- Comparison tool to detect ranking changes

**Implementation**:
- Created `scripts/benchmark.py` for automated performance measurement
- Created `scripts/compare_golden.py` for correctness verification
- Created `scripts/golden_queries.json` with 5 categories × 8 queries
- Created `scripts/explain_queries.py` for database query plan analysis
- Added `backend/app/core/query_counter.py` for DB query tracking
- Modified `backend/app/api/v1/search.py` to expose query count in benchmark mode

**Impact**:
- Performance: None (measurement only, no optimization yet)
- Correctness: Unchanged (instrumentation behind PERF_BENCHMARK flag)
- Trade-offs: None

**Verification**: All changes are read-only instrumentation or behind environment flag

---

## Template for Future Entries

### YYYY-MM-DD: [Brief Title]
**Context**: [Why was this decision needed?]

**Decision**: [What did we decide to do?]

**Rationale**: [Why this approach?]

**Impact**: 
- Performance: [Measured impact on latency/throughput]
- Correctness: [Any ranking changes? Verified how?]
- Trade-offs: [What did we give up?]

**Verification**: [How did we confirm correctness?]

---

## Ranking Change Policy

Any change that affects search result order MUST:
1. Be explicitly approved by user
2. Be logged here with full justification
3. Include before/after comparison on golden query set
4. Document the specific ranking delta (# of position changes)
5. Explain why the change improves search quality

**Current ranking changes**: None
