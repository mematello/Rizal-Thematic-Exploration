# Project: Rizal Thematic Explorer

Intelligent search engine for José Rizal's novels (Noli Me Tangere, El Filibusterismo) using hybrid semantic + lexical search with pgvector and custom XLM-RoBERTa DAPT model.

## Key Rules
1. **Never commit to main** — always work on feature branches
2. **Read `docs/perf/STATE.md` first** every session before any work
3. **Update memory after every completed task**: Update `STATE.md` + `DECISIONS.md` and commit them with code changes
4. **Preserve correctness**: Never change ranking output or search behavior without explicit approval and logging in `DECISIONS.md`

## Tech Stack
- **Backend**: FastAPI, SQLAlchemy, PostgreSQL + pgvector, Redis, sentence-transformers
- **Frontend**: Next.js 16 (App Router), TailwindCSS, Zustand, React Query
- **ML**: Custom XLM-RoBERTa DAPT model, base XLM-R for embeddings

## Architecture
- Database: Postgres with pgvector extension on port 5433
- Redis cache on port 6379
- Backend API: http://localhost:8000
- Frontend: http://localhost:3000

## Current Focus
Performance optimization for public deployment while maintaining search quality and correctness.
- Phase 2 Steps 1–A completed and verified (0/40 ranking changes vs `golden_snapshot_seed0.json`).
- Step B (determinism via `sorted()` and `ORDER BY`) pending approval.
- Always run backend with `PYTHONHASHSEED=0` and connect via `127.0.0.1` with persistent HTTP sessions.

