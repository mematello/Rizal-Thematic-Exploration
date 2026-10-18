# Project: Rizal Thematic Explorer

Intelligent search engine for José Rizal's novels (Noli Me Tangere, El Filibusterismo) using hybrid semantic + lexical search with pgvector and custom XLM-RoBERTa DAPT model.

## Key Rules
1. **Never commit to main** — always work on feature branches.
2. **Read `docs/perf/STATE.md` first** every session before any work.
3. **Update memory after every completed task**: Update `STATE.md` + `DECISIONS.md` and commit them with code changes.
4. **Preserve correctness**: Never change ranking output or search behavior without explicit approval and logging in `DECISIONS.md`. Reference snapshot is `docs/perf/golden_snapshot_deterministic.json`.
5. **Corpus Fingerprint Rule**: Any modification to `backend/app/api/v1/content.py` or `backend/app/core/robust_aligner.py` invalidates the Sanggunian fingerprint and requires rebuilding Sanggunian bundles (`build_sanggunian_cache.py`). Any modification to `backend/app/core/engine.py` requires rebuilding chapter bundles (`build_chapter_cache.py`).
6. **No HF Space Push Without Verification & Approval**: Never push to `remotes/hf/main` without testing `/health/db`, verifying canonical chapter SHA-256 (`8f5c4830...`), confirming 40 golden search queries 100% rank invariant, and receiving explicit user approval.

## Tech Stack
- **Backend**: Python 3.11+, FastAPI 0.109+, Uvicorn 0.27+, SQLAlchemy 2.0+, PostgreSQL + pgvector, Redis, sentence-transformers 3.3.0
- **Frontend**: Next.js 16.1.6 (App Router), React 19.2.3, TailwindCSS v4, Zustand 5, TanStack React Query 5.90.20, Framer Motion
- **ML**: Custom fine-tuned XLM-RoBERTa DAPT model (`mematello/rizal-xlm-r-dapt`), base XLM-R for embeddings

## Architecture
- Database: Supabase Singapore AWS Pooler (`130 MB` live binary database) or local Postgres on port 5433
- Redis cache on port 6379 (optional local development)
- Backend API: http://localhost:8000 (Local) / https://mematello-rizal-thematic-search-api.hf.space (Production)
- Frontend: http://localhost:3000 (Local) / https://rizalengine.vercel.app (Production)

## Current Focus
- Item G: Documentation & Repository Sync on branch `docs/sync`.
- Production deployment verified across all 16 matrix configurations (0 console errors, 0 axe violations).
- Strictly zero diff on `backend/` and `frontend/` preserved.
