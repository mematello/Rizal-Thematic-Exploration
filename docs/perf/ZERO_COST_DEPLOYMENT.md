# Production Deployment & Infrastructure Guide

This guide details the verified deployment architecture and operational procedure for the **Rizal Thematic Exploration System** across Vercel, Hugging Face Spaces, and Supabase.

---

## 1. High-Level Architecture & Resource Allocation

| Component | Host / Provider | Verified Tier & Specifications | Project Usage | Cost / Plan Status |
| :--- | :--- | :--- | :--- | :---: |
| **Frontend Web UI** | **Vercel** | Hobby Plan (Edge CDN, SSL, Next.js 16.1.6 runtime) | Next.js 16 App (~15 MB static bundle) | Free Tier ($0.00) |
| **Backend API Engine** | **Hugging Face Spaces** | `zero-a10g` (Gradio SDK 5.16.0, `app_file: space.py`) | FastAPI + SentenceTransformers (~1.5 GB RAM) | Free Tier (Community ZeroGPU quota limits; `isPro: false`) |
| **Database + Vectors** | **Supabase** | Free Tier (Singapore AWS Pooler, PostgreSQL 15+ with `pgvector`) | 130 MB DB (16,402 sentences, 545 themes) | Free Tier ($0.00) |
| **Precomputed Cache** | Git-tracked static bundles | Baked in repository: `word_vec_cache_v1.pkl` (73.39 MB), 205 chapter bundles (3.12 MB), 205 Sanggunian bundles (6.06 MB) | 24,770 words vocabulary; all 205 chapters & Sanggunian cross-references | Included in repo |

> [!NOTE]
> **Plan & Quota Verification**:
> The Hugging Face account (`mematello`) operates on the standard free tier (`isPro: false`, verified via HF API `/api/whoami-v2`). Hardware allocation is set to `zero-a10g` (NVIDIA A10G dynamic ZeroGPU assignment under free community quota). While operating under free allowances, community ZeroGPU usage is subject to dynamic daily quota thresholds and queueing during peak hours. Unqualified "$0.00 / month forever" claims are not guaranteed under third-party terms.

---

## 2. Database Sizing & Storage Reconciliation

Database storage involves three distinct figures measured across development and production environments:

1. **Supabase Production Database**: **130 MB** (135,919,283 bytes, **[Verified]** via `SELECT pg_size_pretty(pg_database_size(current_database()))` on live Supabase pooler).
   - `sentences` table: 117 MB (16,402 rows with 768-dimensional float32 vector embeddings).
   - `themes` table: 2.48 MB (545 themes).
2. **Local PostgreSQL Database**: **147 MB** (earlier local measurement before database VACUUM / table optimization).
3. **Plaintext SQL Dump (`rizal_db_backup.sql`)**: **241.8 MB** (253,554,017 bytes on disk, **[Verified]**).
   - **Cause**: Likely text vs binary vectors. In PostgreSQL binary storage, vector embeddings use compact packed 4-byte IEEE 754 floats. In `pg_dump` plaintext SQL, each 768-dimensional vector is expanded into human-readable ASCII text strings (e.g. `[-0.01234567, 0.04567891, ...]`), inflating data footprint from 4 bytes per float to 12–16 bytes per float.

---

## 3. Backend Deployment Architecture (Hugging Face Spaces)

### A. Space Configuration & Runtime
- **Space Repository**: `https://huggingface.co/spaces/mematello/rizal-thematic-search-api` (Remote: `hf`).
- **Space SDK**: **Gradio SDK** (`sdk: gradio`, `sdk_version: 5.16.0`, `app_file: space.py` defined in root `README.md` YAML front matter).
- **Container Entrypoint**: Live server logs show Hugging Face runs **Gunicorn with Uvicorn workers**. Note: The specific launch command is not defined in this repository; it is injected by Hugging Face's official Gradio container supervisor.
- **Docker Usage**: `backend/Dockerfile` exists in the repository for local containerized development, but **is NOT used by the current Hugging Face Space** (the Space is provisioned via the Gradio SDK).
- **ZeroGPU Status**: The Space is assigned the `zero-a10g` hardware tier. In `space.py`, only the dummy Gradio interface function `predict()` is decorated with `@spaces.GPU`. The core FastAPI search engine (`engine.py`) operates via **CPU inference** using precomputed embedding caches.

### B. Environment Variables on Hugging Face Spaces
Configured in Space Settings → **Variables and secrets**:
- `DATABASE_URL`: `postgresql://postgres.[REF]:[PASSWORD]@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres`
- `CORS_ORIGINS`: `["https://rizalengine.vercel.app", "https://*.vercel.app", "http://localhost:3000"]`
- `ENVIRONMENT`: `production`

### C. Rollback Target SHAs on `remotes/hf/main`
If a backend deployment fails verification, roll back to a known-healthy commit from `remotes/hf/main`:
- `7fb5667` (Current Active): `feat(keepalive): add /health/db endpoint and register health router`
- `3117ead`: `feat(perf): configure early logging in main.py to surface cache verification`
- `4bbc31e`: `fix(sanggunian): deploy line-ending normalized fingerprint verification`
- `0d123bc`: `feat(sanggunian): deploy precomputed Sanggunian bundles and chapter endpoint with corpus fingerprint verification`
- `2b15166`: `feat(chapters): deploy precomputed chapter bundles with corpus fingerprint verification and URL sanitization`
- `007116d`: `feat(chapters): optimize chapters endpoint with LRU cache, context batching, and startup warming`
- `0b74a1d`: `feat(themes): optimize themes endpoint with LATERAL query, process cache, startup warming, and static fallback`

---

## 4. Frontend Deployment (Vercel)

- **Framework**: Next.js 16.1.6 (React 19.2.3, TailwindCSS v4, React Query v5.90.20)
- **Root Directory**: `frontend`
- **Build Command**: `npm run build` (`next build --webpack`)
- **Environment Variables**:
  - `NEXT_PUBLIC_API_URL`: `https://mematello-rizal-thematic-search-api.hf.space`

---

## 5. Deployment Order & Verification Protocol

Always execute production deployments in strict dependency order:

```mermaid
graph LR
    Step1[1. Deploy Backend to HF Space] --> Step2[2. Verify Health & Cache Fingerprints]
    Step2 --> Step3[3. Deploy Frontend to Vercel]
    Step3 --> Step4[4. End-to-End Browser & Axe Matrix]
```

### Protocol Steps:
1. **Deploy Backend to Hugging Face First**:
   - Push strictly backend code and data caches to `hf/main`.
   - Wait for Hugging Face build completion (`RUNNING` stage).
2. **Verify Live Backend API**:
   - `GET https://mematello-rizal-thematic-search-api.hf.space/health/db` $\rightarrow$ must return `{"status":"ok","db":"ok"}` with `200 OK`.
   - `GET https://mematello-rizal-thematic-search-api.hf.space/api/v1/chapters/noli/1?mode=buod` $\rightarrow$ canonical SHA-256 hash must match `8f5c4830076befd1225f687fe90d1f47523b0fe7cf4959d84707b285513c25ec`.
   - Confirm startup log outputs: `[CACHE] Chapter static cache verified` and `[CACHE] Sanggunian static cache verified`.
3. **Deploy Frontend to Vercel Second**:
   - Merge approved feature branch into `main` and push to GitHub `origin/main`.
   - Vercel automatically deploys production.
4. **End-to-End Verification**:
   - Run verification suite (`verify_production_live.py`, `verify_production_themes.py`) across 16 matrix configurations (Chromium & WebKit, 390px & 1440px, light & dark, Noli & Fili).
   - Ensure 0 console errors and 0 accessibility violations in axe-core.

---

## 6. Historical Pre-Merge Checklist (Archived)

*The following checklist was completed during the initial Phase 2 deployment milestone (2026-10-16) and is retained for audit lineage:*

- [x] `.gitignore` updated to track `backend/app/data/word_vec_cache_v1.pkl` and JSON cache bundles.
- [x] Backend `Dockerfile` committed for local container parity.
- [x] Database backup `rizal_db_backup.sql` generated and tested (241.8 MB dump).
- [x] Frontend API URL environment variable verified in `frontend/`.
- [x] Merged `perf/speed-optimization` into `main`.
