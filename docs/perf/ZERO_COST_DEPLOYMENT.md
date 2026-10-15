# Zero-Cost ($0) Production Deployment Guide

This guide details the complete procedure for deploying the **Rizal Thematic Exploration System** publicly to production at **$0.00 / month forever**, preserving all search performance (~480 ms latency) and full model accuracy.

---

## 1. High-Level Architecture & Cost Breakdown

| Component | Host / Provider | Free Tier Allowance | Project Usage | Monthly Cost |
| :--- | :--- | :--- | :--- | :---: |
| **Frontend Web UI** | **Vercel** (Hobby Plan) | Unlimited deployments, 100 GB bandwidth, automatic SSL, Edge CDN | Next.js 16 Web App (~15 MB) | **$0.00** |
| **Backend API Engine** | **Hugging Face Spaces** (Docker) | **16 GB RAM**, 2 vCPU, 50 GB storage, persistent public HTTPS | FastAPI + PyTorch + Dual XLM-R (~1.5 GB RAM) | **$0.00** |
| **Database + Vectors** | **Supabase** or **Neon.tech** | 500 MB PostgreSQL storage, native `pgvector`, SSL pooling | 147 MB DB (16,402 sentences, 545 themes) | **$0.00** |
| **Precomputed Cache** | Baked in Docker image | 73.39 MB (`word_vec_cache_v1.pkl`) | 24,770 words precomputed vocabulary | **$0.00** |
| **Total** | | | | **$0.00 / month** |

> [!NOTE]
> Standard cloud providers (Render, Koyeb, Fly.io) restrict free tiers to **512 MB RAM**, which crashes PyTorch with Out-Of-Memory (OOM). **Hugging Face Spaces** provides **16 GB RAM for free** because it is purpose-built for PyTorch / SentenceTransformer applications.

---

## 2. Step 1: Database Setup (Supabase or Neon)

Our current PostgreSQL database (`rizal_db`) is **147 MB** containing:
- 16,402 sentences (with 768-dim float32 embeddings)
- 545 themes
- Chapter titles and text

### A. Create Free Project
1. Go to [supabase.com](https://supabase.com) (or [neon.tech](https://neon.tech)) and create a free account.
2. Click **New Project**, choose a region close to your users (e.g., `Singapore / ap-southeast-1`), and set a strong database password.
3. In the SQL Editor on Supabase, ensure `vector` is enabled:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

### B. Export Local Database
From your local machine, export the database schema and data:
```bash
docker exec -t rizal_postgres pg_dump -U rizal -d rizal_db --no-owner --no-acl > rizal_db_backup.sql
```

### C. Import to Supabase
Import the backup into your Supabase database using `psql`:
```bash
psql "postgresql://postgres:[YOUR-PASSWORD]@db.[YOUR-PROJECT-REF].supabase.co:5432/postgres" < rizal_db_backup.sql
```
*(Copy the connection string from Supabase Dashboard → Settings → Database → Connection String → URI).*

---

## 3. Step 2: Packaging `word_vec_cache_v1.pkl`

To ensure production retains the **~480 ms** search latency without needing 20-second on-demand encoding:
- `backend/app/data/word_vec_cache_v1.pkl` is **73.39 MB** (well under GitHub's 100 MB limit).
- Add an exception in `.gitignore`:
  ```gitignore
  backend/app/data/*.pkl
  !backend/app/data/word_vec_cache_v1.pkl
  ```
- This ensures the precomputed cache is checked into git and automatically copied into the Docker container during build.

---

## 4. Step 3: Backend Deployment (Hugging Face Spaces)

### A. Create Hugging Face Space
1. Sign up / log in to [huggingface.co](https://huggingface.co).
2. Click **New Space** → Name: `rizal-thematic-search-api`.
3. Select **Docker** (Blank) as the Space SDK.
4. License: **MIT** / Open source.
5. Visibility: **Public**.
6. Space Hardware: **CPU Basic (2 vCPU, 16 GB RAM) - FREE**.

### B. Dockerfile
Create `backend/Dockerfile` optimized for Hugging Face Spaces:

```dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    PORT=7860

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install --no-cache-dir poetry

# Configure poetry to not create virtualenvs inside container
RUN poetry config virtualenvs.create false

# Copy poetry config and install dependencies
COPY backend/pyproject.toml backend/poetry.lock* /app/
RUN poetry install --no-root --no-interaction --no-ansi

# Copy backend application source
COPY backend/app /app/app
COPY backend/app/data/word_vec_cache_v1.pkl /app/app/data/word_vec_cache_v1.pkl

# Copy fine-tuned DAPT model if present
COPY backend/app/models /app/app/models

# Expose Hugging Face default port
EXPOSE 7860

# Run Uvicorn server on port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### C. Environment Variables on Hugging Face Spaces
In Hugging Face Space → **Settings** → **Variables and secrets**:
- `DATABASE_URL`: `postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres`
- `CORS_ORIGINS`: `["https://*.vercel.app", "http://localhost:3000"]`
- `ENVIRONMENT`: `production`

Hugging Face builds the container and provides your public API URL:
`https://[username]-rizal-thematic-search-api.hf.space`

---

## 5. Step 4: Frontend Deployment (Vercel)

1. Sign up / log in to [vercel.com](https://vercel.com) using your GitHub account.
2. Click **Add New...** → **Project**.
3. Select your GitHub repository: `Rizal-Thematic-Exploration`.
4. Configure Project settings:
   - **Framework Preset**: `Next.js`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`
5. **Environment Variables**:
   Add in Vercel:
   - `NEXT_PUBLIC_API_URL`: `https://[username]-rizal-thematic-search-api.hf.space`
6. Click **Deploy**.
   Vercel will build the frontend and provide your production link:
   `https://rizal-thematic-exploration.vercel.app`

---

## 6. Pre-Merge Verification Checklist

Before merging `perf/speed-optimization` into `main`:
- [ ] `.gitignore` updated to track `backend/app/data/word_vec_cache_v1.pkl`.
- [ ] Backend `Dockerfile` committed and verified.
- [ ] Database backup `rizal_db_backup.sql` generated and tested.
- [ ] Frontend API URL environment variable verified in `frontend/`.
- [ ] Merge `perf/speed-optimization` into `main`.
