---
title: Rizal Thematic Search API
emoji: 📖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.16.0
app_file: space.py
pinned: false
license: mit
---

# 📖 Rizal Engine

[![Live Demo](https://img.shields.io/badge/Demo-rizalengine.vercel.app-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://rizalengine.vercel.app)
[![HF Space API](https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Spaces-API%20Endpoint-FFD21E?style=for-the-badge)](https://mematello-rizal-thematic-search-api.hf.space)
[![HF Model Hub](https://img.shields.io/badge/%F0%9F%A4%97%20Model-rizal--xlm--r--dapt-yellow?style=for-the-badge)](https://huggingface.co/mematello/rizal-xlm-r-dapt)
[![Dataset Normalization Repo](https://img.shields.io/badge/GitHub-Dataset%20Normalization-24292e?style=for-the-badge&logo=github)](https://github.com/mematello/rizal-thesis-dataset-normalization)

[![Next.js](https://img.shields.io/badge/Next.js-black?logo=next.js)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Supabase pgvector](https://img.shields.io/badge/Supabase-pgvector-3ECF8E?logo=supabase&logoColor=white)](https://supabase.com)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://python.org)

An intelligent, full-stack search and exploration engine for the thematic analysis of José Rizal's master novels: *Noli Me Tangere* and *El Filibusterismo*.

---

### 🌐 Live Deployments & Key Links

| Resource | Description | URL |
| :--- | :--- | :--- |
| **Production Web App** | Live interactive Next.js application hosted on Vercel | [https://rizalengine.vercel.app](https://rizalengine.vercel.app) |
| **Backend REST API** | FastAPI backend with ZeroGPU acceleration on Hugging Face Spaces | [https://mematello-rizal-thematic-search-api.hf.space](https://mematello-rizal-thematic-search-api.hf.space) |
| **Interactive API Docs** | Swagger UI OpenAPI documentation & endpoint tester | [https://mematello-rizal-thematic-search-api.hf.space/docs](https://mematello-rizal-thematic-search-api.hf.space/docs) |
| **Fine-Tuned DAPT Model** | Domain-Adapted `rizal-xlm-r-dapt` model weights on Hugging Face Model Hub | [huggingface.co/mematello/rizal-xlm-r-dapt](https://huggingface.co/mematello/rizal-xlm-r-dapt) |
| **Dataset Normalization Pipeline** | Vital companion repository for corpus normalization & preprocessing | [github.com/mematello/rizal-thesis-dataset-normalization](https://github.com/mematello/rizal-thesis-dataset-normalization) |

---

## 📚 Vital Companion Repository: Dataset Normalization

A fundamental cornerstone of this research is the automated text normalization and preprocessing pipeline hosted in its dedicated repository:
👉 **[mematello/rizal-thesis-dataset-normalization](https://github.com/mematello/rizal-thesis-dataset-normalization)**

### Why this repository is vital to Rizal Engine:
- **Orthographic Modernization & Cleansing**: Preprocesses historical Tagalog translations, cleaning archaic spelling inconsistencies, OCR scanning artifacts, and irregular punctuation across both *Noli Me Tangere* and *El Filibusterismo*.
- **Sentence-Level Granularity**: Produces the 16,400+ cleanly segmented sentences indexed by Rizal Engine's vector database, ensuring accurate passage boundaries.
- **Thematic Ground Truth**: Constructs the 540+ thematic keyword associations used for lexical filtering, thematic density heatmaps, and retrieval evaluation benchmarks.

---

## 🖼️ Application Showcase

![Landing Page Hero](./docs/snapshots/landing_page.png)

![Search Functionality Demo](./docs/snapshots/searching_demo.gif)

---

## ⚙️ Key Engineering Features

- **Semantic & Lexical Hybrid Search**: Utilizes `pgvector` and a custom domain-adapted `XLM-RoBERTa` (DAPT) model to enable robust Tagalog query matching.
- **Dynamic Thematic Mapping**: Translates abstract concepts (e.g., *korapsyon*, *pag-ibig*, *katarungan*) to specific novel chapters and character arcs.
- **Cross-Lingual Support**: Automatically bridges English queries to their Tagalog literary counterparts via custom bilingual semantic mapping and fallback expansion.
- **Contextual Reading Map**: Enables users to explore preceding and following sentences dynamically to evaluate passage continuity within the chapter.
- **Anti-Hallucination Domain Gate**: Active validation barrier that flags out-of-domain queries, ensuring strictly faithful thematic extractions.

### Architectural Differentiators

**RobustAligner Pipeline**  
![RobustAligner Architecture](./docs/snapshots/13-robustaligner-pipeline.png)  
*The RobustAligner utilizes dynamic-programming sequence alignment to accurately map queries directly to deep contextual passages.*

**Reading Map View**  
![Reading Map Expanded Context](./docs/snapshots/06-reading-map-view.png)  
*Contextual expansion allows users to read neighboring sentences, visually mapped to demonstrate thematic continuity within the novel's structure.*

**Staged Retrieval & Dual Scoring**  
![Dual Scoring Result Cards](./docs/snapshots/05-result-cards-scoring.png)  
*Search results show both semantic and lexical scores, so users can see exactly why a result ranked where it did.*

**Anti-Hallucination Gate**  
![Domain Validation Error State](./docs/snapshots/14-empty-state-rejection.png)  
*Out-of-domain queries are actively blocked by the domain validation gate, preventing AI hallucinations and ensuring rigorous thematic relevance.*

---

## 🏗️ System Architecture

```mermaid
graph TD
    User([User / Researcher]) -->|Web UI| Vercel[Frontend: Next.js on Vercel<br/>rizalengine.vercel.app]
    Vercel -->|HTTPS REST| HF[Backend: FastAPI on Hugging Face Spaces<br/>ZeroGPU Accelerated]
    HF -->|Auto-Load Weights| Hub[Hugging Face Model Hub<br/>mematello/rizal-xlm-r-dapt]
    HF -->|Vector Search & Filter| Supabase[(Supabase PostgreSQL + pgvector<br/>Singapore AWS Pooler)]
    DatasetRepo[Dataset Normalization Pipeline<br/>rizal-thesis-dataset-normalization] -.->|Generates Clean Corpus & Themes| Supabase
```

For detailed cloud deployment architecture and zero-cost hosting specifications, see [ZERO_COST_DEPLOYMENT.md](./docs/perf/ZERO_COST_DEPLOYMENT.md).

---

## 🛠️ Tech Stack

- **Frontend:** Next.js 15 (App Router, Turbopack), Tailwind CSS, Framer Motion, Zustand, React Query, Vercel Web Analytics & Speed Insights
- **Backend:** FastAPI, Uvicorn, SQLAlchemy, PostgreSQL + `pgvector`, Redis
- **Machine Learning & NLP:** `sentence-transformers`, fine-tuned domain-adapted `rizal-xlm-r-dapt`, NLTK, spaCy
- **Cloud Infrastructure:** Vercel (Edge Frontend), Hugging Face Spaces (ZeroGPU Backend), Hugging Face Model Hub (Model Storage), Supabase (Vector Database)

---

## 👥 Academic Attribution & Research Team

This project was developed as an undergraduate thesis research project at the **Technological Institute of the Philippines – Manila**:

- **Researchers & Authors:**
  - **Marcus Kent R. Oliver**
  - **Ian Kurby A. Placencia**
  - **Dominic B. Vilog**
- **Thesis Adviser:** **Dr. Melvin A. Ballera**
- **Degree Program:** Bachelor of Science in Computer Science (BSCS)
- **Institution:** Technological Institute of the Philippines – Manila

---

## 🚀 Running Locally

Follow these steps to set up the project on your local machine. This project is cross-platform and supports **Windows (PowerShell), macOS, and Linux**.

### Prerequisites
- **Git**
- **Docker & Docker Compose** (for local PostgreSQL & Redis)
- **Python 3.12+**
- **Node.js 18+** & **npm**

### Option 1: First-Time Setup (Bagong Startup)
Use this if you just cloned the repository or need to reset the environment completely.

#### 1. Clone the Repository
```bash
git clone https://github.com/mematello/Rizal-Thematic-Exploration.git
cd Rizal-Thematic-Exploration
```

#### 2. Start Infrastructure
Start the PostgreSQL (with pgvector) and Redis services using Docker.
> [!NOTE]
> Ensure Docker Desktop is running.

```bash
docker-compose up -d
```

#### 3. Backend Initialization
The backend uses Python and Poetry. The custom ML model (`rizal-xlm-r-dapt`) can be run locally or pulled from Hugging Face Hub.

Open a terminal and run:
```bash
cd backend
poetry install

# Run database migrations
poetry run python scripts/migrate_source_type.py
poetry run python scripts/migrate_is_short.py
poetry run python scripts/migrate_original_index.py
poetry run python scripts/migrate_dapt_column.py
poetry run python scripts/character_index.py

# Seed the database with base XLM embeddings
poetry run python scripts/seed_db.py
poetry run python scripts/seed_full_db.py

# Seed the Sanggunian (DAPT) embeddings
poetry run python scripts/seed_dapt_db.py

# Precompute word vector cache for fast startup (eliminates slow first request)
poetry run python scripts/build_word_vec_cache.py

# Start the API server (PYTHONHASHSEED=0 ensures deterministic set ordering across runs)
export PYTHONHASHSEED=0
poetry run uvicorn app.main:app --reload
```
*(On Windows PowerShell, use `$env:PYTHONHASHSEED="0"` before running uvicorn)*  
*The API will be available at `http://localhost:8000`. Explore the interactive API docs at `http://localhost:8000/docs`.*

![Swagger UI API Documentation](./docs/snapshots/swagger_ui_api.png)

#### 4. Frontend Initialization
Open a **new terminal** window:
```bash
cd frontend
npm install
npm run dev
```

---

### Option 2: Subsequent Runs (Kasunod na Startup)
Use this for daily development after the initial setup is complete.

#### 1. Start Docker (If not already running)
```bash
docker-compose up -d
```

#### 2. Start the Backend
Open a terminal:
```bash
cd backend
export PYTHONHASHSEED=0   # PowerShell: $env:PYTHONHASHSEED="0"
poetry run uvicorn app.main:app --reload
```

#### 3. Start the Frontend
Open a **new terminal**:
```bash
cd frontend
npm run dev
```

🚀 **Access the application locally at:** [http://localhost:3000](http://localhost:3000)

---

### ⚠️ Windows Setup Notes
If you are developing on Windows, please keep the following in mind:
- **Use PowerShell or Windows Terminal**: Command Prompt (cmd) may behave differently.
- **Python Command**: On Windows, `python` is the standard command, whereas macOS/Linux often default to `python3`. The instructions above use `python` or `python -m poetry` for consistency.
- **Poetry Execution**: We use `python -m poetry` to ensure the correct Python environment is used and to avoid PATH issues common on Windows.
- **Dependencies**: Never run `pip install` globally for this project. Always use `poetry add <package>` or `poetry install` to manage dependencies within the virtual environment.

---

### 🔑 Environment Variables

**Backend (`backend/.env`)**
```env
DATABASE_URL=postgresql://rizal:rizalpass123@localhost:5433/rizal_db
REDIS_URL=redis://localhost:6379/0
BERT_MODEL_NAME=sentence-transformers/paraphrase-xlm-r-multilingual-v1
ENVIRONMENT=development
DEBUG=True
```

**Frontend (`frontend/.env.local`)**
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```
*(For production, set `NEXT_PUBLIC_API_URL=https://mematello-rizal-thematic-search-api.hf.space`)*
