# 📖 Rizal Thematic Explorer

[![Next.js](https://img.shields.io/badge/Next.js-black?logo=next.js)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?logo=postgresql&logoColor=white)](https://postgresql.org)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://python.org)

An intelligent, full-stack search and exploration engine for the thematic analysis of José Rizal's novels *Noli Me Tangere* and *El Filibusterismo*. 


## 🖼️ Application Showcase

![Landing Page Hero](./docs/snapshots/landing_page.png)

![Search Functionality Demo](./docs/snapshots/searching_demo.gif)

## ⚙️ Key Engineering Features
- **Semantic & Lexical Hybrid Search**: Utilizes `pgvector` and a custom-tuned `XLM-RoBERTa` (DAPT) model to allow robust Tagalog querying.
- **Dynamic Thematic Mapping**: Translates abstract concepts (e.g., "korapsyon", "pag-ibig") to specific novel contexts using TF-IDF and dense embeddings.
- **Cross-Lingual Support**: Automatically bridges English queries to their Tagalog counterparts via custom dictionary mapping and dynamic fallback.
- **Modern User Interface**: Built with Next.js, Tailwind CSS, and Framer Motion for a responsive and premium academic reading experience.

### Differentiators

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

## 🏗️ Architecture

![Detailed Retrieval Architecture](./docs/snapshots/08-detailed-retrieval-architecture.png)
*This diagram illustrates the staged retrieval flow, demonstrating how initial lexical filtering is combined with deep semantic re-ranking to deliver highly accurate thematic results.*

## 🛠️ Tech Stack
- **Frontend:** Next.js 16 (App Router), TailwindCSS, Zustand, React Query
- **Backend:** FastAPI, SQLAlchemy, PostgreSQL + `pgvector`, Redis
- **Machine Learning:** `sentence-transformers`, custom `XLM-R` model, NLTK, spaCy

## 👥 Contributors & Academic Information
- **Researchers:** Marcus Kent Oliver, Ian Kurby Placencia, Dominic Vilog
- **Adviser:** Dr. Melvin Ballera 
- **Institution:** Technological Institute of the Philippines – Manila (BSCS)


## 🚀 Running Locally

Follow these steps to set up the project on your local machine. This project is cross-platform and supports **Windows (PowerShell), macOS, and Linux**.

### Prerequisites
- **Git**
- **Docker & Docker Compose** (for the database)
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
The backend uses Python and Poetry. The custom ML model (`rizal-xlm-r-dapt`) is pre-trained and bundled, so no training is required.

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

# Start the API server
poetry run uvicorn app.main:app --reload
```
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
poetry run uvicorn app.main:app --reload
```

#### 3. Start the Frontend
Open a **new terminal**:
```bash
cd frontend
npm run dev
```

🚀 **Access the application at:** [http://localhost:3000](http://localhost:3000)

### ⚠️ Windows Setup Notes
If you are developing on Windows, please keep the following in mind:
- **Use PowerShell or Windows Terminal**: Command Prompt (cmd) may behave differently.
- **Python Command**: On Windows, `python` is the standard command, whereas macOS/Linux often default to `python3`. The instructions above use `python` or `python -m poetry` for consistency.
- **Poetry Execution**: We use `python -m poetry` to ensure the correct Python environment is used and to avoid PATH issues common on Windows.
- **Dependencies**: Never run `pip install` globally for this project. Always use `poetry add <package>` or `poetry install` to manage dependencies within the virtual environment.

### 🔑 Environment Variables

**Backend (`backend/.env`)**
Created automatically, but ensure it contains:
```env
DATABASE_URL=postgresql://rizal:rizal123@localhost:5432/rizal_db
REDIS_URL=redis://localhost:6379/0
BERT_MODEL_NAME=sentence-transformers/paraphrase-xlm-r-multilingual-v1
ENVIRONMENT=development
DEBUG=True
```

**Frontend (`frontend/.env.local`)**
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```
