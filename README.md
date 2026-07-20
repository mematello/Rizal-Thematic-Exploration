# Rizal Thematic Analysis

This repository contains the materials, datasets, and code used for our undergraduate thesis on the thematic analysis of José Rizal's novels *Noli Me Tangere* and *El Filibusterismo*.  

## 📖 Project Overview
The study focuses on extracting, categorizing, and analyzing the central themes of Rizal’s works.  
We aim to present thematic explanations in a way that is accessible to both students and researchers.  

## ⚙️ Features
- Automated extraction of themes per chapter  
- CSV dataset containing chapter numbers, titles, and thematic explanations  
- Tools for enhancing thematic content for academic use  

## 🛠️ Tech Stack
- **Python 3** (Pandas, NLTK, NumPy)  
- **GitHub** for version control  

## 👥 Contributors
- Marcus Kent Oliver
- Ian Kurby Placencia
- Dominic Vilog

## 🎓 Academic Information
- **Degree Program:** Bachelor of Science in Computer Science (BSCS)  
- **Section:** CS41S1  
- **Institution:** Technological Institute of the Philippines – Manila  
- **Course:** CCS 401 - Thesis 01
- **Research Adviser:** Dr. Melvin Ballera  


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
