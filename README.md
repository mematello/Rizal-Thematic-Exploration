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

### 1. Clone the Repository
```bash
git clone https://github.com/mematello/Rizal-Thematic-Exploration.git
cd Rizal-Thematic-Exploration
```

### 2. Database Setup
Start the PostgreSQL (with pgvector) and Redis services using Docker.
> [!NOTE]
> Ensure Docker Desktop is running.

```bash
docker compose up -d
```
*If `docker compose` is not recognized, try `docker-compose up -d`.*

This will start the database on port `5432`.

### 3. Backend Setup (FastAPI)
The backend requires Python dependencies managed by Poetry.

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

2.  **Install Poetry** (if not already installed):
    
    *macOS / Linux / WSL:*
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

    *Windows (PowerShell):*
    ```powershell
    python -m pip install --user poetry
    ```

3.  **Configure Poetry & Install Dependencies:**
    It is recommended to create the virtual environment inside the project directory to avoid conflicts.
    ```bash
    # Configure in-project virtualenv
    python -m poetry config virtualenvs.in-project true

    # Install dependencies
    python -m poetry install --no-root
    ```
    
    *Verify installation:*
    ```bash
    python -m poetry --version
    ```

4.  **Seed the database** with Noli & Fili chapters and themes:
    ```bash
    python -m poetry run python scripts/seed_db.py
    ```

5.  **Start the backend server:**
    ```bash
    python -m poetry run uvicorn app.main:app --reload
    ```
   
The API will be available at `http://localhost:8000`.  
Explore the interactive API docs at `http://localhost:8000/docs`.

### 4. Frontend Setup (Next.js)
1.  **Open a new terminal** and navigate to the frontend directory:
    ```bash
    cd frontend
    ```

2.  **Install Node.js dependencies:**
    ```bash
    npm install
    ```

3.  **Start the development server:**
    ```bash
    npm run dev
    ```

    > [!TIP]
    > If you see an error saying `'next' is not recognized`, ensure you ran `npm install` first. Then use `npm run dev` (which uses the local `next` binary) instead of trying to run `next` globally.

    
Open [http://localhost:3000](http://localhost:3000) in your browser to view the application.

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
