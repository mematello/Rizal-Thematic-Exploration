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

Follow these steps to set up the project on your local machine.

### Prerequisites
- **Git**
- **Docker & Docker Compose** (for the database)
- **Python 3.12+** (recommended to use `pyenv` or `conda`)
- **Node.js 18+** & **npm**

### 1. Clone the Repository
```bash
git clone https://github.com/mematello/Rizal-Thematic-Exploration.git
cd Rizal-Thematic-Exploration
```

### 2. Database Setup
Start the PostgreSQL (with pgvector) and Redis services using Docker:
```bash
docker-compose up -d
```
This will start the database on port `5432`.

### 3. Backend Setup (FastAPI)
The backend requires Python dependencies managed by Poetry.

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Install dependencies (including PyTorch and Transformers):
   ```bash
   # If you don't have poetry installed, install it first:
   # curl -sSL https://install.python-poetry.org | python3 -
   
   python3 -m poetry install --no-root
   ```
3. Seed the database with Noli & Fili chapters and themes:
   ```bash
   python3 -m poetry run python scripts/seed_db.py
   ```
4. Start the backend server:
   ```bash
   python3 -m poetry run uvicorn app.main:app --reload
   ```
   
The API will be available at `http://localhost:8000`.  
Explore the interactive API docs at `http://localhost:8000/docs`.

### 4. Frontend Setup (Next.js)
1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install Node.js dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
   
Open [http://localhost:3000](http://localhost:3000) in your browser to view the application.

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

