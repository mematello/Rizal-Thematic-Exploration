# Rizal Thematic Exploration - Backend

FastAPI backend for performing semantic and lexical search on Rizal's novels using `sentence-transformers` and `pgvector`.

## Setup

1. **Install Dependencies**
   ```bash
   poetry install
   ```

2. **Start Infrastructure**
   ```bash
   docker-compose up -d
   ```

3. **Configure Environment**
   Ensure `.env` matches your docker-compose credentials:
   ```env
   DATABASE_URL=postgresql://rizal:dev123@localhost:5432/rizal_db
   ```

4. **Seed Database**
   Downloads model and ingests CSV data from `csvFiles/`.
   ```bash
   poetry run python scripts/seed_db.py
   ```

5. **Run Development Server**
   ```bash
   poetry run uvicorn app.main:app --reload
   ```

## API Documentation
Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
