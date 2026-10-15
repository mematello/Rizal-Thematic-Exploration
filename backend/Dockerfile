FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    PORT=7860

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install --no-cache-dir poetry

# Configure poetry to not create virtualenvs inside container
RUN poetry config virtualenvs.create false

# Copy dependency definitions and install
COPY pyproject.toml poetry.lock* /app/
RUN poetry install --no-root --no-interaction --no-ansi --without dev

# Copy application code
COPY app /app/app

# Ensure precomputed vocabulary cache is in place
COPY app/data/word_vec_cache_v1.pkl /app/app/data/word_vec_cache_v1.pkl

# Expose default port (7860 for Hugging Face Spaces, 8000 for standard Docker)
EXPOSE 7860

# Launch Uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
