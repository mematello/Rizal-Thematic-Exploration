from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    PROJECT_NAME: str = "Rizal Thematic Exploration API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: str
    
    # Redis
    REDIS_URL: str
    
    # ML
    BERT_MODEL_NAME: str = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"

    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
