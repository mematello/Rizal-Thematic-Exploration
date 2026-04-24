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
    TEST_GATE_THRESHOLD_OOV: float = 0.45
    
    # Matching Logic Thresholds (Paksa)
    PAKSA_MIN_KWS_WITH_CHARS: int = 2
    PAKSA_MIN_KWS_NO_CHARS: int = 3
    PAKSA_WEIGHT_KEYWORD_DENSITY: float = 0.40
    PAKSA_WEIGHT_SEMANTIC: float = 0.60
    PAKSA_THEME_THRESHOLD: float = 0.70
    PAKSA_OVERRIDE_THRESHOLD: float = 0.85
    
    # Matching Logic Thresholds (Sanggunian)
    SANGGUNIAN_WEIGHT_LEXICAL: float = 0.55
    SANGGUNIAN_WEIGHT_SEMANTIC: float = 0.45
    SANGGUNIAN_FALLBACK_THRESHOLD: float = 0.45
    SANGGUNIAN_DYNAMIC_BFR_MULTIPLIER: float = 2.0
    SANGGUNIAN_CHAR_BOOST_PER_MATCH: float = 0.05
    SANGGUNIAN_CHAR_BOOST_MAX: float = 0.20
    SANGGUNIAN_SPACY_MODEL: str = "xx_ent_wiki_sm"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
