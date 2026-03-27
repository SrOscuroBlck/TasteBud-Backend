import os
from typing import Optional, List
from dotenv import load_dotenv
from pathlib import Path

# Load .env from project folder explicitly (works even if CWD differs)
_BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_BASE_DIR / ".env")


class Settings:
    # Database
    DATABASE_URL: str = os.getenv("TASTEBUD_DATABASE_URL", "sqlite:///./tastebud.db")

    # OpenAI (bounded usage)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8010"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # CORS
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")

    # Onboarding
    ONBOARDING_MAX_QUESTIONS: int = int(os.getenv("ONBOARDING_MAX_QUESTIONS", "7"))
    ONBOARDING_EARLY_STOP_CONFIDENCE: float = float(os.getenv("ONBOARDING_EARLY_STOP_CONFIDENCE", "0.8"))
    ONBOARDING_K: float = float(os.getenv("ONBOARDING_K", "0.4"))
    ONBOARDING_SIGMA_STEP: float = float(os.getenv("ONBOARDING_SIGMA_STEP", "0.20"))

    # Recommendation weights
    LAMBDA_CUISINE: float = float(os.getenv("LAMBDA_CUISINE", "0.2"))
    LAMBDA_POP: float = float(os.getenv("LAMBDA_POP", "0.2"))
    MMR_ALPHA: float = float(os.getenv("MMR_ALPHA", "0.7"))
    GPT_CONFIDENCE_DISCOUNT: float = float(os.getenv("GPT_CONFIDENCE_DISCOUNT", "0.3"))
    EXPLORATION_COEFFICIENT: float = float(os.getenv("EXPLORATION_COEFFICIENT", "0.2"))

    # Temporal decay
    FEEDBACK_HALF_LIFE_DAYS: int = int(os.getenv("FEEDBACK_HALF_LIFE_DAYS", "21"))

    # Popularity
    DECAY_HALF_LIFE_DAYS: int = int(os.getenv("DECAY_HALF_LIFE_DAYS", "30"))

    # FAISS
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "data/faiss_indexes/")
    FAISS_DIMENSION: int = int(os.getenv("FAISS_DIMENSION", "1536"))

    # Frontend
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173")

    # JWT
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "30"))

    # SMTP
    SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    SMTP_FROM_EMAIL: str = os.getenv("SMTP_FROM_EMAIL", "")
    SMTP_FROM_NAME: str = os.getenv("SMTP_FROM_NAME", "TasteBud")

    # OTP
    OTP_EXPIRE_MINUTES: int = int(os.getenv("OTP_EXPIRE_MINUTES", "10"))
    OTP_MAX_ATTEMPTS: int = int(os.getenv("OTP_MAX_ATTEMPTS", "3"))

    # Early Access
    EARLY_ACCESS_ENABLED: bool = os.getenv("EARLY_ACCESS_ENABLED", "False").lower() == "true"
    EARLY_ACCESS_ADMIN_EMAIL: str = os.getenv("EARLY_ACCESS_ADMIN_EMAIL", "admin@tastebud-co.com")
    EARLY_ACCESS_EXEMPT_EMAIL: str = os.getenv("EARLY_ACCESS_EXEMPT_EMAIL", "camargogustavoa@gmail.com")
    EARLY_ACCESS_TOKEN_SECRET: str = os.getenv("EARLY_ACCESS_TOKEN_SECRET", "change-this-secret")

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "True").lower() == "true"
    RATE_LIMIT_REDIS_URL: str = os.getenv("RATE_LIMIT_REDIS_URL", "redis://localhost:6379/1")
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "60"))
    RATE_LIMIT_REQUESTS_PER_HOUR: int = int(os.getenv("RATE_LIMIT_REQUESTS_PER_HOUR", "300"))

    # Backend base URL (for email links pointing to API endpoints)
    BACKEND_BASE_URL: str = os.getenv("BACKEND_BASE_URL", "http://localhost:8010")

    # Phase 3: Query-based recommendations
    QUERY_RETRIEVAL_CANDIDATES_MULTIPLIER: int = int(os.getenv("QUERY_RETRIEVAL_CANDIDATES_MULTIPLIER", "3"))
    QUERY_DEFAULT_DIVERSITY_WEIGHT: float = float(os.getenv("QUERY_DEFAULT_DIVERSITY_WEIGHT", "0.3"))
    QUERY_ENABLE_CROSS_ENCODER: bool = os.getenv("QUERY_ENABLE_CROSS_ENCODER", "False").lower() == "true"
    QUERY_CROSS_ENCODER_MODEL: str = os.getenv("QUERY_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Phase 3: MMR diversity
    MMR_DEFAULT_DIVERSITY_WEIGHT: float = float(os.getenv("MMR_DEFAULT_DIVERSITY_WEIGHT", "0.3"))
    MMR_MAX_ITEMS_PER_CUISINE: Optional[int] = int(os.getenv("MMR_MAX_ITEMS_PER_CUISINE")) if os.getenv("MMR_MAX_ITEMS_PER_CUISINE") else None
    MMR_MAX_ITEMS_PER_RESTAURANT: Optional[int] = int(os.getenv("MMR_MAX_ITEMS_PER_RESTAURANT")) if os.getenv("MMR_MAX_ITEMS_PER_RESTAURANT") else None
    
    # MMR in main recommendation flow
    USE_MMR_DIVERSITY: bool = os.getenv("USE_MMR_DIVERSITY", "True").lower() == "true"
    RECOMMENDATION_DIVERSITY_WEIGHT: float = float(os.getenv("RECOMMENDATION_DIVERSITY_WEIGHT", "0.2"))
    
    # Phase 4: Explanations & Evaluation
    EXPLANATION_USE_LLM_FIRST: bool = os.getenv("EXPLANATION_USE_LLM_FIRST", "True").lower() == "true"
    EXPLANATION_MAX_HISTORY_ITEMS: int = int(os.getenv("EXPLANATION_MAX_HISTORY_ITEMS", "5"))
    
    EVALUATION_MIN_AB_TEST_SAMPLES: int = int(os.getenv("EVALUATION_MIN_AB_TEST_SAMPLES", "30"))
    EVALUATION_DEFAULT_TIME_PERIOD_DAYS: int = int(os.getenv("EVALUATION_DEFAULT_TIME_PERIOD_DAYS", "30"))


settings = Settings()
