from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pathlib import Path
from sqlalchemy import text
from config import settings, create_db_and_tables, engine
from routes.api import router as api_router
from routes.ingestion import router as ingestion_router
from routes.sessions import router as sessions_router
from routes.feedback import router as feedback_router
from routes.admin_rebuild import router as admin_rebuild_router
from routes.early_access import router as early_access_router
# from routes.recommendation_session import router as recommendation_session_router
from services.features.faiss_service import FAISSService
from scripts.migrations.migrate_add_permanently_excluded_items import add_permanently_excluded_items_column
from scripts.migrations.migrate_fix_permanently_excluded_items_type import fix_permanently_excluded_items_type
from scripts.migrations.migrate_add_feedback_indexes import add_feedback_performance_indexes
from scripts.migrations.migrate_add_course_cuisine import add_course_and_cuisine_columns
from scripts.migrations.migrate_add_ingredient_penalties import add_ingredient_penalties_column
from scripts.migrations.migrate_add_onboarding_choices import add_onboarding_choices_column
from scripts.migrations.add_access_request_table import add_access_request_table
from utils.logger import setup_logger

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application lifespan")
    
    # Enable pgvector extension before creating tables
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    logger.info("Enabled pgvector extension")
    
    # Create all tables first (handles fresh databases)
    create_db_and_tables()
    logger.info("Database tables synchronized")
    
    # Run migrations for existing tables (handles schema updates)
    add_permanently_excluded_items_column()
    fix_permanently_excluded_items_type()  # Fix TEXT -> JSONB conversion
    add_feedback_performance_indexes()
    add_course_and_cuisine_columns()  # Add meal type filtering columns
    add_ingredient_penalties_column()
    add_onboarding_choices_column()
    add_access_request_table()

    # Load FAISS index for similarity search
    faiss_service = FAISSService()
    index_loaded = False
    
    try:
        faiss_service.load(index_name="current", dimension=64)
        logger.info("FAISS index loaded successfully", extra={"dimension": 64, "count": faiss_service.index_size})
        index_loaded = True
    except FileNotFoundError:
        logger.info("64D FAISS index not found, trying 1536D fallback")
        try:
            faiss_service.load(index_name="current", dimension=1536)
            logger.info("FAISS index loaded successfully", extra={"dimension": 1536, "count": faiss_service.index_size})
            index_loaded = True
        except FileNotFoundError:
            logger.warning("FAISS index not found, similarity search will be unavailable")
        except Exception as e:
            logger.error("Failed to load 1536D FAISS index", extra={"error": str(e)}, exc_info=True)
    except Exception as e:
        logger.error("Failed to load 64D FAISS index", extra={"error": str(e)}, exc_info=True)
    
    app.state.faiss_service = faiss_service if index_loaded else None
    
    yield
    
    logger.info("Application shutdown complete")


app = FastAPI(title="TasteBud API", version="0.1.0", debug=settings.DEBUG, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware execution order (outermost first): CorrelationId → RateLimit → RequestTiming → handler
# Starlette reverses add_middleware order, so add in reverse.
from middleware.logging_middleware import CorrelationIdMiddleware, RequestTimingMiddleware
from middleware.rate_limit_middleware import RateLimitMiddleware

app.add_middleware(RequestTimingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(CorrelationIdMiddleware)

app.include_router(api_router, prefix="/api/v1")
app.include_router(ingestion_router, prefix="/api/v1")
app.include_router(sessions_router, prefix="/api/v1")
app.include_router(feedback_router, prefix="/api/v1")
app.include_router(admin_rebuild_router, prefix="/api/v1")
app.include_router(early_access_router, prefix="/api/v1")
# app.include_router(recommendation_session_router, prefix="/api/v1")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    return {"app": "TasteBud", "status": "running"}

