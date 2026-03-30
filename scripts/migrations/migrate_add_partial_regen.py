"""
Add pending_partial_regen column to RecommendationSession table.
Stores accepted/rejected course info for partial meal regeneration.
"""

from sqlmodel import Session, text
from config.database import engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_migration():
    logger.info("Adding pending_partial_regen to RecommendationSession...")

    with Session(engine) as session:
        try:
            session.exec(text("""
                ALTER TABLE recommendationsession
                ADD COLUMN IF NOT EXISTS pending_partial_regen JSON
            """))
            session.commit()
            logger.info("pending_partial_regen column added successfully")
        except Exception as e:
            logger.warning(f"Column may already exist: {e}")
            session.rollback()


if __name__ == "__main__":
    run_migration()
