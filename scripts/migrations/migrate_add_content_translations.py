"""
Migration: Add content_translations column to menuitem table

Stores localized user-facing dish content (description, ingredients) keyed by
language code. Safe to run multiple times - checks for column existence.
"""

from sqlalchemy import text, inspect
from config.database import engine
from utils.logger import setup_logger

logger = setup_logger(__name__)


def table_exists(table_name: str) -> bool:
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def column_exists(table_name: str, column_name: str) -> bool:
    if not table_exists(table_name):
        return False
    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns(table_name)]
    return column_name in columns


def add_content_translations_column():
    table_name = "menuitem"

    if not table_exists(table_name):
        logger.info(f"Table '{table_name}' does not exist yet - will be created with column by SQLModel")
        return

    if column_exists(table_name, "content_translations"):
        logger.info(f"Column 'content_translations' already exists in {table_name} table")
        return

    logger.info(f"Adding 'content_translations' column to {table_name} table")

    with engine.connect() as conn:
        conn.execute(text(f"""
            ALTER TABLE "{table_name}"
            ADD COLUMN content_translations JSON NOT NULL DEFAULT '{{}}'
        """))
        conn.commit()

    logger.info(f"Successfully added 'content_translations' column to {table_name} table")


if __name__ == "__main__":
    add_content_translations_column()
