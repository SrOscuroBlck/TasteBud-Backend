"""
Migration: Add preferred_language column to users table

Adds a preferred_language column (VARCHAR, default 'en') to the users table.
Safe to run multiple times - checks for column existence before altering.
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


def add_preferred_language_column():
    table_name = "user"

    if not table_exists(table_name):
        logger.info(f"Table '{table_name}' does not exist yet - will be created with column by SQLModel")
        return

    if column_exists(table_name, "preferred_language"):
        logger.info(f"Column 'preferred_language' already exists in {table_name} table")
        return

    logger.info(f"Adding 'preferred_language' column to {table_name} table")

    with engine.connect() as conn:
        conn.execute(text(f"""
            ALTER TABLE "{table_name}"
            ADD COLUMN preferred_language VARCHAR NOT NULL DEFAULT 'en'
        """))
        conn.commit()

    logger.info(f"Successfully added 'preferred_language' column to {table_name} table")


if __name__ == "__main__":
    add_preferred_language_column()
