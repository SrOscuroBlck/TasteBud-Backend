from sqlalchemy import text, inspect
from config.database import engine
from utils.logger import setup_logger

logger = setup_logger(__name__)


def table_exists(table_name: str) -> bool:
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def add_access_request_table():
    if table_exists("access_request"):
        logger.info("access_request table already exists, skipping migration")
        return

    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS access_request (
                id UUID PRIMARY KEY,
                email VARCHAR NOT NULL UNIQUE,
                status VARCHAR NOT NULL DEFAULT 'pending',
                requested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP,
                admin_token VARCHAR NOT NULL,
                token_used BOOLEAN NOT NULL DEFAULT FALSE,
                notes VARCHAR
            )
        """))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_access_request_email ON access_request (email)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_access_request_admin_token ON access_request (admin_token)"))
        conn.commit()

    logger.info("access_request table created successfully")


if __name__ == "__main__":
    add_access_request_table()
