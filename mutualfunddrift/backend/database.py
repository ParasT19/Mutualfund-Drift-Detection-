"""
MutualFundDrift — database connection and session management.
Provides the SQLAlchemy engine, session factory, and FastAPI dependency.
"""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlmodel import Session, SQLModel

from backend.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,          # verify connection health before using from pool
    pool_size=10,
    max_overflow=20,
    echo=(settings.log_level.upper() == "DEBUG"),
)

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=Session,
)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session and ensures it is
    properly closed after the request, regardless of success or failure.
    """
    db: Session = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Context manager variant (for use in scripts / scheduler)
# ---------------------------------------------------------------------------

@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager version of get_db for use outside FastAPI request lifecycle.
    Commits on clean exit, rolls back on exception.
    """
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("Database transaction rolled back due to: %s", exc)
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Table initialisation
# ---------------------------------------------------------------------------

def create_db_and_tables() -> None:
    """
    Create all SQLModel/SQLAlchemy tables in the database if they don't exist.
    Safe to call multiple times — uses CREATE IF NOT EXISTS semantics.
    """
    try:
        # Import models here to ensure they are registered with SQLModel metadata
        import backend.models  # noqa: F401
        SQLModel.metadata.create_all(engine)
        logger.info("Database tables created / verified successfully.")
    except Exception as exc:
        logger.error("Failed to create database tables: %s", exc)
        raise


def check_db_connection() -> bool:
    """
    Attempt a lightweight query to verify the database connection is alive.
    Returns True if connected, False otherwise.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("Database connection check failed: %s", exc)
        return False
