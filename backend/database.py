"""
CartSense — SQLAlchemy Database Engine & Session Factory
"""
from __future__ import annotations

import logging
from typing import Any, Generator

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from backend.config import settings

logger = logging.getLogger("cartsense.database")

# ── Engine ────────────────────────────────────────────────────────────────────
_engine_kwargs: dict[str, Any] = {"pool_pre_ping": True, "pool_recycle": 1800}
if settings.is_sqlite:
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    _engine_kwargs["pool_size"] = 5
    _engine_kwargs["max_overflow"] = 10

engine = create_engine(settings.database_url, **_engine_kwargs)

# ── Session factory ───────────────────────────────────────────────────────────
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ── ORM base ──────────────────────────────────────────────────────────────────
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database_health() -> bool:
    """Return True if PostgreSQL responds to a simple query."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("Database health check failed: %s", exc)
        return False


def read_sql_safe(query: str, params: dict[str, Any] | None = None) -> list[dict]:
    """Execute a SELECT and return rows as list-of-dicts, with one retry."""
    import pandas as pd

    stmt = text(query)
    for attempt in range(2):
        try:
            with engine.begin() as conn:
                df = pd.read_sql(stmt, conn, params=params or {})
            return df.to_dict(orient="records")
        except OperationalError:
            if attempt == 0:
                engine.dispose()
                continue
            raise
        except SQLAlchemyError as exc:
            raise RuntimeError(f"SQL query failed: {exc}") from exc
    return []


def scalar_sql_safe(query: str, params: dict[str, Any] | None = None) -> Any:
    """Execute a scalar query with one retry."""
    stmt = text(query)
    for attempt in range(2):
        try:
            with engine.begin() as conn:
                return conn.execute(stmt, params or {}).scalar_one()
        except OperationalError:
            if attempt == 0:
                engine.dispose()
                continue
            raise
        except SQLAlchemyError as exc:
            raise RuntimeError(f"Scalar query failed: {exc}") from exc
    return None
