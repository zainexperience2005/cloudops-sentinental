"""
CloudOps Sentinel - SQLAlchemy Database Module (Neon PostgreSQL)
================================================================

Manages audit logging, query trace persistence, and operational metrics
using SQLAlchemy 2.0 with Neon Serverless PostgreSQL.
"""

from datetime import datetime, timezone
from functools import lru_cache
import json
from typing import List, Optional

from sqlalchemy import (
    Integer,
    String,
    Text,
    create_engine,
    desc,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

from src.config import get_settings


class Base(DeclarativeBase):
    """Declarative base class for SQLAlchemy models."""
    pass


class RagAudit(Base):
    """
    SQLAlchemy ORM Model representing an incident resolution query and Self-RAG execution trace.
    """
    __tablename__ = "rag_audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    route: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    used_web: Mapped[int] = mapped_column(Integer, default=0)
    support_status: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    usefulness: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    trace_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sources_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def to_dict(self) -> dict:
        """Serializes the ORM record to a dictionary matching the API contract."""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "question": self.question,
            "answer": self.answer,
            "route": self.route or "",
            "used_web": self.used_web,
            "support_status": self.support_status or "",
            "usefulness": self.usefulness or "",
            "trace_json": self.trace_json or "[]",
            "sources_json": self.sources_json or "[]",
        }


def _get_normalized_db_url() -> str:
    """
    Normalizes the database URL to use the appropriate SQLAlchemy driver.
    Ensures postgresql:// URLs use psycopg (v3) via postgresql+psycopg://.
    """
    s = get_settings()
    url = getattr(s, "database_url", None)
    if not url:
        return f"sqlite:///{s.database_file}"

    # If raw postgresql:// is provided, normalize to postgresql+psycopg://
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


@lru_cache
def get_engine():
    """
    Creates and caches the SQLAlchemy Engine instance.
    Configured with connection pooling optimized for Neon serverless PostgreSQL.
    """
    db_url = _get_normalized_db_url()
    if db_url.startswith("sqlite"):
        return create_engine(db_url, connect_args={"check_same_thread": False})

    return create_engine(
        db_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=300,
    )


@lru_cache
def get_session_factory():
    """Returns a cached sessionmaker bound to the active engine."""
    engine = get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Initializes the database schema by creating all registered tables."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def clean_db() -> None:
    """Cleans all audit records from the database table."""
    init_db()
    session_factory = get_session_factory()
    with session_factory() as session:
        session.query(RagAudit).delete()
        session.commit()


def save_audit(question: str, result: dict) -> None:
    """Saves a Self-RAG query execution and evaluation trace to the database."""
    session_factory = get_session_factory()
    with session_factory() as session:
        audit = RagAudit(
            created_at=datetime.now(timezone.utc).isoformat(),
            question=question,
            answer=result.get("answer", ""),
            route=result.get("route", ""),
            used_web=int(bool(result.get("used_web_search"))),
            support_status=result.get("support_status", ""),
            usefulness=result.get("usefulness", ""),
            trace_json=json.dumps(result.get("trace", []), ensure_ascii=False),
            sources_json=json.dumps(result.get("sources", []), ensure_ascii=False),
        )
        session.add(audit)
        session.commit()


def latest_audits(limit: int = 25) -> List[dict]:
    """Retrieves the most recent audit records from the database."""
    session_factory = get_session_factory()
    with session_factory() as session:
        records = (
            session.query(RagAudit)
            .order_by(desc(RagAudit.id))
            .limit(limit)
            .all()
        )
        return [r.to_dict() for r in records]