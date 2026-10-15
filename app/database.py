"""Relational storage: users (auth/RBAC) and the asset catalog.

Sessions are synchronous SQLAlchemy sessions backed by a connection pool; FastAPI runs
sync dependencies in its threadpool so they do not block the event loop.

CLI:
    python -m app.database create-user <username> --role viewer   (password read from stdin)
"""

import logging
import os
import time
import uuid
from datetime import UTC, datetime

import bcrypt
from sqlalchemy import JSON, BigInteger, Column, DateTime, Float, ForeignKey, String, Text, create_engine
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

from app import config

logger = logging.getLogger("dam.database")

ROLES = ("admin", "viewer")


def make_engine(url: str):
    if url.startswith("sqlite"):
        # Used by the unit tests: one shared in-memory database across threads.
        return create_engine(url, connect_args={"check_same_thread": False}, poolclass=StaticPool)
    return create_engine(url, pool_size=10, max_overflow=10, pool_pre_ping=True, pool_recycle=1800)


engine = make_engine(config.database_url())
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)
Base = declarative_base()


def utcnow() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    username = Column(String(64), primary_key=True)
    hashed_password = Column(String, nullable=False)
    role = Column(String(16), nullable=False)

    def verify_password(self, plain_password: str) -> bool:
        return bcrypt.checkpw(plain_password.encode("utf-8"), self.hashed_password.encode("utf-8"))


class Asset(Base):
    """Catalog entry for one uploaded file. Vectors in Qdrant reference `id` as `asset_id`."""

    __tablename__ = "assets"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)  # sanitized original name, for display
    object_key = Column(String(512), nullable=False, unique=True)
    media_type = Column(String(16), nullable=False, index=True)
    content_type = Column(String(64), nullable=False)
    size_bytes = Column(BigInteger, nullable=True)
    # pending_upload -> queued -> processing -> ready | failed
    status = Column(String(16), nullable=False, default="queued", index=True)
    task_id = Column(String(64), nullable=True)
    error = Column(Text, nullable=True)
    caption = Column(Text, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    stats = Column(JSON, nullable=True)
    summary = Column(JSON, nullable=True)  # cached summary result
    uploaded_by = Column(String(64), ForeignKey("users.username", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "media_type": self.media_type,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "status": self.status,
            "task_id": self.task_id,
            "error": self.error,
            "caption": self.caption,
            "duration_seconds": self.duration_seconds,
            "stats": self.stats or {},
            "has_summary": bool(self.summary),
            "uploaded_by": self.uploaded_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def update_asset(asset_id: str, **fields) -> None:
    """Used by the worker to record status transitions and ingestion stats."""
    with SessionLocal() as db:
        asset = db.get(Asset, asset_id)
        if asset is None:
            raise LookupError(f"Asset {asset_id} not found")
        for key, value in fields.items():
            setattr(asset, key, value)
        db.commit()


def _wait_for_database(max_attempts: int = 20, delay: float = 3.0) -> None:
    for attempt in range(1, max_attempts + 1):
        try:
            with engine.connect():
                return
        except OperationalError:
            if attempt == max_attempts:
                raise
            logger.warning("Database not ready (attempt %d/%d), retrying in %.0fs", attempt, max_attempts, delay)
            time.sleep(delay)


def _seed_user(db: Session, username: str, password: str, role: str) -> None:
    if db.get(User, username):
        return
    db.add(User(username=username, hashed_password=hash_password(password), role=role))
    try:
        db.commit()
        logger.info("Seeded %s account '%s'", role, username)
    except IntegrityError:  # another replica seeded it first
        db.rollback()


def init_db() -> None:
    """Creates tables and seeds accounts from ADMIN_* / VIEWER_* environment variables."""
    _wait_for_database()
    Base.metadata.create_all(bind=engine)

    with SessionLocal() as db:
        admin_user = os.getenv("ADMIN_USERNAME", "admin")
        admin_password = os.getenv("ADMIN_PASSWORD", "")
        has_admin = db.query(User).filter(User.role == "admin").first() is not None
        if not has_admin:
            if len(admin_password) < 8:
                raise config.ConfigError("No admin account exists: set ADMIN_PASSWORD (8+ characters) to seed one.")
            _seed_user(db, admin_user, admin_password, "admin")

        viewer_user = os.getenv("VIEWER_USERNAME", "viewer")
        viewer_password = os.getenv("VIEWER_PASSWORD", "")
        if viewer_password:
            _seed_user(db, viewer_user, viewer_password, "viewer")


def _cli() -> None:
    import argparse
    import getpass

    parser = argparse.ArgumentParser(prog="python -m app.database")
    sub = parser.add_subparsers(dest="cmd", required=True)
    create = sub.add_parser("create-user", help="create a user or reset an existing user's password")
    create.add_argument("username")
    create.add_argument("--role", choices=ROLES, default="viewer")
    args = parser.parse_args()

    password = getpass.getpass("Password: ")
    if len(password) < 8:
        raise SystemExit("Password must be at least 8 characters.")
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        user = db.get(User, args.username)
        if user:
            user.hashed_password, user.role = hash_password(password), args.role
        else:
            db.add(User(username=args.username, hashed_password=hash_password(password), role=args.role))
        db.commit()
    print(f"Saved {args.role} '{args.username}'.")


if __name__ == "__main__":
    _cli()
