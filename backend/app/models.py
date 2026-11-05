from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    """Stores naive UTC, always returns timezone-aware UTC (so JSON gets a +00:00 suffix)."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    def process_result_value(self, value, dialect):
        return None if value is None else value.replace(tzinfo=timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    projects: Mapped[list["Project"]] = relationship(back_populates="user")


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    # Paths below are relative to storage/projects/{id}/
    video_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    video_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    video_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    video_duration: Mapped[Optional[float]] = mapped_column(nullable=True)
    scene_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    scene_ply_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    scene_run_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    scene_meta: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON text

    user: Mapped[User] = relationship(back_populates="projects")
    tasks: Mapped[list["Task"]] = relationship(back_populates="project", order_by="Task.id.desc()")


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    type: Mapped[str] = mapped_column(String(32), default="reconstruction")
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)  # pending|running|completed|failed
    stage: Mapped[str] = mapped_column(String(64), default="queued")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    steps: Mapped[int] = mapped_column(Integer, default=7000)
    run_name: Mapped[str] = mapped_column(String(64), default="stable_7000")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    claim_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    last_heartbeat: Mapped[Optional[datetime]] = mapped_column(UTCDateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime, nullable=True)

    project: Mapped[Project] = relationship(back_populates="tasks")
