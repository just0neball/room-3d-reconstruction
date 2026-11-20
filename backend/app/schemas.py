import json
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from . import config
from .models import Task
from .stages import group_for, label_for


class SignupIn(BaseModel):
    email: str = Field(max_length=255)
    password: str = Field(min_length=8, max_length=128)


class LoginIn(BaseModel):
    email: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    email: str


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class ReconstructIn(BaseModel):
    steps: int = Field(config.DEFAULT_STEPS, ge=config.MIN_STEPS, le=config.MAX_STEPS)
    # used in file paths and CLI args by the pipeline, so keep it strictly safe
    run_name: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9_-]{1,40}$")
    frame_fps: int = Field(1, ge=1, le=2)  # frames per second used for reconstruction (poses + splat + edit frames)


class ProgressIn(BaseModel):
    """Same shape as the event pipeline/common.py:send_progress() posts."""

    model_config = ConfigDict(extra="ignore")

    stage: str = Field(max_length=64)
    progress: int = Field(ge=0, le=100)
    status: str = Field("running", pattern=r"^(running|completed|failed)$")
    error: Optional[str] = Field(None, max_length=8000)
    timestamp_utc: Optional[str] = None


class TaskOut(BaseModel):
    id: int
    project_id: int
    edit_id: Optional[int] = None
    type: str
    status: str
    stage: str
    stage_group: str
    stage_label: str
    progress: int
    detail: Optional[str] = None
    error: Optional[str]
    steps: int
    run_name: str
    attempts: int
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    metrics: Optional[dict] = None


def task_out(task: Task) -> TaskOut:
    group = group_for(task.stage) if task.type == "reconstruction" else task.type
    return TaskOut(
        id=task.id,
        project_id=task.project_id,
        edit_id=task.edit_id,
        type=task.type,
        status=task.status,
        stage=task.stage,
        stage_group=group,
        stage_label=label_for(task.stage),
        progress=task.progress,
        detail=task.detail,
        error=task.error,
        steps=task.steps,
        run_name=task.run_name,
        attempts=task.attempts,
        created_at=task.created_at,
        started_at=task.started_at,
        finished_at=task.finished_at,
        metrics=json.loads(task.metrics) if task.metrics else None,
    )


class ProjectOut(BaseModel):
    id: int
    name: str
    created_at: datetime
    has_video: bool
    video_name: Optional[str]
    video_size: Optional[int]
    video_duration: Optional[float]
    scene_ready: bool
    latest_task: Optional[TaskOut]
