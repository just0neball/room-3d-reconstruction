import json
import mimetypes
import os
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import config, storage
from ..db import get_db
from ..models import Project, Task, User
from ..schemas import ProjectCreate, ProjectOut, ReconstructIn, TaskOut, task_out
from ..security import (
    bearer,
    create_file_token,
    decode_token,
    get_current_user,
    get_owned_project,
    user_from_credentials,
)

router = APIRouter(prefix="/projects", tags=["projects"])

ACTIVE = ("pending", "running")
MIN_RECOMMENDED_S, MAX_RECOMMENDED_S = 60, 360  # brief: 2-5 min; warn only outside a loose band


def _project_out(project: Project) -> ProjectOut:
    latest = project.tasks[0] if project.tasks else None
    return ProjectOut(
        id=project.id,
        name=project.name,
        created_at=project.created_at,
        has_video=project.video_path is not None,
        video_name=project.video_name,
        video_size=project.video_size,
        video_duration=project.video_duration,
        scene_ready=project.scene_path is not None,
        latest_task=task_out(latest) if latest else None,
    )


@router.post("", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
def create_project(body: ProjectCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    project = Project(user_id=user.id, name=body.name.strip())
    db.add(project)
    db.commit()
    storage.project_dir(project.id)
    return _project_out(project)


@router.get("", response_model=list[ProjectOut])
def list_projects(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    projects = db.scalars(select(Project).where(Project.user_id == user.id).order_by(Project.id.desc())).all()
    return [_project_out(p) for p in projects]


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return _project_out(get_owned_project(db, project_id, user))


def _active_task(db: Session, project_id: int) -> Optional[Task]:
    return db.scalar(select(Task).where(Task.project_id == project_id, Task.status.in_(ACTIVE)))


@router.post("/{project_id}/video")
def upload_video(
    project_id: int,
    request: Request,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    project = get_owned_project(db, project_id, user)
    if _active_task(db, project.id):
        raise HTTPException(status.HTTP_409_CONFLICT, "A reconstruction is in progress for this project")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in storage.VIDEO_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported video type '{ext}'. Allowed: {', '.join(sorted(storage.VIDEO_EXTENSIONS))}",
        )
    # The multipart parser spools the body to disk before we run, so reject early by header too.
    declared = request.headers.get("content-length")
    if declared and int(declared) > config.MAX_VIDEO_BYTES + 1024 * 1024:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Video is too large")

    video_dir = storage.project_dir(project.id) / "video"
    video_dir.mkdir(exist_ok=True)
    tmp = video_dir / "input.part"
    size = 0
    with open(tmp, "wb") as out:
        while chunk := file.file.read(1 << 20):
            size += len(chunk)
            if size > config.MAX_VIDEO_BYTES:
                out.close()
                tmp.unlink(missing_ok=True)
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Video is too large")
            out.write(chunk)
    if size == 0:
        tmp.unlink(missing_ok=True)
        raise HTTPException(422, "Empty file")

    for old in video_dir.glob("input.*"):
        if old != tmp:
            old.unlink(missing_ok=True)
    final = video_dir / f"input{ext}"
    os.replace(tmp, final)

    duration = storage.probe_duration(final)
    project.video_path = f"video/input{ext}"
    project.video_name = Path(file.filename).name[:255]
    project.video_size = size
    project.video_duration = duration
    db.commit()

    warnings = []
    if duration is not None and not (MIN_RECOMMENDED_S <= duration <= MAX_RECOMMENDED_S):
        warnings.append(
            f"Video is {duration:.0f}s long; the pipeline is tuned for 2-5 minutes of slow, well-lit room footage."
        )
    return {"project": _project_out(project), "warnings": warnings}


@router.post("/{project_id}/reconstruct", response_model=TaskOut, status_code=status.HTTP_202_ACCEPTED)
def reconstruct(
    project_id: int,
    body: ReconstructIn | None = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    project = get_owned_project(db, project_id, user)
    body = body or ReconstructIn()
    if project.video_path is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Upload a video before starting reconstruction")
    if _active_task(db, project.id):
        raise HTTPException(status.HTTP_409_CONFLICT, "A reconstruction is already queued or running")
    task = Task(
        project_id=project.id,
        type="reconstruction",
        steps=body.steps,
        run_name=body.run_name or f"stable_{body.steps}",
    )
    db.add(task)
    db.commit()
    return task_out(task)


@router.get("/{project_id}/tasks", response_model=list[TaskOut])
def project_tasks(project_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    project = get_owned_project(db, project_id, user)
    return [task_out(t) for t in project.tasks]


@router.get("/{project_id}/scene")
def get_scene(project_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    project = get_owned_project(db, project_id, user)
    if project.scene_path is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Scene is not ready yet")
    token = create_file_token(project.id)

    def url(rel: Optional[str]) -> Optional[str]:
        return f"/projects/{project.id}/files/{quote(rel)}?t={token}" if rel else None

    return {
        "project_id": project.id,
        "ready": True,
        "format": "splat",
        "splat_url": url(project.scene_path),  # relative to the API base URL
        "ply_url": url(project.scene_ply_path),
        "url_expires_minutes": config.FILE_TOKEN_MINUTES,
        "metadata": json.loads(project.scene_meta or "{}"),
        "run_name": project.scene_run_name,
        # Week 2 contract: exact locations of frames, poses, intrinsics, splat
        "handoff": storage.handoff_index(project.id, project.scene_run_name or ""),
    }


@router.get("/{project_id}/files/{file_path:path}")
def get_file(
    project_id: int,
    file_path: str,
    t: Optional[str] = Query(None, description="signed file token from /scene"),
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
):
    """Serves stored artifacts (supports HTTP Range). Auth: signed ?t= token or the user's bearer token."""
    if t is not None:
        if decode_token(t, "files").get("pid") != project_id:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Token is for a different project")
    else:
        user = user_from_credentials(creds, db)
        if user is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
        get_owned_project(db, project_id, user)
    target = storage.resolve_inside(project_id, file_path)
    if target is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found")
    media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return FileResponse(target, media_type=media_type, headers={"Cache-Control": "private, max-age=3600"})
