"""Task status for the frontend plus the pull-based worker protocol.

Worker flow (all worker calls carry  Authorization: Bearer <ROOM3D_WORKER_TOKEN>):
    POST /worker/next-task            claim the oldest pending task (atomic)
    GET  /worker/tasks/{id}/video     download the source video
    POST /tasks/{id}/heartbeat        every ~30 s while working (separate from progress!)
    POST /tasks/{id}/progress         stage events from pipeline/common.py:send_progress
    PUT  /tasks/{id}/artifacts        upload Person_A_backend_delivery.zip in parts
    POST /tasks/{id}/complete         finalize once artifacts are stored
"""
import json
import os
import shutil
import uuid
import zipfile
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .. import config, storage
from ..db import get_db
from ..models import Project, Task, User, utcnow
from ..schemas import ProgressIn, TaskOut, task_out
from ..security import get_current_user, get_owned_project, require_worker

router = APIRouter(tags=["tasks"])
PART_NAME = "part-{:05d}"


def _get_task(db: Session, task_id: int) -> Task:
    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found")
    return task


# ---------------- frontend ----------------
@router.get("/tasks/{task_id}", response_model=TaskOut)
def get_task(task_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    task = _get_task(db, task_id)
    get_owned_project(db, task.project_id, user)
    return task_out(task)


# ---------------- worker ----------------
def _requeue_stale(db: Session) -> None:
    """Running tasks whose worker went silent (Kaggle session died) go back to the queue."""
    cutoff = utcnow() - timedelta(seconds=config.STALE_SECONDS)
    stale = db.scalars(select(Task).where(Task.status == "running", Task.last_heartbeat < cutoff)).all()
    for task in stale:
        if task.attempts >= config.MAX_ATTEMPTS:
            task.status, task.finished_at = "failed", utcnow()
            task.error = f"Worker stopped responding (gave up after {task.attempts} attempts)"
        else:
            task.status, task.stage, task.progress, task.claim_id = "pending", "queued", 0, None
    if stale:
        db.commit()


@router.post("/worker/next-task", dependencies=[Depends(require_worker)])
def next_task(db: Session = Depends(get_db)):
    _requeue_stale(db)
    claim, now = uuid.uuid4().hex, utcnow()
    oldest_pending = select(Task.id).where(Task.status == "pending").order_by(Task.id).limit(1).scalar_subquery()
    result = db.execute(
        update(Task)
        .where(Task.id == oldest_pending, Task.status == "pending")  # compare-and-set => atomic claim
        .values(
            status="running", stage="starting", progress=0, error=None, claim_id=claim,
            attempts=Task.attempts + 1, last_heartbeat=now, started_at=now,
        )
    )
    db.commit()
    if result.rowcount != 1:
        return {"task": None}
    # The bulk UPDATE bypasses the identity map (and _requeue_stale may have loaded this row),
    # so drop cached instances or the SELECT below returns pre-update values.
    db.expire_all()
    task = db.scalar(select(Task).where(Task.claim_id == claim))
    return {
        "task": {
            "id": task.id,
            "project_id": task.project_id,
            "steps": task.steps,
            "run_name": task.run_name,
            "attempt": task.attempts,
            "video_url": f"/worker/tasks/{task.id}/video",
        }
    }


@router.get("/worker/tasks/{task_id}/video", dependencies=[Depends(require_worker)])
def worker_video(task_id: int, db: Session = Depends(get_db)):
    task = _get_task(db, task_id)
    project = db.get(Project, task.project_id)
    target = storage.resolve_inside(project.id, project.video_path or "")
    if target is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Video missing")
    return FileResponse(target, media_type="application/octet-stream", filename=target.name)


@router.post("/tasks/{task_id}/heartbeat", dependencies=[Depends(require_worker)])
def heartbeat(task_id: int, db: Session = Depends(get_db)):
    """Long stages (COLMAP matching ran ~25 min with no progress event) need liveness
    that does not depend on progress. The status in the reply lets the worker stop
    if the task was failed/cancelled server-side."""
    task = _get_task(db, task_id)
    if task.status == "running":
        task.last_heartbeat = utcnow()
        db.commit()
    return {"status": task.status}


@router.post("/tasks/{task_id}/progress", dependencies=[Depends(require_worker)])
def post_progress(task_id: int, event: ProgressIn, db: Session = Depends(get_db)):
    task = _get_task(db, task_id)
    if task.status in ("completed", "failed"):
        return {"ignored": True, "status": task.status}  # late events must not resurrect a task

    task.last_heartbeat = utcnow()
    if event.status == "failed":
        # keep the last real stage so the UI can say where it failed
        task.status, task.error, task.finished_at = "failed", event.error or "Pipeline failed", utcnow()
    else:
        task.status = "running"
        task.stage = event.stage
        # Monotonic. Also, the pipeline's last script reports "completed" at 100 before the
        # artifacts are uploaded; hold at 99 until POST /complete confirms they are stored.
        task.progress = max(task.progress, min(event.progress, 99))
    db.commit()
    return {"ignored": False, "status": task.status, "progress": task.progress}


@router.put("/tasks/{task_id}/artifacts", dependencies=[Depends(require_worker)])
async def upload_artifacts(
    task_id: int,
    request: Request,
    part: int = Query(0, ge=0),
    parts: int = Query(1, ge=1, le=2000),
    db: Session = Depends(get_db),
):
    """Raw-body upload of the delivery zip, split into parts so each request stays below
    the 100 MB request-body limit Cloudflare applies on its free plan. Idempotent per part;
    the zip is assembled and extracted when the last missing part arrives."""
    task = _get_task(db, task_id)
    if task.status != "running":
        raise HTTPException(status.HTTP_409_CONFLICT, f"Task is {task.status}")
    if part >= parts:
        raise HTTPException(422, "part must be < parts")

    tmp = storage.upload_tmp_dir(task.project_id, task.id)
    tmp.mkdir(parents=True, exist_ok=True)
    staging = tmp / (PART_NAME.format(part) + ".tmp")
    written = 0
    with open(staging, "wb") as out:
        async for chunk in request.stream():
            written += len(chunk)
            if written > config.MAX_PART_BYTES:
                out.close()
                staging.unlink(missing_ok=True)
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Part too large")
            out.write(chunk)
    if written == 0:
        staging.unlink(missing_ok=True)
        raise HTTPException(422, "Empty part")
    os.replace(staging, tmp / PART_NAME.format(part))

    present = [i for i in range(parts) if (tmp / PART_NAME.format(i)).is_file()]
    if len(present) < parts:
        return {"received": len(present), "parts": parts, "extracted": False}

    try:
        written_total = await run_in_threadpool(storage.assemble_and_extract, task.project_id, task.id, parts)
    except (zipfile.BadZipFile, ValueError) as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        raise HTTPException(422, f"Bad delivery archive: {exc}")
    task.last_heartbeat = utcnow()
    db.commit()
    return {"received": parts, "parts": parts, "extracted": True, "bytes_extracted": written_total}


@router.post("/tasks/{task_id}/complete", response_model=TaskOut, dependencies=[Depends(require_worker)])
def complete(task_id: int, db: Session = Depends(get_db)):
    task = _get_task(db, task_id)
    if task.status == "completed":
        return task_out(task)
    if task.status != "running":
        raise HTTPException(status.HTTP_409_CONFLICT, f"Task is {task.status}")

    project = db.get(Project, task.project_id)
    root = storage.project_dir(project.id)
    splat_rel = f"artifacts/exports/{task.run_name}/scene.splat"
    ply_rel = f"artifacts/exports/{task.run_name}/scene.ply"
    if not (root / splat_rel).is_file():
        raise HTTPException(status.HTTP_409_CONFLICT, f"{splat_rel} not found; upload artifacts first")

    run = task.run_name
    metadata = {
        "scene": storage.read_json(root / f"artifacts/exports/{run}/scene_metadata.json"),
        "training": storage.read_json(root / f"artifacts/3dgs_training/{run}/training_metadata.json"),
        "validation": storage.read_json(root / "artifacts/reports/validation_status.json"),
    }
    project.scene_path = splat_rel
    project.scene_ply_path = ply_rel if (root / ply_rel).is_file() else None
    project.scene_run_name = run
    project.scene_meta = json.dumps(metadata)

    task.status, task.stage, task.progress, task.error = "completed", "done", 100, None
    task.finished_at = utcnow()
    db.commit()
    return task_out(task)
