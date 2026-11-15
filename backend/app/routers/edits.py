"""Edit = one "click an object -> prompt -> edited scene" run. See docs/API.md for the contract and state machine."""
import base64
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import quote

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import geometry, storage
from ..db import get_db
from ..models import Edit, Project, Task, User, utcnow
from ..schemas import TaskOut, task_out
from ..security import create_file_token, get_current_user, get_owned_project
from ..services import dispatcher, edit_ops
from ..services.edit_ops import edit_dir, included_frames, jl, stem
from ..services.gpu_client import GpuClient, GpuError, GpuUnreachable

router = APIRouter(tags=["edits"])
ACTIVE = ("pending", "running")
DEFAULT_SETTINGS = {"seeds": 4, "quant": "nf4", "rebuild_mode": "finetune", "rebuild_steps": 3000}


# ------------------------------------------------------------------ helpers
def _project_for(db, project_id, user) -> Project:
    p = get_owned_project(db, project_id, user)
    if p.scene_path is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Reconstruct the room first: the project has no scene yet")
    return p


def _edit_for(db, edit_id, user) -> Edit:
    e = db.get(Edit, edit_id)
    if e is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Edit not found")
    get_owned_project(db, e.project_id, user)
    return e


def _active_task(db, project_id):
    return db.scalar(select(Task).where(Task.project_id == project_id, Task.status.in_(ACTIVE)))


def _no_active(db, edit):
    if _active_task(db, edit.project_id):
        raise HTTPException(status.HTTP_409_CONFLICT, "Another task is already running for this project")


def _gpu() -> GpuClient:
    g = GpuClient()
    if not g.configured:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "No GPU server connected")
    return g


def _url(pid, rel, token, exists_in: Path | None = None):
    if exists_in is not None and not exists_in.is_file():
        return None
    return f"/projects/{pid}/files/{quote(rel)}?t={token}&v={int(exists_in.stat().st_mtime) if exists_in else 0}"


def edit_out(db: Session, e: Edit) -> dict:
    pid, token = e.project_id, create_file_token(e.project_id)
    d = edit_dir(pid, e.id)
    rel = f"edits/{e.id}"
    cams = edit_ops.cameras(pid)
    tr = jl(e.tracking, None)
    tracking = None
    if tr and tr.get("frames"):
        frames = []
        for f in tr["frames"]:
            s = stem(f["name"])
            frames.append({
                **{k: f.get(k) for k in ("index", "name", "seconds", "included", "source", "score", "area")},
                "thumb_url": f"/projects/{pid}/thumb/{quote(f['name'])}?w=240&t={token}",
                "mask_url": _url(pid, f"{rel}/masks/{s}.png", token, d / "masks" / f"{s}.png"),
                "overlay_url": _url(pid, f"{rel}/overlays/{s}.jpg", token, d / "overlays" / f"{s}.jpg"),
                "edited_url": _url(pid, f"{rel}/edited/{s}.png", token, d / "edited" / f"{s}.png"),
            })
        tracking = {"method": tr.get("method"), "message": tr.get("message", ""), "included_count": sum(f["included"] for f in frames),
                    "total": len(frames), "frames": frames}
    cands = []
    for c in jl(e.candidates, []):
        rp = f"{rel}/candidates/candidate_{c['seed']}.png"
        cands.append({"seed": c["seed"], "url": _url(pid, rp, token, d / "candidates" / f"candidate_{c['seed']}.png"),
                      "thumb_url": f"/projects/{pid}/thumb-edit/{e.id}/{c['seed']}?t={token}"})
    res = jl(e.result, None)
    result = None
    if res and (d / "exports" / "scene.splat").is_file():
        proj = e.project
        result = {**res, "splat_url": _url(pid, f"{rel}/exports/scene.splat", token, d / "exports" / "scene.splat"),
                  "ply_url": _url(pid, f"{rel}/exports/scene.ply", token, d / "exports" / "scene.ply"),
                  "original_splat_url": f"/projects/{pid}/files/{quote(proj.scene_path)}?t={token}"}
    included = [f for f in (tracking["frames"] if tracking else []) if f["included"]]
    done = sum(1 for f in included if f["edited_url"])
    active = db.scalar(select(Task).where(Task.edit_id == e.id, Task.status.in_(ACTIVE)).order_by(Task.id.desc()))
    return {
        "id": e.id, "project_id": pid, "name": e.name, "state": e.state, "error": e.error, "failed_step": e.failed_step, "prompt": e.prompt,
        "created_at": e.created_at, "updated_at": e.updated_at,
        "image_size": {"width": cams[0]["width"], "height": cams[0]["height"]},
        "ref_frame": e.ref_frame, "points": jl(e.points, []), "box": jl(e.box, None),
        "ref_mask_url": _url(pid, f"{rel}/masks/{stem(e.ref_frame)}.png", token, d / "masks" / f"{stem(e.ref_frame)}.png") if e.ref_frame else None,
        "ref_overlay_url": _url(pid, f"{rel}/overlays/{stem(e.ref_frame)}.jpg", token, d / "overlays" / f"{stem(e.ref_frame)}.jpg") if e.ref_frame else None,
        "tracking": tracking, "candidates": cands, "chosen_seed": e.chosen_seed,
        "edit_frame": edit_ops.edit_frame(e) if e.ref_frame else None, "reference_url": _url(pid, f"{rel}/reference.png", token, d / "reference.png"),
        "edit_progress": {"done": done, "total": len(included)},
        "result": result, "settings": {**DEFAULT_SETTINGS, **jl(e.settings, {})}, "metrics": jl(e.metrics, {}),
        "active_task": task_out(active).model_dump(mode="json") if active else None,
    }


def _queue(db, edit, type_, params=None) -> Task:
    task = Task(project_id=edit.project_id, edit_id=edit.id, type=type_, steps=0, run_name=edit.project.scene_run_name or "", params=json.dumps(params or {}))
    db.add(task)
    db.commit()
    dispatcher.wake()
    return task


# ------------------------------------------------------------------ CRUD
class EditCreate(BaseModel):
    name: Optional[str] = Field(None, max_length=200)
    prompt: str = ""


class EditPatch(BaseModel):
    name: Optional[str] = Field(None, max_length=200)
    prompt: Optional[str] = None
    settings: Optional[dict] = None


@router.post("/projects/{project_id}/edits", status_code=status.HTTP_201_CREATED)
def create_edit(project_id: int, body: EditCreate | None = None, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    project = _project_for(db, project_id, user)
    body = body or EditCreate()
    n = len(project.edits) + 1
    e = Edit(project_id=project.id, name=(body.name or f"Edit {n}").strip(), prompt=body.prompt, settings=json.dumps(DEFAULT_SETTINGS))
    db.add(e)
    db.commit()
    return edit_out(db, e)


@router.get("/projects/{project_id}/edits")
def list_edits(project_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    project = get_owned_project(db, project_id, user)
    if project.scene_path is None:
        return []
    return [edit_out(db, e) for e in project.edits]


@router.get("/edits/{edit_id}")
def get_edit(edit_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return edit_out(db, _edit_for(db, edit_id, user))


@router.patch("/edits/{edit_id}")
def patch_edit(edit_id: int, body: EditPatch, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    e = _edit_for(db, edit_id, user)
    if body.name is not None:
        e.name = body.name.strip() or e.name
    if body.prompt is not None:
        e.prompt = body.prompt
    if body.settings:
        s = {**DEFAULT_SETTINGS, **jl(e.settings, {})}
        for k, v in body.settings.items():
            if k in DEFAULT_SETTINGS:
                s[k] = v
        if s["quant"] not in ("nf4", "int8", "bf16") or s["rebuild_mode"] not in ("finetune", "scratch") or not (1 <= int(s["seeds"]) <= 6):
            raise HTTPException(422, "invalid settings")
        e.settings = json.dumps(s)
    e.updated_at = utcnow()
    db.commit()
    return edit_out(db, e)


@router.delete("/edits/{edit_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_edit(edit_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    e = _edit_for(db, edit_id, user)
    if db.scalar(select(Task).where(Task.edit_id == e.id, Task.status.in_(ACTIVE))):
        raise HTTPException(status.HTTP_409_CONFLICT, "Cancel the running task first")
    shutil.rmtree(edit_dir(e.project_id, e.id), ignore_errors=True)
    db.delete(e)
    db.commit()


# ------------------------------------------------------------------ pick the object
class Pick3D(BaseModel):
    point: list[float] = Field(..., min_length=3, max_length=3)


@router.post("/edits/{edit_id}/pick3d")
def pick3d(edit_id: int, body: Pick3D, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    e = _edit_for(db, edit_id, user)
    ranked = geometry.rank_views(body.point, edit_ops.cameras(e.project_id))
    if not ranked:
        raise HTTPException(status.HTTP_409_CONFLICT, "No training frame sees that point; click on the room surface")
    return {**ranked[0], "candidates": ranked[:6]}


class ClickIn(BaseModel):
    x: float
    y: float
    label: int = Field(1, ge=0, le=1)


class SegmentBody(BaseModel):
    frame: str
    points: list[ClickIn] = []
    box: list[float] | None = Field(None, min_length=4, max_length=4)
    set_reference: bool = True


@router.post("/edits/{edit_id}/segment")
def segment(edit_id: int, body: SegmentBody, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    e = _edit_for(db, edit_id, user)
    if not body.points and not body.box:
        raise HTTPException(422, "Click on the object (or draw a box)")
    cams = edit_ops.cameras(e.project_id)
    if body.frame not in {c["name"] for c in cams}:
        raise HTTPException(404, "Unknown frame")
    if _active_task(db, e.project_id):
        raise HTTPException(status.HTTP_409_CONFLICT, "A task is running for this project")
    gpu = _gpu()
    try:
        edit_ops.ensure_dataset(db, gpu, e.project)
        r = gpu.segment(edit_ops.dataset_key(e.project_id), body.frame, [p.model_dump() for p in body.points], body.box)
    except GpuUnreachable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    except GpuError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc))
    mask = cv2.imdecode(np.frombuffer(base64.b64decode(r["mask_png_b64"]), np.uint8), cv2.IMREAD_GRAYSCALE) > 127
    if not mask.any():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "The model found no object there; click closer to its centre")
    edit_ops.save_mask(e.project_id, e.id, body.frame, mask)
    if body.set_reference:
        changed = e.ref_frame != body.frame or e.tracking is not None
        e.ref_frame, e.points, e.box = body.frame, json.dumps([p.model_dump() for p in body.points]), json.dumps(body.box) if body.box else None
        if e.tracking or e.candidates != "[]":  # a new reference invalidates tracking and everything after it
            e.tracking = None
            edit_ops.reset_downstream(e)
        e.state = "masked"
    else:
        tr = edit_ops.get_tracking(e)
        for f in tr["frames"]:
            if f["name"] == body.frame:
                f.update(included=True, source="manual", area=float(mask.mean()))
        e.tracking = json.dumps(tr)
        up = jl(e.user_points, {})
        up[body.frame] = [p.model_dump() for p in body.points]
        e.user_points = json.dumps(up)
    e.updated_at = utcnow()
    db.commit()
    token = create_file_token(e.project_id)
    s, rel = stem(body.frame), f"edits/{e.id}"
    d = edit_dir(e.project_id, e.id)
    return {"mask_url": _url(e.project_id, f"{rel}/masks/{s}.png", token, d / "masks" / f"{s}.png"),
            "overlay_url": _url(e.project_id, f"{rel}/overlays/{s}.jpg", token, d / "overlays" / f"{s}.jpg"),
            "area": float(mask.mean()), "score": r.get("score"), "frame": body.frame}


# ------------------------------------------------------------------ tracking
class TrackBody(BaseModel):
    method: Literal["auto", "video", "lift3d", "neighbor", "manual"] = "auto"
    frames: list[str] = []


@router.post("/edits/{edit_id}/track", response_model=TaskOut, status_code=status.HTTP_202_ACCEPTED)
def track(edit_id: int, body: TrackBody, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    e = _edit_for(db, edit_id, user)
    _no_active(db, e)
    if not e.ref_frame or not (edit_dir(e.project_id, e.id) / "masks" / f"{stem(e.ref_frame)}.png").is_file():
        raise HTTPException(status.HTTP_409_CONFLICT, "Pick the object first (click it on a frame)")
    names = {c["name"] for c in edit_ops.cameras(e.project_id)}
    if body.method == "manual":
        if not body.frames or any(n not in names for n in body.frames):
            raise HTTPException(422, "Tick at least one valid frame for manual segmentation")
    edit_ops.reset_downstream(e)  # candidates / edited frames no longer match the frame set
    return task_out(_queue(db, e, "track", {"method": body.method, "frames": body.frames}))


class FramesPatch(BaseModel):
    include: list[str] = []
    exclude: list[str] = []


@router.patch("/edits/{edit_id}/frames")
def patch_frames(edit_id: int, body: FramesPatch, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    e = _edit_for(db, edit_id, user)
    tr = edit_ops.get_tracking(e)
    masks = edit_dir(e.project_id, e.id) / "masks"
    for f in tr["frames"]:
        if f["name"] in body.exclude and f["name"] != e.ref_frame:
            f["included"] = False
        elif f["name"] in body.include and (masks / f"{stem(f['name'])}.png").is_file():
            f["included"] = True
            if f["source"] == "none":
                f["source"] = "manual"
    e.tracking = json.dumps(tr)
    edit_ops.reset_downstream(e)
    if e.state in ("reference_ready", "editing_frames", "edited", "done", "rebuilding"):
        e.state = "tracked"
    e.updated_at = utcnow()
    db.commit()
    return edit_out(db, e)


# ------------------------------------------------------------------ diffusion
class RefBody(BaseModel):
    prompt: Optional[str] = None
    seeds: Optional[int] = Field(None, ge=1, le=6)
    frame: Optional[str] = None


@router.post("/edits/{edit_id}/edit/reference", response_model=TaskOut, status_code=status.HTTP_202_ACCEPTED)
def edit_reference(edit_id: int, body: RefBody | None = None, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    e = _edit_for(db, edit_id, user)
    _no_active(db, e)
    body = body or RefBody()
    if body.prompt is not None:
        e.prompt = body.prompt
    if not e.prompt.strip():
        raise HTTPException(422, "Type what the object should become (the edit prompt)")
    if not e.ref_frame:
        raise HTTPException(status.HTTP_409_CONFLICT, "Pick the object first")
    if not e.tracking:
        raise HTTPException(status.HTTP_409_CONFLICT, "Track the object first")
    if body.frame:
        ok = any(f["name"] == body.frame for f in included_frames(e)) and (edit_dir(e.project_id, e.id) / "masks" / f"{stem(body.frame)}.png").is_file()
        if not ok:
            raise HTTPException(422, "That frame is not part of the edit set (needs an included frame with a mask)")
        st = jl(e.settings, {})
        st["edit_frame"] = body.frame
        e.settings = json.dumps(st)
    return task_out(_queue(db, e, "edit_reference", {"seeds": body.seeds or jl(e.settings, {}).get("seeds", 4)}))


class AcceptBody(BaseModel):
    seed: int


@router.post("/edits/{edit_id}/edit/accept")
def accept(edit_id: int, body: AcceptBody, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    e = _edit_for(db, edit_id, user)
    src = edit_dir(e.project_id, e.id) / "candidates" / f"candidate_{body.seed}.png"
    if not src.is_file():
        raise HTTPException(404, "No such candidate")
    shutil.copy2(src, edit_dir(e.project_id, e.id) / "reference.png")
    shutil.rmtree(edit_dir(e.project_id, e.id) / "edited", ignore_errors=True)
    shutil.rmtree(edit_dir(e.project_id, e.id) / "exports", ignore_errors=True)
    e.chosen_seed, e.result, e.state = body.seed, None, "reference_ready"
    e.updated_at = utcnow()
    db.commit()
    return edit_out(db, e)


@router.post("/edits/{edit_id}/edit/run", response_model=TaskOut, status_code=status.HTTP_202_ACCEPTED)
def edit_run(edit_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    e = _edit_for(db, edit_id, user)
    _no_active(db, e)
    if e.chosen_seed is None or not (edit_dir(e.project_id, e.id) / "reference.png").is_file():
        raise HTTPException(status.HTTP_409_CONFLICT, "Accept a reference edit first")
    if not edit_ops.included_frames(e):
        raise HTTPException(status.HTTP_409_CONFLICT, "No frames selected")
    shutil.rmtree(edit_dir(e.project_id, e.id) / "edited", ignore_errors=True)
    return task_out(_queue(db, e, "edit_rest"))


class RebuildBody(BaseModel):
    mode: Optional[Literal["finetune", "scratch"]] = None
    steps: Optional[int] = Field(None, ge=200, le=30000)


@router.post("/edits/{edit_id}/rebuild", response_model=TaskOut, status_code=status.HTTP_202_ACCEPTED)
def rebuild(edit_id: int, body: RebuildBody | None = None, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    e = _edit_for(db, edit_id, user)
    _no_active(db, e)
    body = body or RebuildBody()
    if not any((edit_dir(e.project_id, e.id) / "edited").glob("*.png")):
        raise HTTPException(status.HTTP_409_CONFLICT, "Edit the frames first")
    st = {**DEFAULT_SETTINGS, **jl(e.settings, {})}
    st["rebuild_mode"] = body.mode or st["rebuild_mode"]
    st["rebuild_steps"] = body.steps or st["rebuild_steps"]
    e.settings = json.dumps(st)
    return task_out(_queue(db, e, "rebuild", {"mode": st["rebuild_mode"], "steps": st["rebuild_steps"]}))
