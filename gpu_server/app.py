"""GPU server. Run on Kaggle (see notebooks/gpu_server.ipynb) or locally with GPU_MOCK=1:

    uvicorn gpu_server.app:app --host 0.0.0.0 --port 8001

The website backend calls this API (push model): upload inputs -> POST /jobs -> poll GET /jobs/{id} -> GET /jobs/{id}/artifact.
Interactive segmentation (/segment) is synchronous and bypasses the job queue.
"""
import base64
import re
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import datasets, models, settings
from .handlers import HANDLERS
from .jobs import JobManager

app = FastAPI(title="Room3D GPU server", version="2.0")
manager = JobManager(HANDLERS)
STARTED = time.time()
UID = re.compile(r"^[A-Za-z0-9_-]{6,64}$")
KEY = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def auth(x_room3d_token: str | None = Header(None)):
    if settings.TOKEN and x_room3d_token != settings.TOKEN:
        raise HTTPException(401, "Bad or missing X-Room3D-Token")


@app.get("/health", dependencies=[Depends(auth)])
def health():
    devs = []
    if not settings.MOCK:
        try:
            import torch
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                devs.append({"name": torch.cuda.get_device_name(i), "total_gb": round(total / 1024 ** 3, 1), "free_gb": round(free / 1024 ** 3, 1)})
        except Exception:
            pass
    d, a = settings.devices()
    return {"ok": True, "mock": settings.MOCK, "devices": devs, "uptime_s": int(time.time() - STARTED),
            "models": {**models.loaded(), "configured_segmenter": settings.SAM_MODEL_ID, "configured_diffusion": settings.DIFFUSION_MODEL_ID,
                       "quant": settings.DEFAULT_QUANT, "diffusion_device": d, "aux_device": a},
            "jobs": {"running": [j.id for j in manager.jobs.values() if j.status == "running"], "queued": sum(j.status == "queued" for j in manager.jobs.values())}}


# ---------------------------------------------------------------- uploads (chunked: each request stays < Cloudflare's 100 MB body limit)
@app.put("/uploads/{uid}", dependencies=[Depends(auth)])
async def upload(uid: str, request: Request, part: int = Query(0, ge=0), parts: int = Query(1, ge=1, le=4000)):
    if not UID.match(uid) or part >= parts:
        raise HTTPException(422, "bad upload id or part")
    base = settings.DATA_DIR / "uploads"
    tmp = base / f"{uid}.parts"
    tmp.mkdir(parents=True, exist_ok=True)
    staging = tmp / f"part-{part:05d}.tmp"
    size = 0
    with open(staging, "wb") as out:
        async for chunk in request.stream():
            size += len(chunk)
            out.write(chunk)
    if size == 0:
        raise HTTPException(422, "empty part")
    staging.replace(tmp / f"part-{part:05d}")
    have = sum((tmp / f"part-{i:05d}").is_file() for i in range(parts))
    if have < parts:
        return {"complete": False, "received": have, "parts": parts}
    final = base / uid
    with open(final.with_suffix(".assembling"), "wb") as out:
        for i in range(parts):
            out.write((tmp / f"part-{i:05d}").read_bytes())
    final.with_suffix(".assembling").replace(final)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    return {"complete": True, "bytes": final.stat().st_size}


# ---------------------------------------------------------------- datasets
@app.get("/datasets/{key}", dependencies=[Depends(auth)])
def dataset_info(key: str):
    if not KEY.match(key):
        raise HTTPException(422, "bad key")
    return datasets.info(key) or {"key": key, "version": None}


class DatasetIn(BaseModel):
    upload: str
    version: str


@app.post("/datasets/{key}", dependencies=[Depends(auth)])
def dataset_create(key: str, body: DatasetIn):
    if not KEY.match(key) or not UID.match(body.upload):
        raise HTTPException(422, "bad key/upload")
    path = settings.DATA_DIR / "uploads" / body.upload
    if not path.is_file():
        raise HTTPException(409, "upload is not complete")
    return datasets.extract(key, path, body.version)


# ---------------------------------------------------------------- interactive segmentation
class Point(BaseModel):
    x: float
    y: float
    label: int = Field(1, ge=0, le=1)


class SegmentIn(BaseModel):
    dataset: str
    frame: str
    points: list[Point] = []
    box: list[float] | None = None


@app.post("/segment", dependencies=[Depends(auth)])
def segment(body: SegmentIn):
    try:
        ds = datasets.get(body.dataset)
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc))
    if body.frame not in ds.index:
        raise HTTPException(404, "unknown frame")
    if not body.points and not body.box:
        raise HTTPException(422, "give at least one point or a box")
    seg = models.segmenter()
    t0 = time.perf_counter()
    mask, score = seg.segment(ds.image(ds.index[body.frame]), [(p.x, p.y, p.label) for p in body.points], tuple(body.box) if body.box else None)
    ok, png = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    return {"mask_png_b64": base64.b64encode(png.tobytes()).decode(), "score": round(float(score), 4), "area": float(mask.mean()),
            "seconds": round(time.perf_counter() - t0, 3), "segmenter": seg.name}


# ---------------------------------------------------------------- jobs
class JobIn(BaseModel):
    kind: str
    params: dict = {}
    uploads: dict[str, str] = {}


@app.post("/jobs", dependencies=[Depends(auth)], status_code=202)
def create_job(body: JobIn):
    for uid in body.uploads.values():
        if not UID.match(uid) or not (settings.DATA_DIR / "uploads" / uid).is_file():
            raise HTTPException(409, f"upload {uid} is not complete")
    try:
        job = manager.submit(body.kind, body.params, body.uploads)
    except KeyError:
        raise HTTPException(422, f"unknown job kind {body.kind}")
    return job.public()


def _job(job_id):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job (the GPU server may have restarted)")
    return job


@app.get("/jobs/{job_id}", dependencies=[Depends(auth)])
def get_job(job_id: str):
    return _job(job_id).public()


@app.post("/jobs/{job_id}/cancel", dependencies=[Depends(auth)])
def cancel_job(job_id: str):
    _job(job_id)
    return manager.cancel(job_id).public()


@app.get("/jobs/{job_id}/artifact", dependencies=[Depends(auth)])
def job_artifact(job_id: str):
    path = _job(job_id).dir / "artifact.zip"
    if not path.is_file():
        raise HTTPException(409, "no artifact yet")
    return FileResponse(path, media_type="application/zip", filename=f"{job_id}.zip")


@app.get("/jobs/{job_id}/files/{rel:path}", dependencies=[Depends(auth)])
def job_file(job_id: str, rel: str):
    base = (_job(job_id).dir / "out").resolve()
    target = (base / rel).resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(target)


@app.post("/models/unload", dependencies=[Depends(auth)])
def unload():
    models.free_diffusion()
    models.free_segmenter()
    return models.loaded()
