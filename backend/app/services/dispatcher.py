"""Background thread: takes pending Tasks in order, pushes each to the GPU server, mirrors its progress into the Task row,
and turns the result back into files. If the GPU server is offline the task simply stays pending ("Waiting for GPU server")."""
import json
import threading
import time
import traceback

from .. import config, storage
from ..db import SessionLocal
from ..models import Edit, Task, utcnow
from . import edit_ops
from .gpu_client import GpuClient, GpuError, GpuUnreachable

POLL = 1.5
START_STATE = {"track": "tracking", "edit_reference": "editing_reference", "edit_rest": "editing_frames", "rebuild": "rebuilding"}
REVERT_STATE = {"track": "masked", "edit_reference": "tracked", "edit_rest": "reference_ready", "rebuild": "edited"}

_wake = threading.Event()
_stop = threading.Event()


def wake():
    _wake.set()


_thread: threading.Thread | None = None


def start():
    global _thread
    if _thread is not None and _thread.is_alive():
        return  # exactly one dispatcher, or two would run the same task
    _stop.clear()
    with SessionLocal() as db:  # a restart while a job was running: poll the same remote job again
        for t in db.query(Task).filter(Task.status == "running"):
            t.status = "pending"
        db.commit()
    _thread = threading.Thread(target=_loop, daemon=True, name="gpu-dispatcher")
    _thread.start()


def stop():
    _stop.set()
    _wake.set()
    if _thread is not None:
        _thread.join(timeout=10)


def _loop():
    while not _stop.is_set():
        try:
            if not _run_next():
                _wake.wait(2)
                _wake.clear()
        except Exception:
            traceback.print_exc()
            time.sleep(3)


def _set_edit(db, task, state=None, **fields):
    if task.edit_id is None:
        return
    edit = db.get(Edit, task.edit_id)
    if edit is None:
        return
    if state:
        edit.state = state
    for k, v in fields.items():
        setattr(edit, k, v)
    edit.updated_at = utcnow()


def _run_next() -> bool:
    with SessionLocal() as db:
        task = db.query(Task).filter(Task.status == "pending").order_by(Task.id).first()
        if task is None:
            return False
        gpu = GpuClient()
        try:
            if not gpu.configured:
                raise GpuUnreachable("GPU server URL is not set")
            gpu.health()
        except GpuError as exc:
            msg = "Waiting for GPU server" if isinstance(exc, GpuUnreachable) else str(exc)
            if task.detail != msg:
                task.detail = msg
                db.commit()
            time.sleep(3)
            return True
        _execute(db, gpu, task)
    return True


def _fail(db, task, message):
    task.status, task.error, task.finished_at, task.detail = "failed", message[-4000:], utcnow(), None
    _set_edit(db, task, "failed", error=message[-2000:], failed_step=task.type)
    db.commit()


def _execute(db, gpu: GpuClient, task: Task):
    task.status, task.started_at, task.attempts, task.detail = "running", task.started_at or utcnow(), task.attempts + 1, None
    task.last_heartbeat = utcnow()
    if task.type in START_STATE:
        _set_edit(db, task, START_STATE[task.type], error=None, failed_step=None)
    db.commit()
    try:
        job = gpu.job(task.remote_job_id) if task.remote_job_id else None
        if job is None:
            task.remote_job_id, task.stage, task.progress = None, "starting", 0
            task.detail = "Uploading inputs to the GPU server"
            db.commit()
            kind, params, uploads = edit_ops.prepare(db, task, gpu)
            job = gpu.submit(kind, params, uploads)
            task.remote_job_id, task.detail = job["id"], None
            db.commit()
        unreachable_since = None
        while True:
            db.refresh(task)
            if task.status != "running":  # cancelled (or failed) from outside: stop working on it
                if task.status == "cancelled":
                    gpu.cancel(task.remote_job_id)
                    _set_edit(db, task, REVERT_STATE.get(task.type))
                    db.commit()
                return
            try:
                job = gpu.job(task.remote_job_id)
                unreachable_since = None
            except GpuUnreachable:
                unreachable_since = unreachable_since or time.time()
                if time.time() - unreachable_since > config.GPU_LOST_SECONDS:
                    raise GpuError("GPU server unreachable for too long; is the Kaggle notebook still running?")
                task.detail = "GPU server unreachable, retrying…"
                db.commit()
                time.sleep(POLL)
                continue
            if job is None:  # the GPU server restarted and forgot the job
                if task.attempts < config.MAX_ATTEMPTS:
                    task.status, task.remote_job_id, task.stage, task.progress, task.detail = "pending", None, "queued", 0, "GPU job lost, retrying"
                    db.commit()
                    return
                raise GpuError("The GPU server lost the job (did the Kaggle session restart?)")
            task.stage, task.progress, task.detail = job["stage"], max(task.progress, min(job["progress"], 99)), job.get("detail")
            task.last_heartbeat = utcnow()
            edit_ops_partial(db, task, gpu, job)
            db.commit()
            if job["status"] == "completed":
                break
            if job["status"] in ("failed", "cancelled"):
                raise GpuError(job.get("error") or f"GPU job {job['status']}")
            time.sleep(POLL)

        task.stage, task.detail = "downloading_results", None
        db.commit()
        zip_path = storage.project_dir(task.project_id) / f"_download-{task.id}.zip"
        gpu.download_artifact(task.remote_job_id, zip_path)
        try:
            edit_ops.finish(db, task, zip_path, job.get("result") or {})
        finally:
            zip_path.unlink(missing_ok=True)
        task.status, task.stage, task.progress, task.error, task.finished_at = "completed", "done", 100, None, utcnow()
        db.commit()
    except Exception as exc:  # GpuError, upload problems, bad archives ...
        if not isinstance(exc, GpuError):
            traceback.print_exc()
        db.rollback()
        task = db.get(Task, task.id)
        if task.status == "cancelled":
            return
        _fail(db, task, str(exc))


def edit_ops_partial(db, task, gpu, job):
    try:
        edit_ops.on_partial(db, task, gpu, job)
    except Exception:  # previews are best effort
        pass
