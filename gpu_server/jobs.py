"""Single-worker job queue. GPU work is serialised (one heavy job at a time); interactive /segment calls bypass it."""
import queue
import shutil
import threading
import time
import traceback
import uuid
import zipfile
from pathlib import Path

from . import settings


class Cancelled(Exception):
    pass


class Job:
    def __init__(self, kind, params, uploads):
        self.id = uuid.uuid4().hex[:12]
        self.kind, self.params, self.uploads = kind, params, uploads
        self.status, self.stage, self.progress, self.detail = "queued", "queued", 0, None
        self.error, self.result, self.partial = None, {}, []
        self.cancel = threading.Event()
        self.created, self.started, self.finished = time.time(), None, None
        self.dir = settings.DATA_DIR / "jobs" / self.id
        self.dir.mkdir(parents=True, exist_ok=True)

    def public(self):
        return {"id": self.id, "kind": self.kind, "status": self.status, "stage": self.stage, "progress": self.progress,
                "detail": self.detail, "error": self.error, "result": self.result, "partial": self.partial,
                "has_artifact": (self.dir / "artifact.zip").is_file(), "created": self.created, "started": self.started, "finished": self.finished}


class Context:
    def __init__(self, job):
        self.job = job
        self.dir = job.dir
        self.out = job.dir / "out"
        self.out.mkdir(exist_ok=True)
        self.artifact = None  # a handler may hand over a ready-made zip instead of zipping out/

    def update(self, stage=None, progress=None, detail=None):
        if self.job.cancel.is_set():
            raise Cancelled()
        if stage is not None:
            self.job.stage = stage
        if progress is not None:
            self.job.progress = max(self.job.progress, min(int(progress), 99))
        self.job.detail = detail

    def upload_path(self, name):
        return settings.DATA_DIR / "uploads" / self.job.uploads[name]

    def unzip(self, name):
        dest = self.dir / "in" / name
        if not dest.exists():
            dest.mkdir(parents=True)
            with zipfile.ZipFile(self.upload_path(name)) as z:
                z.extractall(dest)
        return dest

    def mark_partial(self, rel):
        self.job.partial.append(rel)

    def cancelled(self):
        return self.job.cancel.is_set()


def zip_dir(src, dest):
    src = Path(src)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED) as z:  # frames are already compressed (png/jpg)
        for f in sorted(src.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(src).as_posix())


class JobManager:
    def __init__(self, handlers):
        self.handlers, self.jobs, self.q = handlers, {}, queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, kind, params, uploads):
        if kind not in self.handlers:
            raise KeyError(kind)
        job = Job(kind, params, uploads)
        self.jobs[job.id] = job
        self.q.put(job.id)
        return job

    def get(self, job_id):
        return self.jobs.get(job_id)

    def cancel(self, job_id):
        job = self.jobs.get(job_id)
        if job and job.status in ("queued", "running"):
            job.cancel.set()
            if job.status == "queued":
                job.status, job.finished = "cancelled", time.time()
        return job

    def _loop(self):
        while True:
            job = self.jobs[self.q.get()]
            if job.status != "queued":
                continue
            job.status, job.started = "running", time.time()
            ctx = Context(job)
            try:
                job.result = self.handlers[job.kind](ctx, job.params) or {}
                if ctx.artifact:
                    shutil.move(str(ctx.artifact), job.dir / "artifact.zip")
                else:
                    zip_dir(ctx.out, job.dir / "artifact.zip")
                job.status, job.progress, job.stage = "completed", 100, "done"
            except Cancelled:
                job.status = "cancelled"
            except Exception as exc:
                traceback.print_exc()
                job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"[-4000:]
            job.finished = time.time()
