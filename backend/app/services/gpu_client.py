"""HTTP client for the GPU server (Kaggle notebook behind a Cloudflare tunnel). Push model:
upload inputs -> create a job -> poll it -> download the artifact zip."""
import io
import json
import os
import threading
import time
import uuid
import zipfile
from pathlib import Path

import requests

from .. import config

_cfg_lock = threading.Lock()


class GpuError(RuntimeError):
    pass


class GpuUnreachable(GpuError):
    pass


def _cfg_path() -> Path:
    return config.DATA_DIR / "gpu_config.json"


def load_config() -> dict:
    with _cfg_lock:
        try:
            return json.loads(_cfg_path().read_text())
        except (OSError, ValueError):
            return {"url": config.GPU_URL, "token": config.GPU_TOKEN}


def save_config(url: str, token: str | None) -> dict:
    cur = load_config()
    new = {"url": url.strip().rstrip("/"), "token": cur.get("token", "") if token is None else token}
    with _cfg_lock:
        _cfg_path().write_text(json.dumps(new))
    return new


def new_uid(prefix="u") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


class GpuClient:
    def __init__(self, url=None, token=None):
        cfg = load_config()
        self.url = (url if url is not None else cfg.get("url", "")).rstrip("/")
        self.token = token if token is not None else cfg.get("token", "")
        self.http = requests.Session()
        if self.token:
            self.http.headers["X-Room3D-Token"] = self.token

    @property
    def configured(self):
        return bool(self.url)

    def _req(self, method, path, retries=3, timeout=60, **kw):
        if not self.url:
            raise GpuUnreachable("GPU server URL is not set")
        last = None
        for attempt in range(retries):
            try:
                r = self.http.request(method, self.url + path, timeout=timeout, **kw)
                if r.status_code in (502, 503, 504, 520, 521, 522, 523, 524, 530):  # tunnel hiccups / origin down
                    last = f"HTTP {r.status_code}"
                else:
                    return r
            except requests.RequestException as exc:
                last = str(exc)
            time.sleep(min(2 ** attempt, 5))
        raise GpuUnreachable(f"GPU server unreachable: {last}")

    def _json(self, r):
        if r.status_code == 401:
            raise GpuError("GPU server rejected the token (check ROOM3D_GPU_TOKEN / the Connect dialog)")
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise GpuError(f"GPU server: {detail}")
        return r.json()

    def health(self):
        return self._json(self._req("GET", "/health", retries=1, timeout=8))

    # ---- uploads
    def upload(self, source, uid=None) -> str:
        """source: Path or bytes. Chunked so every request stays below the tunnel's body limit."""
        uid = uid or new_uid()
        part_bytes = config.UPLOAD_PART_MB * 1024 * 1024
        if isinstance(source, (bytes, bytearray)):
            fh, size = io.BytesIO(source), len(source)
        else:
            fh, size = open(source, "rb"), Path(source).stat().st_size
        parts = max(1, -(-size // part_bytes))
        try:
            for i in range(parts):
                data = fh.read(part_bytes) or b"\0"
                self._json(self._req("PUT", f"/uploads/{uid}", params={"part": i, "parts": parts}, data=data, timeout=300,
                                     headers={"Content-Type": "application/octet-stream"}))
        finally:
            fh.close()
        return uid

    # ---- datasets
    def dataset_info(self, key):
        return self._json(self._req("GET", f"/datasets/{key}"))

    def create_dataset(self, key, uid, version):
        return self._json(self._req("POST", f"/datasets/{key}", json={"upload": uid, "version": version}, timeout=300))

    # ---- interactive
    def segment(self, dataset, frame, points, box=None):
        return self._json(self._req("POST", "/segment", json={"dataset": dataset, "frame": frame, "points": points, "box": box}, timeout=120))

    # ---- jobs
    def submit(self, kind, params, uploads=None):
        return self._json(self._req("POST", "/jobs", json={"kind": kind, "params": params, "uploads": uploads or {}}))

    def job(self, job_id):
        r = self._req("GET", f"/jobs/{job_id}")
        if r.status_code == 404:
            return None
        return self._json(r)

    def cancel(self, job_id):
        try:
            self._req("POST", f"/jobs/{job_id}/cancel", retries=1, timeout=10)
        except GpuError:
            pass

    def download_artifact(self, job_id, dest: Path):
        r = self._req("GET", f"/jobs/{job_id}/artifact", stream=True, timeout=600, retries=4)
        if r.status_code >= 400:
            raise GpuError(f"artifact download failed: HTTP {r.status_code} {r.text[:200]}")
        with open(dest, "wb") as out:
            for chunk in r.iter_content(1 << 20):
                out.write(chunk)
        return dest

    def download_file(self, job_id, rel, dest: Path):
        r = self._req("GET", f"/jobs/{job_id}/files/{rel}", timeout=120)
        if r.status_code != 200:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return True
