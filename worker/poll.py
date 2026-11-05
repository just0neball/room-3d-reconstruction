#!/usr/bin/env python3
"""Pull-based worker. Runs inside the Kaggle notebook (or any GPU machine).

It only makes OUTBOUND HTTP calls to the backend, so nothing on Kaggle has to be
exposed. Loop: claim task -> download video -> run Person A's pipeline as a
subprocess (progress is reported by the pipeline itself) -> upload the delivery
zip in parts -> mark complete.

Required env:  BACKEND_URL, WORKER_TOKEN
Optional env:  PIPELINE_DIR (default ./pipeline), WORK_DIR (default /kaggle/working/jobs),
               POLL_INTERVAL (s, 10), UPLOAD_PART_MB (32), HEARTBEAT_INTERVAL (s, 30),
               COLMAP_BIN, ONCE=1 (process a single task and exit; handy for testing)
"""
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

BACKEND_URL = os.environ["BACKEND_URL"].rstrip("/")
WORKER_TOKEN = os.environ["WORKER_TOKEN"]
PIPELINE_DIR = Path(os.environ.get("PIPELINE_DIR", "pipeline")).resolve()
WORK_DIR = Path(os.environ.get("WORK_DIR", "/kaggle/working/jobs"))
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "10"))
PART_BYTES = int(float(os.environ.get("UPLOAD_PART_MB", "32")) * 1024 * 1024)
HEARTBEAT_INTERVAL = float(os.environ.get("HEARTBEAT_INTERVAL", "30"))
HEADERS = {"Authorization": f"Bearer {WORKER_TOKEN}"}
session = requests.Session()
session.headers.update(HEADERS)


def log(message: str) -> None:
    print(f"[worker {time.strftime('%H:%M:%S')}] {message}", flush=True)


def api(method: str, path: str, retries: int = 4, **kwargs) -> requests.Response:
    """HTTP call with exponential backoff; quick tunnels drop connections occasionally."""
    kwargs.setdefault("timeout", 60)
    for attempt in range(retries):
        try:
            response = session.request(method, f"{BACKEND_URL}{path}", **kwargs)
            if response.status_code < 500:
                return response
            error = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            error = str(exc)
        if attempt == retries - 1:
            raise RuntimeError(f"{method} {path} failed after {retries} attempts: {error}")
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def report(task_id: int, stage: str, progress: int, status: str = "running", error: str | None = None) -> None:
    body = {"stage": stage, "progress": progress, "status": status, "error": error}
    try:
        api("POST", f"/tasks/{task_id}/progress", json=body, retries=3)
    except RuntimeError as exc:
        log(f"could not report progress: {exc}")


class Heartbeat(threading.Thread):
    """Separate liveness signal: GLOMAP matching alone can run 25 min without a progress event."""

    def __init__(self, task_id: int, process_ref: dict):
        super().__init__(daemon=True)
        self.task_id, self.process_ref, self._halt = task_id, process_ref, threading.Event()

    def run(self):
        while not self._halt.wait(HEARTBEAT_INTERVAL):
            try:
                status = api("POST", f"/tasks/{self.task_id}/heartbeat", retries=2).json().get("status")
                if status != "running":  # failed/cancelled server-side -> stop burning GPU time
                    log(f"task is '{status}' on the server; terminating pipeline")
                    proc = self.process_ref.get("proc")
                    if proc and proc.poll() is None:
                        proc.terminate()
                    return
            except Exception as exc:  # never let the heartbeat thread die noisily
                log(f"heartbeat error: {exc}")

    def stop(self):
        self._halt.set()


def download_video(task: dict, dest: Path) -> None:
    with session.get(f"{BACKEND_URL}{task['video_url']}", stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(dest, "wb") as out:
            for chunk in response.iter_content(1 << 20):
                out.write(chunk)


def run_pipeline(task: dict, project_dir: Path, video: Path, log_path: Path, process_ref: dict) -> int:
    command = [
        sys.executable, str(PIPELINE_DIR / "run_pipeline.py"),
        "--project_dir", str(project_dir),
        "--video", str(video),
        "--stage", "production",
        "--steps", str(task["steps"]),
        "--run_name", task["run_name"],
        "--backend_url", BACKEND_URL,
        "--task_id", str(task["id"]),
    ]
    if os.environ.get("COLMAP_BIN"):
        command += ["--colmap_bin", os.environ["COLMAP_BIN"]]
    # pipeline/common.py reads this and sends it as "Authorization: Bearer ..." with each progress event
    env = {**os.environ, "PERSON_A_BACKEND_TOKEN": WORKER_TOKEN}
    log(f"running: {' '.join(command)}")
    with open(log_path, "w", encoding="utf-8") as logfile:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
        process_ref["proc"] = proc
        for line in proc.stdout:
            print(line, end="", flush=True)
            logfile.write(line)
        return proc.wait()


def upload_delivery(task_id: int, zip_path: Path) -> None:
    size = zip_path.stat().st_size
    parts = max(1, -(-size // PART_BYTES))
    log(f"uploading {size / 1024**2:.1f} MB in {parts} part(s)")
    with open(zip_path, "rb") as source:
        for index in range(parts):
            data = source.read(PART_BYTES)
            response = api(
                "PUT", f"/tasks/{task_id}/artifacts", params={"part": index, "parts": parts},
                data=data, headers={"Content-Type": "application/octet-stream"}, timeout=300,
            )
            if response.status_code != 200:
                raise RuntimeError(f"artifact part {index + 1}/{parts} rejected: {response.status_code} {response.text[:300]}")
            report(task_id, "uploading_results", 90 + int(9 * (index + 1) / parts))


def process(task: dict) -> None:
    task_id = task["id"]
    job = WORK_DIR / str(task_id)
    shutil.rmtree(job, ignore_errors=True)
    project_dir = job / "project"
    project_dir.mkdir(parents=True)
    video, log_path = job / "input.mp4", job / "pipeline.log"
    process_ref: dict = {}
    heartbeat = Heartbeat(task_id, process_ref)
    heartbeat.start()
    try:
        log(f"task {task_id} (project {task['project_id']}, attempt {task['attempt']}): downloading video")
        report(task_id, "starting", 1)
        download_video(task, video)
        code = run_pipeline(task, project_dir, video, log_path, process_ref)
        if code != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
            raise RuntimeError(f"pipeline exited with code {code}:\n{tail}")
        zip_path = project_dir / "Person_A_backend_delivery.zip"
        if not zip_path.is_file():
            raise RuntimeError("pipeline finished but Person_A_backend_delivery.zip is missing")
        upload_delivery(task_id, zip_path)
        response = api("POST", f"/tasks/{task_id}/complete")
        if response.status_code != 200:
            raise RuntimeError(f"complete rejected: {response.status_code} {response.text[:300]}")
        log(f"task {task_id} completed")
        shutil.rmtree(job, ignore_errors=True)  # Kaggle /working is small; backend now holds everything
    except Exception as exc:
        log(f"task {task_id} failed: {exc}")
        report(task_id, "failed", 0, status="failed", error=str(exc)[-4000:])
    finally:
        heartbeat.stop()


def main() -> None:
    log(f"polling {BACKEND_URL} every {POLL_INTERVAL}s (pipeline: {PIPELINE_DIR})")
    while True:
        try:
            task = api("POST", "/worker/next-task").json().get("task")
        except Exception as exc:
            log(f"poll error: {exc}")
            task = None
        if task:
            process(task)
            if os.environ.get("ONCE"):
                return
        else:
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
