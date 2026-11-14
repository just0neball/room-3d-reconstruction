import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
_tmp = tempfile.mkdtemp(prefix="room3d-test-")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


GPU_PORT = _free_port()
# Must be set before the app is imported (config reads env at import time).
os.environ["ROOM3D_DATA_DIR"] = _tmp
os.environ["ROOM3D_JWT_SECRET"] = "test-secret-test-secret-test-secret-123"
os.environ["ROOM3D_AUTH_REQUIRED"] = "true"
os.environ["ROOM3D_GPU_URL"] = f"http://127.0.0.1:{GPU_PORT}"
os.environ["ROOM3D_GPU_TOKEN"] = "test-gpu-token"
os.environ["ROOM3D_FRONTEND_DIR"] = str(Path(_tmp) / "no-frontend")

sys.path.insert(0, str(ROOT / "backend"))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session", autouse=True)
def mock_gpu_server():
    """The real GPU server app in GPU_MOCK=1 mode (synthetic room, CPU stand-in models) on a local port."""
    env = {**os.environ, "GPU_MOCK": "1", "GPU_DATA_DIR": str(Path(_tmp) / "gpu"), "GPU_SERVER_TOKEN": "test-gpu-token", "PYTHONPATH": str(ROOT)}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "gpu_server.app:app", "--port", str(GPU_PORT), "--log-level", "warning"], env=env, cwd=ROOT)
    for _ in range(60):
        try:
            if requests.get(f"http://127.0.0.1:{GPU_PORT}/health", headers={"X-Room3D-Token": "test-gpu-token"}, timeout=1).ok:
                break
        except requests.RequestException:
            time.sleep(0.5)
    else:
        proc.kill()
        raise RuntimeError("mock GPU server did not start")
    yield
    proc.terminate()


@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def drain_queue(client):
    from app.db import SessionLocal
    from app.models import Task
    with SessionLocal() as db:
        for task in db.query(Task).filter(Task.status.in_(["pending", "running"])):
            task.status = "cancelled"
        db.commit()
    yield
