import os
import sys
import tempfile
from pathlib import Path

# Must be set before the app is imported (config reads env at import time).
_tmp = tempfile.mkdtemp(prefix="room3d-test-")
os.environ["ROOM3D_DATA_DIR"] = _tmp
os.environ["ROOM3D_WORKER_TOKEN"] = "test-worker-token"
os.environ["ROOM3D_JWT_SECRET"] = "test-secret-test-secret-test-secret-123"
os.environ["ROOM3D_AUTH_REQUIRED"] = "true"
os.environ["ROOM3D_STALE_SECONDS"] = "600"
os.environ["ROOM3D_MAX_ATTEMPTS"] = "2"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def drain_queue(client):
    """The queue is global FIFO (as in production), so tests must not inherit tasks
    left pending/running by earlier tests."""
    from app.db import SessionLocal
    from app.models import Task

    with SessionLocal() as db:
        for task in db.query(Task).filter(Task.status.in_(["pending", "running"])):
            task.status = "failed"
        db.commit()
    yield
