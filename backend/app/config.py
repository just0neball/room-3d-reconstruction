"""Central configuration, read from environment variables (see backend/.env.example)."""
import os
import warnings
from pathlib import Path


def _int(name, default):
    return int(os.environ.get(name, default))


def _bool(name, default):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


DATA_DIR = Path(os.environ.get("ROOM3D_DATA_DIR", "storage")).resolve()
DATABASE_URL = os.environ.get("ROOM3D_DATABASE_URL") or f"sqlite:///{DATA_DIR / 'room3d.db'}"

# Auth is a bonus in the brief. With ROOM3D_AUTH_REQUIRED=false every request
# runs as a built-in "demo@local" user, but projects are still scoped by user_id
# so turning auth on later needs no data migration.
AUTH_REQUIRED = _bool("ROOM3D_AUTH_REQUIRED", True)
JWT_SECRET = os.environ.get("ROOM3D_JWT_SECRET", "dev-secret-change-me-dev-secret-change-me")
ACCESS_TOKEN_MINUTES = _int("ROOM3D_ACCESS_TOKEN_MINUTES", 60 * 24)
FILE_TOKEN_MINUTES = _int("ROOM3D_FILE_TOKEN_MINUTES", 120)

# Shared secret between backend and the Kaggle worker. Person A's pipeline sends it
# as "Authorization: Bearer <token>" (env PERSON_A_BACKEND_TOKEN) on progress events.
WORKER_TOKEN = os.environ.get("ROOM3D_WORKER_TOKEN", "dev-worker-token")

MAX_VIDEO_BYTES = _int("ROOM3D_MAX_VIDEO_MB", 600) * 1024 * 1024
MAX_PART_BYTES = _int("ROOM3D_MAX_PART_MB", 128) * 1024 * 1024
MAX_EXTRACT_BYTES = _int("ROOM3D_MAX_EXTRACT_MB", 6144) * 1024 * 1024
KEEP_DELIVERY_ZIP = _bool("ROOM3D_KEEP_DELIVERY_ZIP", False)

# A running task with no heartbeat for this long is re-queued (or failed once
# MAX_ATTEMPTS is used up). Must be well above the worker heartbeat interval (30 s).
STALE_SECONDS = _int("ROOM3D_STALE_SECONDS", 600)
MAX_ATTEMPTS = _int("ROOM3D_MAX_ATTEMPTS", 2)

CORS_ORIGINS = [o.strip() for o in os.environ.get("ROOM3D_CORS_ORIGINS", "*").split(",") if o.strip()]

DEFAULT_STEPS = 7000
MIN_STEPS, MAX_STEPS = 1000, 30000


def warn_insecure_defaults():
    if JWT_SECRET.startswith("dev-secret"):
        warnings.warn("ROOM3D_JWT_SECRET is the development default; set a real secret.", stacklevel=2)
    if WORKER_TOKEN == "dev-worker-token":
        warnings.warn("ROOM3D_WORKER_TOKEN is the development default; set a real token.", stacklevel=2)
