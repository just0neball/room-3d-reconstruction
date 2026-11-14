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

# GPU server (Kaggle notebook exposed through a Cloudflare tunnel). Can also be set from the UI (PUT /gpu/config).
GPU_URL = os.environ.get("ROOM3D_GPU_URL", "").rstrip("/")
GPU_TOKEN = os.environ.get("ROOM3D_GPU_TOKEN", "")
FRONTEND_DIR = Path(os.environ.get("ROOM3D_FRONTEND_DIR", Path(__file__).resolve().parents[2] / "frontend"))
UPLOAD_PART_MB = _int("ROOM3D_UPLOAD_PART_MB", 32)  # per request to the GPU server (Cloudflare free plan limit is 100 MB)

MAX_VIDEO_BYTES = _int("ROOM3D_MAX_VIDEO_MB", 600) * 1024 * 1024
MAX_EXTRACT_BYTES = _int("ROOM3D_MAX_EXTRACT_MB", 6144) * 1024 * 1024
KEEP_DELIVERY_ZIP = _bool("ROOM3D_KEEP_DELIVERY_ZIP", False)

MAX_ATTEMPTS = _int("ROOM3D_MAX_ATTEMPTS", 2)  # a task whose GPU job is lost (Kaggle session died) is retried this often
GPU_LOST_SECONDS = _int("ROOM3D_GPU_LOST_SECONDS", 600)  # running task, GPU server unreachable this long -> fail

CORS_ORIGINS = [o.strip() for o in os.environ.get("ROOM3D_CORS_ORIGINS", "*").split(",") if o.strip()]

DEFAULT_STEPS = 7000
MIN_STEPS, MAX_STEPS = 1000, 30000


def warn_insecure_defaults():
    if JWT_SECRET.startswith("dev-secret"):
        warnings.warn("ROOM3D_JWT_SECRET is the development default; set a real secret.", stacklevel=2)
