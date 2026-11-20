# Person B: Backend and Job System

FastAPI + SQLite backend and a pull-based Kaggle worker. Built against Person A's actual
pipeline (`pipeline/run_pipeline.py`, `common.py`, `package.py`) and their `week2_handoff.md`.

## Repo layout (drop these folders next to Person A's `pipeline/`)

```
room-3d-reconstruction/
├── pipeline/            # Person A (unchanged)
├── backend/             # this folder: FastAPI app + tests
│   └── app/{main,config,db,models,schemas,security,storage,stages}.py, routers/{auth,projects,tasks}.py
└── worker/poll.py       # runs on Kaggle
```

## Run the backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # set ROOM3D_JWT_SECRET and ROOM3D_WORKER_TOKEN to long random strings
set -a; source .env; set +a
uvicorn app.main:app --host 0.0.0.0 --port 8000      # docs at http://localhost:8000/docs
```

Tests (no GPU, no Kaggle needed): `pip install -r requirements-dev.txt && pytest -q`

Auth is a bonus in the brief: `ROOM3D_AUTH_REQUIRED=false` runs every request as a built-in
demo user. Projects are still scoped by `user_id`, so switching auth on later needs no migration.

## Connect Kaggle

The notebook only makes **outbound** calls, so only the backend needs a public URL:

```bash
cloudflared tunnel --url http://localhost:8000     # prints https://<random>.trycloudflare.com
```

A quick tunnel gets a **new URL every restart**, so update the `BACKEND_URL` Kaggle secret each time
(a named tunnel gives a stable URL but needs a Cloudflare account and domain).

In the Kaggle notebook: enable **Internet** in the notebook settings, add `BACKEND_URL` and `WORKER_TOKEN`
under Add-ons → Secrets, then:

```python
import os
from kaggle_secrets import UserSecretsClient
s = UserSecretsClient()
os.environ["BACKEND_URL"] = s.get_secret("BACKEND_URL")
os.environ["WORKER_TOKEN"] = s.get_secret("WORKER_TOKEN")
!git clone <your team repo> /kaggle/working/repo
%cd /kaggle/working/repo
!python worker/poll.py          # blocks and polls; Ctrl-C / session end stops it
```

Use `ONCE=1` to process a single task and exit while testing.

## How a job flows

```
user: POST /projects/{id}/reconstruct  →  Task(pending)
worker: POST /worker/next-task         →  claims it atomically (pending→running, attempts+1)
worker: GET  /worker/tasks/{id}/video  →  downloads the video
worker: runs  pipeline/run_pipeline.py --stage production --backend_url … --task_id …
        pipeline itself POSTs /tasks/{id}/progress  (Bearer = worker token, via PERSON_A_BACKEND_TOKEN)
worker: POST /tasks/{id}/heartbeat every 30 s    (liveness, independent of progress)
worker: PUT  /tasks/{id}/artifacts?part=i&parts=n   uploads Person_A_backend_delivery.zip in 32 MB parts
worker: POST /tasks/{id}/complete      →  backend verifies scene.splat exists, task = completed
```

Design decisions that come from Person A's actual code:

| Observation in the pipeline | What the backend does |
|---|---|
| GLOMAP matching took ~1465 s with no progress event | Separate heartbeat thread; a task is only re-queued after `ROOM3D_STALE_SECONDS` (600) **without a heartbeat**, not without progress |
| `package.py` sends `status=completed, progress=100` before anything is uploaded | Progress is held at 99 until `POST /complete` confirms the artifacts are stored |
| `evaluate` and `report` stages send no progress | The bar simply holds; no fake timer |
| Progress values are per-script and can arrive out of order | Progress is monotonic (`max`), late events after a terminal state are ignored |
| Delivery zip can exceed 100 MB for 2-5 min videos (frames are stored twice) | Chunked upload. Cloudflare's free plan limits request bodies to 100 MB (verify on your plan); chunk size is `UPLOAD_PART_MB` |
| Worker dies mid-job (session limit) | Task goes back to `pending` up to `ROOM3D_MAX_ATTEMPTS` (2), then `failed` |

## Storage (Week 2 contract)

Person A's paths are preserved exactly under `artifacts/`, so `week2_handoff.md` stays valid:

```
storage/projects/{id}/
├── video/input.mp4
└── artifacts/
    ├── 3dgs_dataset/images/                              undistorted frames
    ├── 3dgs_dataset/sparse_txt/{cameras,images,points3D}.txt   pinhole intrinsics, world-to-camera poses, points
    ├── 3dgs_dataset/sparse/0/                            binary COLMAP model
    ├── full_reconstruction/{images/, vggt/vggt_poses.npz}
    ├── exports/<run_name>/{scene.ply, scene.splat, scene_metadata.json}
    ├── 3dgs_training/<run_name>/{inference_weights.pt, split.json, ...}
    └── reports/{week2_handoff.md, comparison.md, runtime_vram.csv, validation_status.json}
```

`GET /projects/{id}/scene` returns a `handoff` object listing each of these with an `exists` flag.
Camera conventions (COLMAP world-to-camera, +Y down, +Z forward) are in Person A's `week2_handoff.md`.
Set `ROOM3D_KEEP_DELIVERY_ZIP=true` to also keep the original zip under `delivery/`.

## API contract for Person C

Base URL = backend URL. Send `Authorization: Bearer <access_token>` on everything except signup/login.

| Method | Path | Notes |
|---|---|---|
| POST | `/auth/signup`, `/auth/login` | `{email, password(≥8)}` → `{access_token}` (409 duplicate, 401 bad login) |
| GET | `/auth/me` | |
| POST | `/projects` | `{name}` → project |
| GET | `/projects` | only the caller's projects; each has `has_video`, `scene_ready`, `latest_task` |
| POST | `/projects/{id}/video` | multipart field `file` (mp4/mov/m4v/webm/mkv, ≤600 MB). Returns `{project, warnings[]}` |
| POST | `/projects/{id}/reconstruct` | optional `{steps: 1000-30000 (default 7000), run_name}` → 202 + task. 409 if no video or a job is active |
| GET | `/tasks/{id}` | poll every ~2 s |
| GET | `/projects/{id}/tasks` | history |
| GET | `/projects/{id}/scene` | 409 until ready |

**Task object** (progress page): `status` (`pending|running|completed|failed`), `progress` 0-100,
`stage` (raw name), `stage_group` (`extract|poses|train|export`), `stage_label` (display text), `error`.
Stepper order: `extract → poses → train → export`. On `failed`, `stage_label` is where it failed and `error` has the tail of the pipeline log.

**Scene response**: `splat_url` is **relative** to the API base (`new URL(scene.splat_url, API_BASE)`), contains a
signed `?t=` token (valid `url_expires_minutes`, 120) so the viewer can fetch without an Authorization header, and supports
HTTP Range. Re-call `/scene` for a fresh URL if it expires. `ply_url` is the higher-quality PLY (SH degree 2); the `.splat` keeps DC colour only.

CORS defaults to `*`; set `ROOM3D_CORS_ORIGINS` to your frontend origin(s) before sharing the URL.

## Known limits

- One global FIFO queue, one active job per project, no cancel endpoint (heartbeat reply already lets the worker stop if a task is failed server-side).
- Re-running a project replaces its `artifacts/` folder (previous scene is overwritten on successful upload).
- File tokens are bearer URLs: anyone holding a `splat_url` can fetch that project's files until it expires.
- The video-length warning needs `ffprobe` on the backend host; without it the check is skipped.
- Kaggle weekly GPU quota and per-session time limit apply; check the current numbers before planning runs.
