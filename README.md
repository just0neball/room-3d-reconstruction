# From Video to a Browsable 3D Room — Week 2: click → segment → track → edit → rebuild

Upload a room video, browse the reconstructed Gaussian splat in the browser, then **click an object, describe a change
("make the sofa dark green leather")** and get a re-built scene with a before/after toggle.

```
 Browser (frontend/)  ──HTTP──►  Website backend (backend/, FastAPI + SQLite)  ──HTTPS tunnel──►  GPU server (gpu_server/, FastAPI)
  3D viewer + edit wizard         projects, edits, task queue/dispatcher                            runs on Kaggle (2×T4) from notebooks/gpu_server.ipynb
                                  serves the frontend and all files                                 SAM 2/3 · FLUX.2 klein · gsplat · reconstruction
```

The website never runs a model. It uploads inputs to the GPU server, starts a job, polls it, and downloads the result
(push model). If the GPU server is offline a task simply waits ("Waiting for GPU server") and starts when it is back.

## Repository layout

| Folder | What is in it |
|---|---|
| `frontend/` | Static SPA (vanilla JS + three.js + `gaussian-splats-3d`, vendored — no build step). Pages: login, projects, project, viewer. Edit wizard in `js/wizard/`. |
| `backend/` | Website API: auth, projects, upload, edits (`routers/edits.py`), task dispatcher (`services/dispatcher.py`), GPU client. Tests in `backend/tests/`. |
| `gpu_server/` | The server the Kaggle notebook starts. `vision/` (SAM wrappers, depth, **tracking**), `editing/` (FLUX klein editor, mask compositing, benchmark), `rebuild/` (fine-tune vs scratch), `reconstruction/` (Week-1 pipeline stages), `mock/` (synthetic room for demos/tests). |
| `notebooks/` | `gpu_server.ipynb` — **the reusable Kaggle server notebook**. `research/` — Week-1 notebooks. |
| `docs/` | `API.md` (website API contract), `WEEK2_REPORT.md` (tracking methods, VRAM/timing, quantisation), Week-1 reports. |

## Run it

**1. Try everything without a GPU (mock mode, ~1 minute)**

```bash
pip install -r backend/requirements.txt -r gpu_server/requirements.txt   # mock mode needs no torch
GPU_MOCK=1 PYTHONPATH=. uvicorn gpu_server.app:app --port 8001 &
cd backend && ROOM3D_GPU_URL=http://127.0.0.1:8001 ROOM3D_AUTH_REQUIRED=false uvicorn app.main:app --port 8000
# open http://127.0.0.1:8000  → create project → upload any short video → "Build 3D scene" → open → "Edit object"
```
Mock mode generates a synthetic room and uses stand-ins (flood-fill "SAM", colour-word "diffusion"); the UI shows a purple **DEMO MODE** banner. It exercises every step and every API, but the pictures are not model output.

**2. Real run (Kaggle)**
1. Kaggle → new notebook → upload/import `notebooks/gpu_server.ipynb`; Accelerator **GPU T4 ×2**, Internet on; Secrets `GPU_SERVER_TOKEN`, `HF_TOKEN`.
2. *Run all*. It clones this repo, installs `gpu_server/requirements.txt`, starts uvicorn and a Cloudflare tunnel, and prints `URL` + `token`.
3. Start the website backend anywhere (`cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000`; set `ROOM3D_AUTH_REQUIRED=true` and a real `ROOM3D_JWT_SECRET` for anything public).
4. In the website click the **GPU server** badge (top right) → paste URL and token. The badge turns green and shows the GPU names.
5. Projects → upload video → **Build 3D scene** (extract → poses → train → export) → open → **Edit object**.

Environment variables: `backend/app/config.py` (`ROOM3D_*`) and `gpu_server/settings.py` (`SAM_MODEL_ID`, `DIFFUSION_MODEL_ID`, `DIFFUSION_QUANT=nf4|int8|bf16`, `EDIT_LONG_SIDE`, …).

## The edit flow (what the user sees)

1. **Pick object** — click on a frame (grid at every / 2nd / 4th sample) or click the object in the 3D viewer (ray → best-seeing camera → 2D click). Add/exclude points until the pink outline is right.
2. **Describe** — type the change; **Find object in all frames** (auto tracking).
3. **Review frames** — progress, then a grid with per-frame mask overlay and confidence: untick wrong frames, click a frame to fix its mask, or switch to **Mark frames manually** (the fallback: tick the frames yourself and SAM segments exactly those).
4. **First edit** — N diffusion options on the frame with the largest object; accept or regenerate with new seeds.
5. **Edit all frames** — the accepted result is the reference for every other frame; live progress + peak VRAM / s per frame.
6. **Rebuild** — fine-tune the existing splat (fast) or retrain from scratch; then **View before / after** (original splat is always kept).

Edits are listed per project (edit history); the original scene is never modified.

## Guarantees and how they are checked

* **Nothing outside the mask changes:** the diffusion result is alpha-composited back with weight exactly 0 outside the mask (feathered edge only inside it); every edited frame logs `outside_mask_max_diff` (tests assert `0`).
* **No pose re-estimation:** the rebuild reuses the Week-1 COLMAP poses; only pixels change.
* Tests: `cd backend && pytest` (spawns a real mock GPU server and drives the whole flow through the API), `PYTHONPATH=. pytest gpu_server/tests`.
