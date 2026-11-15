# Room3D website API (Week 2 contract)

The website backend (`backend/`, FastAPI) serves both this API and the static frontend (`frontend/`) from one origin.
All paths below are relative to the backend base URL. If `GET /health` reports `auth_required: true`, send
`Authorization: Bearer <token>` on everything except `/auth/*`, `/health` and signed file URLs.

Every `*_url` in a response is **relative to the base URL** and already carries a signed `?t=` token
(valid `url_expires_minutes`, default 120), so it can go straight into `<img src>` / `fetch` / the splat viewer.
Re-fetch the object to get fresh URLs. Image coordinates are **pixels of the undistorted frame**
(`image_size.width x image_size.height`, long side 640-960), origin top-left.

## Meta / GPU server

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | `{status, auth_required, gpu_configured}` |
| GET | `/gpu/status` | `{configured, url, reachable, mock, message, checked_at, devices:[{name,total_gb,free_gb}], models:{segmenter,diffusion,...}, uptime_s}`. Cheap; the UI polls it every ~10 s and shows a banner when `reachable` is false |
| PUT | `/gpu/config` | `{url, token?}` -> same shape as `/gpu/status`. Saves the Kaggle tunnel URL (changes every Kaggle session) |

## Projects (unchanged from Week 1) + new read endpoints

`POST /projects`, `GET /projects`, `GET /projects/{id}`, `POST /projects/{id}/video`,
`POST /projects/{id}/reconstruct` (body `{steps?, run_name?, frame_fps?: 1|2}`), `GET /projects/{id}/tasks`, `GET /projects/{id}/scene`.

`GET /projects/{id}/scene` adds `cameras_url` and `edits: [{id, name, state, splat_url|null}]`.

| Method | Path | Notes |
|---|---|---|
| GET | `/projects/{id}/frames` | `{fps, count, image_size:{width,height}, frames:[{index, name, seconds, url, thumb_url}]}` undistorted training frames in temporal order. 409 until a scene exists |
| GET | `/projects/{id}/thumb/{name}?w=240&t=` | cached JPEG thumbnail |
| GET | `/projects/{id}/cameras` | `{image_size, up:[0,-1,0], cameras:[{index,name,position:[x,y,z],rotation_c2w:[9 floats row-major, COLMAP axes +X right,+Y down,+Z fwd],fx,fy,cx,cy}]}` in the splat's world coordinates, for snapping the viewer to a training camera |

## Tasks

`GET /tasks/{id}` (poll ~1.5 s). Task object:

```
id, project_id, edit_id|null, type: reconstruction|track|edit_reference|edit_rest|rebuild,
status: pending|running|completed|failed|cancelled,
stage, stage_group, stage_label, progress 0-100, detail (e.g. "frame 12 of 80"|null),
error|null, steps, run_name, attempts, created_at, started_at, finished_at, metrics (object|null)
```
Reconstruction stage groups: `extract|poses|train|export`. Other types use their `type` as `stage_group`.
`POST /tasks/{id}/cancel` cancels a pending/running task (best effort).

## Edits (one object = one "click -> prompt -> result" run; several per project, acts as edit history)

State machine (`edit.state`):
`draft` -> `masked` (reference mask exists) -> `tracking` -> `tracked` -> `editing_reference` -> `reference_ready`
(candidates to review) -> `editing_frames` -> `edited` -> `rebuilding` -> `done`; any step may go to `failed`
(`edit.error` set) and the user can retry the step. Going back is allowed: re-running `track` resets later stages.

| Method | Path | Notes |
|---|---|---|
| POST | `/projects/{pid}/edits` | `{name?, prompt?}` -> Edit (`draft`). 409 if the project has no scene |
| GET | `/projects/{pid}/edits` | list of Edit, newest first |
| GET | `/edits/{id}` | Edit |
| PATCH | `/edits/{id}` | `{name?, prompt?, settings?}` |
| DELETE | `/edits/{id}` | removes edit + files |
| POST | `/edits/{id}/pick3d` | `{point:[x,y,z]}` (splat world coords, from a ray hit in the 3D viewer) -> `{frame, index, x, y, candidates:[{frame,index,x,y}]}`: the training frame that sees the point best and its 2D pixel. Frontend then calls `/segment` with that click |
| POST | `/edits/{id}/segment` | **sync**. `{frame, points:[{x,y,label:1|0}], box?:[x0,y0,x1,y1], set_reference?:true}` -> `{mask_url, overlay_url, area, score, frame}`. Click on a frame; positive/negative points refine; with `set_reference` (default) this frame becomes the reference mask and state -> `masked`. With `set_reference:false` it edits that single frame's mask (manual fix, `source:"manual"`) |
| POST | `/edits/{id}/track` | `{method:"auto"|"video"|"lift3d"|"neighbor"|"manual", frames?:[names]}` -> 202 Task `track`. `auto` = lift3d with video-propagation cross-check (server decides); `manual` = SAM on exactly `frames` (the user-marked fallback), each prompted with projected points when available |
| PATCH | `/edits/{id}/frames` | `{include?:[names], exclude?:[names]}` toggles frames in/out of the edit set without running models (user removes wrong frames). Frames included that have no mask need `track` with `method:"manual"` |
| POST | `/edits/{id}/edit/reference` | `{prompt?, seeds?:1-6, frame?}` (`frame` = any included frame with a mask; default = the clicked frame) -> 202 Task `edit_reference`; then `state:reference_ready` and `edit.candidates` filled |
| POST | `/edits/{id}/edit/accept` | `{seed}` picks the reference edit (state stays `reference_ready`, `chosen_seed` set) |
| POST | `/edits/{id}/edit/run` | `{}` -> 202 Task `edit_rest` (needs `chosen_seed`); progress `detail` = "frame x of N" |
| POST | `/edits/{id}/rebuild` | `{mode?:"finetune"|"scratch", steps?}` -> 202 Task `rebuild` |

Edit object:

```
id, project_id, name, state, error|null, prompt, created_at, updated_at,
image_size:{width,height},
ref_frame|null, points:[{x,y,label}], box|null, ref_mask_url|null, ref_overlay_url|null,
tracking: null | {method, message, included_count, total, frames:[
  {index, name, seconds, included, source:"reference"|"auto"|"manual"|"none", score|null, area (0-1 fraction of image)|null,
   thumb_url, mask_url|null, overlay_url|null, edited_url|null}]},
candidates:[{seed, url, thumb_url}], chosen_seed|null, reference_url|null, edit_frame|null (frame the first edit is made on),
edit_progress:{done,total},
result: null | {splat_url, ply_url, mode, steps, build_seconds, original_splat_url},
settings:{seeds, quant:"nf4"|"int8"|"bf16", rebuild_mode, rebuild_steps},
metrics:{track:{...}, edit:{peak_vram_gb, sec_per_frame, model, quant}, rebuild:{...}},
active_task: Task|null
```

`result.original_splat_url` is the untouched scene, so the viewer can toggle before/after without another call.

## Conventions

- Errors: JSON `{detail}` with 4xx/5xx; 409 = wrong state / nothing to do yet; 503 = GPU server unreachable (UI shows the banner).
- A task that cannot start because the GPU server is offline stays `pending` with `detail:"Waiting for GPU server"`; it starts by itself when the server comes back.
- One active task per project.
