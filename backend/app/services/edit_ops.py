"""Everything that turns a Task / Edit into GPU-server requests and back into files + DB rows.

Per-edit files (under storage/projects/{pid}/edits/{eid}/):
    masks/<stem>.png          0/255 masks (reference + tracked + manual)
    overlays/<stem>.jpg       frame with the mask tinted (grids)
    candidates/candidate_<seed>.png    reference-edit candidates (full frame, composited)
    reference.png             accepted reference edit
    edited/<stem>.png         edited frames (outside the mask pixel-identical to the original)
    exports/scene.{splat,ply} rebuilt scene
"""
import io
import json
import shutil
import threading
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

from .. import storage
from ..geometry import read_cameras
from ..models import Edit, Project, Task, utcnow
from .gpu_client import GpuClient

_ds_lock = threading.Lock()
FRAMES = "artifacts/3dgs_dataset/images"


def jl(text, default):
    try:
        return json.loads(text) if text else default
    except ValueError:
        return default


def stem(name):
    return Path(name).stem


def edit_dir(pid, eid) -> Path:
    d = storage.project_dir(pid) / "edits" / str(eid)
    d.mkdir(parents=True, exist_ok=True)
    return d


def cameras(pid):
    root = storage.project_dir(pid)
    return read_cameras(root / "artifacts/3dgs_dataset/sparse_txt")


def frame_fps(pid):
    manifest = storage.read_json(storage.project_dir(pid) / "artifacts/full_reconstruction/dataset_manifest.json") or {}
    return float(manifest.get("fps") or 1)


def image_path(pid, name) -> Path:
    return storage.project_dir(pid) / FRAMES / name


# ------------------------------------------------------------------ masks / overlays
def read_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 127


def save_mask(pid, eid, name, mask) -> Path:
    path = edit_dir(pid, eid) / "masks" / f"{stem(name)}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
    make_overlay(pid, eid, name, mask)
    return path


def make_overlay(pid, eid, name, mask, width=480):
    img = cv2.imread(str(image_path(pid, name)))
    if img is None:
        return
    h, w = img.shape[:2]
    scale = min(1.0, width / w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    tint = np.array([255, 120, 40], np.float32)  # BGR: a warm blue-ish highlight that reads on most rooms
    out = img.astype(np.float32)
    out[mask] = out[mask] * 0.45 + tint * 0.55
    out = out.astype(np.uint8)
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnt, -1, (60, 220, 255), 2)
    path = edit_dir(pid, eid) / "overlays" / f"{stem(name)}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 82])


def thumb(path: Path, width: int, cache_dir: Path) -> Path:
    out = cache_dir / str(width) / (path.stem + ".jpg")
    if out.is_file() and out.stat().st_mtime >= path.stat().st_mtime:
        return out
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


# ------------------------------------------------------------------ tracking records
def fresh_tracking(pid, method="auto"):
    cams = cameras(pid)
    fps = frame_fps(pid)
    return {"method": method, "message": "", "frames": [
        {"index": c["index"], "name": c["name"], "seconds": round(c["index"] / fps, 2), "included": False, "source": "none",
         "score": None, "area": None, "visible_points": None} for c in cams]}


def get_tracking(edit):
    t = jl(edit.tracking, None)
    return t if t and t.get("frames") else fresh_tracking(edit.project_id)


def reset_downstream(edit, keep_masks=True):
    """Going back a step invalidates everything computed from it."""
    d = edit_dir(edit.project_id, edit.id)
    for sub in ("candidates", "edited", "exports"):
        shutil.rmtree(d / sub, ignore_errors=True)
    (d / "reference.png").unlink(missing_ok=True)
    edit.candidates, edit.chosen_seed, edit.result, edit.seed_offset = "[]", None, None, 0


def included_frames(edit):
    return [f for f in get_tracking(edit)["frames"] if f["included"] and (edit_dir(edit.project_id, edit.id) / "masks" / f"{stem(f['name'])}.png").is_file()]


# ------------------------------------------------------------------ dataset sync
def dataset_key(pid):
    return f"p{pid}"


def build_dataset_zip(project: Project) -> Path:
    root = storage.project_dir(project.id)
    art = root / "artifacts"
    cache = root / "_cache"
    cache.mkdir(exist_ok=True)
    out = cache / f"dataset-{project.dataset_version}.zip"
    if out.is_file():
        return out
    for old in cache.glob("dataset-*.zip"):
        old.unlink()
    tmp = out.with_suffix(".tmp")
    run = project.scene_run_name or ""
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for f in sorted((art / "3dgs_dataset/images").glob("*")):
            z.write(f, f"images/{f.name}")
        for f in sorted((art / "3dgs_dataset/sparse_txt").glob("*.txt")):
            z.write(f, f"sparse_txt/{f.name}")
        weights = art / f"3dgs_training/{run}/inference_weights.pt"
        if weights.is_file():
            z.write(weights, "inference_weights.pt")
        splat = art / f"exports/{run}/scene.splat"
        if splat.is_file():
            z.write(splat, "scene.splat")
    tmp.replace(out)
    return out


def ensure_dataset(db, gpu: GpuClient, project: Project) -> str:
    """Make sure the GPU server has this project's frames/poses/weights (it may have been restarted: Kaggle sessions die)."""
    key = dataset_key(project.id)
    with _ds_lock:
        if not project.dataset_version:
            project.dataset_version = f"b{int(time.time())}"
            db.commit()
        info = gpu.dataset_info(key)
        if info.get("version") != project.dataset_version:
            zip_path = build_dataset_zip(project)
            uid = gpu.upload(zip_path)
            gpu.create_dataset(key, uid, project.dataset_version)
    return key


# ------------------------------------------------------------------ bundles
def _zip_bytes(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for arc, path in entries:
            z.write(path, arc)
    return buf.getvalue()


def _masks_entries(edit, names):
    d = edit_dir(edit.project_id, edit.id) / "masks"
    return [(f"masks/{stem(n)}.png", d / f"{stem(n)}.png") for n in names if (d / f"{stem(n)}.png").is_file()]


# ------------------------------------------------------------------ prepare / finish per task type
def prepare(db, task: Task, gpu: GpuClient):
    """-> (job kind, params, {upload name: upload id}). Uploads happen here (the slow part for big inputs)."""
    project = db.get(Project, task.project_id)
    p = jl(task.params, {})
    if task.type == "reconstruction":
        key = dataset_key(project.id)
        uid = gpu.upload(storage.project_dir(project.id) / project.video_path)
        return "reconstruct", {"dataset": key, "steps": task.steps, "run_name": task.run_name, "frame_fps": p.get("frame_fps", 1), "project_id": project.id}, {"video": uid}

    edit = db.get(Edit, task.edit_id)
    key = ensure_dataset(db, gpu, project)
    st = jl(edit.settings, {})
    if task.type == "track":
        existing = [f["name"] for f in included_frames(edit) if f["name"] != edit.ref_frame] if p["method"] == "manual" else []
        entries = _masks_entries(edit, [edit.ref_frame] + existing)
        uid = gpu.upload(_zip_bytes(entries))
        return "track", {"dataset": key, "method": p["method"], "ref_frame": edit.ref_frame, "ref_points": jl(edit.points, []),
                         "frames": p.get("frames", []), "user_points": jl(edit.user_points, {})}, {"bundle": uid}
    if task.type == "edit_reference":
        ef = edit_frame(edit)
        uid = gpu.upload(_zip_bytes(_masks_entries(edit, [ef])))
        return "edit_reference", {"dataset": key, "frame": ef, "prompt": edit.prompt, "seeds": p.get("seeds", st.get("seeds", 4)),
                                  "seed_offset": edit.seed_offset, "quant": st.get("quant", "nf4")}, {"bundle": uid}
    if task.type == "edit_rest":
        frames = [f["name"] for f in included_frames(edit)]
        entries = _masks_entries(edit, frames) + [("reference.png", edit_dir(edit.project_id, edit.id) / "reference.png")]
        uid = gpu.upload(_zip_bytes(entries))
        return "edit_rest", {"dataset": key, "frames": frames, "reference_frame": edit_frame(edit), "prompt": edit.prompt, "seed": edit.chosen_seed or 0,
                             "quant": st.get("quant", "nf4")}, {"bundle": uid}
    if task.type == "rebuild":
        d = edit_dir(edit.project_id, edit.id)
        names = sorted(f.stem for f in (d / "edited").glob("*.png"))
        entries = [(f"edited/{n}.png", d / "edited" / f"{n}.png") for n in names] + _masks_entries(edit, [n + ".jpg" for n in names])
        uid = gpu.upload(_zip_bytes(entries))
        return "rebuild", {"dataset": key, "mode": p.get("mode", st.get("rebuild_mode", "finetune")), "steps": p.get("steps", st.get("rebuild_steps", 3000))}, {"bundle": uid}
    raise ValueError(task.type)


def edit_frame(edit: Edit) -> str:
    """Frame on which the first (reference) edit is made: the user's choice if it is still an included frame with a mask, else the clicked frame."""
    want = jl(edit.settings, {}).get("edit_frame")
    if want and any(f["name"] == want for f in included_frames(edit)) and (edit_dir(edit.project_id, edit.id) / "masks" / f"{stem(want)}.png").is_file():
        return want
    return edit.ref_frame


def on_partial(db, task: Task, gpu: GpuClient, job):
    """Live preview: pull edited frames as the GPU server finishes them."""
    if task.type not in ("edit_rest", "edit_reference"):
        return
    d = edit_dir(task.project_id, task.edit_id)
    for rel in job.get("partial", []):
        dest = d / rel
        if not dest.is_file():
            gpu.download_file(job["id"], rel, dest)


def _extract(zip_path, dest: Path, prefix: str):
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            n = info.filename.replace("\\", "/")
            if info.is_dir() or not n.startswith(prefix) or ".." in n.split("/"):
                continue
            out = dest / n
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(info))


def finish(db, task: Task, zip_path: Path, result: dict):
    project = db.get(Project, task.project_id)
    task.metrics = json.dumps(result)
    if task.type == "reconstruction":
        return finalize_reconstruction(db, task, project, zip_path, result)
    edit = db.get(Edit, task.edit_id)
    d = edit_dir(edit.project_id, edit.id)
    metrics = jl(edit.metrics, {})
    if task.type == "track":
        p = jl(task.params, {})
        tmp = d / "_track"
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        data = json.loads((tmp / "tracking.json").read_text())
        tr = get_tracking(edit) if p["method"] == "manual" else fresh_tracking(edit.project_id)
        by_name = {f["name"]: f for f in tr["frames"]}
        for r in data["frames"]:
            src = tmp / "masks" / f"{stem(r['name'])}.png"
            entry = by_name.get(r["name"])
            if entry is None:
                continue
            if p["method"] == "manual":
                if r["included"] and src.is_file():
                    entry.update(included=True, source="manual", area=r["area"], visible_points=r["visible_points"])
                else:
                    continue
            else:
                entry.update(included=r["included"], source=r["source"], score=r["score"], area=r["area"], visible_points=r["visible_points"])
            if src.is_file():
                m = read_mask(src)
                save_mask(edit.project_id, edit.id, r["name"], m)
        info = data["info"]
        tr["method"] = info.get("method_used", p["method"])
        inc = sum(f["included"] for f in tr["frames"])
        tr["message"] = (info.get("note") + " " if info.get("note") else "") + f"Found the object in {inc} of {len(tr['frames'])} frames."
        edit.tracking = json.dumps(tr)
        metrics["track"] = info
        shutil.rmtree(tmp, ignore_errors=True)
        edit.state = "tracked"
    elif task.type == "edit_reference":
        _extract(zip_path, d, "candidates/")
        seeds = sorted(int(f.stem.split("_")[1]) for f in (d / "candidates").glob("candidate_*.png"))
        edit.candidates = json.dumps([{"seed": s} for s in seeds][-12:])
        edit.seed_offset = max(seeds or [0])
        metrics["edit"] = {**metrics.get("edit", {}), **result}
        edit.state = "reference_ready"
    elif task.type == "edit_rest":
        shutil.rmtree(d / "edited", ignore_errors=True)
        _extract(zip_path, d, "edited/")
        metrics["edit"] = {**metrics.get("edit", {}), **result, "peak_vram_gb": result.get("peak_vram_gb"), "sec_per_frame": result.get("sec_per_frame")}
        edit.state = "edited"
    elif task.type == "rebuild":
        shutil.rmtree(d / "exports", ignore_errors=True)
        (d / "exports").mkdir()
        with zipfile.ZipFile(zip_path) as z:
            for name in ("scene.splat", "scene.ply"):
                if name in z.namelist():
                    (d / "exports" / name).write_bytes(z.read(name))
        metrics["rebuild"] = result
        edit.result = json.dumps({"mode": result.get("mode"), "steps": result.get("steps"), "build_seconds": result.get("build_seconds")})
        edit.state = "done"
    edit.metrics = json.dumps(metrics)
    edit.error, edit.failed_step = None, None
    edit.updated_at = utcnow()


def finalize_reconstruction(db, task, project, zip_path, result):
    root = storage.project_dir(project.id)
    staging = root / "artifacts.new"
    shutil.rmtree(staging, ignore_errors=True)
    written = storage.safe_extract(zip_path, staging)
    if written == 0:
        raise RuntimeError("delivery archive contains no artifacts/ entries")
    shutil.rmtree(root / "artifacts", ignore_errors=True)
    staging.rename(root / "artifacts")
    run = task.run_name
    splat_rel, ply_rel = f"artifacts/exports/{run}/scene.splat", f"artifacts/exports/{run}/scene.ply"
    if not (root / splat_rel).is_file():  # the mock writes a single export folder; be tolerant of run-name differences
        found = sorted((root / "artifacts/exports").glob("*/scene.splat"))
        if not found:
            raise RuntimeError(f"no scene.splat in the delivery archive; found {sorted(str(p.relative_to(root)) for p in root.rglob(chr(42)))[:30]}")
        run = found[0].parent.name
        splat_rel, ply_rel = f"artifacts/exports/{run}/scene.splat", f"artifacts/exports/{run}/scene.ply"
    meta = {"scene": storage.read_json(root / f"artifacts/exports/{run}/scene_metadata.json"),
            "training": storage.read_json(root / f"artifacts/3dgs_training/{run}/training_metadata.json"),
            "validation": storage.read_json(root / "artifacts/reports/validation_status.json")}
    project.scene_path, project.scene_run_name = splat_rel, run
    project.scene_ply_path = ply_rel if (root / ply_rel).is_file() else None
    project.scene_meta = json.dumps(meta)
    project.dataset_version = result.get("dataset_version") or f"b{int(time.time())}"
    for old in db.query(Edit).filter(Edit.project_id == project.id):  # edits belong to the previous frames/poses
        db.delete(old)
    shutil.rmtree(root / "edits", ignore_errors=True)
    shutil.rmtree(root / "_cache", ignore_errors=True)
