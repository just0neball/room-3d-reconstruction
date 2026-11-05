"""On-disk layout. Everything for one project lives under storage/projects/{id}/:

    video/input.<ext>          uploaded source video
    artifacts/                 extracted Person A delivery (paths exactly as in week2_handoff.md)
        3dgs_dataset/images/                undistorted frames
        3dgs_dataset/sparse_txt/{cameras,images,points3D}.txt   pinhole intrinsics + poses + points
        full_reconstruction/images/         extracted frames
        exports/<run_name>/scene.{ply,splat}
        reports/, 3dgs_training/<run_name>/ ...
    delivery/                  (optional) original delivery zip, if ROOM3D_KEEP_DELIVERY_ZIP=true
    _uploads/<task_id>/        temporary chunks of an in-flight artifact upload
"""
import json
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Optional

from . import config

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}


def project_dir(project_id: int) -> Path:
    path = config.DATA_DIR / "projects" / str(project_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def upload_tmp_dir(project_id: int, task_id: int) -> Path:
    return project_dir(project_id) / "_uploads" / str(task_id)


def resolve_inside(project_id: int, rel_path: str) -> Optional[Path]:
    """Resolve a project-relative path; None if it escapes the project folder or isn't a file."""
    root = project_dir(project_id).resolve()
    target = (root / rel_path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    return target if target.is_file() else None


def probe_duration(path: Path) -> Optional[float]:
    """Video duration via ffprobe if installed; None otherwise (it is only a warning aid)."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
        return float(json.loads(out)["format"]["duration"])
    except (subprocess.SubprocessError, KeyError, ValueError):
        return None


def safe_extract(zip_path: Path, dest: Path, prefix: str = "artifacts/") -> int:
    """Extract only entries under `prefix`, stripping it. Rejects path traversal and
    enforces a total-size cap on bytes actually written (zip headers can lie)."""
    written = 0
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir() or not name.startswith(prefix):
                continue
            rel = PurePosixPath(name[len(prefix):])
            if rel.is_absolute() or ".." in rel.parts or not rel.parts:
                raise ValueError(f"Unsafe path in archive: {info.filename}")
            out = dest.joinpath(*rel.parts)
            out.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, open(out, "wb") as dst:
                while chunk := src.read(1 << 20):
                    written += len(chunk)
                    if written > config.MAX_EXTRACT_BYTES:
                        raise ValueError("Archive exceeds the extraction size limit")
                    dst.write(chunk)
    return written


def assemble_and_extract(project_id: int, task_id: int, parts: int) -> int:
    """Join uploaded parts, extract artifacts/ atomically (swap in only on success)."""
    tmp = upload_tmp_dir(project_id, task_id)
    root = project_dir(project_id)
    zip_path = tmp / "delivery.zip"
    with open(zip_path, "wb") as joined:
        for index in range(parts):
            with open(tmp / f"part-{index:05d}", "rb") as part:
                shutil.copyfileobj(part, joined, 1 << 20)

    staging = root / "artifacts.new"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        written = safe_extract(zip_path, staging)
        if written == 0:
            raise ValueError("Archive contains no artifacts/ entries")
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    final = root / "artifacts"
    shutil.rmtree(final, ignore_errors=True)
    staging.rename(final)

    if config.KEEP_DELIVERY_ZIP:
        (root / "delivery").mkdir(exist_ok=True)
        shutil.move(str(zip_path), root / "delivery" / "Person_A_backend_delivery.zip")
    shutil.rmtree(tmp, ignore_errors=True)
    return written


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# Week 2 contract: where each reusable artifact lives, relative to the project folder.
def handoff_index(project_id: int, run_name: str) -> dict:
    paths = {
        "frames_full": "artifacts/full_reconstruction/images",
        "frames_undistorted": "artifacts/3dgs_dataset/images",
        "intrinsics_pinhole": "artifacts/3dgs_dataset/sparse_txt/cameras.txt",
        "poses_world_to_camera": "artifacts/3dgs_dataset/sparse_txt/images.txt",
        "points3d": "artifacts/3dgs_dataset/sparse_txt/points3D.txt",
        "sparse_binary": "artifacts/3dgs_dataset/sparse/0",
        "vggt_poses": "artifacts/full_reconstruction/vggt/vggt_poses.npz",
        "scene_ply": f"artifacts/exports/{run_name}/scene.ply",
        "scene_splat": f"artifacts/exports/{run_name}/scene.splat",
        "inference_weights": f"artifacts/3dgs_training/{run_name}/inference_weights.pt",
        "train_split": f"artifacts/3dgs_training/{run_name}/split.json",
        "handoff_doc": "artifacts/reports/week2_handoff.md",
    }
    root = project_dir(project_id)
    return {key: {"path": rel, "exists": (root / rel).exists()} for key, rel in paths.items()}
