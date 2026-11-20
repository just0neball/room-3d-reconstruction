"""Datasets cached on the GPU server: undistorted frames + COLMAP txt + Gaussian weights (or .splat in mock mode).
The backend is the source of truth; a Kaggle session can vanish, so the backend re-uploads on demand."""
import json
import shutil
import threading
import zipfile
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from . import settings
from .geometry import read_cameras

_lock = threading.Lock()
_loaded = {}


def root(key):
    return settings.DATA_DIR / "datasets" / key


def info(key):
    meta = root(key) / "meta.json"
    return json.loads(meta.read_text()) if meta.is_file() else None


def extract(key, zip_path, version):
    dest = root(key)
    tmp = dest.with_name(dest.name + ".new")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            p = PurePosixPath(member.filename)
            if member.is_dir() or p.is_absolute() or ".." in p.parts:
                continue
            out = tmp.joinpath(*p.parts)
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
    cams = read_cameras(tmp)
    (tmp / "meta.json").write_text(json.dumps({"key": key, "version": version, "frames": len(cams), "width": cams[0].width, "height": cams[0].height}))
    with _lock:
        _loaded.pop(key, None)
        shutil.rmtree(dest, ignore_errors=True)
        tmp.rename(dest)
    return info(key)


def register_dir(key, src_dir, version):
    """Adopt a freshly reconstructed project (images/ sparse_txt/ weights/splat) as a cached dataset."""
    dest = root(key)
    tmp = dest.with_name(dest.name + ".new")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    src = Path(src_dir)
    shutil.copytree(src / "3dgs_dataset" / "images", tmp / "images")
    shutil.copytree(src / "3dgs_dataset" / "sparse_txt", tmp / "sparse_txt")
    for rel, name in (("3dgs_training", None), ):
        pass
    for w in sorted(src.glob("3dgs_training/*/inference_weights.pt")):
        shutil.copy2(w, tmp / "inference_weights.pt")
    for s in sorted(src.glob("exports/*/scene.splat")):
        shutil.copy2(s, tmp / "scene.splat")
    cams = read_cameras(tmp)
    (tmp / "meta.json").write_text(json.dumps({"key": key, "version": version, "frames": len(cams), "width": cams[0].width, "height": cams[0].height}))
    with _lock:
        _loaded.pop(key, None)
        shutil.rmtree(dest, ignore_errors=True)
        tmp.rename(dest)


class Dataset:
    def __init__(self, key):
        self.key = key
        self.dir = root(key)
        if not (self.dir / "meta.json").is_file():
            raise FileNotFoundError(f"dataset '{key}' is not on this GPU server (upload it first)")
        self.cams = read_cameras(self.dir)
        self.names = [c.name for c in self.cams]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.width, self.height = self.cams[0].width, self.cams[0].height
        self._depth = None

    def image(self, i):
        """RGB uint8 HxWx3."""
        img = cv2.imread(str(self.dir / "images" / self.names[i]), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def depth_source(self, mock, device):
        if self._depth is None:
            from .vision.depth import make_depth_source
            self._depth = make_depth_source(self.dir, self.cams, mock, device)
        return self._depth


def get(key):
    with _lock:
        ds = _loaded.get(key)
        if ds is None:
            ds = _loaded[key] = Dataset(key)
        return ds
