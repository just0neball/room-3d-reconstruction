"""Camera maths the website backend needs without importing the GPU server package
(a deliberately small copy of gpu_server/geometry.py: cameras listing and 3D-click -> frame picking)."""
from pathlib import Path

import numpy as np


def _quat_to_rot(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def read_cameras(sparse_txt: Path):
    """-> list of dicts sorted by image name: name,width,height,fx,fy,cx,cy,R (w2c 3x3),t."""
    intr = {}
    for line in (sparse_txt / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        v = line.split()
        p = list(map(float, v[4:]))
        if v[1] == "SIMPLE_PINHOLE":
            fx = fy = p[0]; cx, cy = p[1], p[2]
        elif v[1] == "PINHOLE":
            fx, fy, cx, cy = p
        else:
            raise ValueError(f"{v[1]} is not an undistorted pinhole model")
        intr[int(v[0])] = dict(width=int(v[2]), height=int(v[3]), fx=fx, fy=fy, cx=cx, cy=cy)
    lines = [l.strip() for l in (sparse_txt / "images.txt").read_text().splitlines() if not l.startswith("#")]
    cams = []
    for header in lines[::2]:
        v = header.split()
        if len(v) < 10:
            continue
        cams.append({**intr[int(v[8])], "name": " ".join(v[9:]), "R": _quat_to_rot([float(x) for x in v[1:5]]), "t": np.array([float(x) for x in v[5:8]])})
    cams.sort(key=lambda c: c["name"])
    for i, c in enumerate(cams):
        c["index"] = i
        c["position"] = (-c["R"].T @ c["t"]).tolist()
        c["rotation_c2w"] = c["R"].T.reshape(-1).tolist()
    return cams


def project(point, cam):
    pc = cam["R"] @ np.asarray(point, float) + cam["t"]
    if pc[2] <= 1e-6:
        return None
    return cam["fx"] * pc[0] / pc[2] + cam["cx"], cam["fy"] * pc[1] / pc[2] + cam["cy"], float(pc[2])


def rank_views(point, cams):
    """Frames in which `point` lies in front of the camera and inside the image, best (central, close) first."""
    scored = []
    for cam in cams:
        r = project(point, cam)
        if r is None:
            continue
        u, v, z = r
        if not (0 <= u < cam["width"] and 0 <= v < cam["height"]):
            continue
        centre = float(np.hypot(u / cam["width"] - 0.5, v / cam["height"] - 0.5))
        scored.append((centre + 0.02 * z, cam["index"], cam["name"], float(u), float(v)))
    scored.sort()
    return [{"frame": n, "index": i, "x": u, "y": v} for _, i, n, u, v in scored]
