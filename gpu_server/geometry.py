"""Camera geometry shared by tracking, lifting and the mock renderer.

Conventions (docs/reports/week2_handoff.md): COLMAP, world->camera, +X right, +Y down, +Z forward.
Pixel coordinates are those of the *undistorted* pinhole images, origin top-left.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Camera:
    index: int
    name: str
    width: int
    height: int
    K: np.ndarray  # 3x3
    w2c: np.ndarray  # 4x4

    @property
    def center(self):
        return -self.w2c[:3, :3].T @ self.w2c[:3, 3]

    @property
    def c2w(self):
        m = np.eye(4)
        m[:3, :3] = self.w2c[:3, :3].T
        m[:3, 3] = self.center
        return m


def quat_to_rot(q):
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
        v = [0.0] * 3
        v[i] = 0.25 * s
        v[j] = (R[j, i] + R[i, j]) / s
        v[k] = (R[k, i] + R[i, k]) / s
        q = [(R[k, j] - R[j, k]) / s, *v]
    q = np.array(q)
    return q / np.linalg.norm(q)


def read_cameras(dataset_dir):
    """Parse sparse_txt/{cameras,images}.txt (PINHOLE / SIMPLE_PINHOLE only) -> list[Camera] sorted by name."""
    txt = Path(dataset_dir) / "sparse_txt"
    intr = {}
    for line in (txt / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        v = line.split()
        w, h = int(v[2]), int(v[3])
        p = list(map(float, v[4:]))
        if v[1] == "SIMPLE_PINHOLE":
            fx = fy = p[0]; cx, cy = p[1], p[2]
        elif v[1] == "PINHOLE":
            fx, fy, cx, cy = p
        else:
            raise ValueError(f"Camera model {v[1]} is not undistorted pinhole")
        intr[int(v[0])] = (w, h, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]))
    # keep blank lines: the POINTS2D line after each pose is empty in undistorted models
    lines = [l.strip() for l in (txt / "images.txt").read_text().splitlines() if not l.startswith("#")]
    cams = []
    for header in lines[::2]:
        v = header.split()
        w, h, K = intr[int(v[8])]
        w2c = np.eye(4)
        w2c[:3, :3] = quat_to_rot([float(x) for x in v[1:5]])
        w2c[:3, 3] = [float(x) for x in v[5:8]]
        cams.append(Camera(0, " ".join(v[9:]), w, h, K, w2c))
    cams.sort(key=lambda c: c.name)
    for i, c in enumerate(cams):
        c.index = i
    return cams


def project(points, cam):
    """World points (N,3) -> (uv (N,2), z (N,)). z<=0 means behind the camera."""
    pc = points @ cam.w2c[:3, :3].T + cam.w2c[:3, 3]
    z = pc[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = cam.K[0, 0] * pc[:, 0] / safe + cam.K[0, 2]
    v = cam.K[1, 1] * pc[:, 1] / safe + cam.K[1, 2]
    return np.stack([u, v], 1), z


def backproject(mask, depth, cam, max_points=6000, rng=None):
    """Masked pixels with valid depth -> world points (N,3), robust to mask-edge depth leaks."""
    ys, xs = np.nonzero(mask & (depth > 0))
    if len(xs) == 0:
        return np.zeros((0, 3))
    z = depth[ys, xs].astype(np.float64)
    med = np.median(z)
    q1, q3 = np.percentile(z, [25, 75])
    spread = max(q3 - q1, 0.05 * med)
    keep = np.abs(z - med) <= 2.5 * spread  # drops background pixels that bled into the mask
    xs, ys, z = xs[keep], ys[keep], z[keep]
    if len(xs) > max_points:
        sel = (rng or np.random.default_rng(0)).choice(len(xs), max_points, replace=False)
        xs, ys, z = xs[sel], ys[sel], z[sel]
    x = (xs + 0.5 - cam.K[0, 2]) / cam.K[0, 0] * z
    y = (ys + 0.5 - cam.K[1, 2]) / cam.K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    return (pc - cam.w2c[:3, 3]) @ cam.w2c[:3, :3]  # R^T (pc - t)


def visible_points(points, cam, depth=None, margin=0, rel_tol=0.06, abs_tol=0.05):
    """Frustum + occlusion test. Returns (uv, z, visible_mask over input points).
    Occluded = the camera's depth map shows a surface clearly in front of the point."""
    uv, z = project(points, cam)
    inside = (z > 1e-3) & (uv[:, 0] >= -margin) & (uv[:, 0] < cam.width + margin) & (uv[:, 1] >= -margin) & (uv[:, 1] < cam.height + margin)
    vis = inside.copy()
    if depth is not None and inside.any():
        xi = np.clip(uv[:, 0].astype(int), 0, cam.width - 1)
        yi = np.clip(uv[:, 1].astype(int), 0, cam.height - 1)
        d = depth[yi, xi]
        # (patch-min over 3x3 would be more tolerant; depth edges are noisy, so allow rel_tol)
        occluded = (d > 0) & (z > d * (1 + rel_tol) + abs_tol)
        vis &= ~occluded
    return uv, z, vis


def best_view_for_point(point, cams, depth_fn=None):
    """Rank cameras by how well they see a 3D point: in front, inside, near the centre, close."""
    scored = []
    for cam in cams:
        uv, z = project(point[None], cam)
        u, v, zz = uv[0, 0], uv[0, 1], z[0]
        if zz <= 0.05 or not (0 <= u < cam.width and 0 <= v < cam.height):
            continue
        if depth_fn is not None:
            d = depth_fn(cam.index)
            if d is not None:
                dd = d[int(v), int(u)]
                if dd > 0 and zz > dd * 1.08 + 0.05:
                    continue
        centre = np.hypot((u / cam.width - 0.5), (v / cam.height - 0.5))
        scored.append((centre + 0.02 * zz, cam.index, float(u), float(v)))
    scored.sort()
    return scored
