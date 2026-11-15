"""Synthetic room used by GPU_MOCK=1 and by the tests: a point-cloud room with a few coloured objects,
a camera that spins around so each object enters and leaves the view, rendered with a z-buffer.
Writes the same COLMAP-txt + .splat layout the real pipeline produces."""
import zipfile
from pathlib import Path

import cv2
import os
import numpy as np

from ..geometry import Camera, project, rot_to_quat

W, H, FOCAL = 640, 360, 420.0
OBJECTS = {  # name: (centre xz angle deg, dist, size (x,y,z), colour rgb), y is DOWN (floor at +1.3)
    "sofa": (0, 2.2, (1.6, 0.7, 0.8), (150, 105, 70)),
    "table": (90, 2.0, (1.0, 0.5, 1.0), (60, 110, 170)),
    "lamp": (180, 2.2, (0.35, 1.0, 0.35), (230, 200, 70)),
    "plant": (270, 2.1, (0.5, 0.9, 0.5), (60, 150, 80)),
    "chair": (45, 2.4, (0.6, 0.8, 0.6), (170, 70, 90)),
}


def _box(rng, centre, size, color, n):
    cx, cy, cz = centre
    sx, sy, sz = size
    pts = []
    faces = [(0, 1, sy * sz), (0, -1, sy * sz), (1, -1, sx * sz), (2, 1, sx * sy), (2, -1, sx * sy)]
    total = sum(a for _, _, a in faces)
    for axis, sign, area in faces:
        k = int(n * area / total)
        p = rng.uniform(-0.5, 0.5, (k, 3)) * np.array(size)
        p[:, axis] = sign * 0.5 * size[axis]
        pts.append(p + np.array(centre))
    pts = np.concatenate(pts)
    col = np.clip(np.array(color)[None] + rng.normal(0, 4, (len(pts), 3)), 0, 255)
    return pts, col


def build_scene(seed=7, density=1.0):
    rng = np.random.default_rng(seed)
    P, C, S = [], [], []
    n = int(30000 * density)
    for axis, val, color in ((1, 1.3, (176, 160, 140)), (1, -1.3, (235, 235, 235)), (0, -3, (205, 195, 180)), (0, 3, (190, 200, 205)), (2, -3, (200, 185, 190)), (2, 3, (195, 205, 190))):
        p = rng.uniform(-3, 3, (n, 3))
        p[:, 1] = rng.uniform(-1.3, 1.3, n)
        p[:, axis] = val
        P.append(p)
        S.append(np.full(len(p), 0.07 / np.sqrt(density)))
        base = np.array(color, float)
        col = base[None] + rng.normal(0, 5, (n, 3))
        if axis == 1 and val > 0:  # checker floor so the floor is not one flat colour
            chk = ((np.floor(p[:, 0] * 2) + np.floor(p[:, 2] * 2)) % 2)[:, None] * 22
            col = col - chk
        C.append(np.clip(col, 0, 255))
    for ang, dist, size, color in OBJECTS.values():
        a = np.deg2rad(ang)
        cx, cz = dist * np.sin(a), dist * np.cos(a)
        p, c = _box(rng, (cx, 1.3 - size[1] / 2, cz), size, color, int(14000 * density))
        P.append(p)
        C.append(c)
        S.append(np.full(len(p), 0.03 / np.sqrt(density)))
    return np.concatenate(P), np.concatenate(C).astype(np.uint8), np.concatenate(S)


def make_cameras(count=48):
    cams = []
    for i in range(count):
        yaw = 2 * np.pi * i / count
        c = np.array([0.5 * np.sin(yaw * 2) * 0.5, 0.0, 0.5 * np.cos(yaw * 2) * 0.5])
        pitch = np.deg2rad(16)  # look slightly down so floor-standing objects are in frame
        fwd = np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])
        right = np.array([np.cos(yaw), 0, -np.sin(yaw)])
        down = np.cross(fwd, right)
        R_c2w = np.stack([right, down, fwd], 1)
        w2c = np.eye(4)
        w2c[:3, :3] = R_c2w.T
        w2c[:3, 3] = -R_c2w.T @ c
        K = np.array([[FOCAL, 0, W / 2], [0, FOCAL, H / 2], [0, 0, 1.0]])
        cams.append(Camera(i, f"frame_{i:06d}.jpg", W, H, K, w2c))
    return cams


def render(points, colors, cam, radius=3):
    img = np.full((cam.height, cam.width, 3), 24, np.uint8)
    zbuf = np.full((cam.height, cam.width), np.inf, np.float32)
    uv, z = project(points, cam)
    ok = z > 0.05
    uv, z, col = uv[ok], z[ok].astype(np.float32), colors[ok]
    order = np.argsort(-z)  # far -> near, so within one pass the nearest duplicate is written last
    u, v, z, col = uv[order, 0].astype(int), uv[order, 1].astype(int), z[order], col[order]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            uu, vv = u + dx, v + dy
            m = (uu >= 0) & (uu < cam.width) & (vv >= 0) & (vv < cam.height)
            uu, vv, zz, cc = uu[m], vv[m], z[m], col[m]
            take = zz < zbuf[vv, uu]  # z-test across passes
            zbuf[vv[take], uu[take]] = zz[take]
            img[vv[take], uu[take]] = cc[take]
    return img


def object_mask_truth(name, cam, _unused=None):
    """Ground-truth silhouette: convex hull of the projected object box (ignores occlusion by other objects)."""
    ang, dist, size, _ = OBJECTS[name]
    a = np.deg2rad(ang)
    c = np.array([dist * np.sin(a), 1.3 - size[1] / 2, dist * np.cos(a)])
    corners = np.array([[sx, sy, sz] for sx in (-.5, .5) for sy in (-.5, .5) for sz in (-.5, .5)]) * np.array(size) + c
    uv, z = project(corners, cam)
    mask = np.zeros((cam.height, cam.width), np.uint8)
    if (z <= 0.05).any():  # partially behind the camera: not needed for the tests
        return mask.astype(bool)
    cv2.fillConvexPoly(mask, cv2.convexHull(uv.astype(np.int32)), 1)
    return mask.astype(bool)


def write_colmap_txt(cams, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cameras.txt").write_text(f"1 PINHOLE {W} {H} {FOCAL} {FOCAL} {W / 2} {H / 2}\n")
    lines = []
    for c in cams:
        q = rot_to_quat(c.w2c[:3, :3])
        t = c.w2c[:3, 3]
        lines.append(f"{c.index + 1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {c.name}\n\n")
    (out / "images.txt").write_text("# images\n" + "".join(lines))
    (out / "points3D.txt").write_text("# points\n")


def write_splat(points, colors, path, scale=0.03):
    scale = np.asarray(scale, np.float32)
    scale = scale[:, None] if scale.ndim == 1 else scale
    dt = np.dtype([("pos", "<f4", (3,)), ("scales", "<f4", (3,)), ("color", "u1", (4,)), ("quat", "u1", (4,))])
    d = np.empty(len(points), dt)
    d["pos"], d["scales"] = points, scale
    d["color"][:, :3], d["color"][:, 3] = colors, 255
    d["quat"] = [255, 128, 128, 128]
    d.tofile(path)


def build_project(out_dir, frames=48, progress=None):
    """Create <out>/3dgs_dataset/{images,sparse_txt}, <out>/exports/mock/scene.splat; returns object ground truth."""
    out = Path(out_dir)
    pts, col, scl = build_scene(density=float(os.environ.get('GPU_MOCK_SPLAT_DENSITY', '0.3')))  # light set -> becomes the splat (keeps software WebGL usable)
    hi_pts, hi_col, _ = build_scene(seed=11, density=6)  # dense set -> only used to render solid-looking frames
    cams = make_cameras(frames)
    img_dir = out / "3dgs_dataset" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, cam in enumerate(cams):
        cv2.imwrite(str(img_dir / cam.name), cv2.cvtColor(render(hi_pts, hi_col, cam), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        if progress:
            progress(i + 1, len(cams))
    write_colmap_txt(cams, out / "3dgs_dataset" / "sparse_txt")
    (out / "exports" / "mock").mkdir(parents=True, exist_ok=True)
    write_splat(pts, col, out / "exports" / "mock" / "scene.splat", scl)
    return pts, col, cams


def zip_delivery(project_dir, run_name, zip_path):
    root = Path(project_dir)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for f in sorted((root / "3dgs_dataset").rglob("*")):
            if f.is_file():
                z.write(f, "artifacts/" + str(f.relative_to(root)))
        z.write(root / "exports" / "mock" / "scene.splat", f"artifacts/exports/{run_name}/scene.splat")
        z.writestr(f"artifacts/exports/{run_name}/scene_metadata.json", '{"mock": true, "note": "synthetic room (GPU_MOCK=1)"}')
        z.writestr("artifacts/reports/week2_handoff.md", "# Mock handoff\nSynthetic room generated by GPU_MOCK=1.\n")
        z.writestr("artifacts/reports/validation_status.json", '{"mock": true}')
