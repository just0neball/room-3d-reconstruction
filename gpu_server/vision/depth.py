"""Per-frame depth for lifting masks to 3D and for occlusion tests.

  GsplatDepth  expected depth rendered from the trained Gaussians with gsplat (GPU, accurate)
  SplatDepth   z-buffer of Gaussian centres (CPU; used in mock mode and as a fallback)
Both return float32 HxW arrays in camera units, 0 where there is no surface.
"""
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..geometry import project


def read_splat(path):
    """.splat (32 B/Gaussian) -> positions (N,3), scales (N,3), rgba (N,4) uint8."""
    dt = np.dtype([("pos", "<f4", (3,)), ("scales", "<f4", (3,)), ("color", "u1", (4,)), ("quat", "u1", (4,))])
    data = np.fromfile(path, dtype=dt)
    return data["pos"].astype(np.float64), data["scales"].astype(np.float64), data["color"]


def zbuffer(points, cam, radius=2):
    depth = np.full((cam.height, cam.width), np.inf, np.float32)
    uv, z = project(points, cam)
    ok = z > 0.05
    u, v, z = uv[ok, 0].astype(int), uv[ok, 1].astype(int), z[ok].astype(np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            uu, vv = u + dx, v + dy
            m = (uu >= 0) & (uu < cam.width) & (vv >= 0) & (vv < cam.height)
            np.minimum.at(depth, (vv[m], uu[m]), z[m])
    depth[~np.isfinite(depth)] = 0
    return depth


class _Cached:
    def __init__(self, cams, size=64):
        self.cams, self.cache, self.size = cams, OrderedDict(), size

    def __call__(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        d = self.render(self.cams[index])
        self.cache[index] = d
        if len(self.cache) > self.size:
            self.cache.popitem(last=False)
        return d

    def render(self, cam):
        raise NotImplementedError


class SplatDepth(_Cached):
    def __init__(self, cams, points, radius=4):
        super().__init__(cams)
        self.points, self.radius = points, radius

    def render(self, cam):
        return zbuffer(self.points, cam, self.radius)


class GsplatDepth(_Cached):
    """Expected depth ('ED') from gsplat for the trained inference weights."""

    def __init__(self, cams, weights_path, device):
        super().__init__(cams, size=32)
        import torch
        self.torch, self.device = torch, device
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.sh_degree = int(state["sh_degree"])
        self.splats = {k: v.detach().to(device).float() for k, v in state["splats"].items()}

    def render(self, cam):
        import torch
        import torch.nn.functional as F
        from gsplat.rendering import rasterization
        s = self.splats
        with torch.no_grad():
            out, alphas, _ = rasterization(
                means=s["means"], quats=F.normalize(s["quats"], dim=-1), scales=torch.exp(s["scales"].clamp(-12, 3)),
                opacities=torch.sigmoid(s["opacities"]), colors=torch.cat([s["sh0"], s["shN"]], dim=1), sh_degree=self.sh_degree,
                viewmats=torch.tensor(cam.w2c, dtype=torch.float32, device=self.device)[None],
                Ks=torch.tensor(cam.K, dtype=torch.float32, device=self.device)[None],
                width=cam.width, height=cam.height, render_mode="RGB+ED",
            )
            depth = out[0, ..., 3]
            depth = torch.where(alphas[0, ..., 0] > 0.5, depth, torch.zeros_like(depth))
            return depth.float().cpu().numpy()

    def free(self):
        self.splats = {}
        self.torch.cuda.empty_cache()


def make_depth_source(dataset_dir, cams, mock, device):
    dataset_dir = Path(dataset_dir)
    weights = dataset_dir / "inference_weights.pt"
    if not mock and weights.is_file():
        try:
            return GsplatDepth(cams, weights, device)
        except Exception as exc:  # gsplat missing / CPU only -> fall back to the z-buffer
            print(f"[depth] gsplat unavailable ({exc}); using splat z-buffer", flush=True)
    if weights.is_file() and not (dataset_dir / "scene.splat").is_file():
        import torch
        pts = torch.load(weights, map_location="cpu", weights_only=False)["splats"]["means"].numpy().astype(np.float64)
    else:
        pts, _, _ = read_splat(dataset_dir / "scene.splat")
    return SplatDepth(cams, pts)
