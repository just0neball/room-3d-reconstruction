"""Rebuild the splat from edited frames. Poses and intrinsics are NOT re-estimated.

  finetune  start from the existing Gaussians (inference_weights.pt) and optimise a few thousand steps on the
            edited images (colour-focused LRs, no densification). Fast; the first optimisation to try.
  scratch   naive version: re-run the Week-1 trainer (reconstruction/train.py) on the edited image set.
  mock      CPU: re-colour Gaussians seen inside the edit masks from the edited pixels (GPU_MOCK=1)

Every function writes <out>/exports/<run>/scene.{splat,ply} and returns a metrics dict.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .. import settings
from ..geometry import project, visible_points
from ..vision.depth import read_splat


def build_edit_dataset(ds_dir, edited_dir, out_ds):
    """<out_ds>/images = originals with the edited frames swapped in (same file names), + sparse_txt."""
    ds_dir, edited_dir, out_ds = Path(ds_dir), Path(edited_dir), Path(out_ds)
    shutil.rmtree(out_ds, ignore_errors=True)
    (out_ds / "images").mkdir(parents=True)
    shutil.copytree(ds_dir / "sparse_txt", out_ds / "sparse_txt")
    swapped = 0
    for img in sorted((ds_dir / "images").iterdir()):
        edited = edited_dir / (img.stem + ".png")
        if edited.is_file():
            im = cv2.imread(str(edited), cv2.IMREAD_COLOR)
            cv2.imwrite(str(out_ds / "images" / img.name), im, [cv2.IMWRITE_JPEG_QUALITY, 98])
            swapped += 1
        else:
            shutil.copy2(img, out_ds / "images" / img.name)
    return swapped


def rebuild_mock(ds, edited, masks, out_dir, run, progress):
    """edited: {frame name: RGB array}, masks: {frame name: bool array}."""
    t0 = time.perf_counter()
    pts, scales, rgba = read_splat(ds.dir / "scene.splat")
    acc, cnt = np.zeros((len(pts), 3)), np.zeros(len(pts))
    depth = ds.depth_source(True, "cpu")
    names = list(edited)
    for n, name in enumerate(names):
        progress(n, len(names))
        i = ds.index[name]
        cam = ds.cams[i]
        uv, z, vis = visible_points(pts, cam, depth(i), rel_tol=0.03, abs_tol=0.03)
        u, v = uv[:, 0].astype(int), uv[:, 1].astype(int)
        inside = vis & masks[name][np.clip(v, 0, cam.height - 1), np.clip(u, 0, cam.width - 1)]
        acc[inside] += edited[name][v[inside], u[inside]]
        cnt[inside] += 1
    hit = cnt >= 2
    new = rgba.copy()
    new[hit, :3] = np.clip(acc[hit] / cnt[hit, None], 0, 255).astype(np.uint8)
    out = Path(out_dir) / "exports" / run
    out.mkdir(parents=True, exist_ok=True)
    from ..mock.synthetic import write_splat
    write_splat(pts, new[:, :3], out / "scene.splat", scales)
    progress(len(names), len(names))
    return {"mode": "mock", "gaussians": int(len(pts)), "gaussians_recoloured": int(hit.sum()), "build_seconds": round(time.perf_counter() - t0, 2)}


def rebuild_finetune(ds, edited_ds_dir, masks, out_dir, run, steps, device, progress, cancelled):
    import torch
    import torch.nn.functional as F
    from gsplat.rendering import rasterization

    t0 = time.perf_counter()
    state = torch.load(ds.dir / "inference_weights.pt", map_location="cpu", weights_only=False)
    sh_degree = int(state["sh_degree"])
    splats = {k: torch.nn.Parameter(v.detach().to(device).float()) for k, v in state["splats"].items()}
    cams = ds.cams
    vmat = torch.tensor(np.stack([c.w2c for c in cams]), dtype=torch.float32, device=device)
    K = torch.tensor(np.stack([c.K for c in cams]), dtype=torch.float32, device=device)
    centers = torch.linalg.inv(vmat)[:, :3, 3]
    scene_scale = max(float(torch.linalg.norm(centers - centers.mean(0), dim=-1).max()), 0.1)
    # colour must move a lot (that is the edit); geometry only needs to settle -> lower LRs than the Week-1 trainer
    base = {"means": 4e-5 * scene_scale, "quats": 2.5e-4, "scales": 1.25e-3, "opacities": 1.25e-2, "sh0": 2.5e-3, "shN": 1.25e-4}
    opts = {k: torch.optim.Adam([{"params": [p], "lr": base[k]}], eps=1e-15) for k, p in splats.items()}
    images = [torch.from_numpy(cv2.cvtColor(cv2.imread(str(Path(edited_ds_dir) / "images" / c.name)), cv2.COLOR_BGR2RGB)) for c in cams]
    weight = {}
    for name, m in masks.items():  # 3x loss weight on (slightly grown) edited regions
        weight[ds.index[name]] = torch.from_numpy(1 + 2 * cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32))
    edited_ids = sorted(weight)
    rng = np.random.default_rng(0)
    bg = torch.tensor([0.5, 0.5, 0.5], device=device)
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, steps + 1):
        if cancelled():
            raise RuntimeError("cancelled")
        i = int(rng.choice(edited_ids)) if (edited_ids and rng.random() < 0.5) else int(rng.integers(len(cams)))
        decay = 0.1 ** (step / steps)
        for k, o in opts.items():
            o.param_groups[0]["lr"] = base[k] * decay
            o.zero_grad(set_to_none=True)
        rendered, _, _ = rasterization(
            means=splats["means"], quats=F.normalize(splats["quats"], dim=-1), scales=torch.exp(splats["scales"].clamp(-12, 3)),
            opacities=torch.sigmoid(splats["opacities"]), colors=torch.cat([splats["sh0"], splats["shN"]], dim=1), sh_degree=sh_degree,
            viewmats=vmat[i:i + 1], Ks=K[i:i + 1], width=cams[i].width, height=cams[i].height, backgrounds=bg, packed=True)
        target = images[i].to(device, torch.float32) / 255
        err = (rendered[0] - target).abs()
        if i in weight:
            err = err * weight[i].to(device)[..., None]
        loss = err.mean()
        loss.backward()
        for o in opts.values():
            o.step()
        if step % 50 == 0 or step == steps:
            progress(step, steps)
    out = Path(out_dir)
    ckpt = out / "finetuned.pt"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"step": steps, "splats": {k: v.detach().cpu() for k, v in splats.items()}, "sh_degree": sh_degree}, ckpt)
    peak = round(torch.cuda.max_memory_allocated(device) / 1024 ** 3, 3)
    _export(out, run, ckpt)
    return {"mode": "finetune", "steps": steps, "gaussians": int(len(splats["means"])), "peak_vram_gb": peak, "build_seconds": round(time.perf_counter() - t0, 1)}


def _export(out, run, ckpt):
    subprocess.run([sys.executable, str(settings.RECON_DIR / "export.py"), "--project_dir", str(out), "--run_name", run, "--checkpoint_path", str(ckpt)],
                   check=True, capture_output=True, text=True)


def rebuild_scratch(ds, edited_ds_dir, out_dir, run, steps, progress, cancelled):
    """Naive path: the Week-1 trainer on the edited image set (its own hold-out split, seeded from the sparse points)."""
    t0 = time.perf_counter()
    out = Path(out_dir)
    shutil.rmtree(out / "3dgs_dataset", ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(edited_ds_dir, out / "3dgs_dataset")
    cmd = [sys.executable, str(settings.RECON_DIR / "train.py"), "--project_dir", str(out), "--steps", str(steps), "--schedule_steps", str(max(steps, 15000)),
           "--run_name", run, "--refine_stop_iter", "0", "--ssim_weight", "0", "--scale_regularization", "0"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in proc.stdout:
        tail = (tail + [line])[-30:]
        if line.startswith("Step "):
            try:
                a = line.split()[1].split("/")
                progress(int(a[0]), int(a[1]))
            except ValueError:
                pass
        if cancelled():
            proc.terminate()
    if proc.wait() != 0:
        raise RuntimeError("train.py failed:\n" + "".join(tail))
    _export(out, run, out / "3dgs_training" / run / "latest.pt")
    meta = json.loads((out / "3dgs_training" / run / "training_metadata.json").read_text())
    return {"mode": "scratch", "steps": steps, "gaussians": meta.get("gaussians"), "peak_vram_gb": meta.get("peak_allocated_gpu_vram_gb"), "build_seconds": round(time.perf_counter() - t0, 1)}
