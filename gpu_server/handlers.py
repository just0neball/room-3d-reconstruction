"""Job handlers: reconstruct | track | edit_reference | edit_rest | rebuild.
Each takes (ctx, params) and writes its deliverables into ctx.out (zipped by the job manager)."""
import csv
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import datasets, models, settings
from .editing import compose
from .editing.editors import build_prompt
from .rebuild import rebuild as rb
from .vision import tracking


def stem(name):
    return Path(name).stem


def read_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 127


def write_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def write_rgb(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def log_metrics(kind, metrics):
    f = settings.DATA_DIR / "metrics.csv"
    new = not f.exists()
    f.parent.mkdir(parents=True, exist_ok=True)
    with f.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "kind", "json"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), kind, json.dumps(metrics)])


# ------------------------------------------------------------------ reconstruct
def reconstruct(ctx, p):
    project = ctx.dir / "project"
    project.mkdir(exist_ok=True)
    run = p.get("run_name", "stable_7000")
    version = f"{run}-{int(time.time())}"
    if settings.MOCK:
        from .mock import synthetic
        ctx.update("extracting_frames", 5)
        time.sleep(1)
        synthetic.build_project(project, frames=40, progress=lambda a, b: ctx.update("extracting_frames", 5 + 20 * a // b))
        for stage, lo, hi in (("estimating_poses", 25, 60), ("training_gaussians", 65, 90), ("exporting", 91, 96)):
            for k in range(4):
                ctx.update(stage, lo + (hi - lo) * k // 4)
                time.sleep(0.4)
        zpath = ctx.dir / "delivery.zip"
        synthetic.zip_delivery(project, run, zpath)
        # the mock keeps the splat under exports/mock; register_dir globs exports/*/scene.splat
        ctx.artifact = zpath
        datasets.register_dir(p["dataset"], project, version)
        return {"dataset_version": version, "mock": True, "frames": 40}

    video = project / "input.mp4"
    video.symlink_to(ctx.upload_path("video"))
    cmd = [sys.executable, str(settings.RECON_DIR / "run_pipeline.py"), "--project_dir", str(project), "--video", str(video), "--stage", "production",
           "--steps", str(p.get("steps", 7000)), "--run_name", run, "--fps", str(max(2, p.get("frame_fps", 1))), "--full_fps", str(p.get("frame_fps", 1))]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail, stop = [], threading.Event()

    def pump_progress():  # the Week-1 scripts append every progress event to progress_events.jsonl
        seen = 0
        events = project / "progress_events.jsonl"
        while not stop.wait(2):
            if events.is_file():
                lines = events.read_text().splitlines()
                for line in lines[seen:]:
                    try:
                        e = json.loads(line)
                        if e.get("status") != "failed":
                            ctx.job.stage, ctx.job.progress = e["stage"], max(ctx.job.progress, min(int(e["progress"]), 99))
                    except ValueError:
                        pass
                seen = len(lines)

    t = threading.Thread(target=pump_progress, daemon=True)
    t.start()
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail = (tail + [line])[-40:]
        if ctx.cancelled():
            proc.terminate()
    code = proc.wait()
    stop.set()
    if ctx.cancelled():
        ctx.update()
    if code != 0:
        raise RuntimeError(f"pipeline exited with code {code}:\n" + "".join(tail))
    zpath = project / "Person_A_backend_delivery.zip"
    if not zpath.is_file():
        raise RuntimeError("pipeline finished but Person_A_backend_delivery.zip is missing")
    ctx.artifact = zpath
    datasets.register_dir(p["dataset"], project, version)
    return {"dataset_version": version}


# ------------------------------------------------------------------ track
def track(ctx, p):
    ds = datasets.get(p["dataset"])
    if models.single_gpu():
        models.free_diffusion()
    ctx.update("loading_segmenter", 2)
    seg = models.segmenter()
    depth = ds.depth_source(settings.MOCK, settings.devices()[1])
    bundle = ctx.unzip("bundle")
    ref_idx = ds.index[p["ref_frame"]]
    ref_mask = read_mask(bundle / "masks" / f"{stem(p['ref_frame'])}.png")
    if ref_mask is None or not ref_mask.any():
        raise RuntimeError("reference mask missing or empty")
    existing = []
    for i, name in enumerate(ds.names):
        m = read_mask(bundle / "masks" / f"{stem(name)}.png")
        if m is not None:
            existing.append(tracking.FrameResult(i, name, True, "auto", None, float(m.mean()), None, m))
    frames = [ds.index[n] for n in p.get("frames", [])]
    user_points = {ds.index[n]: pts for n, pts in p.get("user_points", {}).items()}
    ref_points = [(q["x"], q["y"], q["label"]) for q in p.get("ref_points", [])] or [(float(x), float(y), 1) for x, y in [tracking._inner_point(ref_mask)]]
    ctx.update("tracking_frames", 5)
    t0 = time.perf_counter()
    results, info = tracking.run_tracking(
        ds, depth, seg, p.get("method", "auto"), ref_idx, ref_mask, ref_points, frames, user_points, existing,
        lambda a, b: ctx.update("tracking_frames", 5 + int(90 * a / max(b, 1)), f"frame {a} of {b}"))
    out = []
    for r in results:
        if r.mask is not None:
            write_mask(ctx.out / "masks" / f"{stem(r.name)}.png", r.mask)
        out.append({"name": r.name, "index": r.index, "included": r.included, "source": r.source, "score": r.score, "area": r.area, "visible_points": r.visible_points})
    peak = None
    if not settings.MOCK:
        import torch
        peak = round(torch.cuda.max_memory_allocated(settings.devices()[1]) / 1024 ** 3, 3)
    metrics = {**info, "segmenter": seg.name, "peak_vram_gb": peak, "total_seconds": round(time.perf_counter() - t0, 2)}
    (ctx.out / "tracking.json").write_text(json.dumps({"frames": out, "info": metrics}))
    log_metrics("track", metrics)
    return metrics


# ------------------------------------------------------------------ edit
def _prepare_editor(ctx, p):
    quant = p.get("quant") or settings.DEFAULT_QUANT
    if models.single_gpu():
        models.free_segmenter()
    ctx.update("loading_diffusion", 3)
    ed = models.editor(quant, free_text_encoder=bool(p.get("free_text_encoder")))
    t0 = time.perf_counter()
    ed.load()
    if hasattr(ed, "reset_peak"):
        ed.reset_peak()
    return ed, round(time.perf_counter() - t0, 2)


def _edit_one(ed, ds, i, mask, prompt, seed, ref=None):
    """One frame: crop around its mask, edit (optionally conditioned on the reference crop), composite back."""
    img = ds.image(i)
    crop = compose.crop_box(mask, long_side=settings.EDIT_LONG_SIDE)
    imgs = [compose.cut(img, crop)]
    masks = [compose.cut(mask.astype(np.uint8) * 255, crop) > 127]
    if ref is not None:
        imgs.append(ref["img"])
        masks.append(ref["mask"])
    t0 = time.perf_counter()
    edited = ed.edit(imgs, prompt, seed, masks)
    seconds = time.perf_counter() - t0
    full = compose.composite(img, edited, mask, crop)
    return full, seconds, compose.outside_mask_max_diff(img, full, mask)


def edit_reference(ctx, p):
    ds = datasets.get(p["dataset"])
    ed, load_s = _prepare_editor(ctx, p)
    bundle = ctx.unzip("bundle")
    i = ds.index[p["frame"]]
    mask = read_mask(bundle / "masks" / f"{stem(p['frame'])}.png")
    if mask is None or not mask.any():
        raise RuntimeError("mask missing for the reference frame")
    if hasattr(ed, "precompute_prompts"):
        ed.precompute_prompts([build_prompt(p["prompt"], False)])
    seeds = p.get("seeds_list") or [int(p.get("seed_offset", 0)) + k + 1 for k in range(int(p.get("seeds", 4)))]
    times, worst = [], 0
    for k, seed in enumerate(seeds):
        ctx.update("editing_reference", 8 + int(90 * k / len(seeds)), f"seed {k + 1} of {len(seeds)}")
        full, sec, diff = _edit_one(ed, ds, i, mask, p["prompt"], seed)
        write_rgb(ctx.out / "candidates" / f"candidate_{seed}.png", full)
        ctx.mark_partial(f"candidates/candidate_{seed}.png")
        times.append(round(sec, 2))
        worst = max(worst, diff)
    metrics = {"model": ed.name, "quant": ed.quant, "peak_vram_gb": ed.peak_vram_gb(), "load_seconds": load_s, "sec_per_frame": round(float(np.mean(times)), 2),
               "per_frame_seconds": times, "seeds": seeds, "outside_mask_max_diff": worst, "crop_long_side": settings.EDIT_LONG_SIDE}
    log_metrics("edit_reference", metrics)
    return metrics


def edit_rest(ctx, p):
    ds = datasets.get(p["dataset"])
    ed, load_s = _prepare_editor(ctx, p)
    bundle = ctx.unzip("bundle")
    ref_name = p["reference_frame"]
    ref_full = cv2.cvtColor(cv2.imread(str(bundle / "reference.png")), cv2.COLOR_BGR2RGB)
    ref_mask = read_mask(bundle / "masks" / f"{stem(ref_name)}.png")
    rc = compose.crop_box(ref_mask, long_side=settings.EDIT_LONG_SIDE)
    ref = {"img": compose.cut(ref_full, rc), "mask": compose.cut(ref_mask.astype(np.uint8) * 255, rc) > 127}
    if hasattr(ed, "precompute_prompts"):
        ed.precompute_prompts([build_prompt(p["prompt"], True)])
    frames = [n for n in p["frames"] if n != ref_name]
    write_rgb(ctx.out / "edited" / f"{stem(ref_name)}.png", ref_full)
    times, worst = [], 0
    for k, name in enumerate(frames):
        ctx.update("editing_frames", 5 + int(93 * k / max(len(frames), 1)), f"frame {k + 1} of {len(frames)}")
        mask = read_mask(bundle / "masks" / f"{stem(name)}.png")
        if mask is None or not mask.any():
            continue
        full, sec, diff = _edit_one(ed, ds, ds.index[name], mask, p["prompt"], int(p.get("seed", 0)), ref)
        write_rgb(ctx.out / "edited" / f"{stem(name)}.png", full)
        ctx.mark_partial(f"edited/{stem(name)}.png")
        times.append(round(sec, 2))
        worst = max(worst, diff)
    ctx.update("editing_frames", 98, f"frame {len(frames)} of {len(frames)}")
    metrics = {"model": ed.name, "quant": ed.quant, "peak_vram_gb": ed.peak_vram_gb(), "load_seconds": load_s,
               "sec_per_frame": round(float(np.mean(times)), 2) if times else None, "frames": len(times),
               "outside_mask_max_diff": worst, "seed": int(p.get("seed", 0)), "crop_long_side": settings.EDIT_LONG_SIDE}
    log_metrics("edit_rest", metrics)
    return metrics


# ------------------------------------------------------------------ rebuild
def rebuild(ctx, p):
    ds = datasets.get(p["dataset"])
    if models.single_gpu():
        models.free_diffusion()
    bundle = ctx.unzip("bundle")
    work = ctx.dir / "rb"
    run = "edit"
    edited, masks = {}, {}
    for f in sorted((bundle / "edited").glob("*.png")):
        name = f.stem + ".jpg"
        if name in ds.index:
            edited[name] = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            m = read_mask(bundle / "masks" / f"{f.stem}.png")
            masks[name] = m if m is not None else np.zeros(edited[name].shape[:2], bool)
    if not edited:
        raise RuntimeError("no edited frames were provided")
    mode, steps = p.get("mode", "finetune"), int(p.get("steps", 3000))
    ctx.update("preparing_dataset", 3, f"{len(edited)} edited frames")
    edit_ds = work / "dataset"
    rb.build_edit_dataset(ds.dir, bundle / "edited", edit_ds)
    prog = lambda a, b: ctx.update("rebuilding_splat", 8 + int(85 * a / max(b, 1)), f"step {a} of {b}")  # noqa: E731
    if settings.MOCK:
        metrics = rb.rebuild_mock(ds, edited, masks, work, run, prog)
    elif mode == "scratch":
        metrics = rb.rebuild_scratch(ds, edit_ds, work, run, steps, prog, ctx.cancelled)
    else:
        metrics = rb.rebuild_finetune(ds, edit_ds, masks, work, run, steps, settings.devices()[1], prog, ctx.cancelled)
    ctx.update("exporting", 96)
    exp = work / "exports" / run
    for name in ("scene.splat", "scene.ply"):
        if (exp / name).is_file():
            (ctx.out / name).write_bytes((exp / name).read_bytes())
    if not (ctx.out / "scene.splat").is_file():
        raise RuntimeError("rebuild finished without scene.splat")
    metrics["edited_frames"] = len(edited)
    log_metrics("rebuild", metrics)
    return metrics


HANDLERS = {"reconstruct": reconstruct, "track": track, "edit_reference": edit_reference, "edit_rest": edit_rest, "rebuild": rebuild}
