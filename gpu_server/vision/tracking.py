"""Find the target object in every frame.

Methods (cheapest first, see the report for the evaluation):
  neighbor  mask of frame t-1 (box + inner point) prompts SAM on frame t; re-init if IoU collapses
  lift3d    reference mask + depth -> 3D object points -> project into every frame with its pose,
            frustum + occlusion test, visible points/box prompt SAM (also acts as the frustum filter)
  video     SAM video propagation from the reference clicks (forward + backward)
  manual    the user-marked fallback: SAM on exactly the frames the user ticked, hinted by projected points
  auto      lift3d, cross-checked against video propagation when the back-end supports it
"""
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..geometry import backproject, visible_points
from .segmenters import mask_iou


@dataclass
class FrameResult:
    index: int
    name: str
    included: bool = False
    source: str = "none"  # reference | auto | manual | none
    score: float | None = None
    area: float | None = None
    visible_points: int | None = None
    mask: np.ndarray | None = field(default=None, repr=False)


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    return (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None


def _pad_box(box, w, h, frac=0.06, min_px=6):
    x0, y0, x1, y1 = box
    px = max(min_px, (x1 - x0) * frac)
    py = max(min_px, (y1 - y0) * frac)
    return (max(0, x0 - px), max(0, y0 - py), min(w - 1, x1 + px), min(h - 1, y1 + py))


def _inner_point(mask):
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    y, x = np.unravel_index(int(dist.argmax()), dist.shape)
    return float(x), float(y)


def _fps_points(uv, k=6):
    """k well-spread points from the visible projections (farthest-point sampling from the median)."""
    if len(uv) <= k:
        return uv
    start = int(np.argmin(np.linalg.norm(uv - np.median(uv, 0), axis=1)))
    chosen = [start]
    d = np.linalg.norm(uv - uv[start], axis=1)
    for _ in range(k - 1):
        nxt = int(d.argmax())
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(uv - uv[nxt], axis=1))
    return uv[chosen]


def lift_object(ds, depth_fn, ref_idx, ref_mask):
    pts = backproject(ref_mask, depth_fn(ref_idx), ds.cams[ref_idx])
    if len(pts) < 30:
        raise RuntimeError("Not enough depth under the reference mask to lift it to 3D (try a frame where the object is larger)")
    return pts


def _segment_with_points(ds, seg, i, uv_vis, pts_total):
    cam = ds.cams[i]
    box = _pad_box((uv_vis[:, 0].min(), uv_vis[:, 1].min(), uv_vis[:, 0].max(), uv_vis[:, 1].max()) if len(uv_vis) < 20 else
                   (*np.percentile(uv_vis, 2, axis=0), *np.percentile(uv_vis, 98, axis=0)), cam.width, cam.height)
    pos = [(float(x), float(y), 1) for x, y in _fps_points(uv_vis)]
    mask, sscore = seg.segment(ds.image(i), pos, box)
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    keep = np.zeros_like(mask)
    keep[y0:y1 + 1, x0:x1 + 1] = True
    mask = mask & keep  # leak guard: the object cannot extend beyond the projected cloud's box
    inside = 0.0
    if mask.any():
        xi = np.clip(uv_vis[:, 0].astype(int), 0, cam.width - 1)
        yi = np.clip(uv_vis[:, 1].astype(int), 0, cam.height - 1)
        inside = float(mask[yi, xi].mean())
    return mask, inside, sscore


def track_lift3d(ds, depth_fn, seg, ref_idx, ref_mask, progress, min_visible=40, min_frac=0.04, min_inside=0.5):
    pts = lift_object(ds, depth_fn, ref_idx, ref_mask)
    results = []
    for i, cam in enumerate(ds.cams):
        progress(i, len(ds.cams))
        r = FrameResult(i, cam.name)
        if i == ref_idx:
            r.included, r.source, r.score, r.mask = True, "reference", 1.0, ref_mask
        else:
            uv, z, vis = visible_points(pts, cam, depth_fn(i))
            r.visible_points = int(vis.sum())
            if r.visible_points >= min_visible and r.visible_points / len(pts) >= min_frac:  # frustum/occlusion filter
                mask, inside, _ = _segment_with_points(ds, seg, i, uv[vis], len(pts))
                r.score = round(inside, 3)
                if mask.any() and inside >= min_inside:
                    r.included, r.source, r.mask = True, "auto", mask
        if r.mask is not None:
            r.area = float(r.mask.mean())
        results.append(r)
    return results, {"object_points": int(len(pts))}


def track_neighbor(ds, seg, ref_idx, ref_mask, progress, min_iou=0.25, max_misses=3):
    results = [FrameResult(i, c.name) for i, c in enumerate(ds.cams)]
    results[ref_idx] = FrameResult(ref_idx, ds.names[ref_idx], True, "reference", 1.0, float(ref_mask.mean()), None, ref_mask)
    done = 0
    for step in (1, -1):
        prev, misses, i = ref_mask, 0, ref_idx + step
        while 0 <= i < len(ds.cams) and misses < max_misses:
            box = _bbox(prev)
            ok = False
            if box is not None:
                px, py = _inner_point(prev)
                mask, _ = seg.segment(ds.image(i), [(px, py, 1)], _pad_box(box, ds.width, ds.height, 0.25, 12))
                iou = mask_iou(mask, prev)
                results[i].score = round(iou, 3)
                if mask.any() and iou >= min_iou:
                    r = results[i]
                    r.included, r.source, r.mask, r.area = True, "auto", mask, float(mask.mean())
                    prev, misses, ok = mask, 0, True
            if not ok:
                misses += 1  # keep the old prompt: the object may come back after a short occlusion
            done += 1
            progress(done, len(ds.cams))
            i += step
    return results, {}


def track_video(ds, seg, ref_idx, points, progress):
    frames = [ds.image(i) for i in range(len(ds.cams))]
    progress(0, len(frames))
    masks = seg.propagate(frames, ref_idx, points)
    results = []
    for i, cam in enumerate(ds.cams):
        m = masks.get(i)
        r = FrameResult(i, cam.name)
        if m is not None and m.sum() > 0.0005 * m.size:
            r.included, r.source, r.mask, r.area = True, "reference" if i == ref_idx else "auto", m, float(m.mean())
        results.append(r)
    progress(len(frames), len(frames))
    return results, {}


def track_manual(ds, depth_fn, seg, ref_idx, ref_mask, frames, user_points, existing, progress):
    """Segment exactly `frames`. Hint order: user's click on that frame > projected 3D points > box of nearest tracked mask."""
    try:
        pts = lift_object(ds, depth_fn, ref_idx, ref_mask)
    except RuntimeError:
        pts = None
    out = []
    for n, i in enumerate(frames):
        progress(n, len(frames))
        cam = ds.cams[i]
        r = FrameResult(i, cam.name)
        clicks = user_points.get(i)
        hint_points, box = [], None
        if clicks:
            hint_points = [(c["x"], c["y"], c["label"]) for c in clicks]
        elif pts is not None:
            uv, z, vis = visible_points(pts, cam, depth_fn(i))
            r.visible_points = int(vis.sum())
            if vis.sum() >= 5:
                hint_points = [(float(x), float(y), 1) for x, y in _fps_points(uv[vis])]
                box = _pad_box((*np.percentile(uv[vis], 2, axis=0), *np.percentile(uv[vis], 98, axis=0)), cam.width, cam.height)
        if not hint_points:
            near = min((e for e in existing if e.included and e.mask is not None), key=lambda e: abs(e.index - i), default=None)
            if near is not None:
                px, py = _inner_point(near.mask)
                hint_points, box = [(px, py, 1)], _pad_box(_bbox(near.mask), cam.width, cam.height, 0.3, 12)
        if hint_points:
            mask, _ = seg.segment(ds.image(i), hint_points, box)
            if mask.any():
                r.included, r.source, r.mask, r.area, r.score = True, "manual", mask, float(mask.mean()), None
        out.append(r)
    progress(len(frames), len(frames))
    return out, {}


def run_tracking(ds, depth_fn, seg, method, ref_idx, ref_mask, ref_points, frames, user_points, existing, progress):
    """Dispatch + metrics. Returns (results list, info dict)."""
    t0 = time.perf_counter()
    info = {"method": method}
    if method in ("video", "auto") and not seg.supports_video:
        info["note"] = f"{seg.name} has no video propagation; used {'lift3d' if method == 'auto' else 'neighbor'} instead"
        method = "lift3d" if method == "auto" else "neighbor"
    if method == "neighbor":
        results, extra = track_neighbor(ds, seg, ref_idx, ref_mask, progress)
    elif method == "video":
        results, extra = track_video(ds, seg, ref_idx, ref_points, progress)
    elif method == "manual":
        results, extra = track_manual(ds, depth_fn, seg, ref_idx, ref_mask, frames, user_points, existing, progress)
    else:  # lift3d / auto
        try:
            results, extra = track_lift3d(ds, depth_fn, seg, ref_idx, ref_mask, progress)
        except RuntimeError as exc:
            info["note"] = f"{exc}. Fell back to neighbour search."
            results, extra = track_neighbor(ds, seg, ref_idx, ref_mask, progress)
            method = "neighbor"
        if method == "lift3d" and info["method"] == "auto" and seg.supports_video:
            vid, _ = track_video(ds, seg, ref_idx, ref_points, lambda a, b: None)
            ious = [mask_iou(a.mask, b.mask) for a, b in zip(results, vid) if a.included and b.included and a.source != "reference"]
            extra["video_agreement_iou"] = round(float(np.mean(ious)), 3) if ious else None
    info.update(extra)
    info["method_used"] = method
    info["seconds"] = round(time.perf_counter() - t0, 2)
    info["included"] = sum(r.included for r in results)
    return results, info
