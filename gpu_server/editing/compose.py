"""Crop -> edit -> paste back, with a hard guarantee that pixels outside the mask are untouched.

The diffusion model never sees the whole frame: it gets a padded crop around the mask, resized so both sides are
multiples of 16 (long side <= EDIT_LONG_SIDE, which keeps tokens, VRAM and time down). The edited crop is resized back
to the exact crop size and alpha-blended with the ORIGINAL frame using an alpha that is exactly 0 wherever mask == 0.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Crop:
    x0: int
    y0: int
    x1: int
    y1: int  # exclusive
    size: tuple  # (w, h) sent to the model, multiples of 16

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0


def mask_bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def crop_box(mask, pad_frac=0.25, min_pad=24, long_side=768, multiple=16):
    h, w = mask.shape
    x0, y0, x1, y1 = mask_bbox(mask)
    px = max(min_pad, int((x1 - x0) * pad_frac))
    py = max(min_pad, int((y1 - y0) * pad_frac))
    x0, y0, x1, y1 = max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)
    cw, ch = x1 - x0, y1 - y0
    scale = min(1.0 if max(cw, ch) <= long_side else long_side / max(cw, ch), 2.0)
    if max(cw, ch) < 512:  # small objects: upscale a little so the model has detail to work with
        scale = min(2.0, 512 / max(cw, ch))
    tw = max(multiple, int(round(cw * scale / multiple)) * multiple)
    th = max(multiple, int(round(ch * scale / multiple)) * multiple)
    return Crop(x0, y0, x1, y1, (tw, th))


def cut(image, crop):
    sub = image[crop.y0:crop.y1, crop.x0:crop.x1]
    return cv2.resize(sub, crop.size, interpolation=cv2.INTER_CUBIC if crop.size[0] > crop.w else cv2.INTER_AREA)


def soft_alpha(mask, feather=2):
    """Alpha = 1 deep inside the mask, a soft rim near its border, and exactly 0 wherever mask == 0."""
    m = mask.astype(np.uint8)
    alpha = m.astype(np.float32)
    if feather > 0:
        inner = cv2.erode(m, np.ones((2 * feather + 1,) * 2, np.uint8))
        alpha = cv2.GaussianBlur(inner.astype(np.float32), (0, 0), feather)
        rim = (m > 0) & (inner == 0)
        alpha[rim] = np.maximum(alpha[rim], 0.5)  # edited colour must reach the silhouette
        alpha[inner > 0] = 1.0
    alpha[m == 0] = 0.0
    return alpha


def composite(original, edited_crop, mask, crop, feather=2):
    """original: HxWx3 uint8; edited_crop: model output at crop.size; returns the full frame."""
    back = cv2.resize(edited_crop, (crop.w, crop.h), interpolation=cv2.INTER_AREA if edited_crop.shape[1] > crop.w else cv2.INTER_CUBIC)
    full = original.copy()
    alpha = soft_alpha(mask, feather)[crop.y0:crop.y1, crop.x0:crop.x1, None]
    region = original[crop.y0:crop.y1, crop.x0:crop.x1].astype(np.float32)
    blended = region * (1 - alpha) + back.astype(np.float32) * alpha
    full[crop.y0:crop.y1, crop.x0:crop.x1] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    return full


def outside_mask_max_diff(original, final, mask):
    """0 means the guarantee holds (nothing outside the mask changed)."""
    diff = np.abs(original.astype(int) - final.astype(int)).max(axis=2)
    return int(diff[~mask].max()) if (~mask).any() else 0
