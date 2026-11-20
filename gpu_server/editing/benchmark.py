"""VRAM / speed benchmark of the diffusion editor for the Week-2 report.

    python -m gpu_server.editing.benchmark --image some_frame.jpg --quants nf4 int8 bf16 --frames 6

For every quantisation it loads the editor on the diffusion GPU, edits one masked crop `--frames` times and appends a row to
docs/reports/week2_benchmark.csv: load seconds, seconds/frame (mean of frames after the first warm-up), peak VRAM (GB).
Run it on Kaggle (2xT4) - there is nothing meaningful to measure in mock mode.
"""
import argparse
import csv
import gc
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="a frame (any size); a centred box is used as the mask")
    ap.add_argument("--quants", nargs="+", default=["nf4", "int8", "bf16"])
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--long-side", type=int, default=768)
    ap.add_argument("--free-text-encoder", action="store_true")
    ap.add_argument("--prompt", default="make the object dark green leather")
    ap.add_argument("--out", default="docs/reports/week2_benchmark.csv")
    a = ap.parse_args()
    import cv2
    import torch
    from . import compose
    from .editors import FluxKleinEditor, build_prompt
    from .. import settings

    img = cv2.cvtColor(cv2.imread(a.image), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), bool)
    mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
    box = compose.crop_box(mask, long_side=a.long_side)
    crop, cmask = compose.cut(img, box), compose.cut(mask.astype(np.uint8) * 255, box) > 127
    dev = settings.devices()[0]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    gpu = torch.cuda.get_device_name(dev) if dev != "cpu" else "cpu"
    with open(out, "a", newline="") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["gpu", "model", "quant", "free_text_encoder", "crop", "load_s", "first_frame_s", "sec_per_frame", "peak_vram_gb", "status"])
        for q in a.quants:
            row = [gpu, settings.DIFFUSION_MODEL_ID, q, a.free_text_encoder, f"{crop.shape[1]}x{crop.shape[0]}"]
            try:
                ed = FluxKleinEditor(settings.DIFFUSION_MODEL_ID, dev, q, settings.DIFFUSION_STEPS, settings.DIFFUSION_GUIDANCE, free_text_encoder=a.free_text_encoder)
                ed.load()
                ed.precompute_prompts([build_prompt(a.prompt, False)])
                times = []
                for k in range(a.frames):
                    t0 = time.perf_counter()
                    ed.edit([crop], a.prompt, 1000 + k, [cmask])
                    torch.cuda.synchronize(dev)
                    times.append(time.perf_counter() - t0)
                row += [ed.load_seconds, round(times[0], 2), round(float(np.mean(times[1:] or times)), 2), ed.peak_vram_gb(), "ok"]
                print(row, flush=True)
            except Exception as exc:  # OOM etc. is a valid result for the table
                row += ["", "", "", "", f"{type(exc).__name__}: {str(exc)[:80]}"]
                print("FAILED", row, flush=True)
            wr.writerow(row)
            fh.flush()
            ed = None
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
