# Week 2 report — object editing in a reconstructed room

> Status of the numbers: tables marked **MEASURED (mock)** come from the synthetic room run on CPU in this repo and only validate geometry/plumbing.
> Tables marked **TO MEASURE on Kaggle** are empty on purpose — no real-GPU run has been done yet. Fill them from `docs/reports/week2_benchmark.csv`
> (written by `python -m gpu_server.editing.benchmark` / the notebook) and `gpu_data/metrics.csv` (written by every job).

## 1. Pipeline

click (2D frame or 3D ray) → SAM mask on the reference frame → **tracking** to all sampled frames → user review (untick / fix / mark manually) →
diffusion edit of a masked crop on the reference frame (N seeds, user picks one) → same prompt + chosen result as reference image on every other frame →
composite back (alpha = 0 outside mask) → rebuild splat from edited frames with the Week-1 poses → before/after.

Frames are the undistorted pinhole training frames (long side 640–960, both sides divisible by 16). The crop sent to diffusion is the mask box + context, resized to `EDIT_LONG_SIDE` (default 768, multiple of 16).

## 2. Tracking methods (`gpu_server/vision/tracking.py`)

| Method | Idea | Strengths | Weaknesses |
|---|---|---|---|
| `neighbor` | Mask of frame t−1 (box + inner point) prompts SAM on frame t; re-initialise when IoU collapses | no 3D needed; follows the video order | drifts and "leaks" onto similar neighbouring objects; cannot say "object not in this frame" — **false positives**; errors accumulate |
| `video` | SAM 2/3 video predictor, clicks on the reference frame, propagate both ways | best for smooth, short sequences | sampled frames at 1–4 fps are far apart, memory bank loses the object after big viewpoint jumps; no visibility reasoning |
| `lift3d` (**default**) | Reference mask + depth (rendered from the splat) → 3D points of the object → project into every frame with its COLMAP pose; frustum test + z-buffer occlusion test; visible projected points / box prompt SAM | gives a principled **frustum filter** (frames where the object is out of view or hidden are excluded automatically); independent per frame, no drift; uses the poses we already have | depends on depth quality near the object boundary; thin objects project few points; SAM still decides the final outline |
| `auto` | `lift3d`, cross-checked with `video` where the backend supports it | agreement ⇒ confidence | slower |
| `manual` | User ticks the frames; SAM runs on exactly those (hinted with projected points when available) | **fallback that always works**; fully acceptable per the brief | needs user time |

Not implemented (future work, as allowed by the brief): DINO feature re-identification, tracking inside the splat (per-Gaussian masks).

### MEASURED (mock): synthetic room, 40 frames, sofa seen in 13 of them

| Method | frames included | mean IoU vs ground truth | false positives | missed | time |
|---|---:|---:|---:|---:|---:|
| `lift3d` | 12 | 0.888 | 0 | 1 | 0.9 s (CPU) |
| `neighbor` | 35 | 0.196 | 22 | 0 | 0.2 s (CPU) |

Caveat: the mock "SAM" is a colour flood fill, so `neighbor` is penalised more than it would be with real SAM. What the table does show reliably is the structural difference: the geometric method knows when the object is **not** in the frame (0 false positives), the 2D-only method does not. Real-SAM numbers: **TO MEASURE** (run the same two calls in the notebook on a real room and compare to hand-labelled masks on ~10 frames).

## 3. Editing quality guarantees

* Mask composite: `composite()` blends with alpha = 0 exactly outside the mask; `outside_mask_max_diff` is computed for every frame and recorded in `metrics.csv` (mock: 0).
* Consistency: every non-reference frame gets `[crop, reference crop]` as image conditioning plus the same prompt/seed, so colour/material follow the reference.
* The prompt is wrapped (`build_prompt`) to say "edit only the masked object, keep everything else".

## 4. Compute and memory (2×T4, 16 GB each, **no bf16 tensor cores on T4**)

Placement: **cuda:0** diffusion (transformer + text encoder + VAE), **cuda:1** SAM + 3DGS. One-GPU sessions share cuda:0 and swap models (`POST /models/unload`).

Quantisation options implemented in `FluxKleinEditor`:

| Option | How | Intent |
|---|---|---|
| `nf4` (default) | bitsandbytes 4-bit NF4 on transformer + text encoder, bf16 compute | fits comfortably next to the VAE on one T4 |
| `int8` | bitsandbytes 8-bit | quality closer to bf16, ~2× nf4 weight memory, slower matmuls |
| `bf16` | full weights + `enable_model_cpu_offload` | quality reference; slow because of PCIe streaming |
| `free_text_encoder` | encode the (identical) prompt once, drop the encoder before denoising | saves the text encoder's VRAM during the per-frame loop |
| `vae.enable_tiling()` | always | avoids VAE decode spikes |

Choices and why: **bf16, not fp16** for compute (FLUX-style transformers overflow in fp16); crop-around-mask editing instead of whole frames (fewer tokens ⇒ faster, more pixels on the object); 4 denoising steps (klein is step-distilled).
T4 has no native bf16, so bf16 math is emulated — expect it slower than the same model on Ampere; this is the main reason to keep the benchmark table honest.

### TO MEASURE on Kaggle — diffusion (fill from `week2_benchmark.csv`)

| Quant | text enc. freed | crop | load (s) | first frame (s) | s / frame | peak VRAM (GB) | notes |
|---|---|---|---:|---:|---:|---:|---|
| nf4 | no | 768 | | | | | |
| nf4 | yes | 768 | | | | | |
| int8 | no | 768 | | | | | |
| bf16 + offload | no | 768 | | | | | |

### TO MEASURE on Kaggle — other stages (from `gpu_data/metrics.csv`)

| Stage | GPU | peak VRAM (GB) | time |
|---|---|---:|---:|
| SAM click (per frame) | cuda:1 | | |
| Tracking `lift3d` (N frames) | cuda:1 | | |
| Edit all frames (N frames) | cuda:0 | | |
| Rebuild — fine-tune from existing splat, 3000 steps | cuda:1 | | |
| Rebuild — scratch, 3000 steps | cuda:1 | | |

Rebuild comparison to report: build seconds and PSNR/visual quality on held-out views for **fine-tune** (starts from `inference_weights.pt`, colour-focused learning rates, 3× loss weight inside edited masks, no densification) vs **scratch** (`reconstruction/train.py` on the edited image set). Expectation to verify: fine-tune is several times faster and keeps geometry; scratch is the cleaner baseline but may blur the edited region less/more depending on step count.

## 5. Known risks / what to check first on Kaggle

1. **FLUX.2 klein through `diffusers`**: pipeline class, `images=[…]` reference conditioning and `PipelineQuantizationConfig` argument names follow the current docs but were not executed; if loading fails, install diffusers from git (noted in `requirements.txt`) and adjust `FluxKleinEditor.load/edit` only.
2. **`precompute_prompts`** (free text encoder) is guarded and silently falls back to the normal path if the pipeline rejects pre-computed embeds.
3. **SAM 3 / SAM 2.1 via `transformers`** class names (`Sam2Model`, `Sam2VideoModel`, `Sam3Tracker*`) vary by version; `SAM_MODEL_ID` switches between them.
4. **gsplat** rebuild: version-dependent rasteriser signature; the mock rebuild path is only a recolour, so real fine-tune/scratch quality is unverified.
5. Kaggle sessions and tunnel URLs are temporary — the website waits and resumes; re-run the notebook and paste the new URL.
6. Diffusion may change texture subtly across frames even with the reference; the review step (step 3) and re-generate (step 4) are the user-side remedies.

## 6. Bonus items

Done: edit history (several edits per project, each with its own result and before/after), job queue (single GPU worker with a persistent task table, resume on restart), per-edit cancel.
Not done: undo inside the mask editor beyond point-undo; Celery (the dispatcher thread + SQLite table covers the single-GPU case).

## 7. Demo script (≈4 min, include a fallback run)

1. Open the project, orbit/fly in the original splat. 2. **Edit object** → *Click in 3D* on the sofa → refine with one exclude point.
3. Prompt "make the sofa dark green leather" → auto tracking → show frame grid with confidences, untick one bad frame.
4. Generate 4 options → accept one → apply to all frames (show progress, s/frame, VRAM). 5. Rebuild (fine-tune) → **Before / After** toggle.
6. **Fallback run:** new edit, choose *Mark frames manually*, tick 5–6 frames yourself, continue the same way.
