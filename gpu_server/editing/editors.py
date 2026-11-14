"""Diffusion editors. edit(images, prompt, seed, masks) -> RGB uint8 array the size of images[0].

  images[0]  crop of the frame to edit            images[1] (optional)  crop of the accepted reference edit
  masks      object mask for each image crop (used by the mock editor; the real model ignores masks)

FluxKleinEditor  FLUX.2 [klein] 4B (Apache-2.0) through diffusers; multi-reference image editing.
MockEditor       CPU recolouring from the colour words in the prompt (GPU_MOCK=1, tests).
"""
import gc
import time

import cv2
import numpy as np

COLORS = {  # word -> (hue 0-179 OpenCV, saturation scale, value scale)
    "green": (60, 1.0, 0.8), "red": (0, 1.1, 0.85), "blue": (110, 1.0, 0.9), "yellow": (28, 1.0, 1.0),
    "orange": (14, 1.1, 1.0), "purple": (140, 0.9, 0.8), "pink": (165, 0.6, 1.0), "teal": (90, 0.9, 0.8),
    "brown": (12, 0.7, 0.55), "black": (0, 0.0, 0.18), "white": (0, 0.0, 1.0), "gray": (0, 0.0, 0.6), "grey": (0, 0.0, 0.6),
}


def build_prompt(user_prompt, has_reference):
    if has_reference:
        return ("Image 1 is a photo to edit. Image 2 shows the already edited result of the same object seen from another angle. "
                f"Edit the main object in image 1 as follows: {user_prompt}. Make its colour, material and design match the object in image 2. "
                "Keep everything else, the lighting and the camera view exactly as in image 1.")
    return (f"Edit the main object in the centre of the photo as follows: {user_prompt}. "
            "Keep everything else, the lighting and the camera view exactly the same.")


class MockEditor:
    name, quant = "mock-recolor", "none"

    def load(self):
        pass

    def unload(self):
        pass

    def peak_vram_gb(self):
        return None

    def edit(self, images, prompt, seed, masks=None):
        img = images[0]
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        words = prompt.lower().replace(",", " ").split()
        hue, sat, val = 60, 1.0, 0.8
        for w in words:
            if w in COLORS:
                hue, sat, val = COLORS[w]
        if "dark" in words:
            val *= 0.65
        if "light" in words or "pastel" in words:
            val, sat = min(val * 1.2, 1.2), sat * 0.6
        if len(images) > 1 and masks is not None and masks[1] is not None and masks[1].any():
            ref_hsv = cv2.cvtColor(images[1], cv2.COLOR_RGB2HSV).astype(np.float32)[masks[1]]
            hue, sat_ref, val_ref = np.median(ref_hsv[:, 0]), np.median(ref_hsv[:, 1]), np.median(ref_hsv[:, 2])
            base_s = np.median(hsv[..., 1][masks[0]]) if masks and masks[0] is not None and masks[0].any() else 128
            base_v = np.median(hsv[..., 2][masks[0]]) if masks and masks[0] is not None and masks[0].any() else 128
            sat, val = sat_ref / max(base_s, 1), val_ref / max(base_v, 1)
        else:  # first frame: seed gives visible variety so "choose a seed" is meaningful
            rng = np.random.default_rng(seed)
            hue = (hue + rng.integers(-6, 7)) % 180
            val *= 1 + rng.uniform(-0.15, 0.15)
        hsv[..., 0] = hue
        hsv[..., 1] = np.clip(hsv[..., 1] * sat if sat else 0, 0, 255) if sat <= 1.5 else np.clip(hsv[..., 1] * sat, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * val, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        time.sleep(0.05)
        return out


class FluxKleinEditor:
    def __init__(self, model_id, device, quant="nf4", steps=4, guidance=1.0, compute_dtype="bfloat16", free_text_encoder=False):
        self.model_id, self.device, self.quant = model_id, device, quant
        self.steps, self.guidance, self.compute_dtype = steps, guidance, compute_dtype
        self.free_text_encoder = free_text_encoder
        self.pipe = None
        self.embeds = {}
        self.name = model_id
        self.load_seconds = 0.0

    def load(self):
        if self.pipe is not None:
            return
        import torch
        from diffusers import DiffusionPipeline
        t0 = time.perf_counter()
        dtype = getattr(torch, self.compute_dtype)  # bf16, NOT fp16: FLUX-style models overflow in fp16
        kwargs = {"torch_dtype": dtype}
        if self.quant in ("nf4", "int8"):
            from diffusers.quantizers import PipelineQuantizationConfig
            if self.quant == "nf4":
                cfg = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", components_to_quantize=["transformer", "text_encoder"],
                                                 quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": dtype})
            else:
                cfg = PipelineQuantizationConfig(quant_backend="bitsandbytes_8bit", components_to_quantize=["transformer", "text_encoder"],
                                                 quant_kwargs={"load_in_8bit": True})
            kwargs["quantization_config"] = cfg
        self.pipe = DiffusionPipeline.from_pretrained(self.model_id, **kwargs)
        if self.quant == "bf16":  # ~13 GB of weights: stream submodules to the GPU on demand
            self.pipe.enable_model_cpu_offload(gpu_id=torch.device(self.device).index or 0)
        else:
            try:
                self.pipe.to(self.device)
            except Exception:  # some bnb components refuse .to(); offload handles placement itself
                self.pipe.enable_model_cpu_offload(gpu_id=torch.device(self.device).index or 0)
        try:
            self.pipe.vae.enable_tiling()
        except Exception:
            pass
        torch.cuda.reset_peak_memory_stats(self.device)
        self.load_seconds = round(time.perf_counter() - t0, 2)

    def precompute_prompts(self, prompts):
        """The prompt is identical for every frame, so encode once and (optionally) free the text encoder
        before the transformer starts denoising. Guarded: any API mismatch silently falls back to the normal path."""
        if not self.free_text_encoder:
            return
        import torch
        try:
            with torch.no_grad():
                for p in prompts:
                    out = self.pipe.encode_prompt(prompt=p, device=self.device)
                    self.embeds[p] = out[0] if isinstance(out, (tuple, list)) else out
            # sanity: the pipeline must accept the embeds before we throw the encoder away
            self.pipe.text_encoder = None
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[editor] text-encoder free skipped: {exc}", flush=True)
            self.embeds = {}

    def edit(self, images, prompt, seed, masks=None):
        import torch
        from PIL import Image
        pil = [Image.fromarray(i) for i in images]
        w, h = pil[0].size
        text = build_prompt(prompt, len(images) > 1)
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        kw = dict(image=pil if len(pil) > 1 else pil[0], height=h, width=w, num_inference_steps=self.steps, guidance_scale=self.guidance, generator=gen)
        if text in self.embeds:
            kw["prompt_embeds"] = self.embeds[text]
        else:
            kw["prompt"] = text
        with torch.inference_mode():
            out = self.pipe(**kw).images[0]
        return np.array(out.convert("RGB").resize((w, h)))

    def peak_vram_gb(self):
        import torch
        return round(torch.cuda.max_memory_allocated(self.device) / 1024 ** 3, 3)

    def reset_peak(self):
        import torch
        torch.cuda.reset_peak_memory_stats(self.device)

    def unload(self):
        self.pipe, self.embeds = None, {}
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
