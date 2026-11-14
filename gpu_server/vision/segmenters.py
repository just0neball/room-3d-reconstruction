"""Segmentation back-ends behind one interface.

  SamSegmenter   SAM 2.1 / SAM 3 via Hugging Face transformers (GPU)   -> image prompts + video propagation
  MockSegmenter  OpenCV flood-fill, CPU only (GPU_MOCK=1, tests)       -> image prompts only

segment(image_rgb, points, box) -> (mask bool HxW, score 0-1)
  points: [(x, y, label)] label 1 = positive, 0 = negative; box: (x0, y0, x1, y1) or None
propagate(frames_rgb, ref_idx, points) -> {frame_idx: mask}   (video back-ends only)
"""
import gc
import threading

import cv2
import numpy as np


class Segmenter:
    name = "base"
    supports_video = False

    def segment(self, image, points, box=None):
        raise NotImplementedError

    def propagate(self, frames, ref_idx, points):
        raise NotImplementedError

    def unload(self):
        pass


class MockSegmenter(Segmenter):
    """Colour flood-fill from the positive clicks (works on the flat-coloured synthetic room)."""

    name = "mock-floodfill"

    def __init__(self, tol=14):
        self.tol = tol

    def segment(self, image, points, box=None):
        h, w = image.shape[:2]
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        mask = np.zeros((h, w), bool)
        pos = [(x, y) for x, y, l in points if l == 1]
        if not pos and box is not None:
            pos = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)]
        for x, y in pos:
            x, y = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
            ff = np.zeros((h + 2, w + 2), np.uint8)
            flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
            cv2.floodFill(blur.copy(), ff, (x, y), 0, (self.tol,) * 3, (self.tol,) * 3, flags)
            mask |= ff[1:-1, 1:-1] > 0
        for x, y, l in points:
            if l == 0:
                x, y = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
                ff = np.zeros((h + 2, w + 2), np.uint8)
                flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
                cv2.floodFill(blur.copy(), ff, (x, y), 0, (self.tol,) * 3, (self.tol,) * 3, flags)
                mask &= ~(ff[1:-1, 1:-1] > 0)
        if box is not None:
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            keep = np.zeros_like(mask)
            keep[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = True
            mask &= keep
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        return mask, (0.9 if mask.any() else 0.0)


class SamSegmenter(Segmenter):
    """SAM 2.1 (default) or SAM 3 through transformers. Picks the class family from the model id.
    The image model is kept resident (interactive clicks must be fast); the video model is loaded on demand."""

    supports_video = True

    def __init__(self, model_id, device, dtype="bfloat16"):
        import torch
        self.torch = torch
        self.model_id, self.device = model_id, device
        # T4 has no native bf16; fp16 is fine for SAM (unlike the diffusion model).
        self.dtype = torch.float16 if ("cuda" in device and torch.cuda.get_device_capability(device)[0] < 8) else getattr(torch, dtype)
        self.name = model_id
        self.family = "sam3" if "sam3" in model_id.lower() else "sam2"
        self._image = None
        self._video = None
        self._lock = threading.Lock()

    def _classes(self, video):
        import transformers as T
        if self.family == "sam3":
            return (T.Sam3TrackerVideoModel, T.Sam3TrackerVideoProcessor) if video else (T.Sam3TrackerModel, T.Sam3TrackerProcessor)
        return (T.Sam2VideoModel, T.Sam2VideoProcessor) if video else (T.Sam2Model, T.Sam2Processor)

    def _load_image(self):
        if self._image is None:
            M, P = self._classes(False)
            self._image = (M.from_pretrained(self.model_id, torch_dtype=self.dtype).to(self.device).eval(), P.from_pretrained(self.model_id))
        return self._image

    def segment(self, image, points, box=None):
        torch = self.torch
        with self._lock:
            model, proc = self._load_image()
            kw = {}
            if points:
                kw["input_points"] = [[[[float(x), float(y)] for x, y, _ in points]]]
                kw["input_labels"] = [[[int(l) for _, _, l in points]]]
            if box is not None:
                kw["input_boxes"] = [[[float(v) for v in box]]]
            inputs = proc(images=image, return_tensors="pt", **kw).to(self.device)
            inputs = {k: (v.to(self.dtype) if torch.is_floating_point(v) else v) for k, v in inputs.items()}
            with torch.inference_mode():
                out = model(**inputs, multimask_output=len(points) <= 1 and box is None)
            masks = proc.post_process_masks(out.pred_masks.float().cpu(), inputs["original_sizes"])[0]  # (objects, n, H, W)
            scores = out.iou_scores.float().cpu()[0, 0]
            best = int(scores.argmax())
            return masks[0, best].numpy().astype(bool), float(scores[best])

    def propagate(self, frames, ref_idx, points):
        """Track the object from frame `ref_idx` in both directions with the SAM video tracker."""
        torch = self.torch
        M, P = self._classes(True)
        with self._lock:
            model = M.from_pretrained(self.model_id, torch_dtype=self.dtype).to(self.device).eval()
            proc = P.from_pretrained(self.model_id)
            session = proc.init_video_session(video=list(frames), inference_device=self.device, video_storage_device="cpu", dtype=self.dtype)
            proc.add_inputs_to_inference_session(
                inference_session=session, frame_idx=ref_idx, obj_ids=1,
                input_points=[[[[float(x), float(y)] for x, y, _ in points]]],
                input_labels=[[[int(l) for _, _, l in points]]],
            )
            h, w = session.video_height, session.video_width
            result = {}

            def run(**kw):
                for out in model.propagate_in_video_iterator(session, **kw):
                    m = proc.post_process_masks([out.pred_masks], original_sizes=[[h, w]], binarize=False)[0]
                    arr = m[0].float().cpu().numpy().squeeze()
                    result[int(out.frame_idx)] = arr > (0.0 if arr.min() < 0 else 0.5)

            with torch.inference_mode():
                model(inference_session=session, frame_idx=ref_idx)  # run the prompted frame first
                run(start_frame_idx=ref_idx)
                run(start_frame_idx=ref_idx, reverse=True)
            del model, session
            gc.collect()
            torch.cuda.empty_cache()
            return result

    def unload(self):
        self._image = None
        gc.collect()
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0
