"""Lazy model registry. 2xT4 plan: GPU0 = diffusion (+text encoder + VAE), GPU1 = SAM + 3DGS / depth.
With a single GPU the diffusion model is unloaded before SAM/3DGS jobs run (and vice versa)."""
import threading

from . import settings

_lock = threading.Lock()
_seg = None
_editor = None


def segmenter():
    global _seg
    with _lock:
        if _seg is None:
            if settings.MOCK:
                from .vision.segmenters import MockSegmenter
                _seg = MockSegmenter()
            else:
                from .vision.segmenters import SamSegmenter
                _seg = SamSegmenter(settings.SAM_MODEL_ID, settings.devices()[1])
        return _seg


def editor(quant=None, free_text_encoder=False):
    global _editor
    quant = quant or settings.DEFAULT_QUANT
    with _lock:
        if _editor is not None and (_editor.quant != quant and not settings.MOCK):
            _editor.unload()
            _editor = None
        if _editor is None:
            if settings.MOCK:
                from .editing.editors import MockEditor
                _editor = MockEditor()
            else:
                from .editing.editors import FluxKleinEditor
                _editor = FluxKleinEditor(settings.DIFFUSION_MODEL_ID, settings.devices()[0], quant, settings.DIFFUSION_STEPS,
                                          settings.DIFFUSION_GUIDANCE, free_text_encoder=free_text_encoder)
        return _editor


def single_gpu():
    d, a = settings.devices()
    return d == a and d != "cpu"


def free_diffusion():
    global _editor
    with _lock:
        if _editor is not None:
            _editor.unload()
            _editor = None


def free_segmenter():
    with _lock:
        if _seg is not None:
            _seg.unload()


def loaded():
    return {"segmenter": getattr(_seg, "name", None), "diffusion": getattr(_editor, "name", None) if _editor else None,
            "diffusion_quant": getattr(_editor, "quant", None) if _editor else None}
