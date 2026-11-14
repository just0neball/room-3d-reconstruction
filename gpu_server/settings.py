"""Runtime configuration of the GPU server (env vars, all optional)."""
import os
from pathlib import Path


def _b(name, default=False):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


MOCK = _b("GPU_MOCK", False)  # synthetic scene + CPU stand-in models; no torch needed
DATA_DIR = Path(os.environ.get("GPU_DATA_DIR", "/kaggle/working/gpu_data" if Path("/kaggle").exists() else "gpu_data")).resolve()
TOKEN = os.environ.get("GPU_SERVER_TOKEN", "")  # shared secret; empty = open (dev only)
RECON_DIR = Path(__file__).resolve().parent / "reconstruction"

SAM_MODEL_ID = os.environ.get("SAM_MODEL_ID", "facebook/sam2.1-hiera-small")  # "facebook/sam3" also supported (gated)
DIFFUSION_MODEL_ID = os.environ.get("DIFFUSION_MODEL_ID", "black-forest-labs/FLUX.2-klein-4B")
DEFAULT_QUANT = os.environ.get("DIFFUSION_QUANT", "nf4")  # nf4 | int8 | bf16
DIFFUSION_STEPS = int(os.environ.get("DIFFUSION_STEPS", "4"))
DIFFUSION_GUIDANCE = float(os.environ.get("DIFFUSION_GUIDANCE", "1.0"))
EDIT_LONG_SIDE = int(os.environ.get("EDIT_LONG_SIDE", "768"))  # crop sent to the diffusion model (multiple of 16)

# Device split for 2xT4: GPU0 = diffusion model + text encoder + VAE, GPU1 = SAM + 3DGS.
DIFFUSION_DEVICE = os.environ.get("DIFFUSION_DEVICE", "")
AUX_DEVICE = os.environ.get("AUX_DEVICE", "")


def devices():
    """(diffusion_device, aux_device) as torch device strings; 'cpu' in mock mode."""
    if MOCK:
        return "cpu", "cpu"
    import torch
    n = torch.cuda.device_count()
    if n == 0:
        return "cpu", "cpu"
    diff = DIFFUSION_DEVICE or "cuda:0"
    aux = AUX_DEVICE or ("cuda:1" if n > 1 else "cuda:0")
    return diff, aux
