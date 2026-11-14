from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..models import User
from ..security import get_current_user
from ..services import dispatcher
from ..services.gpu_client import GpuClient, GpuError, load_config, save_config

router = APIRouter(prefix="/gpu", tags=["gpu"])


class GpuConfigIn(BaseModel):
    url: str
    token: str | None = None


def status_dict(client: GpuClient) -> dict:
    base = {"configured": client.configured, "url": client.url, "reachable": False, "mock": False, "message": "",
            "checked_at": datetime.now(timezone.utc).isoformat(), "devices": [], "models": {}, "uptime_s": None}
    if not client.configured:
        base["message"] = "No GPU server connected. Start the Kaggle server notebook and paste its public URL."
        return base
    try:
        h = client.health()
    except GpuError as exc:
        base["message"] = str(exc)
        return base
    base.update(reachable=True, mock=bool(h.get("mock")), devices=h.get("devices", []), models=h.get("models", {}), uptime_s=h.get("uptime_s"),
                message="Connected" + (" (demo mode: synthetic models)" if h.get("mock") else ""))
    return base


@router.get("/status")
def gpu_status(_: User = Depends(get_current_user)):
    return status_dict(GpuClient())


@router.put("/config")
def gpu_config(body: GpuConfigIn, _: User = Depends(get_current_user)):
    cfg = save_config(body.url, body.token)
    dispatcher.wake()
    return status_dict(GpuClient(cfg["url"], cfg["token"]))
