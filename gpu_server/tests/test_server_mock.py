"""Full GPU-server API in mock mode: reconstruct -> segment -> track -> edit -> rebuild, through HTTP."""
import io
import os
import time
import zipfile

os.environ["GPU_MOCK"] = "1"

import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["GPU_DATA_DIR"] = str(tmp_path_factory.mktemp("gpu"))
    from gpu_server import settings
    settings.DATA_DIR = type(settings.DATA_DIR)(os.environ["GPU_DATA_DIR"])
    from gpu_server.app import app
    return TestClient(app)


def wait(client, job_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/jobs/{job_id}").json()
        if j["status"] in ("completed", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise TimeoutError(job_id)


def upload(client, uid, data):
    r = client.put(f"/uploads/{uid}", params={"part": 0, "parts": 1}, content=data)
    assert r.json()["complete"], r.text


def zip_bytes(files):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for k, v in files.items():
            z.writestr(k, v)
    return b.getvalue()


def png(mask):
    return cv2.imencode(".png", mask.astype(np.uint8) * 255)[1].tobytes()


def test_full_flow(client):
    assert client.get("/health").json()["mock"] is True
    upload(client, "video-0001", b"not a real video")
    job = client.post("/jobs", json={"kind": "reconstruct", "params": {"dataset": "p1", "run_name": "stable_7000"}, "uploads": {"video": "video-0001"}}).json()
    j = wait(client, job["id"])
    assert j["status"] == "completed", j
    art = zipfile.ZipFile(io.BytesIO(client.get(f"/jobs/{job['id']}/artifact").content))
    assert "artifacts/exports/stable_7000/scene.splat" in art.namelist()
    assert client.get("/datasets/p1").json()["frames"] == 40

    frame = "frame_000000.jpg"
    r = client.post("/segment", json={"dataset": "p1", "frame": frame, "points": [{"x": 320, "y": 250, "label": 1}]}).json()
    mask = cv2.imdecode(np.frombuffer(base64.b64decode(r["mask_png_b64"]), np.uint8), 0) > 127
    assert 0.1 < r["area"] < 0.6

    upload(client, "bundle-trk1", zip_bytes({"masks/frame_000000.png": png(mask)}))
    t = wait(client, client.post("/jobs", json={"kind": "track", "uploads": {"bundle": "bundle-trk1"},
             "params": {"dataset": "p1", "method": "auto", "ref_frame": frame, "ref_points": [{"x": 320, "y": 250, "label": 1}]}}).json()["id"])
    assert t["status"] == "completed", t
    tz = zipfile.ZipFile(io.BytesIO(client.get(f"/jobs/{t['id']}/artifact").content))
    import json
    frames = json.loads(tz.read("tracking.json"))["frames"]
    inc = [f["index"] for f in frames if f["included"]]
    assert 0 in inc and len(inc) >= 5 and 20 not in inc

    masks = {n: tz.read(n) for n in tz.namelist() if n.startswith("masks/")}
    upload(client, "bundle-ref1", zip_bytes({k: v for k, v in masks.items() if k.endswith("frame_000000.png")}))
    e = wait(client, client.post("/jobs", json={"kind": "edit_reference", "uploads": {"bundle": "bundle-ref1"},
             "params": {"dataset": "p1", "frame": frame, "prompt": "make the sofa dark green leather", "seeds": 3}}).json()["id"])
    assert e["status"] == "completed" and e["result"]["outside_mask_max_diff"] == 0
    ez = zipfile.ZipFile(io.BytesIO(client.get(f"/jobs/{e['id']}/artifact").content))
    cands = [n for n in ez.namelist() if n.startswith("candidates/")]
    assert len(cands) == 3

    upload(client, "bundle-rest1", zip_bytes({**masks, "reference.png": ez.read(cands[0])}))
    names = [f["name"] for f in frames if f["included"]]
    e2 = wait(client, client.post("/jobs", json={"kind": "edit_rest", "uploads": {"bundle": "bundle-rest1"},
              "params": {"dataset": "p1", "reference_frame": frame, "frames": names, "prompt": "make the sofa dark green leather", "seed": 1}}).json()["id"])
    assert e2["status"] == "completed" and e2["result"]["outside_mask_max_diff"] == 0, e2
    e2z = zipfile.ZipFile(io.BytesIO(client.get(f"/jobs/{e2['id']}/artifact").content))
    edited = {n: e2z.read(n) for n in e2z.namelist() if n.startswith("edited/")}
    assert len(edited) == len(names)

    upload(client, "bundle-rb1", zip_bytes({**masks, **edited}))
    rb = wait(client, client.post("/jobs", json={"kind": "rebuild", "uploads": {"bundle": "bundle-rb1"}, "params": {"dataset": "p1", "mode": "finetune"}}).json()["id"])
    assert rb["status"] == "completed", rb
    assert rb["result"]["gaussians_recoloured"] > 100
    rz = zipfile.ZipFile(io.BytesIO(client.get(f"/jobs/{rb['id']}/artifact").content))
    assert "scene.splat" in rz.namelist()
