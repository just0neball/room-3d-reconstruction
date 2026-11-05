"""End-to-end test of the whole contract with a fake worker that speaks exactly the
protocol worker/poll.py and Person A's pipeline/common.py use."""
import io
import json
import zipfile
from datetime import timedelta

import pytest

from app.db import SessionLocal
from app.models import Task, utcnow

WORKER = {"Authorization": "Bearer test-worker-token"}
RUN = "stable_7000"
SPLAT = b"SPLAT" * 2000  # 10 KB stand-in


def signup(client, email):
    r = client.post("/auth/signup", json={"email": email, "password": "correct-horse-1"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def make_delivery_zip(extra=None) -> bytes:
    """Mimics pipeline/package.py output: everything under artifacts/."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"artifacts/exports/{RUN}/scene.splat", SPLAT)
        z.writestr(f"artifacts/exports/{RUN}/scene.ply", b"ply\n" + b"x" * 500)
        z.writestr(f"artifacts/exports/{RUN}/scene_metadata.json", json.dumps({"gaussians": 23350}))
        z.writestr(f"artifacts/3dgs_training/{RUN}/training_metadata.json", json.dumps({"steps": 7000}))
        z.writestr("artifacts/3dgs_dataset/sparse_txt/cameras.txt", "1 PINHOLE 960 540 800 800 480 270\n")
        z.writestr("artifacts/3dgs_dataset/sparse_txt/images.txt", "# poses\n")
        z.writestr("artifacts/3dgs_dataset/images/00001.jpg", b"jpg")
        z.writestr("artifacts/reports/week2_handoff.md", "# handoff")
        z.writestr("artifacts/reports/validation_status.json", json.dumps({"ok": True}))
        z.writestr("pipeline/run_pipeline.py", "# not extracted")
        for name, data in (extra or {}).items():
            z.writestr(name, data)
    return buf.getvalue()


def upload_parts(client, task_id, blob, part_size):
    parts = max(1, -(-len(blob) // part_size))
    last = None
    for i in range(parts):
        last = client.put(
            f"/tasks/{task_id}/artifacts", params={"part": i, "parts": parts},
            content=blob[i * part_size:(i + 1) * part_size], headers=WORKER,
        )
        assert last.status_code == 200, last.text
    return last.json()


def new_project_with_video(client, headers, name="Living room"):
    pid = client.post("/projects", json={"name": name}, headers=headers).json()["id"]
    r = client.post(f"/projects/{pid}/video", files={"file": ("room.mp4", b"\x00" * 4096, "video/mp4")}, headers=headers)
    assert r.status_code == 200, r.text
    return pid


def test_full_happy_path(client):
    user = signup(client, "nada@example.com")
    pid = new_project_with_video(client, user)

    # cannot read the scene before it exists
    assert client.get(f"/projects/{pid}/scene", headers=user).status_code == 409

    r = client.post(f"/projects/{pid}/reconstruct", json={}, headers=user)
    assert r.status_code == 202
    task = r.json()
    assert task["status"] == "pending" and task["run_name"] == RUN and task["steps"] == 7000

    # second request while one is active is rejected
    assert client.post(f"/projects/{pid}/reconstruct", json={}, headers=user).status_code == 409

    # worker claims it
    claimed = client.post("/worker/next-task", headers=WORKER).json()["task"]
    assert claimed["id"] == task["id"] and claimed["attempt"] == 1 and claimed["run_name"] == RUN
    assert client.post("/worker/next-task", headers=WORKER).json()["task"] is None  # nothing else queued

    video = client.get(claimed["video_url"], headers=WORKER)
    assert video.status_code == 200 and len(video.content) == 4096

    # pipeline progress events, exactly as send_progress() emits them (incl. extra timestamp field)
    for stage, pct, label in [
        ("extracting_frames", 5, "Extracting frames"),
        ("filtering_frames", 15, "Extracting frames"),
        ("estimating_poses", 34, "Estimating camera poses"),
        ("preparing_training_data", 65, "Estimating camera poses"),
        ("training_gaussians", 80, "Training Gaussian splat"),
        ("exporting", 96, "Exporting scene"),
    ]:
        ev = {"stage": stage, "progress": pct, "status": "running", "error": None, "timestamp_utc": "2026-11-05T10:00:00+00:00"}
        assert client.post(f"/tasks/{task['id']}/progress", json=ev, headers=WORKER).status_code == 200
        polled = client.get(f"/tasks/{task['id']}", headers=user).json()
        assert polled["progress"] == pct and polled["stage"] == stage and polled["stage_label"] == label

    # out-of-order event never moves the bar backwards
    client.post(f"/tasks/{task['id']}/progress", json={"stage": "training_gaussians", "progress": 70, "status": "running"}, headers=WORKER)
    assert client.get(f"/tasks/{task['id']}", headers=user).json()["progress"] == 96

    # package.py sends "completed" at 100 BEFORE upload: must hold at 99, not finish
    client.post(f"/tasks/{task['id']}/progress", json={"stage": "packaging", "progress": 100, "status": "completed"}, headers=WORKER)
    held = client.get(f"/tasks/{task['id']}", headers=user).json()
    assert held["status"] == "running" and held["progress"] == 99

    # complete without artifacts is refused
    assert client.post(f"/tasks/{task['id']}/complete", headers=WORKER).status_code == 409

    # chunked upload (small parts to exercise assembly)
    blob = make_delivery_zip()
    result = upload_parts(client, task["id"], blob, part_size=max(1, len(blob) // 3))
    assert result["extracted"] is True

    done = client.post(f"/tasks/{task['id']}/complete", headers=WORKER).json()
    assert done["status"] == "completed" and done["progress"] == 100 and done["stage_label"] == "Done"
    # idempotent
    assert client.post(f"/tasks/{task['id']}/complete", headers=WORKER).status_code == 200

    scene = client.get(f"/projects/{pid}/scene", headers=user).json()
    assert scene["format"] == "splat" and scene["run_name"] == RUN
    assert scene["metadata"]["scene"]["gaussians"] == 23350
    assert scene["handoff"]["intrinsics_pinhole"]["exists"] is True
    assert scene["handoff"]["poses_world_to_camera"]["exists"] is True
    assert scene["handoff"]["frames_undistorted"]["exists"] is True
    assert scene["handoff"]["vggt_poses"]["exists"] is False  # not in this fake delivery

    # viewer downloads via signed URL, no Authorization header, with Range support
    full = client.get(scene["splat_url"])
    assert full.status_code == 200 and full.content == SPLAT
    ranged = client.get(scene["splat_url"], headers={"Range": "bytes=0-9"})
    assert ranged.status_code == 206 and ranged.content == SPLAT[:10]

    # project list shows the finished state
    listed = client.get("/projects", headers=user).json()
    assert listed[0]["scene_ready"] is True and listed[0]["latest_task"]["status"] == "completed"


def test_failure_reporting(client):
    user = signup(client, "fail@example.com")
    pid = new_project_with_video(client, user)
    tid = client.post(f"/projects/{pid}/reconstruct", json={"steps": 1000}, headers=user).json()["id"]
    client.post("/worker/next-task", headers=WORKER)
    client.post(f"/tasks/{tid}/progress", json={"stage": "estimating_poses", "progress": 30}, headers=WORKER)
    client.post(f"/tasks/{tid}/progress", json={"stage": "failed", "progress": 0, "status": "failed", "error": "COLMAP crashed"}, headers=WORKER)
    t = client.get(f"/tasks/{tid}", headers=user).json()
    assert t["status"] == "failed" and t["error"] == "COLMAP crashed"
    assert t["stage"] == "estimating_poses" and t["stage_label"] == "Estimating camera poses"  # where it failed
    # late event cannot resurrect it; heartbeat tells worker to stop
    assert client.post(f"/tasks/{tid}/progress", json={"stage": "training_gaussians", "progress": 70}, headers=WORKER).json()["ignored"]
    assert client.post(f"/tasks/{tid}/heartbeat", headers=WORKER).json()["status"] == "failed"
    # user can retry after failure
    assert client.post(f"/projects/{pid}/reconstruct", json={}, headers=user).status_code == 202


def test_stale_task_is_requeued_then_failed(client):
    user = signup(client, "stale@example.com")
    pid = new_project_with_video(client, user)
    tid = client.post(f"/projects/{pid}/reconstruct", json={}, headers=user).json()["id"]

    def age_heartbeat():
        with SessionLocal() as db:
            db.get(Task, tid).last_heartbeat = utcnow() - timedelta(seconds=3600)
            db.commit()

    assert client.post("/worker/next-task", headers=WORKER).json()["task"]["attempt"] == 1
    age_heartbeat()  # Kaggle session died
    again = client.post("/worker/next-task", headers=WORKER).json()["task"]
    assert again and again["id"] == tid and again["attempt"] == 2
    age_heartbeat()  # died again; MAX_ATTEMPTS=2 exhausted
    assert client.post("/worker/next-task", headers=WORKER).json()["task"] is None
    t = client.get(f"/tasks/{tid}", headers=user).json()
    assert t["status"] == "failed" and "stopped responding" in t["error"]


def test_heartbeat_keeps_long_stage_alive(client):
    user = signup(client, "beat@example.com")
    pid = new_project_with_video(client, user)
    tid = client.post(f"/projects/{pid}/reconstruct", json={}, headers=user).json()["id"]
    client.post("/worker/next-task", headers=WORKER)
    with SessionLocal() as db:
        db.get(Task, tid).last_heartbeat = utcnow() - timedelta(seconds=3600)
        db.commit()
    client.post(f"/tasks/{tid}/heartbeat", headers=WORKER)  # worker is alive
    assert client.post("/worker/next-task", headers=WORKER).json()["task"] is None
    assert client.get(f"/tasks/{tid}", headers=user).json()["status"] == "running"


def test_isolation_and_auth(client):
    a = signup(client, "a@example.com")
    b = signup(client, "b@example.com")
    pid = new_project_with_video(client, a)
    tid = client.post(f"/projects/{pid}/reconstruct", json={}, headers=a).json()["id"]

    assert client.get("/projects", headers=b).json() == []
    for url in (f"/projects/{pid}", f"/projects/{pid}/scene", f"/projects/{pid}/tasks", f"/tasks/{tid}"):
        assert client.get(url, headers=b).status_code == 404, url
    assert client.post(f"/projects/{pid}/reconstruct", json={}, headers=b).status_code == 404

    assert client.get("/projects").status_code == 401  # no token
    assert client.get("/projects", headers={"Authorization": "Bearer garbage"}).status_code == 401
    # a user token is not a worker token and vice versa
    assert client.post("/worker/next-task", headers=a).status_code == 401
    assert client.post("/worker/next-task").status_code == 401
    assert client.post(f"/tasks/{tid}/progress", json={"stage": "x", "progress": 1}, headers=a).status_code == 401
    assert client.get("/projects", headers=WORKER).status_code == 401

    # login works / wrong password rejected / duplicate signup rejected
    assert client.post("/auth/login", json={"email": "a@example.com", "password": "correct-horse-1"}).status_code == 200
    assert client.post("/auth/login", json={"email": "a@example.com", "password": "wrong-password"}).status_code == 401
    assert client.post("/auth/signup", json={"email": "A@example.com", "password": "correct-horse-1"}).status_code == 409


def test_file_tokens_are_project_scoped_and_paths_are_confined(client):
    a = signup(client, "files-a@example.com")
    pid1 = new_project_with_video(client, a, "one")
    pid2 = new_project_with_video(client, a, "two")
    # finish project 1 quickly
    tid = client.post(f"/projects/{pid1}/reconstruct", json={}, headers=a).json()["id"]
    client.post("/worker/next-task", headers=WORKER)
    upload_parts(client, tid, make_delivery_zip(), 10**6)
    client.post(f"/tasks/{tid}/complete", headers=WORKER)
    url = client.get(f"/projects/{pid1}/scene", headers=a).json()["splat_url"]
    token = url.split("?t=")[1]

    # token for project 1 does not open project 2, and a bare path needs some credential
    assert client.get(f"/projects/{pid2}/files/video/input.mp4?t={token}").status_code == 403
    assert client.get(f"/projects/{pid1}/files/video/input.mp4").status_code == 401
    assert client.get(f"/projects/{pid1}/files/video/input.mp4?t=nonsense").status_code == 401
    # owner can fetch with bearer instead of token
    assert client.get(f"/projects/{pid1}/files/video/input.mp4", headers=a).status_code == 200
    # traversal attempts
    assert client.get(f"/projects/{pid1}/files/..%2F..%2Froom3d.db?t={token}").status_code == 404
    assert client.get(f"/projects/{pid1}/files/../{pid2}/video/input.mp4?t={token}").status_code in (403, 404)


def test_upload_validation(client):
    user = signup(client, "up@example.com")
    pid = client.post("/projects", json={"name": "p"}, headers=user).json()["id"]
    # no video yet -> cannot reconstruct
    assert client.post(f"/projects/{pid}/reconstruct", json={}, headers=user).status_code == 409
    bad_ext = client.post(f"/projects/{pid}/video", files={"file": ("x.exe", b"abc", "application/octet-stream")}, headers=user)
    assert bad_ext.status_code == 415
    empty = client.post(f"/projects/{pid}/video", files={"file": ("x.mp4", b"", "video/mp4")}, headers=user)
    assert empty.status_code == 422
    # bad reconstruct params
    client.post(f"/projects/{pid}/video", files={"file": ("x.mov", b"abc", "video/quicktime")}, headers=user)
    assert client.post(f"/projects/{pid}/reconstruct", json={"steps": 5}, headers=user).status_code == 422
    assert client.post(f"/projects/{pid}/reconstruct", json={"run_name": "../evil"}, headers=user).status_code == 422
    assert client.post(f"/projects/{pid}/reconstruct", json={"steps": 15000, "run_name": "hq_15k"}, headers=user).json()["run_name"] == "hq_15k"
    # cannot swap the video while a task is queued
    assert client.post(f"/projects/{pid}/video", files={"file": ("y.mp4", b"abc", "video/mp4")}, headers=user).status_code == 409


def test_malicious_archives_rejected(client):
    user = signup(client, "zip@example.com")
    pid = new_project_with_video(client, user)
    tid = client.post(f"/projects/{pid}/reconstruct", json={}, headers=user).json()["id"]
    client.post("/worker/next-task", headers=WORKER)

    evil = make_delivery_zip({"artifacts/../../escape.txt": b"pwned"})
    r = client.put(f"/tasks/{tid}/artifacts", params={"part": 0, "parts": 1}, content=evil, headers=WORKER)
    assert r.status_code == 422
    r = client.put(f"/tasks/{tid}/artifacts", params={"part": 0, "parts": 1}, content=b"not a zip", headers=WORKER)
    assert r.status_code == 422
    nothing = io.BytesIO()
    with zipfile.ZipFile(nothing, "w") as z:
        z.writestr("other/file.txt", "x")
    r = client.put(f"/tasks/{tid}/artifacts", params={"part": 0, "parts": 1}, content=nothing.getvalue(), headers=WORKER)
    assert r.status_code == 422
    # a valid upload still works afterwards (bad attempts left no residue)
    assert upload_parts(client, tid, make_delivery_zip(), 10**6)["extracted"]
