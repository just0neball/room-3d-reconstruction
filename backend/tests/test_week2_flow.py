"""Whole Week-2 flow through the website API against the real GPU server running in mock mode:
upload -> reconstruct -> frames/cameras -> click-segment -> track -> edit reference -> accept -> edit all -> rebuild -> before/after."""
import time

from test_security import new_project_with_video, signup


def wait_task(client, h, tid, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        t = client.get(f"/tasks/{tid}", headers=h).json()
        if t["status"] in ("completed", "failed", "cancelled"):
            return t
        time.sleep(0.4)
    raise TimeoutError(client.get(f"/tasks/{tid}", headers=h).json())


def test_gpu_status_and_config(client):
    h = signup(client, "gpu@example.com")
    s = client.get("/gpu/status", headers=h).json()
    assert s["reachable"] and s["mock"] and s["configured"]
    dead = client.put("/gpu/config", json={"url": "http://127.0.0.1:9"}, headers=h).json()
    assert dead["reachable"] is False and dead["configured"]
    # restore the real one (token is kept)
    from conftest import GPU_PORT
    ok = client.put("/gpu/config", json={"url": f"http://127.0.0.1:{GPU_PORT}", "token": "test-gpu-token"}, headers=h).json()
    assert ok["reachable"]


def test_waits_when_gpu_offline_then_runs(client):
    from conftest import GPU_PORT
    h = signup(client, "offline@example.com")
    client.put("/gpu/config", json={"url": "http://127.0.0.1:9"}, headers=h)
    pid = new_project_with_video(client, h)
    tid = client.post(f"/projects/{pid}/reconstruct", json={}, headers=h).json()["id"]
    t0 = time.time()
    while time.time() - t0 < 60:  # the dispatcher may still be finishing a leftover task from an earlier test
        t = client.get(f"/tasks/{tid}", headers=h).json()
        if t["detail"] == "Waiting for GPU server":
            break
        time.sleep(0.5)
    assert t["status"] == "pending" and t["detail"] == "Waiting for GPU server"
    client.put("/gpu/config", json={"url": f"http://127.0.0.1:{GPU_PORT}", "token": "test-gpu-token"}, headers=h)
    assert wait_task(client, h, tid)["status"] == "completed"


def test_full_edit_flow(client):
    h = signup(client, "flow@example.com")
    pid = new_project_with_video(client, h, "Mock room")
    tid = client.post(f"/projects/{pid}/reconstruct", json={"frame_fps": 1}, headers=h).json()["id"]
    seen = set()
    t0 = time.time()
    while True:
        t = client.get(f"/tasks/{tid}", headers=h).json()
        seen.add(t["stage_group"])
        if t["status"] in ("completed", "failed"):
            break
        assert time.time() - t0 < 120
        time.sleep(0.3)
    assert t["status"] == "completed", t
    assert {"extract", "poses", "train"} <= seen

    scene = client.get(f"/projects/{pid}/scene", headers=h).json()
    assert scene["splat_url"] and scene["cameras_url"] and scene["edits"] == []
    assert client.get(scene["splat_url"]).status_code == 200

    fr = client.get(f"/projects/{pid}/frames", headers=h).json()
    assert fr["count"] == 40 and fr["image_size"] == {"width": 640, "height": 360}
    assert client.get(fr["frames"][3]["thumb_url"]).headers["content-type"] == "image/jpeg"
    assert client.get(fr["frames"][3]["url"]).status_code == 200
    cams = client.get(f"/projects/{pid}/cameras", headers=h).json()
    assert cams["up"] == [0, -1, 0] and len(cams["cameras"]) == 40

    e = client.post(f"/projects/{pid}/edits", json={"name": "Green sofa"}, headers=h).json()
    eid = e["id"]
    assert e["state"] == "draft"

    # option 2: a 3D point on the sofa -> best training frame + pixel
    pick = client.post(f"/edits/{eid}/pick3d", json={"point": [0.0, 1.0, 2.2]}, headers=h).json()
    assert pick["frame"] and 0 <= pick["x"] < 640

    frame = fr["frames"][0]["name"]
    seg = client.post(f"/edits/{eid}/segment", json={"frame": frame, "points": [{"x": 320, "y": 250, "label": 1}]}, headers=h)
    assert seg.status_code == 200, seg.text
    assert 0.1 < seg.json()["area"] < 0.6
    assert client.get(seg.json()["overlay_url"]).status_code == 200
    assert client.get(f"/edits/{eid}", headers=h).json()["state"] == "masked"

    # tracking
    task = client.post(f"/edits/{eid}/track", json={"method": "auto"}, headers=h)
    assert task.status_code == 202, task.text
    assert wait_task(client, h, task.json()["id"])["status"] == "completed"
    e = client.get(f"/edits/{eid}", headers=h).json()
    assert e["state"] == "tracked" and e["tracking"]["included_count"] >= 5
    inc = {f["index"] for f in e["tracking"]["frames"] if f["included"]}
    assert 0 in inc and 20 not in inc
    assert all(client.get(f["overlay_url"]).status_code == 200 for f in e["tracking"]["frames"] if f["included"])

    # user removes a wrong frame, then the manual fallback adds frame 20 (no sofa there) -> segmented from a hint or nearest mask
    victim = next(f["name"] for f in e["tracking"]["frames"] if f["included"] and f["source"] == "auto")
    e = client.patch(f"/edits/{eid}/frames", json={"exclude": [victim]}, headers=h).json()
    assert not next(f for f in e["tracking"]["frames"] if f["name"] == victim)["included"]
    e = client.patch(f"/edits/{eid}/frames", json={"include": [victim]}, headers=h).json()
    assert next(f for f in e["tracking"]["frames"] if f["name"] == victim)["included"]
    mt = client.post(f"/edits/{eid}/track", json={"method": "manual", "frames": [fr["frames"][2]["name"]]}, headers=h).json()
    assert wait_task(client, h, mt["id"])["status"] == "completed"
    e = client.get(f"/edits/{eid}", headers=h).json()
    assert next(f for f in e["tracking"]["frames"] if f["index"] == 2)["source"] in ("manual", "auto")

    # reference edit
    assert client.post(f"/edits/{eid}/edit/reference", json={}, headers=h).status_code == 422  # no prompt yet
    rt = client.post(f"/edits/{eid}/edit/reference", json={"prompt": "make the sofa dark green leather", "seeds": 3}, headers=h)
    assert rt.status_code == 202, rt.text
    assert wait_task(client, h, rt.json()["id"])["status"] == "completed"
    e = client.get(f"/edits/{eid}", headers=h).json()
    assert e["state"] == "reference_ready" and len(e["candidates"]) == 3
    assert client.get(e["candidates"][0]["thumb_url"]).status_code == 200
    assert client.post(f"/edits/{eid}/edit/run", headers=h).status_code == 409  # not accepted yet
    e = client.post(f"/edits/{eid}/edit/accept", json={"seed": e["candidates"][1]["seed"]}, headers=h).json()
    assert e["chosen_seed"] and e["reference_url"]

    # all frames
    rn = client.post(f"/edits/{eid}/edit/run", headers=h)
    assert rn.status_code == 202, rn.text
    assert client.post(f"/edits/{eid}/edit/run", headers=h).status_code == 409  # one task per project
    done = wait_task(client, h, rn.json()["id"])
    assert done["status"] == "completed", done
    e = client.get(f"/edits/{eid}", headers=h).json()
    assert e["state"] == "edited" and e["edit_progress"]["done"] == e["edit_progress"]["total"] > 3
    assert e["metrics"]["edit"]["outside_mask_max_diff"] == 0

    # rebuild
    rb = client.post(f"/edits/{eid}/rebuild", json={"mode": "finetune"}, headers=h)
    assert wait_task(client, h, rb.json()["id"])["status"] == "completed"
    e = client.get(f"/edits/{eid}", headers=h).json()
    assert e["state"] == "done" and e["result"]["splat_url"] and e["result"]["original_splat_url"]
    new, old = client.get(e["result"]["splat_url"]).content, client.get(e["result"]["original_splat_url"]).content
    assert len(new) == len(old) and new != old  # same Gaussians, recoloured
    scene = client.get(f"/projects/{pid}/scene", headers=h).json()
    assert scene["edits"][0]["splat_url"]
    assert len(client.get(f"/projects/{pid}/edits", headers=h).json()) == 1

    # a second edit is independent (edit history)
    e2 = client.post(f"/projects/{pid}/edits", json={}, headers=h).json()
    assert e2["id"] != eid and e2["name"] == "Edit 2"
    assert client.delete(f"/edits/{eid}", headers=h).status_code == 204
