"""Auth, ownership, signed file URLs, upload validation and archive safety (no GPU needed)."""
import io
import zipfile

import pytest

from app import storage


def signup(client, email):
    r = client.post("/auth/signup", json={"email": email, "password": "correct-horse-1"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def new_project_with_video(client, headers, name="Living room"):
    pid = client.post("/projects", json={"name": name}, headers=headers).json()["id"]
    r = client.post(f"/projects/{pid}/video", files={"file": ("room.mp4", b"\x00" * 4096, "video/mp4")}, headers=headers)
    assert r.status_code == 200, r.text
    return pid


def test_isolation_and_auth(client):
    a, b = signup(client, "a@example.com"), signup(client, "b@example.com")
    pid = new_project_with_video(client, a)
    tid = client.post(f"/projects/{pid}/reconstruct", json={}, headers=a).json()["id"]
    assert client.get("/projects", headers=b).json() == []
    for url in (f"/projects/{pid}", f"/projects/{pid}/scene", f"/projects/{pid}/tasks", f"/tasks/{tid}", f"/projects/{pid}/frames", f"/projects/{pid}/edits"):
        assert client.get(url, headers=b).status_code == 404, url
    assert client.post(f"/projects/{pid}/edits", json={}, headers=b).status_code == 404
    assert client.post(f"/tasks/{tid}/cancel", headers=b).status_code == 404
    assert client.get("/projects").status_code == 401
    assert client.get("/projects", headers={"Authorization": "Bearer garbage"}).status_code == 401
    assert client.get("/gpu/status").status_code == 401
    assert client.post("/auth/login", json={"email": "a@example.com", "password": "wrong-password"}).status_code == 401
    assert client.post("/auth/signup", json={"email": "A@example.com", "password": "correct-horse-1"}).status_code == 409


def test_edit_needs_a_scene(client):
    a = signup(client, "noscene@example.com")
    pid = new_project_with_video(client, a)
    assert client.post(f"/projects/{pid}/edits", json={}, headers=a).status_code == 409
    assert client.get(f"/projects/{pid}/frames", headers=a).status_code == 409


def test_upload_validation(client):
    user = signup(client, "up@example.com")
    pid = client.post("/projects", json={"name": "p"}, headers=user).json()["id"]
    assert client.post(f"/projects/{pid}/reconstruct", json={}, headers=user).status_code == 409
    assert client.post(f"/projects/{pid}/video", files={"file": ("x.exe", b"abc", "application/octet-stream")}, headers=user).status_code == 415
    assert client.post(f"/projects/{pid}/video", files={"file": ("x.mp4", b"", "video/mp4")}, headers=user).status_code == 422
    client.post(f"/projects/{pid}/video", files={"file": ("x.mov", b"abc", "video/quicktime")}, headers=user)
    assert client.post(f"/projects/{pid}/reconstruct", json={"steps": 5}, headers=user).status_code == 422
    assert client.post(f"/projects/{pid}/reconstruct", json={"run_name": "../evil"}, headers=user).status_code == 422
    assert client.post(f"/projects/{pid}/reconstruct", json={"frame_fps": 9}, headers=user).status_code == 422
    assert client.post(f"/projects/{pid}/reconstruct", json={"steps": 15000, "run_name": "hq_15k"}, headers=user).json()["run_name"] == "hq_15k"
    assert client.post(f"/projects/{pid}/video", files={"file": ("y.mp4", b"abc", "video/mp4")}, headers=user).status_code == 409


def test_malicious_archives_rejected(tmp_path):
    def zipped(files):
        p = tmp_path / "z.zip"
        with zipfile.ZipFile(p, "w") as z:
            for k, v in files.items():
                z.writestr(k, v)
        return p
    with pytest.raises(ValueError):
        storage.safe_extract(zipped({"artifacts/../../escape.txt": b"pwned"}), tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()
    assert storage.safe_extract(zipped({"other/file.txt": "x"}), tmp_path / "out2") == 0
    assert storage.safe_extract(zipped({"artifacts/a.txt": "x"}), tmp_path / "out3") == 1
