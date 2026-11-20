# Week 2 — Backend and Viewer Handoff

**Project root:** `/kaggle/working/room_3d_handoff_v2`

| Artifact | Relative path |
|---|---|
| Full video frames | `full_reconstruction/images/` |
| Undistorted frames | `3dgs_dataset/images/` |
| Undistorted pinhole intrinsics | `3dgs_dataset/sparse_txt/cameras.txt` |
| COLMAP registered poses | `3dgs_dataset/sparse_txt/images.txt` |
| Sparse 3D points | `3dgs_dataset/sparse_txt/points3D.txt` |
| Binary sparse model | `3dgs_dataset/sparse/0/` |
| VGGT full trajectory and internal intrinsics | `full_reconstruction/vggt/vggt_poses.npz` |
| SH degree 2 Gaussian model | `exports/stable_7000/scene.ply` |
| Compact DC-only-color scene | `exports/stable_7000/scene.splat` |
| Inference-only Gaussian weights | `3dgs_training/stable_7000/inference_weights.pt` |
| Train/monitor/test frame split | `3dgs_training/stable_7000/split.json` |
| Held-out metrics and images | `3dgs_training/stable_7000/evaluation/` |
| Individual backend-ready stages | `pipeline/*.py` in the ZIP |

**Camera coordinates.** COLMAP uses local camera +X right, +Y down, +Z forward. The stored quaternion and translation are world-to-camera. World-space camera centers are `-R.T @ t`. Convert to OpenGL-style camera-to-world coordinates (+Y up and -Z forward) by right-multiplying the camera-to-world matrix by `diag(1,-1,-1,1)`. World alignment is arbitrary. If a world transform is needed, apply it to the Gaussian scene and each camera-to-world matrix consistently.

**Intrinsics integrity.** Undistorted images from `image_undistorter` must be paired with the resulting pinhole calibration. Do not apply the undistorted camera intrinsics to pre-undistortion frames. The PLY contains higher-degree spherical harmonic coefficients while the legacy `.splat` stores only DC colors. Viewer rendering and the initial camera must be validated independently.

**Backend interface.** Stage scripts accept `--project_dir`. Optional `--backend_url` and `--task_id` enable `POST /tasks/{id}/progress`; events are also logged locally in `progress_events.jsonl`. Work is invoked by subprocesses; scheduling, API authentication, and persistent task queuing belong to the backend worker implementation. Persist the ZIP in backend storage before Kaggle sessions expire.

**Evaluation.** The model reached 7000 iterations with 23350 Gaussians. Gaussian optimization excluded 18 held-out frames, with 10 test frames not used for best-checkpoint selection; all frames participated in the SfM solution. Mean PSNR across all held-out views: 23.925 dB; separately monitored-test PSNR: 24.553 dB. These view-interpolation metrics do not prove unrestricted novel-view quality. A longer 15,000-iteration run must be separately recorded if performed.
