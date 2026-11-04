import argparse
import csv
import json
import math
from common import add_service_args, project_root, save_json


def main():
    parser = add_service_args(argparse.ArgumentParser())
    parser.add_argument('--run_name', default='stable_7000')
    args = parser.parse_args()
    root = project_root(args.project_dir)
    reports = root / 'reports'
    reports.mkdir(parents=True, exist_ok=True)

    def get(name):
        p = root / name
        return json.loads(p.read_text()) if p.is_file() else {}

    def fmt(val):
        return 'not measured' if val is None else str(val)

    pre = get('preprocessing_metadata.json')
    vb = get('pose_comparison/vggt/metadata.json')
    vf = get('full_reconstruction/vggt/metadata.json')
    gb = get('pose_comparison/glomap/mapping_metadata.json')
    gf = get('full_reconstruction/glomap/mapping_metadata.json')
    und = get('3dgs_dataset/metadata.json')
    train = get(f'3dgs_training/{args.run_name}/training_metadata.json')
    ev = get(f'3dgs_training/{args.run_name}/evaluation/heldout_summary.json')
    ex = get(f'exports/{args.run_name}/scene_metadata.json')

    rows = [
        ['Extraction and filtering', pre.get('processing_seconds'), 0, 'CPU decode and filtering'],
        ['VGGT 32-frame benchmark', vb.get('total_seconds'), vb.get('peak_gpu_allocated_vram_gb'), 'Includes loading; PyTorch allocated peak'],
        ['GLOMAP 32-frame benchmark', gb.get('total_seconds'), 0, 'CPU SIFT/matching 0 GB; global mapping GPU peak unmeasured'],
        ['VGGT full chunk merge', vf.get('total_seconds'), vf.get('peak_gpu_allocated_vram_gb'), 'Includes model loading; PyTorch allocated peak'],
        ['GLOMAP full reconstruction', gf.get('total_seconds'), 0, 'CPU SIFT/matching 0 GB; global mapping GPU peak unmeasured'],
        ['COLMAP image undistortion', und.get('seconds'), 0, 'CPU stage'],
        ['Gaussian training', train.get('cumulative_training_seconds', train.get('time_this_run_seconds')), train.get('peak_allocated_gpu_vram_gb'), 'PyTorch allocated peak; excludes JIT compilation'],
        ['Held-out evaluation', ev.get('runtime_seconds'), ev.get('peak_allocated_gpu_vram_gb'), 'Unmeasured values left blank'],
        ['Gaussian PLY/SPLAT export', ex.get('runtime_seconds'), 0, 'CPU model export']
    ]
    with (reports / 'runtime_vram.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['stage', 'runtime_seconds', 'allocated_gpu_vram_gb', 'measurement_notes'])
        writer.writerows(rows)

    chunk_count = len(vf.get('chunk_seconds', [])) if vf.get('chunk_seconds') else None
    if chunk_count is None and vf.get('frames') and vf.get('batch_chunk_size') and vf.get('overlap'):
        chunk_count = 1 + math.ceil(max(0, vf['frames'] - vf['batch_chunk_size']) / (vf['batch_chunk_size'] - vf['overlap']))

    comparison = f'''# VGGT vs GLOMAP — Reconstruction Benchmark

VGGT and GLOMAP were evaluated on matching frame identities from the same video. VGGT uses internally resized images and predicts poses without SfM registration; GLOMAP uses extracted frames and verifies image registration geometrically.

| Metric | VGGT benchmark | GLOMAP benchmark | VGGT full sequence | GLOMAP full sequence |
|---|---:|---:|---:|---:|
| Input frames | {fmt(vb.get('frames'))} | {fmt(gb.get('input_images'))} | {fmt(vf.get('frames'))} | {fmt(gf.get('input_images'))} |
| Predicted / registered frames | {fmt(vb.get('predicted_poses'))} predicted | {fmt(gb.get('registered_frames'))} registered | {fmt(vf.get('predicted_poses'))} predicted | {fmt(gf.get('registered_frames'))} registered |
| Sparse triangulated points | N/A (dense predicted cloud) | {fmt(gb.get('sparse_points'))} | N/A (dense predicted cloud) | {fmt(gf.get('sparse_points'))} |
| Runtime (seconds) | {fmt(vb.get('total_seconds'))} | {fmt(gb.get('total_seconds'))} | {fmt(vf.get('total_seconds'))} | {fmt(gf.get('total_seconds'))} |
| GPU VRAM (GB) | {fmt(vb.get('peak_gpu_allocated_vram_gb'))} allocated | 0 CPU matching; mapper unmeasured | {fmt(vf.get('peak_gpu_allocated_vram_gb'))} allocated | 0 CPU matching; mapper unmeasured |
| Resolution | {fmt(vb.get('model_image_hw'))} H×W | Extracted frames | {fmt(vf.get('model_image_hw'))} H×W | Extracted frames |
| Relative verification | Chunk residual not applicable | {fmt(gb.get('verified_pairs'))} verified pairs | {fmt(vf.get('alignment_residual_mean'))} mean chunk-alignment residual | {fmt(gf.get('verified_pairs'))} verified pairs |

**VGGT chunking:** {fmt(chunk_count)} overlapping chunks were used on the full sequence. The alignment residual measures internal agreement between overlapping chunks; it is not a ground-truth absolute trajectory error.

**Selected method:** GLOMAP supplies geometrically registered cameras, COLMAP intrinsics, and sparse geometry suitable for undistortion and Gaussian training. VGGT is substantially faster on the full sequence but its camera estimates and merged point map require additional validation before using them for reconstruction. The full GLOMAP run used {fmt(gf.get('matching_strategy'))} feature matching and spent {fmt(gf.get('matching_seconds'))} seconds matching. Production defaults to sequential matching for lower CPU cost, with less loop-closure coverage as a trade-off.

**Measurement boundary:** 0 GB refers to CPU SIFT feature extraction and matching. GLOMAP global mapper peak GPU allocation was not independently instrumented and is not reported as a measured zero. The methods also use different effective image resolutions, so the runtimes are not resolution-controlled.

**Visual checks:** `pose_comparison/vggt/vggt_pose_pointcloud.png`, `pose_comparison/glomap/sparse_preview.png`, `full_reconstruction/vggt/vggt_pose_pointcloud.png`, `full_reconstruction/glomap/sparse_preview.png`.
'''
    (reports / 'comparison.md').write_text(comparison)

    handoff = f'''# Week 2 — Backend and Viewer Handoff

**Project root:** `{root}`

| Artifact | Relative path |
|---|---|
| Full video frames | `full_reconstruction/images/` |
| Undistorted frames | `3dgs_dataset/images/` |
| Undistorted pinhole intrinsics | `3dgs_dataset/sparse_txt/cameras.txt` |
| COLMAP registered poses | `3dgs_dataset/sparse_txt/images.txt` |
| Sparse 3D points | `3dgs_dataset/sparse_txt/points3D.txt` |
| Binary sparse model | `3dgs_dataset/sparse/0/` |
| VGGT full trajectory and internal intrinsics | `full_reconstruction/vggt/vggt_poses.npz` |
| SH degree {fmt(ex.get('sh_degree'))} Gaussian model | `exports/{args.run_name}/scene.ply` |
| Compact DC-only-color scene | `exports/{args.run_name}/scene.splat` |
| Inference-only Gaussian weights | `3dgs_training/{args.run_name}/inference_weights.pt` |
| Train/monitor/test frame split | `3dgs_training/{args.run_name}/split.json` |
| Held-out metrics and images | `3dgs_training/{args.run_name}/evaluation/` |
| Individual backend-ready stages | `pipeline/*.py` in the ZIP |

**Camera coordinates.** COLMAP uses local camera +X right, +Y down, +Z forward. The stored quaternion and translation are world-to-camera. World-space camera centers are `-R.T @ t`. For an OpenGL-style camera-to-world view, use `T_gl = inv(T_colmap_world_to_camera) @ diag(1,-1,-1,1)`. The explicit camera-axis correction is a 180-degree rotation around local X: `[[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]`. This camera-axis correction does not itself require rotating the scene points. If a viewer needs an additional world transform, apply the same chosen transform to both scene Gaussians and camera-to-world matrices. No unique scene-world flip was calibrated in SuperSplat; import compatibility was visually checked but the initial view was not aligned against an exact source pose.

**Intrinsics integrity.** Undistorted images from `image_undistorter` must be paired with the resulting pinhole calibration. Do not apply the undistorted camera intrinsics to pre-undistortion frames. The PLY contains higher-degree spherical harmonic coefficients while the legacy `.splat` stores only DC colors. Viewer rendering and the initial camera must be validated independently.

**Backend interface.** Stage scripts accept `--project_dir`. Optional `--backend_url` and `--task_id` enable `POST /tasks/{{id}}/progress`; events are also logged locally in `progress_events.jsonl`. Work is invoked by subprocesses; scheduling, API authentication, and persistent task queuing belong to the backend worker implementation. Persist the ZIP in backend storage before Kaggle sessions expire.

**Evaluation.** The model reached {fmt(ex.get('checkpoint_step'))} iterations with {fmt(ex.get('gaussian_count'))} Gaussians. Gaussian optimization excluded {fmt(ev.get('evaluated_frames'))} held-out frames, with {fmt(ev.get('test_frames'))} test frames not used for best-checkpoint selection; all frames participated in the SfM solution. Mean PSNR across all held-out views: {fmt(ev.get('mean_psnr_db'))} dB; separately monitored-test PSNR: {fmt(ev.get('test_mean_psnr_db'))} dB. These view-interpolation metrics do not prove unrestricted novel-view quality. A longer 15,000-iteration run must be separately recorded if performed.
'''
    (reports / 'week2_handoff.md').write_text(handoff)
    viewer_path=reports/'viewer_validation.json'
    viewer_record=json.loads(viewer_path.read_text()) if viewer_path.is_file() else {}
    status = {
        'run_name': args.run_name,
        'training_steps': train.get('steps_completed'),
        'gaussians': ex.get('gaussian_count'),
        'heldout_mean_psnr_db': ev.get('mean_psnr_db'),
        'heldout_mean_ssim': ev.get('mean_ssim'),
        'test_mean_psnr_db': ev.get('test_mean_psnr_db'),
        'test_mean_ssim': ev.get('test_mean_ssim'),
        'viewer_note': viewer_record.get('notes','Viewer quality not recorded for this project'),
        'completed_15000_steps': bool(ex.get('checkpoint_step', 0) >= 15000),
        'test_scope': 'Test frames withheld from 3DGS training and checkpoint selection, but not from SfM'
    }
    save_json(reports / 'validation_status.json', status)
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
