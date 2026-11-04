import argparse
import json
import os
import shutil
import sqlite3
import struct
import subprocess
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import add_service_args, elapsed, load_json, project_root, save_json, send_progress


def read_count(path):
    with Path(path).open('rb') as stream:
        return struct.unpack('<Q', stream.read(8))[0]


def run_command(binary, args, destination, env):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='utf-8') as stream:
        result = subprocess.run([str(binary)] + args, stdout=stream, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(f'COLMAP failed: {destination}\n{destination.read_text()[-4000:]}')


def get_option(binary, command, candidates, env):
    result = subprocess.run([str(binary), command, '-h'], capture_output=True, text=True, env=env)
    help_text = result.stdout + result.stderr
    for name in candidates:
        if name in help_text:
            return name
    raise RuntimeError(f'No supported CPU option for {command}')


def plot_sparse(folder, text_dir):
    points = []
    for line in (text_dir / 'points3D.txt').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            words = line.split()
            points.append([float(v) for v in words[1:4]])
    centers = []
    for line in (text_dir / 'images.txt').read_text().splitlines():
        words = line.split()
        if len(words) < 10 or not words[9].lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        q = np.asarray([float(x) for x in words[1:5]], dtype=np.float64)
        w, x, y, z = q / np.linalg.norm(q)
        R = np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
        t = np.array([float(x) for x in words[5:8]])
        centers.append(-R.T @ t)
    if not points or not centers:
        return
    xyz = np.asarray(points)
    cs = np.asarray(centers)
    lo, hi = np.percentile(xyz, [1, 99], axis=0)
    xyz = xyz[np.all((xyz >= lo) & (xyz <= hi), axis=1)]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, alpha=.55)
    ax.scatter(cs[:, 0], cs[:, 1], cs[:, 2], s=9, c='red')
    ax.set_title('COLMAP global mapping: sparse point cloud and cameras')
    ax.set_box_aspect(np.maximum(np.ptp(np.concatenate((xyz, cs)), axis=0), .001))
    fig.savefig(folder / 'sparse_preview.png', dpi=150)
    plt.close(fig)


def main():
    parser = add_service_args(argparse.ArgumentParser())
    parser.add_argument('--mode', choices=['benchmark', 'full'], required=True)
    parser.add_argument('--colmap_bin', default='/kaggle/working/reconstruction_tools/colmap_env/bin/colmap')
    parser.add_argument('--camera_model', choices=['SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'OPENCV'], default='SIMPLE_RADIAL')
    parser.add_argument('--matching', choices=['sequential', 'exhaustive', 'both'], default='both')
    parser.add_argument('--reprocess', action='store_true')
    args = parser.parse_args()
    start = time.perf_counter()
    root = project_root(args.project_dir)
    parent = root / ('pose_comparison' if args.mode == 'benchmark' else 'full_reconstruction')
    images = parent / 'images'
    folder = parent / 'glomap'
    folder.mkdir(parents=True, exist_ok=True)
    binary = Path(args.colmap_bin)
    if not binary.is_file():
        raise FileNotFoundError(binary)
    filenames = sorted(p.name for p in images.glob('*.jpg'))
    if len(filenames) < 3:
        raise RuntimeError(f'Not enough images in {images}')
    meta_path = folder / 'mapping_metadata.json'
    sparse = folder / 'sparse' / '0'
    if meta_path.is_file() and (sparse / 'cameras.bin').is_file() and not args.reprocess:
        print(meta_path.read_text())
        send_progress(args, 'estimating_poses', 60 if args.mode == 'full' else 42)
        return
    env = os.environ.copy()
    env['QT_QPA_PLATFORM'] = 'offscreen'
    database = folder / 'database.db'
    if args.reprocess:
        for suffix in ('', '-wal', '-shm'):
            candidate = Path(str(database) + suffix)
            if candidate.exists():
                candidate.unlink()
    with sqlite3.connect(database) as con:
        has_images = con.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='images'").fetchone()[0] > 0
        current_count = con.execute('SELECT COUNT(*) FROM images').fetchone()[0] if has_images else 0
        has_matches = con.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='two_view_geometries'").fetchone()[0] > 0
        pair_count = con.execute('SELECT COUNT(*) FROM two_view_geometries WHERE rows>0').fetchone()[0] if has_matches else 0
    feature_seconds = matching_seconds = 0.
    if current_count != len(filenames) or pair_count == 0 or args.reprocess:
        for suffix in ('', '-shm', '-wal'):
            file = Path(str(database) + suffix)
            if file.exists():
                file.unlink()
        feature_option = get_option(binary, 'feature_extractor', ('--FeatureExtraction.use_gpu', '--SiftExtraction.use_gpu'), env)
        start_stage = time.perf_counter()
        run_command(binary, ['feature_extractor', '--database_path', str(database), '--image_path', str(images), '--ImageReader.single_camera', '1', '--ImageReader.camera_model', args.camera_model, feature_option, '0'], folder / 'feature_extraction.log', env)
        feature_seconds = elapsed(start_stage)
        send_progress(args, 'estimating_poses', 34 if args.mode == 'full' else 29)
        start_stage = time.perf_counter()
        if args.matching in ('sequential', 'both'):
            match_option = get_option(binary, 'sequential_matcher', ('--FeatureMatching.use_gpu', '--SiftMatching.use_gpu'), env)
            run_command(binary, ['sequential_matcher', '--database_path', str(database), '--SequentialMatching.overlap', '10', match_option, '0'], folder / 'sequential_matching.log', env)
        if args.matching in ('exhaustive', 'both'):
            match_option = get_option(binary, 'exhaustive_matcher', ('--FeatureMatching.use_gpu', '--SiftMatching.use_gpu'), env)
            run_command(binary, ['exhaustive_matcher', '--database_path', str(database), match_option, '0'], folder / 'exhaustive_matching.log', env)
        matching_seconds = elapsed(start_stage)
    with sqlite3.connect(database) as con:
        image_count = con.execute('SELECT COUNT(*) FROM images').fetchone()[0]
        feature_count = con.execute('SELECT COUNT(*) FROM keypoints WHERE rows>0').fetchone()[0]
        pairs = con.execute('SELECT COUNT(*) FROM two_view_geometries WHERE rows>0').fetchone()[0]
    if image_count != len(filenames) or feature_count < 3 or pairs == 0:
        raise RuntimeError('COLMAP database is incomplete')
    send_progress(args, 'estimating_poses', 45 if args.mode == 'full' else 35)
    candidate = folder / 'sparse_candidate'
    if candidate.exists():
        shutil.rmtree(candidate)
    candidate.mkdir()
    mapping_start = time.perf_counter()
    run_command(binary, ['global_mapper', '--database_path', str(database), '--image_path', str(images), '--output_path', str(candidate)], folder / 'global_mapping.log', env)
    mapping_seconds = elapsed(mapping_start)
    models = []
    for cameras_bin in candidate.rglob('cameras.bin'):
        directory = cameras_bin.parent
        if (directory/'images.bin').is_file() and (directory/'points3D.bin').is_file():
            models.append((int(read_count(directory/'images.bin')), int(read_count(directory/'points3D.bin')), directory))
    if not models:
        raise RuntimeError('No sparse model produced')
    registered, points, winner = max(models, key=lambda x: (x[0], x[1]))
    if registered < 2 or points < 1:
        raise RuntimeError('Insufficient registered cameras or sparse 3D points')
    final_sparse = folder / 'sparse'
    if final_sparse.exists():
        shutil.rmtree(final_sparse)
    final_sparse.mkdir()
    shutil.copytree(winner, final_sparse / '0', dirs_exist_ok=True)
    shutil.rmtree(candidate)
    text = folder / 'sparse_txt'
    if text.exists():
        shutil.rmtree(text)
    text.mkdir()
    run_command(binary, ['model_converter', '--input_path', str(final_sparse/'0'), '--output_path', str(text), '--output_type', 'TXT'], folder/'model_converter.log', env)
    plot_sparse(folder, text)
    info = {'stage': 'glomap_'+args.mode, 'input_images': len(filenames), 'registered_frames': registered, 'registration_ratio': round(registered/len(filenames),4), 'sparse_points': points, 'verified_pairs': pairs, 'feature_extraction_seconds': feature_seconds, 'matching_seconds': matching_seconds, 'global_mapping_seconds': mapping_seconds, 'total_seconds': elapsed(start), 'matching_strategy': args.matching, 'camera_model_used_for_feature_extraction': args.camera_model, 'gpu_sift': False, 'feature_matching_gpu_vram_gb': 0.0, 'global_mapper_peak_vram_gb': None, 'sparse_path': str(final_sparse/'0'), 'sparse_text_path': str(text), 'preview': str(folder/'sparse_preview.png'), 'notes': 'CPU SIFT/matching explicitly uses 0 GPU VRAM; global mapper GPU memory not instrumented. Re-run if comparing precise full-runtime metrics.'}
    save_json(meta_path, info)
    send_progress(args, 'estimating_poses', 60 if args.mode=='full' else 42)
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
