import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from common import add_service_args, elapsed, project_root, save_json, send_progress


def align_similarity(source, target, src_R=None, dst_R=None):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    xm, ym = source.mean(0), target.mean(0)
    X, Y = source - xm, target - ym
    cov = Y.T @ X / max(len(X), 1)
    U, d, VT = np.linalg.svd(cov)
    D = np.diag([1., 1., np.linalg.det(U @ VT)])
    R = U @ D @ VT
    if np.linalg.matrix_rank(X) < 2 and src_R is not None and dst_R is not None:
        relative = dst_R @ np.transpose(src_R, (0, 2, 1))
        R = Rotation.from_matrix(relative).mean().as_matrix()
    variance = float(np.mean(np.sum(X * X, axis=1)))
    if variance < 1e-9:
        raise RuntimeError('Chunk overlap has negligible camera displacement; cannot estimate reliable scale')
    scale = float(np.sum((X @ R.T) * Y) / np.sum(X * X))
    if not np.isfinite(scale) or scale <= 1e-7:
        raise RuntimeError('Invalid VGGT chunk alignment scale')
    t = ym - scale * (R @ xm)
    residual = np.linalg.norm(scale * (source @ R.T) + t - target, axis=1)
    return scale, R, t, float(np.mean(residual))


def write_ply(path, points, colors):
    points = np.asarray(points, dtype=np.float32)
    colors = np.clip(np.asarray(colors) * 255., 0, 255).astype(np.uint8)
    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    array = np.empty(len(points), dtype=dtype)
    for j, key in enumerate(('x','y','z')):
        array[key] = points[:,j]
    for j, key in enumerate(('red','green','blue')):
        array[key] = colors[:,j]
    header = f'ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'
    with path.open('wb') as stream:
        stream.write(header.encode('ascii'))
        array.tofile(stream)


def main():
    parser = add_service_args(argparse.ArgumentParser())
    parser.add_argument('--mode', required=True, choices=['benchmark','full'])
    parser.add_argument('--chunk_size', type=int, default=24)
    parser.add_argument('--overlap', type=int, default=8)
    parser.add_argument('--repository', default='/kaggle/working/vggt')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    start = time.perf_counter()
    root = project_root(args.project_dir)
    parent = root / ('pose_comparison' if args.mode=='benchmark' else 'full_reconstruction')
    paths = sorted((parent / 'images').glob('*.jpg'))
    if len(paths) < 3:
        raise RuntimeError('Need at least three images for VGGT')
    if args.chunk_size <= args.overlap + 2 or args.overlap < 3:
        raise ValueError('chunk_size must exceed overlap+2, and overlap must be at least 3')
    output = parent / 'vggt'
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / 'vggt_poses.npz'
    metadata_path = output / 'metadata.json'
    if results_path.exists() and metadata_path.exists() and not args.force:
        print(metadata_path.read_text())
        return
    if not torch.cuda.is_available():
        raise RuntimeError('VGGT needs CUDA')
    repo = Path(args.repository)
    if not (repo / 'vggt' / 'models' / 'vggt.py').is_file():
        subprocess.run(['git','clone','https://github.com/facebookresearch/vggt.git',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'checkout','a288dd0f14786c93483e45524328726ab7b1b4ce'],check=True)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    send_progress(args,'estimating_poses_vggt',30)
    model_load_start = time.perf_counter()
    model = VGGT.from_pretrained('facebook/VGGT-1B').to('cuda:0').eval()
    load_seconds = elapsed(model_load_start)
    all_Rcw = [None] * len(paths)
    all_C = [None] * len(paths)
    all_K = [None] * len(paths)
    residuals = []
    sampled_points = []
    sampled_colors = []
    chunk_seconds = []
    torch.cuda.set_device('cuda:0')
    step = args.chunk_size - args.overlap
    begins = list(range(0, len(paths), step)) if args.mode=='full' else [0]
    for chunk_index, begin in enumerate(begins):
        end = min(begin + (args.chunk_size if args.mode=='full' else len(paths)), len(paths))
        indices = list(range(begin, end))
        if not indices:
            continue
        if args.mode=='full' and end - begin < args.overlap + 1 and chunk_index:
            indices = list(range(max(0, len(paths)-args.chunk_size), len(paths)))
            if all_C[indices[-1]] is not None:
                break
        batch_start = time.perf_counter()
        batch = load_and_preprocess_images([str(paths[j]) for j in indices]).to('cuda:0')
        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                prediction = model(batch)
            E,K = pose_encoding_to_extri_intri(prediction['pose_enc'], batch.shape[-2:])
        torch.cuda.synchronize()
        e = E.squeeze(0).float().cpu().numpy()
        k = K.squeeze(0).float().cpu().numpy()
        local_Rcw = np.transpose(e[:,:,:3], (0,2,1))
        local_C = -np.einsum('nij,nj->ni', local_Rcw, e[:,:,3])
        overlap_local = [j for j, global_index in enumerate(indices) if all_C[global_index] is not None]
        if chunk_index == 0:
            scale, global_rot, global_t, error = 1., np.eye(3), np.zeros(3), 0.
        elif len(overlap_local) >= 3:
            scale,global_rot,global_t,error = align_similarity(
                local_C[overlap_local],np.asarray([all_C[indices[j]] for j in overlap_local]),
                local_Rcw[overlap_local],np.asarray([all_Rcw[indices[j]] for j in overlap_local])
            )
        else:
            raise RuntimeError(f'Insufficient overlap in VGGT chunk starting at frame {begin}')
        residuals.append(error)
        global_C = scale * (local_C @ global_rot.T) + global_t
        global_Rcw = global_rot @ local_Rcw
        for local_index, global_index in enumerate(indices):
            if all_C[global_index] is None:
                all_C[global_index] = global_C[local_index]
                all_Rcw[global_index] = global_Rcw[local_index]
                all_K[global_index] = k[local_index]
        points_tensor = prediction.get('world_points')
        if points_tensor is not None:
            world = points_tensor.detach().float().cpu().numpy().squeeze(0)
            rgb_images = batch.detach().float().cpu().numpy()
            if world.ndim == 4 and world.shape[1:3] == tuple(batch.shape[-2:]):
                for local_index, global_index in enumerate(indices):
                    if global_index not in indices[:args.overlap] or chunk_index == 0:
                        p = world[local_index][::12,::12].reshape(-1,3)
                        colors = np.moveaxis(rgb_images[local_index],0,-1)[::12,::12].reshape(-1,3)
                        good = np.isfinite(p).all(axis=1)
                        sampled_points.append((scale * (p[good] @ global_rot.T) + global_t).astype(np.float32))
                        sampled_colors.append(np.clip(colors[good],0,1).astype(np.float32))
        chunk_seconds.append(elapsed(batch_start))
        print(f'VGGT chunk {chunk_index+1}: {begin}..{end-1}, residual={error:.5f}, seconds={chunk_seconds[-1]:.2f}', flush=True)
        del prediction,batch,E,K
        gc.collect()
        torch.cuda.empty_cache()
        if args.mode == 'benchmark':
            break
    if any(x is None for x in all_C):
        raise RuntimeError(f'VGGT chunks covered only {sum(x is not None for x in all_C)}/{len(paths)} frames')
    C = np.asarray(all_C, dtype=np.float32)
    Rcw = np.asarray(all_Rcw, dtype=np.float32)
    Rc = np.transpose(Rcw, (0,2,1))
    extr = np.concatenate((Rc,-np.einsum('nij,nj->ni',Rc,C)[:,:,None]),axis=2)
    intr = np.asarray(all_K, dtype=np.float32)
    np.savez_compressed(results_path, extrinsics=extr, intrinsics=intr, image_names=np.array([x.name for x in paths]))
    plt.figure(figsize=(8,6))
    ax = plt.axes(projection='3d')
    ax.plot(C[:,0],C[:,1],C[:,2],'-o',markersize=2)
    if sampled_points:
        cloud = np.concatenate(sampled_points,axis=0)
        colors = np.concatenate(sampled_colors,axis=0)
        write_ply(output/'vggt_pointcloud_preview.ply', cloud,colors)
        preview = cloud[::max(1,len(cloud)//20000)]
        ax.scatter(preview[:,0],preview[:,1],preview[:,2],s=.1,alpha=.15)
    ax.set_title(f'VGGT {args.mode}: chunk-aligned camera trajectory / points')
    plt.savefig(output/'vggt_pose_pointcloud.png',dpi=170)
    plt.close()
    info = {'stage': 'vggt_'+args.mode, 'frames': len(paths), 'predicted_poses': len(C), 'geometrically_verified_registered_frames': None, 'input_frame_size': 'VGGT internally resized; check model_image_hw', 'model_image_hw': list(rgb_images.shape[-2:]) if sampled_points else None, 'batch_chunk_size': args.chunk_size if args.mode=='full' else len(paths), 'overlap': args.overlap if args.mode=='full' else 0, 'alignment_residual_mean': float(np.mean(residuals)), 'alignment_residual_max': float(np.max(residuals)), 'inference_chunk_seconds': round(sum(chunk_seconds),2), 'model_loading_seconds': load_seconds, 'total_seconds': elapsed(start), 'peak_gpu_allocated_vram_gb': round(torch.cuda.max_memory_allocated()/1024**3,3), 'pose_path': str(results_path), 'point_cloud_preview': str(output/'vggt_pointcloud_preview.ply') if sampled_points else None, 'notes': 'Similarity-aligned overlapping VGGT chunks are research estimates; validate for drift, scale consistency and point-cloud quality. Camera predictions are not equivalent to SfM registration.'}
    save_json(metadata_path, info)
    send_progress(args,'estimating_poses_vggt',42 if args.mode=='benchmark' else 62)
    print(json.dumps(info,indent=2))


if __name__=='__main__':
    main()
