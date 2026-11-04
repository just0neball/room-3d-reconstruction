import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from PIL import Image
from common import add_service_args, elapsed, project_root, save_json, send_progress


def run(argv, log):
    env = os.environ.copy()
    env['QT_QPA_PLATFORM'] = 'offscreen'
    with log.open('w') as stream:
        result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(f'COLMAP failed: {log}\n{log.read_text()[-4000:]}')


def main():
    parser = add_service_args(argparse.ArgumentParser())
    parser.add_argument('--colmap_bin', default='/kaggle/working/reconstruction_tools/colmap_env/bin/colmap')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    start = time.perf_counter()
    root = project_root(args.project_dir)
    model = root/'full_reconstruction'/'glomap'/'sparse'/'0'
    source_images = root/'full_reconstruction'/'images'
    dataset = root/'3dgs_dataset'
    if not (model/'cameras.bin').is_file() or not list(source_images.glob('*.jpg')):
        raise RuntimeError('Full GLOMAP reconstruction is missing')
    if (dataset/'sparse'/'0'/'cameras.bin').exists() and not args.force:
        print((dataset/'metadata.json').read_text())
        return
    if dataset.exists() and args.force:
        shutil.rmtree(dataset)
    candidate = root/'3dgs_dataset_candidate'
    if candidate.exists():
        shutil.rmtree(candidate)
    candidate.mkdir()
    send_progress(args,'preparing_training_data',63)
    bin_path = Path(args.colmap_bin)
    if not bin_path.exists():
        raise FileNotFoundError(bin_path)
    run([str(bin_path),'image_undistorter','--image_path',str(source_images),'--input_path',str(model),'--output_path',str(candidate),'--output_type','COLMAP'],root/'undistort.log')
    images_dir = candidate/'images'
    sparse_dir = candidate/'sparse'
    if not images_dir.is_dir() or not sparse_dir.is_dir():
        raise RuntimeError('Undistortion did not produce expected COLMAP dataset')
    if (sparse_dir/'cameras.bin').exists():
        sparse_dir.rename(candidate/'sparse_direct')
        (candidate/'sparse').mkdir()
        (candidate/'sparse_direct').rename(candidate/'sparse'/'0')
    elif not (sparse_dir/'0'/'cameras.bin').exists():
        raise RuntimeError('Undistorted sparse cameras are missing')
    txt = candidate/'sparse_txt'
    txt.mkdir()
    run([str(bin_path),'model_converter','--input_path',str(candidate/'sparse'/'0'),'--output_path',str(txt),'--output_type','TXT'],root/'undistorted_model_converter.log')
    camera_lines = [x.split() for x in (txt/'cameras.txt').read_text().splitlines() if x.strip() and not x.startswith('#')]
    if not camera_lines or any(x[1] not in ('PINHOLE','SIMPLE_PINHOLE') for x in camera_lines):
        raise RuntimeError('Image undistorter did not yield pinhole camera models')
    count = len(list(images_dir.glob('*.jpg')))
    meta = {'stage':'dataset_preparation','image_count':count,'original_camera_model':'see full_reconstruction/glomap/sparse_txt/cameras.txt','undistorted_camera_models':sorted({x[1] for x in camera_lines}),'image_dimensions':list(Image.open(next(images_dir.glob('*.jpg'))).size),'seconds':elapsed(start),'undistortion':'COLMAP image_undistorter; use these transformed intrinsics and images together'}
    save_json(candidate/'metadata.json',meta)
    if dataset.exists():
        shutil.rmtree(dataset)
    candidate.rename(dataset)
    send_progress(args,'preparing_training_data',65)
    print(json.dumps(meta, indent=2))


if __name__=='__main__':
    main()
