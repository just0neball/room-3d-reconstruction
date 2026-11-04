import argparse
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from common import add_service_args, elapsed, project_root, save_json, send_progress


def execute(argv):
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{' '.join(map(str, argv))}\n{result.stderr[-3000:]}")
    return result.stdout


def main():
    parser = add_service_args(argparse.ArgumentParser())
    parser.add_argument('--video', required=True)
    parser.add_argument('--fps', type=int, default=2)
    parser.add_argument('--long_side', type=int, default=960)
    parser.add_argument('--blur_threshold', type=float, default=60)
    parser.add_argument('--max_reject_percent', type=float, default=15)
    parser.add_argument('--full_fps', type=int, default=1)
    parser.add_argument('--benchmark_frames', type=int, default=32)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    start = time.perf_counter()
    root = project_root(args.project_dir)
    source = Path(args.video)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not 1 <= args.fps <= 4 or not 640 <= args.long_side <= 960:
        raise ValueError('fps must be 1–4 and long_side must be 640–960')
    if args.full_fps < 1 or args.fps % args.full_fps:
        raise ValueError('full_fps must divide fps')
    if not 0 <= args.max_reject_percent <= 50:
        raise ValueError('max_reject_percent must be 0–50')
    send_progress(args, 'extracting_frames', 5)
    probe = json.loads(execute(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,side_data_list:format=duration', '-of', 'json', str(source)]))
    raw_dir = root / 'frames_raw'
    clean_dir = root / 'frames_clean'
    full_dir = root / 'full_reconstruction' / 'images'
    benchmark_dir = root / 'pose_comparison' / 'images'
    qc_dir = root / 'qc'
    for directory in (raw_dir, clean_dir, full_dir, benchmark_dir, qc_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if args.force or not list(raw_dir.glob('frame_*.jpg')):
        for image in raw_dir.glob('frame_*.jpg'):
            image.unlink()
        filter_graph = f'fps={args.fps},scale={args.long_side}:{args.long_side}:force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos'
        execute(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source), '-vf', filter_graph, '-q:v', '2', '-start_number', '0', str(raw_dir / 'frame_%06d.jpg')])
    raw_paths = sorted(raw_dir.glob('frame_*.jpg'))
    if not raw_paths:
        raise RuntimeError('No frames extracted')
    image_size = Image.open(raw_paths[0]).size
    send_progress(args, 'filtering_frames', 15)
    scored = []
    for path in raw_paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f'Unreadable frame: {path}')
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        scored.append((path, score))
    cutoff = min(args.blur_threshold, float(np.percentile([score for _, score in scored], args.max_reject_percent)))
    selected_clean = [p for p, score in scored if score >= cutoff]
    if args.force or set(p.name for p in clean_dir.glob('frame_*.jpg')) != set(p.name for p in selected_clean):
        for p in clean_dir.glob('frame_*.jpg'):
            p.unlink()
        for p in selected_clean:
            shutil.copy2(p, clean_dir / p.name)
    with (root / 'frame_quality.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['filename', 'laplacian_variance', 'kept'])
        for path, score in scored:
            writer.writerow([path.name, round(score, 4), int(score >= cutoff)])
    frame_names = sorted(p.name for p in selected_clean)
    stride = args.fps // args.full_fps
    selected_full = frame_names[::stride]
    if len(selected_full) < 3:
        raise RuntimeError('Insufficient frames to reconstruct')
    args.benchmark_frames = min(args.benchmark_frames, len(selected_full))
    middle = max(0, (len(selected_full) - args.benchmark_frames) // 2)
    selected_benchmark = selected_full[middle:middle + args.benchmark_frames]
    for directory, filenames in ((full_dir, selected_full), (benchmark_dir, selected_benchmark)):
        if args.force or {p.name for p in directory.glob('frame_*.jpg')} != set(filenames):
            for p in directory.glob('frame_*.jpg'):
                p.unlink()
            for filename in filenames:
                shutil.copy2(clean_dir / filename, directory / filename)
    for directory, filenames in ((full_dir.parent, selected_full), (benchmark_dir.parent, selected_benchmark)):
        manifest = [{'filename': name, 'approx_seconds': round(int(Path(name).stem.split('_')[-1]) / args.fps, 3)} for name in filenames]
        save_json(directory / 'dataset_manifest.json', {'source_video': source.name, 'fps': args.full_fps, 'count': len(filenames), 'image_resolution': list(image_size), 'frames': manifest})
    sample = [selected_clean[round(j*(len(selected_clean)-1)/15)] for j in range(16)]
    sheet = Image.new('RGB', (960, 640), 'white')
    for i, frame in enumerate(sample):
        with Image.open(frame) as im:
            thumb = ImageOps.fit(im.convert('RGB'), (240, 135))
        x, y = i % 4 * 240, i // 4 * 160
        sheet.paste(thumb, (x,y))
        ImageDraw.Draw(sheet).text((x+5,y+140), frame.name, fill='black')
    sheet.save(qc_dir / 'frames_preview.png')
    metadata = {'stage': 'extract', 'source_video': str(source), 'raw_frames': len(raw_paths), 'clean_frames': len(selected_clean), 'full_frames': len(selected_full), 'benchmark_frames': len(selected_benchmark), 'frames_resolution': list(image_size), 'fps_extraction': args.fps, 'fps_reconstruction': args.full_fps, 'blur_threshold': cutoff, 'duration_seconds': float(probe.get('format', {}).get('duration', 0)), 'processing_seconds': elapsed(start), 'rotation_handling': 'ffmpeg automatic rotation and aspect-ratio-preserving scale; dimensions read from extracted frames'}
    save_json(root / 'preprocessing_metadata.json', metadata)
    send_progress(args, 'extracting_frames', 25)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
