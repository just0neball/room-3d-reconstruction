import argparse
import json
import zipfile
from pathlib import Path
import torch
from common import add_service_args, project_root, save_json, send_progress


def main():
    parser=add_service_args(argparse.ArgumentParser())
    parser.add_argument('--scripts_dir',default=None)
    parser.add_argument('--notebook_path',default=None)
    parser.add_argument('--run_name',default='stable_7000')
    parser.add_argument('--include_video',action='store_true')
    args=parser.parse_args()
    root=project_root(args.project_dir)
    run=root/'3dgs_training'/args.run_name
    checkpoint=run/'latest.pt'
    scene=root/'exports'/args.run_name/'scene.ply'
    if not checkpoint.is_file() or not scene.is_file():
        raise RuntimeError('Training checkpoint and exported scene are required before packaging')
    required=[
        root/'3dgs_dataset/images',
        root/'3dgs_dataset/sparse_txt/cameras.txt',
        root/'3dgs_dataset/sparse_txt/images.txt',
        root/'3dgs_dataset/sparse_txt/points3D.txt',
        root/'full_reconstruction/images',
        root/'exports'/args.run_name/'scene.splat'
    ]
    missing=[str(item) for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError('Missing required handoff artifacts: '+str(missing))
    weights_path=run/'inference_weights.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    torch.save({'step':state['step'],'splats':{name:value.detach().cpu() for name,value in state['splats'].items()},'sh_degree':state['sh_degree'],'holdout_stride':state['holdout_stride'],'schedule_steps':state['schedule_steps']},weights_path)
    fixed=[
        'preprocessing_metadata.json','frame_quality.csv','colmap_environment.json',
        'pose_comparison/dataset_manifest.json','full_reconstruction/dataset_manifest.json',
        'pose_comparison/vggt/metadata.json','pose_comparison/vggt/vggt_poses.npz',
        'pose_comparison/vggt/vggt_pose_pointcloud.png','pose_comparison/vggt/vggt_pointcloud_preview.ply',
        'pose_comparison/glomap/mapping_metadata.json','pose_comparison/glomap/sparse_preview.png',
        'full_reconstruction/vggt/metadata.json','full_reconstruction/vggt/vggt_poses.npz',
        'full_reconstruction/vggt/vggt_pose_pointcloud.png','full_reconstruction/vggt/vggt_pointcloud_preview.ply',
        'full_reconstruction/glomap/mapping_metadata.json','full_reconstruction/glomap/sparse_preview.png',
        '3dgs_dataset/metadata.json',
        f'3dgs_training/{args.run_name}/training_metadata.json',
        f'3dgs_training/{args.run_name}/history.csv',
        f'3dgs_training/{args.run_name}/split.json',
        f'3dgs_training/{args.run_name}/inference_weights.pt',
        f'3dgs_training/{args.run_name}/evaluation/heldout_metrics.csv',
        f'3dgs_training/{args.run_name}/evaluation/heldout_summary.json',
        f'3dgs_training/{args.run_name}/evaluation/heldout_contact_sheet.jpg',
        f'exports/{args.run_name}/scene.ply',f'exports/{args.run_name}/scene.splat',
        f'exports/{args.run_name}/scene_metadata.json',
        'reports/comparison.md','reports/runtime_vram.csv','reports/week2_handoff.md',
        'reports/validation_status.json','qc/frames_preview.png'
    ]
    folders=[
        'full_reconstruction/images',
        '3dgs_dataset/images','3dgs_dataset/sparse/0','3dgs_dataset/sparse_txt',
        'full_reconstruction/glomap/sparse/0','full_reconstruction/glomap/sparse_txt',
        'pose_comparison/glomap/sparse/0','pose_comparison/glomap/sparse_txt'
    ]
    files={root/item for item in fixed if (root/item).is_file()}
    for folder in folders:
        base=root/folder
        if base.is_dir():
            files.update(x for x in base.rglob('*') if x.is_file())
    script_dir=Path(args.scripts_dir) if args.scripts_dir else Path(__file__).resolve().parent
    output=root/'Person_A_backend_delivery.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for file in sorted(files):
            archive.write(file,arcname=str(Path('artifacts')/file.relative_to(root)))
        for script in sorted(script_dir.glob('*.py')):
            archive.write(script,arcname=f'pipeline/{script.name}')
        if args.notebook_path and Path(args.notebook_path).is_file():
            archive.write(args.notebook_path,arcname='notebooks/Person_A_Reconstruction.ipynb')
        archive.writestr('README.txt','Person A ML reconstruction package. CLI entrypoint: pipeline/run_pipeline.py. Undistorted photographs: artifacts/3dgs_dataset/images. Pinhole intrinsics and world-to-camera extrinsics: artifacts/3dgs_dataset/sparse_txt. Exported scene: artifacts/exports/'+args.run_name+'/scene.ply and scene.splat. Splat colors contain SH DC only, whereas PLY preserves higher SH terms. Inference weights do not contain optimizer state. Full resume checkpoint remains in the Kaggle project folder and is not transferred. Open artifacts/reports/week2_handoff.md for camera conventions and integration paths.\n')
        if args.include_video:
            for candidate in sorted(root.glob('*.mp4')):
                archive.write(candidate,arcname='source/'+candidate.name)
    record={'stage':'backend_delivery','zip_path':str(output),'zip_size_mb':round(output.stat().st_size/1024**2,2),'files':len(files),'scripts':len(list(script_dir.glob('*.py'))),'run_name':args.run_name,'contains_optimizer_state':False,'contains_clean_images':(root/'3dgs_dataset/images').is_dir(),'contains_colmap_poses':(root/'3dgs_dataset/sparse_txt/images.txt').is_file()}
    save_json(root/'delivery_manifest.json',record)
    send_progress(args,'packaging',100,status='completed')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
