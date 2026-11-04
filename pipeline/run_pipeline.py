import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project_dir',required=True)
    parser.add_argument('--video',default=None)
    parser.add_argument('--stage',choices=['extract','install_colmap','vggt_benchmark','vggt_full','glomap_benchmark','glomap_full','prepare','train','evaluate','export','report','package','production','all'],required=True)
    parser.add_argument('--steps',type=int,default=7000)
    parser.add_argument('--run_name',default='stable_7000')
    parser.add_argument('--backend_url',default=None)
    parser.add_argument('--task_id',default=None)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--matching',choices=['sequential','both','exhaustive'],default=None)
    parser.add_argument('--colmap_bin',default='/kaggle/working/reconstruction_tools/colmap_env/bin/colmap')
    args=parser.parse_args()
    script_dir=Path(__file__).parent
    common=['--project_dir',args.project_dir]
    if args.backend_url:
        common+=['--backend_url',args.backend_url]
    if args.task_id:
        common+=['--task_id',args.task_id]
    commands={
        'extract':['extract.py','--video',args.video] if args.video else None,
        'install_colmap':['install_colmap.py'],
        'vggt_benchmark':['vggt_compare.py','--mode','benchmark'],
        'vggt_full':['vggt_compare.py','--mode','full'],
        'glomap_benchmark':['poses.py','--mode','benchmark','--colmap_bin',args.colmap_bin],
        'glomap_full':['poses.py','--mode','full','--colmap_bin',args.colmap_bin,'--matching',args.matching or ('sequential' if args.stage=='production' else 'both')],
        'prepare':['prepare_dataset.py','--colmap_bin',args.colmap_bin],
        'train':['train.py','--steps',str(args.steps),'--schedule_steps','15000','--run_name',args.run_name,'--refine_stop_iter','0','--ssim_weight','0','--scale_regularization','0']+(['--resume'] if args.resume else []),
        'evaluate':['evaluate.py','--run_name',args.run_name,'--checkpoint','latest'],
        'export':['export.py','--run_name',args.run_name,'--checkpoint','latest'],
        'package':['package.py','--scripts_dir',str(script_dir),'--run_name',args.run_name],
        'report':['report.py','--run_name',args.run_name]
    }
    if args.stage=='production':
        stages=['extract','install_colmap','glomap_full','prepare','train','evaluate','export','report','package']
    elif args.stage=='all':
        stages=['extract','install_colmap','vggt_benchmark','glomap_benchmark','vggt_full','glomap_full','prepare','train','evaluate','export','report','package']
    else:
        stages=[args.stage]
    for stage in stages:
        if stage=='train':
            checkpoint=Path(args.project_dir)/'3dgs_training'/args.run_name/'latest.pt'
            if checkpoint.is_file():
                import torch
                saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
                if int(saved['step'])>=args.steps:
                    print(f'SKIP train: checkpoint already at step {saved["step"]}',flush=True)
                    continue
                if not args.resume:
                    raise RuntimeError('Training checkpoint exists; use --resume to continue')
        item=commands[stage]
        if item is None:
            raise ValueError(f'{stage} requires --video')
        command=[sys.executable,str(script_dir/item[0]),*common,*item[1:]]
        print(f'RUNNING {stage}: {command}',flush=True)
        subprocess.run(command,check=True)
    print(json.dumps({'completed_stages':stages,'project_dir':args.project_dir},indent=2))


if __name__=='__main__':
    main()
