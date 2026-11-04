import argparse
import csv
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy
from scene_data import read_scene, holdout_indices
from common import add_service_args, elapsed, project_root, save_json, send_progress


def ssim(a,b):
    a = a.permute(2,0,1).unsqueeze(0)
    b = b.permute(2,0,1).unsqueeze(0)
    ma, mb = F.avg_pool2d(a, 11, stride=1, padding=5), F.avg_pool2d(b, 11, stride=1, padding=5)
    va = F.avg_pool2d(a*a,11,stride=1,padding=5)-ma*ma
    vb = F.avg_pool2d(b*b,11,stride=1,padding=5)-mb*mb
    covariance = F.avg_pool2d(a*b,11,stride=1,padding=5)-ma*mb
    C1,C2 = .01**2,.03**2
    value = ((2*ma*mb+C1)*(2*covariance+C2))/((ma*ma+mb*mb+C1)*(va+vb+C2)).clamp_min(1e-8)
    return value.clamp(-1,1).mean()


def main():
    parser = add_service_args(argparse.ArgumentParser())
    parser.add_argument('--steps', type=int, default=7000)
    parser.add_argument('--schedule_steps', type=int, default=15000)
    parser.add_argument('--run_name', default='stable_7000')
    parser.add_argument('--holdout_stride', type=int, default=10)
    parser.add_argument('--sh_degree', type=int, default=2, choices=[1,2,3])
    parser.add_argument('--ssim_weight', type=float, default=0.0)
    parser.add_argument('--scale_regularization', type=float, default=0.0)
    parser.add_argument('--scale_cap_fraction', type=float, default=0.20)
    parser.add_argument('--lr_final_factor', type=float, default=0.1)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--validate_every', type=int, default=500)
    parser.add_argument('--refine_stop_iter', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.steps<1 or args.schedule_steps<args.steps or not 0<=args.ssim_weight<=1:
        raise ValueError('Invalid step count or SSIM weight')
    root = project_root(args.project_dir)
    records,xyz,rgb,errors = read_scene(root/'3dgs_dataset')
    train_ids,val_ids = holdout_indices(records,args.holdout_stride)
    if not val_ids:
        raise RuntimeError('No held-out frames for evaluation')
    run = root/'3dgs_training'/args.run_name
    run.mkdir(parents=True,exist_ok=True)
    latest, best = run/'latest.pt',run/'best.pt'
    device='cuda:0'
    if not torch.cuda.is_available():
        raise RuntimeError('Enable Kaggle Tesla T4')
    keep = np.isfinite(xyz).all(axis=1)&np.isfinite(rgb).all(axis=1)&np.isfinite(errors)
    keep &= errors<=np.percentile(errors[keep],99) if keep.any() else keep
    xyz,rgb=xyz[keep],rgb[keep]
    distances = cKDTree(xyz).query(xyz,k=2)[0][:,1]
    positive=distances[(distances>0)&np.isfinite(distances)]
    if len(positive)<3:
        raise RuntimeError('Seed point distances are invalid')
    lo,hi=np.percentile(positive,[10,90])
    init_scales=np.log(np.clip(distances,lo,hi)*0.75).astype(np.float32)
    sh_dim=(args.sh_degree+1)**2
    if args.resume and latest.exists():
        state=torch.load(latest,map_location=device,weights_only=False)
        if state['holdout_stride'] != args.holdout_stride or state['sh_degree'] != args.sh_degree or state['schedule_steps'] != args.schedule_steps:
            raise RuntimeError('Cannot resume with different split/SH/scheduler; choose a new run_name')
        splats={name:torch.nn.Parameter(value.detach().to(device)) for name,value in state['splats'].items()}
        start_step=int(state['step'])
    else:
        if latest.exists():
            raise RuntimeError('Run exists. Supply --resume or a different --run_name')
        splats={
            'means':torch.nn.Parameter(torch.tensor(xyz,device=device)),
            'quats':torch.nn.Parameter(torch.tensor([[1.,0.,0.,0.]],device=device).repeat(len(xyz),1)),
            'scales':torch.nn.Parameter(torch.tensor(init_scales[:,None].repeat(3,axis=1),device=device)),
            'opacities':torch.nn.Parameter(torch.full((len(xyz),),-1.5,device=device)),
            'sh0':torch.nn.Parameter(torch.tensor(((rgb-.5)/0.28209479177387814)[:,None,:],device=device)),
            'shN':torch.nn.Parameter(torch.zeros((len(xyz),sh_dim-1,3),device=device))
        }
        start_step=0
        state=None
    rates={'means':1e-4,'quats':1e-3,'scales':3e-3,'opacities':2e-2,'sh0':1e-2,'shN':5e-4}
    opts={name:torch.optim.Adam([{'params':[value],'lr':rates[name]}],eps=1e-15) for name,value in splats.items()}
    strategy=DefaultStrategy(refine_start_iter=500,refine_stop_iter=args.refine_stop_iter,refine_every=100,reset_every=3000,verbose=False)
    strategy.check_sanity(splats,opts)
    vmat=torch.tensor(np.stack([x['viewmat'] for x in records]),device=device)
    K=torch.tensor(np.stack([x['K'] for x in records]),device=device)
    centers=torch.linalg.inv(vmat)[:,:3,3]
    scene_scale=max(float(torch.linalg.norm(centers-centers.mean(0),dim=-1).max().item()),.1)
    rates.update({'means':1.6e-4*scene_scale,'quats':1e-3,'scales':5e-3,'opacities':5e-2,'sh0':2.5e-3,'shN':1.25e-4})
    strategy_state=strategy.initialize_state(scene_scale=scene_scale)
    if state is not None:
        for name,opt in opts.items():
            opt.load_state_dict(state['optimizers'][name])
        strategy_state=state['strategy_state']
        best_score=float(state.get('best_psnr',float('-inf')))
    else:
        best_score=float('-inf')
    image_cache=[]
    for r in records:
        with Image.open(r['path']) as im:
            image_cache.append(torch.from_numpy(np.array(im.convert('RGB'),dtype=np.uint8)))
    width,height=records[0]['width'],records[0]['height']
    if any(x['width']!=width or x['height']!=height for x in records):
        raise ValueError('Mixed image dimensions are not supported')
    background=torch.tensor([.5,.5,.5],device=device)
    def render(index,grad=False):
        return rasterization(
            means=splats['means'],quats=F.normalize(splats['quats'],dim=-1),
            scales=torch.exp(splats['scales'].clamp(-12,3)),opacities=torch.sigmoid(splats['opacities']),
            colors=torch.cat([splats['sh0'],splats['shN']],dim=1),sh_degree=args.sh_degree,
            viewmats=vmat[index:index+1],Ks=K[index:index+1],width=width,height=height,
            backgrounds=background,packed=True,absgrad=grad and strategy.absgrad
        )
    def validate():
        indices=np.linspace(0,len(val_ids)-1,min(len(val_ids),8)).round().astype(int)
        selected=[val_ids[i] for i in sorted(set(indices))]
        scores=[]
        with torch.no_grad():
            for frame in selected:
                prediction,_,_=render(frame)
                truth=image_cache[frame].to(device,dtype=torch.float32)/255
                mse=F.mse_loss(prediction[0].clamp(0,1),truth).clamp_min(1e-10)
                scores.append(float((-10*torch.log10(mse)).item()))
        return float(np.mean(scores))
    def training_psnr():
        scores=[]
        with torch.no_grad():
            for frame in train_ids:
                prediction,_,_=render(frame)
                truth=image_cache[frame].to(device,dtype=torch.float32)/255
                mse=F.mse_loss(prediction[0].clamp(0,1),truth).clamp_min(1e-10)
                scores.append(float((-10*torch.log10(mse)).item()))
        return float(np.mean(scores))
    def checkpoint(path,step,best_psnr):
        temporary=path.with_suffix('.tmp')
        torch.save({'step':step,'splats':{k:v.detach().cpu() for k,v in splats.items()},'optimizers':{k:o.state_dict() for k,o in opts.items()},'strategy_state':strategy_state,'sh_degree':args.sh_degree,'schedule_steps':args.schedule_steps,'holdout_stride':args.holdout_stride,'best_psnr':best_psnr,'train_count':len(train_ids),'heldout_count':len(val_ids)},temporary)
        temporary.replace(path)
    monitor_positions = np.linspace(0,len(val_ids)-1,min(8,len(val_ids))).round().astype(int)
    monitor_ids = sorted({val_ids[int(i)] for i in monitor_positions})
    test_ids = [i for i in val_ids if i not in set(monitor_ids)]
    train_list_path=run/'split.json'
    save_json(train_list_path,{'trained_frames':[records[i]['name'] for i in train_ids],'heldout_frames':[records[i]['name'] for i in val_ids], 'monitor_frames':[records[i]['name'] for i in monitor_ids], 'test_frames':[records[i]['name'] for i in test_ids],'note':'Held-out images are excluded from splat optimization but were used in SfM pose reconstruction.'})
    rng=np.random.default_rng(2026)
    train_order=np.concatenate([
        rng.permutation(train_ids)
        for _ in range((args.steps+len(train_ids)-1)//len(train_ids))
    ])
    history=run/'history.csv'
    history_exists=history.exists()
    torch.cuda.set_device('cuda:0')
    start=time.perf_counter()
    send_progress(args,'training_gaussians',65)
    if start_step>=args.steps:
        print(f'Already completed {start_step} steps')
        return
    with history.open('a',newline='') as stream:
        writer=csv.writer(stream)
        if not history_exists:
            writer.writerow(['step','train_l1','train_ssim','total_loss','gaussians','heldout_sample_psnr_db','lr_multiplier','train_mean_psnr_db'])
        for step in range(start_step+1,args.steps+1):
            frame=int(train_order[step-1])
            target=image_cache[frame].to(device,dtype=torch.float32)/255.
            lr_factor=args.lr_final_factor**(step/args.schedule_steps)
            for name,opt in opts.items():
                opt.param_groups[0]['lr']=rates[name]*lr_factor
                opt.zero_grad(set_to_none=True)
            rendered,alphas,info=render(frame,True)
            strategy.step_pre_backward(splats,opts,strategy_state,step,info)
            prediction=rendered[0]
            l1=F.l1_loss(prediction,target)
            structural=ssim(prediction,target)
            spread=torch.exp(splats['scales'].clamp(-12,3)).max(dim=1).values
            penalty=torch.relu(spread/(scene_scale*args.scale_cap_fraction)-1).square().mean()
            loss=(1-args.ssim_weight)*l1+args.ssim_weight*(1-structural)+args.scale_regularization*penalty
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at step {step}')
            loss.backward()
            for opt in opts.values():
                opt.step()
            strategy.step_post_backward(splats,opts,strategy_state,step,info,packed=True)
            if step%args.validate_every==0 or step==args.steps:
                heldout_psnr=validate()
                train_psnr=training_psnr()
                writer.writerow([step,round(l1.item(),6),round(structural.item(),6),round(loss.item(),6),len(splats['means']),round(heldout_psnr,3),round(lr_factor,6),round(train_psnr,3)])
                stream.flush()
                print(f'Step {step}/{args.steps} | gaussians={len(splats["means"])} | train PSNR={train_psnr:.3f} dB | heldout PSNR={heldout_psnr:.3f} dB',flush=True)
                if heldout_psnr>best_score:
                    best_score=heldout_psnr
                    checkpoint(best,step,best_score)
            if step%args.save_every==0 or step==args.steps:
                checkpoint(latest,step,best_score)
                send_progress(args,'training_gaussians',65+int(25*step/args.steps))
    torch.cuda.synchronize()
    meta={'stage':'gaussian_training','run':args.run_name,'steps_completed':args.steps,'starting_step':start_step,'time_this_run_seconds':elapsed(start),'cumulative_training_seconds':round(elapsed(start)+(json.loads((run/'training_metadata.json').read_text()).get('cumulative_training_seconds',0) if (run/'training_metadata.json').exists() else 0),3),'gaussians':len(splats['means']),'peak_allocated_gpu_vram_gb':round(torch.cuda.max_memory_allocated()/1024**3,3),'sh_degree':args.sh_degree,'loss':'(1-ssim_weight)*L1 + ssim_weight*(1-SSIM) + scale_regularization*penalty','heldout_frames':len(val_ids),'training_frames':len(train_ids),'best_heldout_sample_psnr_db':round(best_score,3),'best_checkpoint':str(best),'latest_checkpoint':str(latest),'lr_schedule_steps':args.schedule_steps,'note':'Fixed seeded training schedule, L1 with unclipped loss, and no densification by default. Monitoring views are separate from final test views; COLMAP SfM used all images.'}
    save_json(run/'training_metadata.json',meta)
    send_progress(args,'training_gaussians',90)
    print(json.dumps(meta,indent=2))


if __name__=='__main__':
    main()
