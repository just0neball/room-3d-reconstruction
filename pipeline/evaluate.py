import argparse
import csv
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from gsplat.rendering import rasterization
from scene_data import holdout_indices,read_scene
from train import ssim
from common import add_service_args, project_root, save_json


def main():
    parser=add_service_args(argparse.ArgumentParser())
    parser.add_argument('--run_name',default='stable_7000')
    parser.add_argument('--checkpoint',choices=['best','latest'],default='latest')
    args=parser.parse_args()
    start=time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is unavailable. Enable a Kaggle T4 accelerator before evaluation.')
    torch.cuda.set_device('cuda:0')
    root=project_root(args.project_dir)
    folder=root/'3dgs_training'/args.run_name
    path=folder/f'{args.checkpoint}.pt'
    if not path.exists():
        raise FileNotFoundError(path)
    state=torch.load(path,map_location='cuda:0',weights_only=False)
    records,_,_,_=read_scene(root/'3dgs_dataset')
    train_ids,heldout_ids=holdout_indices(records,state['holdout_stride'])
    monitor_positions=np.linspace(0,len(heldout_ids)-1,min(8,len(heldout_ids))).round().astype(int)
    monitor_set={heldout_ids[int(i)] for i in monitor_positions}
    test_set=set(heldout_ids)-monitor_set
    splats=state['splats']
    shdegree=int(state['sh_degree'])
    device='cuda:0'
    means=splats['means'].float().to(device)
    quats=F.normalize(splats['quats'].float().to(device),dim=-1)
    scales=torch.exp(splats['scales'].float().to(device).clamp(-12,3))
    opacities=torch.sigmoid(splats['opacities'].float().to(device))
    colors=torch.cat([splats['sh0'],splats['shN']],dim=1).float().to(device)
    mats=torch.tensor(np.stack([r['viewmat'] for r in records]),device=device)
    Ks=torch.tensor(np.stack([r['K'] for r in records]),device=device)
    background=torch.tensor([.5,.5,.5],device=device)
    rows=[]
    saved_panels=[]
    with torch.inference_mode():
        for index in heldout_ids:
            rec=records[index]
            with Image.open(rec['path']) as im:
                target=torch.tensor(np.array(im.convert('RGB'),dtype=np.uint8),device=device).float()/255
            prediction,alpha,_=rasterization(means=means,quats=quats,scales=scales,opacities=opacities,colors=colors,sh_degree=shdegree,viewmats=mats[index:index+1],Ks=Ks[index:index+1],width=rec['width'],height=rec['height'],backgrounds=background,packed=True)
            pred=prediction[0].clamp(0,1)
            mse=F.mse_loss(pred,target).clamp_min(1e-10)
            psnr=(-10*torch.log10(mse)).item()
            similarity=ssim(pred,target).item()
            rows.append({'image':rec['name'],'split':'monitor' if index in monitor_set else 'test','psnr_db':round(psnr,4),'ssim':round(similarity,5),'l1':round(F.l1_loss(pred,target).item(),5),'mean_alpha':round(alpha.mean().item(),5)})
            if len(saved_panels)<6:
                saved_panels.append((rec['name'],np.array((target.cpu().numpy()*255).astype(np.uint8)),np.array((pred.cpu().numpy()*255).astype(np.uint8))))
    report=folder/'evaluation'
    report.mkdir(parents=True,exist_ok=True)
    with (report/'heldout_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if saved_panels:
        canvas=Image.new('RGB',(2*480,len(saved_panels)*300),'white')
        draw=ImageDraw.Draw(canvas)
        for row,(name,gt,pred) in enumerate(saved_panels):
            yy=row*300
            draw.text((8,yy+4),f'{name} | heldout original',fill='black')
            draw.text((488,yy+4),f'{name} | prediction',fill='black')
            canvas.paste(Image.fromarray(gt).resize((480,270)),(0,yy+30))
            canvas.paste(Image.fromarray(pred).resize((480,270)),(480,yy+30))
        canvas.save(report/'heldout_contact_sheet.jpg',quality=85)
    test_rows=[row for row in rows if row['split']=='test']
    monitor_rows=[row for row in rows if row['split']=='monitor']
    summary={'test_frames':len(test_rows),'monitor_frames':len(monitor_rows),'test_mean_psnr_db':round(float(np.mean([r['psnr_db'] for r in test_rows])),3) if test_rows else None,'test_mean_ssim':round(float(np.mean([r['ssim'] for r in test_rows])),4) if test_rows else None,'monitor_mean_psnr_db':round(float(np.mean([r['psnr_db'] for r in monitor_rows])),3) if monitor_rows else None,'stage':'heldout_evaluation','run':args.run_name,'evaluated_frames':len(rows),'training_frames':len(train_ids),'mean_psnr_db':round(float(np.mean([r['psnr_db'] for r in rows])),3),'mean_ssim':round(float(np.mean([r['ssim'] for r in rows])),4),'mean_l1':round(float(np.mean([r['l1'] for r in rows])),5),'worst_5':sorted(rows,key=lambda r:r['psnr_db'])[:5],'checkpoint':str(path),'split_note':'Heldout images were not used for GS optimization but did participate in SfM camera registration; free-camera generalization requires visual review.'}
    summary['runtime_seconds']=round(time.perf_counter()-start,3)
    summary['peak_allocated_gpu_vram_gb']=round(torch.cuda.max_memory_allocated()/1024**3,3)
    split_file=folder/'split.json'
    split_data=json.loads(split_file.read_text()) if split_file.is_file() else {}
    split_data['trained_frames']=[records[i]['name'] for i in train_ids]
    split_data['heldout_frames']=[records[i]['name'] for i in heldout_ids]
    split_data['monitor_frames']=[records[i]['name'] for i in heldout_ids if i in monitor_set]
    split_data['test_frames']=[records[i]['name'] for i in heldout_ids if i in test_set]
    save_json(split_file,split_data)
    save_json(report/'heldout_summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
