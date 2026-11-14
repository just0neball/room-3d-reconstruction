import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from common import add_service_args, project_root, save_json,send_progress


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-30,30)))


def main():
    parser=add_service_args(argparse.ArgumentParser())
    parser.add_argument('--run_name',default='stable_7000')
    parser.add_argument('--checkpoint',choices=['best','latest'],default='latest')
    parser.add_argument('--checkpoint_path',default=None)
    args=parser.parse_args()
    start=time.perf_counter()
    root=project_root(args.project_dir)
    path=Path(args.checkpoint_path) if args.checkpoint_path else root/'3dgs_training'/args.run_name/f'{args.checkpoint}.pt'
    if not path.is_file():
        raise FileNotFoundError(path)
    state=torch.load(path,map_location='cpu',weights_only=False)
    p=state.get('splats',state.get('parameters'))
    def array(key):
        return p[key].detach().float().cpu().numpy().astype(np.float32)
    is_new='sh0' in p
    means=array('means')
    log_scales=np.clip(array('scales' if is_new else 'scales'),-12,3)
    q=array('quats')
    q_norm=np.linalg.norm(q,axis=1,keepdims=True)
    if np.any(q_norm<1e-6):
        raise RuntimeError('Invalid rotations')
    q=q/q_norm
    opacity=array('opacities')
    if is_new:
        sh0=array('sh0')[:,0,:]
        shn=array('shN')
        degree=int(state['sh_degree'])
    else:
        rgb=sigmoid(array('colors'))
        sh0=(rgb-.5)/.28209479177387814
        shn=np.empty((len(means),0,3),dtype=np.float32)
        degree=0
    rgb=np.clip(.5+sh0*.28209479177387814,0,1).astype(np.float32)
    alpha=sigmoid(opacity).astype(np.float32)
    scales=np.exp(log_scales).astype(np.float32)
    n=len(means)
    if not all(len(x)==n for x in (log_scales,q,opacity,sh0,alpha)):
        raise ValueError('Parameter count mismatch')
    if not all(np.isfinite(x).all() for x in (means,log_scales,q,opacity,sh0,shn)):
        raise RuntimeError('Nonfinite gaussian values')
    rest=shn.transpose(0,2,1).reshape(n,-1)
    fields=['x','y','z','nx','ny','nz']+[f'f_dc_{j}' for j in range(3)]+[f'f_rest_{j}' for j in range(rest.shape[1])]+['opacity']+[f'scale_{j}' for j in range(3)]+[f'rot_{j}' for j in range(4)]
    dtype=np.dtype([(name,'<f4') for name in fields])
    vertices=np.zeros(n,dtype=dtype)
    for j,key in enumerate(('x','y','z')):
        vertices[key]=means[:,j]
    for j in range(3):
        vertices[f'f_dc_{j}']=sh0[:,j]
        vertices[f'scale_{j}']=log_scales[:,j]
    for j in range(rest.shape[1]):
        vertices[f'f_rest_{j}']=rest[:,j]
    for j in range(4):
        vertices[f'rot_{j}']=q[:,j]
    vertices['opacity']=opacity
    export_dir=root/'exports'/args.run_name
    export_dir.mkdir(parents=True,exist_ok=True)
    header='\n'.join(['ply','format binary_little_endian 1.0',f'element vertex {n}']+[f'property float {key}' for key in fields]+['end_header'])+'\n'
    ply=export_dir/'scene.ply'
    with ply.open('wb') as stream:
        stream.write(header.encode('ascii'))
        vertices.tofile(stream)
    splat_dtype=np.dtype([('pos','<f4',(3,)),('scales','<f4',(3,)),('color','u1',(4,)),('quat','u1',(4,))])
    if splat_dtype.itemsize!=32:
        raise RuntimeError('Unexpected SPLAT byte layout')
    order=np.argsort(alpha*np.prod(scales,axis=1))[::-1]
    data=np.empty(n,dtype=splat_dtype)
    data['pos']=means[order]
    data['scales']=scales[order]
    rgba=np.concatenate((rgb,alpha[:,None]),axis=1)
    data['color']=np.clip(np.round(rgba[order]*255),0,255).astype('u1')
    data['quat']=np.clip(np.round(128+128*q[order]),0,255).astype('u1')
    splat=export_dir/'scene.splat'
    data.tofile(splat)
    if ply.stat().st_size!=len(header.encode('ascii'))+n*dtype.itemsize or splat.stat().st_size!=n*32:
        raise RuntimeError('Export size validation failed')
    record={'stage':'export','checkpoint_step':int(state['step']),'checkpoint_path':str(path),'gaussian_count':n,'sh_degree':degree,'ply':str(ply),'splat':str(splat),'ply_size_mb':round(ply.stat().st_size/1024**2,2),'splat_size_mb':round(splat.stat().st_size/1024**2,2),'notes':'PLY preserves SH coefficients; legacy SPLAT bakes only SH DC and needs viewer compatibility testing. Camera orientation is COLMAP convention, so set the viewer transform explicitly.'}
    record['runtime_seconds']=round(time.perf_counter()-start,3)
    save_json(export_dir/'scene_metadata.json',record)
    send_progress(args,'exporting',96)
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
