import numpy as np
from pathlib import Path
from PIL import Image


def quat_rotation(q):
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]],dtype=np.float32)


def read_scene(dataset):
    dataset = Path(dataset)
    txt = dataset/'sparse_txt'
    cameras = {}
    for line in (txt/'cameras.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        v = line.split()
        w,h = map(int, v[2:4])
        values = list(map(float,v[4:]))
        if v[1]=='SIMPLE_PINHOLE':
            f,cx,cy = values
            fx,fy = f,f
        elif v[1]=='PINHOLE':
            fx,fy,cx,cy = values
        else:
            raise RuntimeError(f'Image is not undistorted: {v[1]}')
        cameras[int(v[0])] = (w,h,np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=np.float32))
    lines = [line.strip() for line in (txt/'images.txt').read_text().splitlines() if not line.startswith('#')]
    if len(lines)%2:
        raise RuntimeError('Invalid COLMAP images.txt pairs')
    records = []
    for header in lines[::2]:
        v = header.split()
        if len(v)<10:
            raise RuntimeError(f'Invalid COLMAP image: {header}')
        name = ' '.join(v[9:])
        width,height,K = cameras[int(v[8])]
        path = dataset/'images'/name
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            if image.size!=(width,height):
                raise ValueError(f'Image and intrinsics dimensions do not match: {name}')
        w2c = np.eye(4,dtype=np.float32)
        w2c[:3,:3] = quat_rotation([float(x) for x in v[1:5]])
        w2c[:3,3] = [float(x) for x in v[5:8]]
        records.append({'name':name,'path':str(path),'width':width,'height':height,'viewmat':w2c,'K':K})
    records.sort(key=lambda item:item['name'])
    if len(records)<3:
        raise RuntimeError('Dataset has fewer than 3 cameras')
    points=[]
    colors=[]
    errors=[]
    for line in (txt/'points3D.txt').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            v=line.split()
            points.append([float(x) for x in v[1:4]])
            colors.append([float(x)/255 for x in v[4:7]])
            errors.append(float(v[7]))
    points=np.asarray(points,dtype=np.float32)
    colors=np.asarray(colors,dtype=np.float32)
    errors=np.asarray(errors,dtype=np.float32)
    if len(points)<50:
        raise RuntimeError('Not enough 3D seed points')
    return records,points,colors,errors


def holdout_indices(records,stride):
    if stride<3:
        raise ValueError('Holdout stride must be >=3')

    n = len(records)
    if n < 3:
        raise RuntimeError(
            f'Need at least 3 registered images for training and validation; got {n}'
        )

    held = set(range(stride-1,n,stride))

    # Small datasets may not reach the configured holdout stride.
    # Reserve the final image for validation in that case.
    if not held:
        held = {n-1}

    train = [i for i in range(n) if i not in held]
    return train, sorted(held)
