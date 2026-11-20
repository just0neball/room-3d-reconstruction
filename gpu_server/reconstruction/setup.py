import argparse
import importlib.util
import subprocess
import sys
import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--vggt',action='store_true')
    parser.add_argument('--gsplat',action='store_true')
    args=parser.parse_args()
    dependencies=[('cv2','opencv-python-headless'),('PIL','Pillow'),('scipy','scipy'),('matplotlib','matplotlib'),('ninja','ninja')]
    if args.vggt:
        dependencies.extend([('einops','einops'),('huggingface_hub','huggingface_hub'),('safetensors','safetensors'),('omegaconf','omegaconf')])
    missing=[package for module,package in dependencies if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.run([sys.executable,'-m','pip','install','--no-input','-q',*missing],check=True)
    if args.gsplat and importlib.util.find_spec('gsplat') is None:
        subprocess.run([sys.executable,'-m','pip','install','-q','--no-deps','gsplat==1.5.3'],check=True)
    print({'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'installed':missing})


if __name__=='__main__':
    main()
