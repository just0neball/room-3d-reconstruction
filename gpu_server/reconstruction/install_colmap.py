import argparse
import json
import os
import subprocess
from pathlib import Path
from common import add_service_args, project_root, save_json


def main():
    parser = add_service_args(argparse.ArgumentParser())
    parser.add_argument('--tools_dir', default='/kaggle/working/reconstruction_tools')
    args = parser.parse_args()
    root = project_root(args.project_dir)
    tools = Path(args.tools_dir)
    mamba = tools / 'bin' / 'micromamba'
    env_dir = tools / 'colmap_env'
    binary = env_dir / 'bin' / 'colmap'
    tools.mkdir(parents=True, exist_ok=True)
    if not mamba.exists():
        subprocess.run(['bash', '-lc', f'mkdir -p "{tools}" && curl -fLsS https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "{tools}" bin/micromamba'], check=True)
    env = os.environ.copy()
    env['MAMBA_ROOT_PREFIX'] = str(tools / 'mamba_root')
    if not binary.is_file():
        subprocess.run([str(mamba), 'create', '-y', '-p', str(env_dir), '-c', 'conda-forge', '--override-channels', 'colmap=4.2.1'], env=env, check=True)
    help_result = subprocess.run([str(binary), '-h'], capture_output=True, text=True, env=env)
    if help_result.returncode or 'global_mapper' not in help_result.stdout + help_result.stderr:
        raise RuntimeError('COLMAP global_mapper missing: ' + (help_result.stdout + help_result.stderr)[-1000:])
    save_json(root / 'colmap_environment.json', {'colmap': str(binary), 'global_mapper_available': True, 'environment': str(env_dir), 'provider': 'conda-forge'})
    print(json.dumps({'colmap': str(binary), 'global_mapper_available': True}, indent=2))


if __name__ == '__main__':
    main()
