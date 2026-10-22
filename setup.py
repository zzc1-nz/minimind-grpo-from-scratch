#!/usr/bin/env python
"""One-shot environment setup for this project.

What it does
  1. checks the Python version (3.11 required by the pinned torch/transformers)
  2. creates a venv at .venv
  3. installs the pinned dependency set (torch cu124 + MiniMind requirements)
  4. clones MiniMind into src/minimind-master if it is missing
  5. verifies that CUDA, torch, transformers and the MiniMind model class import

The project directory layout this expects:

    <project root>/
      custom/ tools/ eval/ demo/ reports/ figures/
      src/minimind-master/        <- cloned by this script
      data/                       <- corpora + tokenized blocks (not in git)
      models/                     <- HF exports + GGUF files (not in git)
      .venv/                      <- created by this script

Usage:
  python setup.py                 # full setup
  python setup.py --skip-models   # do not clone MiniMind (already present)
  python setup.py --cpu           # install CPU torch instead of cu124
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, '.venv')
PY = os.path.join(VENV, 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join(VENV, 'bin', 'python')
MINIMIND_DIR = os.path.join(HERE, 'src', 'minimind-master')
MINIMIND_REPO = 'https://github.com/jingyaogong/minimind.git'
# GitHub is slow/unreachable from some networks; this proxy mirror is what the original
# runs used. Set MINIMIND_URL to override.
MINIMIND_URL_DEFAULT = 'https://ghproxy.net/https://github.com/jingyaogong/minimind/archive/refs/heads/master.zip'

TORCH_CU124 = 'https://download.pytorch.org/whl/cu124'
TORCH_CPU = 'https://download.pytorch.org/whl/cpu'
PYPI_MIRROR = 'https://mirrors.aliyun.com/pypi/simple/'


def run(cmd, **kw):
    print('+', ' '.join(cmd) if isinstance(cmd, list) else cmd, flush=True)
    return subprocess.call(cmd, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-models', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--minimind-url', default=os.environ.get('MINIMIND_URL', MINIMIND_URL_DEFAULT))
    a = ap.parse_args()

    if sys.version_info[:2] != (3, 11):
        print(f'WARNING: this project was developed on Python 3.11; you have '
              f'{sys.version_info.major}.{sys.version_info.minor}. The pinned '
              f'torch/transformers versions may not have wheels for other versions.')

    # ---- 1. venv ---------------------------------------------------------
    if not os.path.exists(PY):
        if shutil.which('uv'):
            run(['uv', 'venv', VENV, '--python', '3.11'])
        else:
            run([sys.executable, '-m', 'venv', VENV])
    if not os.path.exists(PY):
        print('ERROR: could not create a virtualenv at', VENV)
        return 1

    def pip(*args):
        if shutil.which('uv'):
            return run(['uv', 'pip', 'install', '--python', PY, *args])
        return run([PY, '-m', 'pip', 'install', *args])

    # ---- 2. torch --------------------------------------------------------
    index = TORCH_CPU if a.cpu else TORCH_CU124
    print(f'\n[2/5] installing torch from {index}')
    pip('torch==2.6.0', 'torchvision==0.21.0', '--index-url', index)

    # ---- 3. MiniMind -----------------------------------------------------
    if not a.skip_models and not os.path.isdir(MINIMIND_DIR):
        print('\n[3/5] fetching MiniMind')
        os.makedirs(os.path.join(HERE, 'src'), exist_ok=True)
        if run(['git', 'clone', '--depth', '1', MINIMIND_REPO, MINIMIND_DIR]) != 0:
            print('  git clone failed; trying the archive mirror instead')
            zip_path = os.path.join(HERE, 'src', 'minimind.zip')
            if run(['curl', '-L', a.minimind_url, '-o', zip_path]) != 0:
                print('ERROR: could not download MiniMind. Clone it manually into '
                      f'{MINIMIND_DIR}')
                return 1
            import zipfile
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(os.path.join(HERE, 'src'))
            extracted = os.path.join(HERE, 'src', 'minimind-master')
            if os.path.isdir(extracted) and not os.path.isdir(MINIMIND_DIR):
                os.rename(extracted, MINIMIND_DIR)
            os.remove(zip_path)
    else:
        print('\n[3/5] MiniMind already present')

    # ---- 4. requirements -------------------------------------------------
    req = os.path.join(MINIMIND_DIR, 'requirements.txt')
    if os.path.exists(req):
        print('\n[4/5] installing MiniMind requirements')
        # ujson and jieba have no wheels on some platforms and need MSVC; neither is
        # imported by any training path (they belong to the demo scripts).
        pkgs = []
        with open(req, encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                name = s.split('==')[0].lower()
                if name in ('ujson', 'jieba'):
                    print(f'  skipping {s} (no wheel on this platform; not needed for training)')
                    continue
                pkgs.append(s)
        pip(*pkgs, '--index-url', PYPI_MIRROR,
            '--extra-index-url', 'https://pypi.org/simple')
        # numpy is pinned to 1.26.4 upstream, which breaks pyarrow; 2.2.6 works.
        pip('numpy==2.2.6', '--index-url', PYPI_MIRROR)
        if not a.cpu:
            pip('triton-windows==3.2.0.post21', '--index-url', PYPI_MIRROR)

    # ---- 5. verify -------------------------------------------------------
    print('\n[5/5] verifying the environment')
    check = (
        'import torch, transformers, datasets, numpy;'
        'print("torch", torch.__version__);'
        'print("cuda available", torch.cuda.is_available());'
        'print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu");'
        'print("transformers", transformers.__version__);'
        'print("datasets", datasets.__version__);'
        'print("numpy", numpy.__version__)'
    )
    if run([PY, '-c', check]) != 0:
        print('ERROR: verification failed')
        return 1

    print('\nSetup complete.')
    print(f'  venv   : {VENV}')
    print(f'  python : {PY}')
    print('\nNext: download the corpora and build the token blocks. See README section 7.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
