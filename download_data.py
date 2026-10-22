#!/usr/bin/env python
"""Download the MiniMind corpora from ModelScope.

ModelScope is the fastest reachable mirror for this project (~30-65 MB/s with parallel
range requests vs ~0.9 MB/s through hf-mirror). HuggingFace direct is unreachable on the
network this was developed on.

Files fetched (into data/):
  pretrain_t2t.jsonl        7.71 GB   full pretraining corpus (2.209 B tokens)
  sft_t2t_mini.jsonl        1.66 GB   SFT conversations
  rlaif.jsonl               22.7 MB   (not used by the rule-reward GRPO path; kept for
                                      comparison with MiniMind's stock trainer)
  dpo.jsonl                 51.2 MB   (optional, not used here)

Usage:
  python download_data.py                # pretrain + sft
  python download_data.py --all          # also rlaif + dpo
  python download_data.py --only sft     # one file
  python download_data.py --parts 16     # parallel range requests (default 16)
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import urllib.request

BASE = ('https://www.modelscope.cn/api/v1/datasets/gongjy/minimind_dataset/repo'
        '?Revision=master&FilePath=')

FILES = {
    # sizes are resolved at run time via HEAD; the values here are only hints used for
    # the human-readable progress line.
    'pretrain_t2t': 'pretrain_t2t.jsonl',
    'sft_t2t_mini': 'sft_t2t_mini.jsonl',
    'rlaif': 'rlaif.jsonl',
    'dpo': 'dpo.jsonl',
}

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')


def remote_size(name):
    """Content-Length of a file, via curl when available (urllib HEAD is flaky here)."""
    curl = shutil.which('curl')
    if curl:
        out = subprocess.run(
            [curl, '-sIL', '--max-time', '60', BASE + name],
            capture_output=True, text=True, errors='replace')
        for line in out.stdout.splitlines():
            low = line.lower()
            if low.startswith('content-length:'):
                try:
                    return int(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
        raise RuntimeError(f'could not read Content-Length for {name}')
    req = urllib.request.Request(BASE + name, method='HEAD')
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers['Content-Length'])


def fetch_range(url, start, end, path):
    """Download one byte range.

    curl is used when available: Python's urllib managed only ~0.2 MB/s on this network
    while curl.exe reached 30+ MB/s, so the download would take days otherwise.
    """
    curl = shutil.which('curl')
    if curl:
        rc = subprocess.call([curl, '-s', '-L', '--retry', '3', '--max-time', '1800',
                              '-r', f'{start}-{end}', '-o', path, url])
        if rc != 0:
            raise RuntimeError(f'curl failed with {rc} for range {start}-{end}')
        return
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(req, timeout=600) as r, open(path, 'wb') as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def download(name, parts, outdir):
    size = remote_size(name)
    out = os.path.join(outdir, name)
    if os.path.exists(out) and os.path.getsize(out) == size:
        print(f'  {name}: already complete ({size/1e9:.2f} GB)')
        return True
    tmp = os.path.join(outdir, f'_tmp_{name}')
    os.makedirs(tmp, exist_ok=True)
    chunk = size // parts
    jobs = []
    for i in range(parts):
        s = i * chunk
        e = size - 1 if i == parts - 1 else s + chunk - 1
        jobs.append((s, e, os.path.join(tmp, f'p{i:03d}.bin')))

    print(f'  {name}: {size/1e9:.2f} GB in {parts} parallel range requests')
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=parts) as ex:
        futs = [ex.submit(fetch_range, BASE + name, s, e, p) for s, e, p in jobs]
        for f in futs:
            f.result()

    with open(out, 'wb') as fo:
        for _s, _e, p in jobs:
            with open(p, 'rb') as fi:
                while True:
                    b = fi.read(1 << 22)
                    if not b:
                        break
                    fo.write(b)
            os.remove(p)
    os.rmdir(tmp)

    got = os.path.getsize(out)
    dt = time.time() - t0
    ok = got == size
    print(f'  {name}: {"OK" if ok else "SIZE MISMATCH"} {got/1e9:.2f} GB '
          f'in {dt:.0f}s ({got/1e6/max(dt,1):.1f} MB/s)')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--only', default=None, choices=list(FILES))
    ap.add_argument('--parts', type=int, default=16)
    a = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    if a.only:
        todo = [a.only]
    elif a.all:
        todo = list(FILES)
    else:
        todo = ['pretrain_t2t', 'sft_t2t_mini']

    print('downloading from ModelScope into', DATA)
    failed = []
    for key in todo:
        name = FILES[key]
        if not download(name, a.parts, DATA):
            failed.append(name)
    if failed:
        print('\nFAILED:', failed)
        return 1
    print('\ndone. next: python custom/pretokenize_packed.py --src data/pretrain_t2t.jsonl '
          '--out data/tokens --seq-len 512 --tag pretrain')
    return 0


if __name__ == '__main__':
    sys.exit(main())
