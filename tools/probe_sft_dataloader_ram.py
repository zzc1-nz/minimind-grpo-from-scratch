"""Measure SFT dataloader RAM cost with multiple workers (16 GB hard constraint).

The Phase-2 run uses `--num_workers 4` on a 200,000-record jsonl. On Windows the
`datasets` object is pickled into each worker (spawn), so the resident cost is
4 x (dataset copy) + overhead. If that exceeds the free RAM, the run dies mid-flight
inside the unattended chain -- exactly the kind of failure worth finding now.

Runs on CPU only (the GPU is busy with Phase 1). Does not build a model.

Usage: python probe_sft_dataloader_ram.py --workers 4 --batches 40
"""
import argparse
import json
import os
import subprocess
import sys
import time

# --- project root, derived from this file's location (portable across machines) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

REPO = os.path.join(PROJECT_ROOT, 'src', 'minimind-maste')
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset.lm_dataset import SFTDataset

p = argparse.ArgumentParser()
p.add_argument('--data', default=os.path.join(PROJECT_ROOT, 'data\\sft\\sft_train.jsonl'))
p.add_argument('--workers', type=int, default=4)
p.add_argument('--batches', type=int, default=40)
p.add_argument('--batch-size', type=int, default=16)
p.add_argument('--seq', type=int, default=768)
args = p.parse_args()


def rss_mib():
    """Total working set of this process + all python children (Windows)."""
    try:
        out = subprocess.check_output(
            ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV', '/NH'],
            text=True, timeout=30, errors='replace')
    except Exception:
        return None
    total = 0
    for line in out.splitlines():
        parts = [x.strip('"') for x in line.split('","')]
        if len(parts) < 5:
            continue
        try:
            total += int(parts[4].replace(',', '').replace(' K', '').replace('K', ''))
        except Exception:
            pass
    return total / 1024.0


def avail_ram_mib():
    try:
        out = subprocess.check_output(
            ['powershell', '-NoProfile', '-Command',
             '(Get-Counter "\\Memory\\Available MBytes").CounterSamples[0].CookedValue'],
            text=True, timeout=30)
        return float(out.strip())
    except Exception:
        return None


t0 = time.time()
print(f'[probe] loading tokenizer + SFTDataset ({args.data})')
tok = AutoTokenizer.from_pretrained(os.path.join(REPO, 'model'))
ds = SFTDataset(args.data, tok, max_length=args.seq)
t_load = time.time() - t0
print(f'[probe] dataset len={len(ds):,} load={t_load:.1f}s')
print(f'[probe] RSS after load: {rss_mib():.0f} MiB | available RAM: {avail_ram_mib():.0f} MiB')

loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=False, drop_last=True,
                    persistent_workers=(args.workers > 0))

peak_rss = 0
peak_avail_drop = None
avail0 = avail_ram_mib()
t0 = time.time()
n_tok = 0
n_sup = 0
for i, (ids, labels) in enumerate(loader):
    n_tok += int(ids.numel())
    n_sup += int((labels != -100).sum())
    if i % 5 == 0:
        r = rss_mib()
        a = avail_ram_mib()
        peak_rss = max(peak_rss or 0, r or 0)
        if avail0 and a:
            peak_avail_drop = max(peak_avail_drop or 0, avail0 - a)
        print(f'[probe] batch {i:3d} rss={r:.0f} MiB avail={a:.0f} MiB', flush=True)
    if i + 1 >= args.batches:
        break
dt = time.time() - t0
peak_rss = max(peak_rss or 0, rss_mib() or 0)

rec = dict(workers=args.workers, batch_size=args.batch_size, seq=args.seq,
           batches=args.batches, batches_done=i + 1,
           dataset_records=len(ds), dataset_load_sec=round(t_load, 1),
           tokens_total=n_tok, supervised_tokens=n_sup,
           supervised_fraction=round(n_sup / max(n_tok, 1), 4),
           peak_rss_mib=round(peak_rss),
           initial_avail_ram_mib=round(avail0) if avail0 else None,
           peak_available_ram_drop_mib=round(peak_avail_drop) if peak_avail_drop else None,
           sec_per_batch=round(dt / max(i + 1, 1), 3),
           batches_per_sec=round((i + 1) / max(dt, 1e-6), 2),
           wall_sec=round(dt, 1))
print('[probe] ' + json.dumps(rec))
os.makedirs(r'D:\llm-rl-lab\logs', exist_ok=True)
with open(r'D:\llm-rl-lab\logs\probe_sft_dataloader_ram.json', 'w') as f:
    json.dump(rec, f, indent=2)
