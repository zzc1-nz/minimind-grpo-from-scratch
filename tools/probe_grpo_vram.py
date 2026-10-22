"""Measure GRPO peak VRAM at the EXACT Phase-3 configuration, without training.

The extrapolation from two small measured points says ~3.5 GiB for bs=4/group=16/gen=48,
but both calibration points used max_prompt_len=192 while Phase 3 uses 256, and the
extrapolation is linear in total tokens -- an assumption that can fail because the
attention cost is quadratic in sequence length.

This script runs ONE real GRPO step (rollout + forward + backward + optimizer) at the
production shape and reports the actual peak. It does NOT save weights and does NOT
continue, so it cannot disturb a real run.

Needs the GPU to be free. It refuses to start if another process holds it.

Usage: python probe_grpo_vram.py [--steps 2]
"""
import argparse
import json
import os
import subprocess
import sys
import time

PY = r'D:\llm-rl-lab\.venv\Scripts\python.exe'
ROOT = r'D:\llm-rl-lab'
OUT = rf'{ROOT}\src\minimind-master\out'
LOGS = rf'{ROOT}\logs'
TRAINER_CWD = rf'{ROOT}\src\minimind-master\trainer'


def free_vram_mib():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            text=True, timeout=30)
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--group', type=int, default=16)
    p.add_argument('--bs', type=int, default=4)
    p.add_argument('--gen', type=int, default=48)
    p.add_argument('--prompt-len', type=int, default=256)
    p.add_argument('--force', action='store_true')
    a = p.parse_args()

    fv = free_vram_mib()
    print(f'free VRAM: {fv} MiB')
    if fv < 5000 and not a.force:
        print('ABORT: GPU is busy (need >= 5000 MiB free). Close the game / other app, '
              'or pass --force to measure anyway (the number will be unreliable).')
        return 1

    os.makedirs(LOGS, exist_ok=True)
    metrics = os.path.join(LOGS, 'probe_grpo_vram.jsonl')
    if os.path.exists(metrics):
        os.remove(metrics)

    # Use the stage1 curriculum (easiest) so a real rollout produces a mix of
    # correct/incorrect rather than all-garbage; the VRAM cost is the same either way.
    cmd = [PY, rf'{ROOT}\custom\train_grpo_rule.py',
           '--data_path', rf'{ROOT}\data\grpo\stage1_easy.jsonl',
           '--from_weight', 'pretrain',
           '--save_weight', '_vram_probe',
           '--save_dir', OUT,
           '--hidden_size', '768', '--num_hidden_layers', '8',
           '--batch_size', str(a.bs), '--num_generations', str(a.group),
           '--max_prompt_len', str(a.prompt_len), '--max_gen_len', str(a.gen),
           '--epochs', '1', '--max_steps', str(a.steps),
           '--learning_rate', '5e-7', '--beta', '0.01', '--temperature', '0.9',
           '--save_interval', '1000000', '--log_interval', '1',
           '--device', 'cuda', '--seed', '42',
           '--metrics_path', metrics]
    print('launching:', ' '.join(cmd[:6]), '...')
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=TRAINER_CWD)
    dt = time.time() - t0
    if rc != 0:
        print(f'probe failed with exit {rc} after {dt:.1f}s')
        return 1

    rows = []
    if os.path.exists(metrics):
        with open(metrics, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    summary = metrics.replace('.jsonl', '_summary.json')
    s = None
    if os.path.exists(summary):
        with open(summary, encoding='utf-8') as f:
            s = json.load(f)

    peak = max((r.get('peak_reserved_MiB') or 0) for r in rows) if rows else None
    # the summary carries the authoritative torch.cuda.max_memory_reserved
    peak_alloc = (s or {}).get('peak_alloc_MiB')
    peak_resv = (s or {}).get('peak_reserved_MiB')
    rec = {
        'config': {'bs': a.bs, 'group': a.group, 'gen': a.gen,
                   'prompt_len': a.prompt_len},
        'samples_per_step': a.bs * a.group,
        'tokens_per_step': a.bs * a.group * (a.prompt_len + a.gen),
        'steps': a.steps, 'wall_sec': round(dt, 1),
        'peak_alloc_MiB': peak_alloc, 'peak_reserved_MiB': peak_resv,
        'free_vram_before_MiB': fv,
        'headroom_MiB': (fv - (peak_resv or 0)) if peak_resv else None,
        'step_metrics': rows,
    }
    print()
    print(json.dumps({k: v for k, v in rec.items() if k != 'step_metrics'}, indent=2))
    with open(os.path.join(LOGS, 'probe_grpo_vram.json'), 'w', encoding='utf-8') as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)

    # clean up the probe weight so it can never be mistaken for a real checkpoint
    for suffix in (f'_vram_probe_768.pth',):
        wp = os.path.join(OUT, suffix)
        if os.path.exists(wp):
            os.remove(wp)
            print(f'removed probe weight {wp}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
