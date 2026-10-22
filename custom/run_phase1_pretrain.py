"""Phase-1 pretraining launcher: packed corpus, 64M MiniMind, RTX 4060 Laptop.

Measured config (see D:/llm-rl-lab/reports/env_report.md for the sweep):
  hidden=768, layers=8  -> 63.91M params
  seq_len=512, batch=32, accumulation=1  -> 16,384 tokens/step
  bf16 autocast + torch.compile (triton-windows 3.2.0.post21)
  peak reserved VRAM 5,288 MiB (of ~6.7 GiB usable), ~23-25k tokens/sec

Corpus: MiniMind pretrain_t2t.jsonl, pre-tokenized to 4,314,165 blocks of 512
        = 2,208,852,480 tokens (exact, from the tokenizer).

One epoch over the whole corpus = 4,314,165 / 32 = 134,817 steps.
At ~23k tok/s that is ~26.7 h, so this script is started as a managed background job
and checkpoints every 2000 steps (~32.8M tokens) so it can be inspected or resumed.

Usage:
  pwsh> python D:/llm-rl-lab/custom/run_phase1_pretrain.py --epochs 1
"""
import argparse
import os
import subprocess
import sys
import time

# --- project root, derived from this file's location (portable across machines) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

PY = os.path.join(PROJECT_ROOT, '.venv', 'Scripts', 'python.exe')
TRAINER = os.path.join(PROJECT_ROOT, 'custom', 'train_pretrain_packed.py')
WORKDIR = os.path.join(PROJECT_ROOT, 'src', 'minimind-master', 'trainer')
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--seq-len', type=int, default=512)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--save-interval', type=int, default=2000)
    p.add_argument('--save-weight', default='pretrain')
    p.add_argument('--use-compile', type=int, default=1)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--metrics', default=os.path.join(PROJECT_ROOT, 'logs', 'pretrain_metrics.jsonl'))
    args = p.parse_args()

    cmd = [PY, TRAINER,
           '--data_path', 'D:/llm-rl-lab/data/tokens/pretrain_blocks.bin',
           '--max_seq_len', str(args.seq_len),
           '--batch_size', str(args.batch_size),
           '--epochs', str(args.epochs),
           '--learning_rate', str(args.lr),
           '--accumulation_steps', '1',
           '--grad_clip', '1.0',
           '--log_interval', '50',
           '--save_interval', str(args.save_interval),
           '--num_workers', str(args.workers),
           '--use_compile', str(args.use_compile),
           '--hidden_size', '768',
           '--num_hidden_layers', '8',
           '--seed', '42',
           '--save_dir', 'D:/llm-rl-lab/src/minimind-master/out',
           '--save_weight', args.save_weight,
           '--metrics_path', args.metrics]
    if args.max_steps:
        cmd += ['--max_steps', str(args.max_steps)]
    print('[launch] ' + ' '.join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=WORKDIR)
    print(f'[launch] exit={rc} wall={time.time()-t0:.1f}s', flush=True)
    return rc


if __name__ == '__main__':
    sys.exit(main())
