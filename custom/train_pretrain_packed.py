"""Packed-corpus pretraining for MiniMind (Phase 1).

Thin derivative of the official `trainer/train_pretrain.py`. The model, the optimizer,
the LR schedule (`trainer_utils.get_lr`), the autocast/GradScaler setup and the
checkpoint format are MiniMind's own code and are unchanged. What is different:

  1. Dataset: `PackedBlockDataset` reads pre-tokenized fixed-length uint16 blocks
     (custom/pretokenize_packed.py) instead of re-tokenizing + padding per document.
     This removes dataloader cost and pad waste: every token in every step is a real
     corpus token.
  2. Token accounting: cumulative tokens = steps * batch * seq_len, computed from the
     tensors actually fed to the model, and written per log step to a JSONL file.
     reports/pretrain.md is generated from that file, never estimated.
  3. torch.compile is optional (--use_compile 1). It requires triton-windows 3.2.x on
     this machine; measured 1.47x speedup and LOWER peak VRAM.
  4. Periodic weight checkpoints are saved with a step suffix so a long run can be
     resumed / inspected without clobbering the final weight.

Usage (run from D:/llm-rl-lab/src/minimind-master/trainer):
  python D:/llm-rl-lab/custom/train_pretrain_packed.py \
      --data_path D:/llm-rl-lab/data/tokens/pretrain_blocks.bin ...
"""
import argparse
import json
import math
import os
import sys
import time

# --- project root, derived from this file's location (portable across machines) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

__package__ = 'trainer'
REPO = os.path.join(PROJECT_ROOT, 'src', 'minimind-master')
sys.path.insert(0, REPO)

import datasets  # noqa: F401  # Windows pyarrow/torch DLL ordering workaround (repo issue #771)
import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader, Dataset

from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import (Logger, get_lr, init_distributed_mode, init_model,
                                   is_main_process, lm_checkpoint, setup_seed,
                                   SkipBatchSampler)

import warnings
warnings.filterwarnings('ignore')


class BlockRangeSampler(torch.utils.data.Sampler):
    """Sequential block indices with an optional skip of already-consumed batches.

    Mirrors MiniMind's own resume strategy (`SkipBatchSampler` over a randperm list):
    on resume we must (a) not re-consume the batches already trained on, and (b) keep
    the step counter aligned with the LR schedule. Returning indices in order (rather
    than shuffling) makes a resumed run exactly continue the same data stream, which is
    what makes the cosine LR position meaningful.
    """

    def __init__(self, n_blocks, batch_size, skip_batches=0):
        self.n_blocks = n_blocks
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        start = self.skip_batches * self.batch_size
        yield from range(start, self.n_blocks)

    def __len__(self):
        total = self.n_blocks // self.batch_size
        return max(0, total - self.skip_batches)

    def set_epoch(self, epoch):
        """No-op: the packed stream is deterministic, so an epoch marker changes nothing.
        Present because the training loop calls it unconditionally (DDP convention)."""
        self.epoch = epoch


class PackedBlockDataset(Dataset):
    """Fixed-length token blocks, memmapped from a flat uint16 file.

    The memmap is NOT pickled to DataLoader workers (a multi-GB mmap object cannot be
    serialised on Windows: 'pickle data was truncated'). __getstate__ keeps only the
    path/shape and __setstate__ re-opens the memmap inside each worker.
    """

    def __init__(self, bin_path, seq_len, n_blocks=None):
        self.bin_path = bin_path
        self.seq_len = seq_len
        self.n_blocks_limit = n_blocks
        self._open()

    def _open(self):
        self.arr = np.memmap(self.bin_path, dtype=np.uint16, mode='r')
        if self.arr.size % self.seq_len:
            raise ValueError(f'{self.bin_path}: size {self.arr.size} not divisible by seq_len {self.seq_len}')
        self.n = self.arr.size // self.seq_len
        if self.n_blocks_limit:
            self.n = min(self.n, self.n_blocks_limit)
        self.blocks = self.arr[:self.n * self.seq_len].reshape(self.n, self.seq_len)

    def __getstate__(self):
        return dict(bin_path=self.bin_path, seq_len=self.seq_len,
                    n_blocks_limit=self.n_blocks_limit)

    def __setstate__(self, state):
        self.bin_path = state['bin_path']
        self.seq_len = state['seq_len']
        self.n_blocks_limit = state['n_blocks_limit']
        self._open()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        ids = torch.from_numpy(self.blocks[i].astype(np.int64))
        return ids, ids.clone()


def main():
    p = argparse.ArgumentParser(description='MiniMind packed pretraining')
    p.add_argument('--save_dir', type=str, default='../out')
    p.add_argument('--save_weight', default='pretrain', type=str)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--learning_rate', type=float, default=5e-4)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--dtype', type=str, default='bfloat16')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--accumulation_steps', type=int, default=1)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--log_interval', type=int, default=50)
    p.add_argument('--save_interval', type=int, default=2000)
    p.add_argument('--hidden_size', default=768, type=int)
    p.add_argument('--num_hidden_layers', default=8, type=int)
    p.add_argument('--max_seq_len', default=512, type=int)
    p.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--data_path', type=str, default=os.path.join(PROJECT_ROOT, 'data', 'tokens', 'pretrain_blocks.bin'))
    p.add_argument('--from_weight', default='none', type=str)
    p.add_argument('--from_resume', default=0, type=int, choices=[0, 1])
    p.add_argument('--use_compile', default=0, type=int, choices=[0, 1])
    p.add_argument('--metrics_path', default=os.path.join(PROJECT_ROOT, 'logs', 'pretrain_metrics.jsonl'))
    p.add_argument('--max_steps', type=int, default=0, help='0 = full epoch')
    p.add_argument('--warmup_steps', type=int, default=0, help='0 = repo get_lr behaviour')
    args = p.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f'cuda:{local_rank}'
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size,
                               num_hidden_layers=args.num_hidden_layers,
                               use_moe=bool(args.use_moe),
                               max_seq_len=args.max_seq_len)

    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    autocast_ctx = (torch.amp.autocast('cuda', dtype=dtype) if device_type == 'cuda'
                    else __import__('contextlib').nullcontext())

    model, tokenizer = init_model(lm_config, args.from_weight, save_dir=args.save_dir, device=args.device)
    ds = PackedBlockDataset(args.data_path, args.max_seq_len)
    sampler = torch.utils.data.DistributedSampler(ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ---- resume: restore weights/optimizer AND the step position -------------------
    # BUG this guards against: `global_step = epoch * iters + step` restarts at 1 on a
    # resume (start_epoch=0, enumerate from 1), which silently resets the cosine LR back
    # to its peak 5e-4 and re-trains the first 66k batches. Three things must be
    # consistent: the data cursor, the step counter, and the token counter.
    #
    # `lm_checkpoint` looks the file up by `--save_weight`, and returns None when it is
    # absent -- i.e. a typo'd/mismatched --save_weight silently trains from scratch.
    # A requested resume that finds nothing must therefore be a hard error.
    resume_path = os.path.join('..', 'checkpoints',
                               f'{args.save_weight}_{lm_config.hidden_size}'
                               f'{"_moe" if lm_config.use_moe else ""}_resume.pth')
    ckp_data = None
    if args.from_resume == 1:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(
                f'--from_resume 1 was requested but no resume checkpoint exists at '
                f'{os.path.abspath(resume_path)}. Refusing to start: continuing silently '
                f'from scratch would reset the cosine LR to its peak and re-train the '
                f'whole corpus. Check that --save_weight matches the interrupted run.')
        ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints')
        if not ckp_data:
            raise RuntimeError(f'resume checkpoint {os.path.abspath(resume_path)} exists '
                               f'but could not be loaded')
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        Logger(f'[pretrain-packed] RESUMED at epoch={start_epoch} step={start_step:,} '
               f'(LR will continue along the cosine schedule at that position)')

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Data order: sequential over the packed blocks, skipping what has already been
    # consumed. `iters` is the FULL per-epoch step count so the LR denominator is right.
    full_iters = len(ds) // args.batch_size
    if args.max_steps:
        full_iters = min(full_iters, args.max_steps)
    total_iters = full_iters * args.epochs
    tokens_per_step = args.batch_size * args.max_seq_len * args.accumulation_steps

    skip = 0
    if start_step > 0:
        skip = start_step if start_epoch == 0 else 0
        if skip >= full_iters:
            Logger(f'[pretrain-packed] resume step {skip:,} >= iters {full_iters:,}: nothing to do')
            return

    sampler = BlockRangeSampler(len(ds), args.batch_size, skip_batches=skip)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True,
                        persistent_workers=(args.num_workers > 0))
    iters = full_iters
    if skip:
        Logger(f'[pretrain-packed] skipping the first {skip:,} batches already trained on; '
               f'{len(loader):,} steps remain this epoch')

    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    # On resume, APPEND to the existing metrics file instead of truncating it: the
    # metric history is the evidence for reports/pretrain.md and must survive a restart.
    metrics_mode = 'a' if args.from_resume == 1 else 'w'
    if args.from_resume == 1 and os.path.exists(args.metrics_path):
        Logger(f'[pretrain-packed] RESUME: appending to existing metrics {args.metrics_path}')
    mw = open(args.metrics_path, metrics_mode, encoding='utf-8')
    Logger(f'[pretrain-packed] blocks={len(ds):,} seq_len={args.max_seq_len} bs={args.batch_size} '
           f'iters/epoch={iters:,} tokens/step={tokens_per_step:,} '
           f'planned_tokens={total_iters*tokens_per_step:,} compile={args.use_compile}')

    # Token accounting must survive a resume. `start_step * tokens_per_step` is only
    # correct if the resumed run uses the SAME batch size and sequence length as the
    # interrupted one; if either changed, it silently under- or over-counts (measured:
    # resuming a bs=32 run with bs=2 reported 6.2 % of the true token count). So the
    # authoritative cumulative token count is persisted in the resume checkpoint and
    # restored verbatim; the step-based product is only a fallback for old checkpoints.
    cum_tokens = 0
    if ckp_data and ckp_data.get('cum_tokens') is not None:
        cum_tokens = int(ckp_data['cum_tokens'])
        Logger(f'[pretrain-packed] resumed cumulative tokens from checkpoint: {cum_tokens:,}')
    elif start_step:
        cum_tokens = start_step * tokens_per_step
        Logger(f'[pretrain-packed] WARNING: resume checkpoint has no cum_tokens field; '
               f'inferred {cum_tokens:,} from step x tokens_per_step, which is only exact '
               f'if batch_size/seq_len are unchanged')
    cum_prompt_tokens = 0
    t_start = time.time()
    run_step = 0
    for epoch in range(start_epoch, args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        # On the resume epoch the first batch index corresponds to `skip + 1`, so the
        # step counter must start there or the cosine LR restarts at its peak.
        first_step = (skip + 1) if (epoch == start_epoch and skip) else 1
        for step, (input_ids, labels) in enumerate(loader, start=first_step):
            if args.max_steps and step > args.max_steps:
                break
            global_step = epoch * iters + step
            run_step += 1
            lr = get_lr(global_step, total_iters, args.learning_rate)
            if args.warmup_steps and global_step <= args.warmup_steps:
                lr = args.learning_rate * global_step / max(1, args.warmup_steps)
            for g in optimizer.param_groups:
                g['lr'] = lr

            input_ids = input_ids.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            cum_tokens += int(input_ids.numel())

            with autocast_ctx:
                res = model(input_ids, labels=labels)
                loss = (res.loss + res.aux_loss) / args.accumulation_steps
            scaler.scale(loss).backward()

            if step % args.accumulation_steps == 0 or step == iters:
                scaler.unscale_(optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                gnorm_val = float(gnorm)
            else:
                gnorm_val = 0.0

            if step % args.log_interval == 0 or step == iters:
                loss_val = float(loss.item()) * args.accumulation_steps
                el = time.time() - t_start
                tps = (run_step * tokens_per_step) / max(el, 1e-6)
                eta_h = (total_iters - global_step) * tokens_per_step / max(tps, 1) / 3600
                rec = dict(step=global_step, epoch=epoch + 1, loss=round(loss_val, 4),
                           lr=round(lr, 8), cum_tokens=int(cum_tokens),
                           tokens_per_step=tokens_per_step,
                           tokens_per_sec=round(tps),
                           elapsed_sec=round(el, 1), eta_hours=round(eta_h, 2),
                           gnorm=round(gnorm_val, 3),
                           peak_reserved_MiB=round(torch.cuda.max_memory_reserved() / 2**20) if torch.cuda.is_available() else 0)
                mw.write(json.dumps(rec) + '\n')
                mw.flush()
                Logger(f'[pretrain-packed] step {global_step:,}/{total_iters:,} '
                       f'loss={loss_val:.4f} lr={lr:.3e} tokens={cum_tokens:,} '
                       f'tok/s={tps:,.0f} elapsed={el/60:.1f}min eta={eta_h:.1f}h '
                       f'vram={rec["peak_reserved_MiB"]}MiB')

            # Save cadence is keyed to the GLOBAL step so a resumed run keeps the same
            # every-2000-steps rhythm instead of firing immediately at run_step=2000.
            if args.save_interval and global_step % args.save_interval == 0 and is_main_process():
                raw = model
                raw = raw.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else raw
                raw = getattr(raw, '_orig_mod', raw)
                sd = raw.state_dict()
                ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}.pth'
                torch.save({k: v.half().cpu() for k, v in sd.items()}, ckp)
                # Filename + resume step must use the GLOBAL step. Using the per-run
                # counter here would write `step2000.pth` at global step 68000 and, far
                # worse, save `step=2000` into the resume file -- the next resume would
                # then silently restart from step 2000 and re-train 66k batches.
                torch.save({k: v.half().cpu() for k, v in sd.items()},
                           f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}_step{global_step}.pth')
                del sd
                lm_checkpoint(lm_config, weight=args.save_weight, model=model,
                              optimizer=optimizer, scaler=scaler, epoch=epoch, step=global_step,
                              save_dir='../checkpoints', cum_tokens=int(cum_tokens))
                Logger(f'[pretrain-packed] saved {ckp} (global step {global_step:,}, '
                       f'tokens {cum_tokens:,})')

    raw = model
    raw = raw.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else raw
    raw = getattr(raw, '_orig_mod', raw)
    sd = raw.state_dict()
    ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}.pth'
    torch.save({k: v.half().cpu() for k, v in sd.items()}, ckp)
    del sd
    summary = dict(steps=global_step, epochs=args.epochs, tokens_per_step=tokens_per_step,
                   total_tokens=int(cum_tokens),
                   wall_sec=round(time.time() - t_start, 1),
                   peak_alloc_MiB=round(torch.cuda.max_memory_allocated() / 2**20),
                   peak_reserved_MiB=round(torch.cuda.max_memory_reserved() / 2**20),
                   final_weight=ckp)
    Logger('[pretrain-packed] SUMMARY ' + json.dumps(summary))
    with open(args.metrics_path.replace('.jsonl', '_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    mw.close()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
