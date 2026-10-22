"""Phase-2 full SFT for MiniMind with exact token accounting.

Thin derivative of the repo's `trainer/train_full_sft.py`. MiniMind's model, dataset
(`dataset/lm_dataset.py::SFTDataset`), optimizer, LR schedule (`trainer_utils.get_lr`),
GradScaler and checkpoint/resume code are unchanged. What is added:

  1. EXACT SFT token accounting. For every iteration the tensors actually fed to the
     model are counted:
         cum_tokens_total       += input_ids.numel()
         cum_tokens_supervised  += (labels != -100).sum()
     Nothing is estimated from steps*batch*seq_len; the numbers come from the tensors.
     Every `--log_interval` iterations a JSONL record is appended to `--metrics_path`:
         {step, loss, lr, cum_tokens_total, cum_tokens_supervised, tokens_per_sec,
          elapsed_sec, eta_hours, gnorm, peak_reserved_MiB}
     plus per-step deltas (`step_tokens_total`, `step_tokens_supervised`), `epoch` and
     `aux_loss`, which make the accounting auditable without recomputing diffs.
  2. `--use_compile` (repo already had it), `--num_workers` (repo), `--max_steps`,
     `--metrics_path`, `--seed` (repo), `--checkpoint_dir`, `--dump_batches`.
  3. Periodic checkpoints are written BOTH as `{save_weight}_{hidden}.pth` (repo
     behaviour, overwritten each time) and as `{save_weight}_{hidden}_step{N}.pth`
     (never overwritten), plus MiniMind's own `_resume.pth` via `lm_checkpoint`.
  4. A final SUMMARY json: total tokens, supervised tokens, wall time, peak VRAM.

PROMPT-FORMAT CONSISTENCY (the one semantic change to the dataset, required by the Lead):
    The canonical held-out-eval prompt (`eval/run_eval.py --prompt-mode chat`, identical
    to the repo's own `eval_llm.py --open_thinking 0`) ends with
        <|im_start|>assistant\n<think>\n\n</think>\n\n
    i.e. an already-closed EMPTY think block. The repo's `SFTDataset.__getitem__` calls
    `post_processing_chat(prompt)` with `remove_empty_think=None`, which strips that block
    80 % of the time (`random.random() > 0.2`), so the repo's SFT training distribution
    would disagree with the eval prompt for 80 % of samples. `SFTDatasetKeepThink` below
    is a byte-for-byte copy of the repo's `__getitem__` with exactly one argument changed:
        post_processing_chat(prompt, remove_empty_think=False)   # always keep the block
    Everything else (pre_processing_chat's random system prompt, create_chat_prompt,
    truncation, padding, generate_labels) is unchanged. Effect: training sees the same
    prompt shape as the `chat` eval mode 100 % of the time, and the
    `<think>\n\n</think>\n\n` tokens inside assistant turns are supervised (they are
    assistant-turn tokens, so supervising them is correct).
    Note: for general conversations that carry `reasoning_content` the think block is
    non-empty and was never stripped anyway, so this change only affects records whose
    assistant turn has no reasoning trace (the aux format-seeding records, and the ~6.6 %
    of general records that have no reasoning_content).

Deliberate modernisations (torch 2.6 emits FutureWarnings for the old API):
    torch.cuda.amp.autocast      -> torch.amp.autocast('cuda', dtype=...)
    torch.cuda.amp.GradScaler    -> torch.amp.GradScaler('cuda', ...)
and, for CPU validation runs, seeding skips `torch.cuda.*` (which would lazily create a
CUDA context and allocate VRAM on a machine whose GPU is busy with the Phase-1 run).

Run (GPU, from D:/llm-rl-lab/src/minimind-master/trainer):
  python D:/llm-rl-lab/custom/train_full_sft_counted.py \
      --data_path D:/llm-rl-lab/data/sft/sft_train.jsonl \
      --from_weight pretrain --save_dir ../out --max_seq_len 768 --batch_size 8 \
      --learning_rate 1e-5 --epochs 1 --num_workers 4 --use_compile 1 \
      --metrics_path D:/llm-rl-lab/logs/sft_metrics.jsonl

CPU validation (tiny model, 200-record slice):
  python D:/llm-rl-lab/custom/train_full_sft_counted.py --device cpu \
      --hidden_size 128 --num_hidden_layers 2 --batch_size 2 --max_seq_len 128 \
      --max_steps 8 --num_workers 0 --from_weight none ...
"""
import argparse
import json
import math
import os
import random
import sys
import time
import warnings

# --- project root, derived from this file's location (portable across machines) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

__package__ = 'trainer'
REPO = os.path.join(PROJECT_ROOT, 'src', 'minimind-master')
sys.path.insert(0, REPO)

import datasets  # noqa: F401  # Windows pyarrow/torch DLL ordering workaround (repo issue #771)
import numpy as np
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset, pre_processing_chat, post_processing_chat
from trainer.trainer_utils import (Logger, get_lr, init_distributed_mode, init_model,
                                   is_main_process, lm_checkpoint, SkipBatchSampler)

warnings.filterwarnings('ignore')

MOE_SUFFIX = ''


class SFTDatasetKeepThink(SFTDataset):
    """`SFTDataset` with the empty `<think>\\n\\n</think>\\n\\n` block always KEPT.

    Verbatim copy of the repo's `__getitem__` (dataset/lm_dataset.py) with a single
    change: `post_processing_chat(prompt)` -> `post_processing_chat(prompt,
    remove_empty_think=False)`. This makes the SFT training prompt identical to the
    canonical `chat` evaluation prompt (`eval/run_eval.py --prompt-mode chat`, i.e.
    `eval_llm.py --open_thinking 0`) instead of matching it only 20 % of the time.
    """

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt, remove_empty_think=False)   # <- only change
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def seed_all(seed, device_type):
    """trainer_utils.setup_seed, minus the torch.cuda.* calls when running on CPU.
    torch.cuda.manual_seed() lazily creates a CUDA context (hundreds of MiB of VRAM);
    on this box the GPU is busy with the Phase-1 pretraining run, so a CPU validation
    must never touch it. On CUDA the behaviour is identical to the repo's."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unwrap(model):
    m = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(m, '_orig_mod', m)


def peak_reserved_mib(device_type):
    if device_type != 'cuda':
        return 0
    return round(torch.cuda.max_memory_reserved() / 2 ** 20)


class RunState:
    """Cumulative, tensor-derived accounting for one run.

    `run_step` and the token counters are seeded from the resume checkpoint so that a
    resumed run continues the global step position (and therefore the cosine LR) instead
    of restarting at step 1 with the LR back at its peak.
    """

    def __init__(self, start_step=0, cum_tokens_total=0, cum_tokens_supervised=0):
        self.run_step = int(start_step)
        self.cum_tokens_total = int(cum_tokens_total)
        self.cum_tokens_supervised = int(cum_tokens_supervised)
        self.resume_step = int(start_step)   # steps <= this are replays after a resume
        self.run_id = time.strftime('%Y%m%dT%H%M%S')
        self.gnorm = 0.0
        self.t_start = time.time()
        self.t_last = time.time()

    def add_batch(self, input_ids, labels):
        total = int(input_ids.numel())
        sup = int((labels != -100).sum().item())
        self.cum_tokens_total += total
        self.cum_tokens_supervised += sup
        return total, sup


class MetricsWriter:
    def __init__(self, path, mode='w'):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.path = path
        # mode='a' on resume: the metrics history is the evidence for the reports and
        # must not be truncated by a restart.
        self.f = open(path, mode, encoding='utf-8', newline='\n')

    def write(self, rec):
        self.f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def save_weights(model, args, lm_config, step=None):
    """Repo-format fp16 state_dict: {save_weight}_{hidden}.pth, plus an immutable
    _step{N}.pth copy when `step` is given."""
    raw = unwrap(model)
    sd = raw.state_dict()
    ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{MOE_SUFFIX}.pth'
    torch.save({k: v.half().cpu() for k, v in sd.items()}, ckp)
    if step is not None:
        torch.save({k: v.half().cpu() for k, v in sd.items()},
                   f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{MOE_SUFFIX}_step{step}.pth')
    del sd
    return ckp


def train_epoch(epoch, loader, iters, state, args, model, optimizer, scaler, autocast_ctx,
                mw, lm_config, total_iters, stop_at):
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    for step, (input_ids, labels) in enumerate(loader, start=1):
        if stop_at and state.run_step >= stop_at:
            break
        state.run_step += 1
        global_step = state.run_step
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        # ---- exact token accounting on the tensors actually fed to the model ----
        step_total, step_sup = state.add_batch(input_ids, labels)

        lr = get_lr(global_step, total_iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0 or step == iters:
            scaler.unscale_(optimizer)
            state.gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters or (stop_at and global_step >= stop_at):
            spend = time.time() - state.t_start
            current_loss = float(loss.item()) * args.accumulation_steps
            current_aux = float(res.aux_loss.item()) if res.aux_loss is not None else 0.0
            tps = state.cum_tokens_total / max(spend, 1e-6)
            planned = total_iters * args.batch_size * args.max_seq_len
            eta_h = max(planned - state.cum_tokens_total, 0) / max(tps, 1e-6) / 3600
            rec = dict(
                step=global_step,
                loss=round(current_loss, 4),
                lr=round(lr, 8),
                cum_tokens_total=int(state.cum_tokens_total),
                cum_tokens_supervised=int(state.cum_tokens_supervised),
                tokens_per_sec=round(tps, 1),
                elapsed_sec=round(spend, 1),
                eta_hours=round(eta_h, 2),
                gnorm=round(state.gnorm, 3),
                peak_reserved_MiB=peak_reserved_mib(device_type),
                # extras (auditable accounting, not estimates)
                step_tokens_total=step_total,
                step_tokens_supervised=step_sup,
                epoch=epoch + 1,
                aux_loss=round(current_aux, 5),
                logits_loss=round(current_loss - current_aux, 4),
                # A resume re-does the batches after the last checkpoint. Because the
                # checkpoint records the step it was saved AT, the first step a resumed
                # run trains is resume_step + 1, which is genuinely new work (the batch
                # that was in flight at the crash is re-consumed). Steps are therefore
                # NOT replayed in the token sense -- but the metrics log can still show
                # the same step number twice if the crash happened after a log line and
                # before the next checkpoint. Consumers must key on `run_id`.
                run_id=state.run_id,
                resumed=bool(state.resume_step > 0),
            )
            mw.write(rec)
            Logger(f'[sft-counted] step {global_step:,}/{total_iters:,} loss={current_loss:.4f} '
                   f'aux={current_aux:.4f} lr={lr:.3e} tok={state.cum_tokens_total:,} '
                   f'sup_tok={state.cum_tokens_supervised:,} tok/s={tps:,.0f} '
                   f'elapsed={spend / 60:.1f}min eta={eta_h:.2f}h gnorm={state.gnorm:.3f} '
                   f'vram={rec["peak_reserved_MiB"]}MiB')

        if args.save_interval and state.run_step % args.save_interval == 0 and is_main_process():
            model.eval()
            ckp = save_weights(model, args, lm_config, step=state.run_step)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                          epoch=epoch, step=state.run_step, save_dir=args.checkpoint_dir,
                          scaler=scaler,
                          cum_tokens_total=int(state.cum_tokens_total),
                          cum_tokens_supervised=int(state.cum_tokens_supervised))
            model.train()
            Logger(f'[sft-counted] saved {ckp} (+ _step{state.run_step}.pth, '
                   f'tokens {state.cum_tokens_total:,})')

        del input_ids, labels, res, loss


def main():
    parser = argparse.ArgumentParser(description='MiniMind full SFT with exact token accounting')
    parser.add_argument('--save_dir', type=str, default='../out')
    parser.add_argument('--save_weight', default='full_sft', type=str)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=768, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--data_path', type=str, default='../dataset/sft_t2t_mini.jsonl')
    parser.add_argument('--from_weight', default='pretrain', type=str)
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1])
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='MiniMind-Full-SFT')
    parser.add_argument('--use_compile', default=0, type=int, choices=[0, 1])
    # ---- added by this thin derivative ----
    parser.add_argument('--max_steps', type=int, default=0, help='0 = full run; else global cap on iterations')
    parser.add_argument('--metrics_path', type=str, default=os.path.join(PROJECT_ROOT, 'logs', 'sft_metrics.jsonl'))
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints',
                        help="MiniMind lm_checkpoint() _resume.pth dir (repo default: ../checkpoints)")
    parser.add_argument('--dump_batches', type=str, default='',
                        help='write the per-step dataset indices here (token reconciliation)')
    args = parser.parse_args()

    global MOE_SUFFIX
    MOE_SUFFIX = '_moe' if args.use_moe else ''

    # ========== 1. env + seed ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f'cuda:{local_rank}'
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    seed_all(args.seed + (dist.get_rank() if dist.is_initialized() else 0), device_type)
    if device_type == 'cpu':
        Logger('[sft-counted] device=cpu -> no CUDA context is created (GPU left untouched)')

    # ========== 2. dirs, config, resume ==========
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size,
                               num_hidden_layers=args.num_hidden_layers,
                               use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight,
                             save_dir=args.checkpoint_dir) if args.from_resume == 1 else None

    # ========== 3. amp ==========
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    autocast_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast('cuda', dtype=dtype)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16' and device_type == 'cuda'))

    # ========== 4. wandb (optional, off by default) ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb  # repo's choice; not used in this project
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        wandb.init(project=args.wandb_project, id=wandb_id,
                   resume='must' if wandb_id else None,
                   name=f'MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}')

    # ========== 5. model / data / optimizer ==========
    model, tokenizer = init_model(lm_config, args.from_weight, save_dir=args.save_dir, device=args.device)
    train_ds = SFTDatasetKeepThink(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. resume ==========
    # `lm_checkpoint` returns None when the file is missing, so a mismatched
    # --save_weight would silently train from scratch. A requested resume that finds
    # nothing must be a hard error.
    start_epoch, start_step = 0, 0
    if args.from_resume == 1 and not ckp_data:
        resume_path = os.path.join(args.checkpoint_dir,
                                   f'{args.save_weight}_{lm_config.hidden_size}'
                                   f'{"_moe" if lm_config.use_moe else ""}_resume.pth')
        raise FileNotFoundError(
            f'--from_resume 1 was requested but no resume checkpoint exists at '
            f'{os.path.abspath(resume_path)}. Refusing to start: continuing silently from '
            f'scratch would reset the cosine LR to its peak and re-train everything. '
            f'Check that --save_weight matches the interrupted run.')
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        Logger(f'[sft-counted] RESUMED at epoch={start_epoch} step={start_step:,}')

    # ========== 7. compile / ddp ==========
    if args.use_compile == 1:
        if device_type == 'cpu':
            Logger('[sft-counted] --use_compile ignored on CPU (pointless, and slow)')
        else:
            model = torch.compile(model)
            Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    state = RunState(start_step=start_step,
                     cum_tokens_total=(ckp_data or {}).get('cum_tokens_total', 0),
                     cum_tokens_supervised=(ckp_data or {}).get('cum_tokens_supervised', 0))
    if start_step and not (ckp_data or {}).get('cum_tokens_total'):
        Logger('[sft-counted] WARNING: resume checkpoint has no cum_tokens_total; token '
               'accounting restarts from 0 for this run (step position is still correct)')
    mw = MetricsWriter(args.metrics_path, mode=('a' if args.from_resume == 1 else 'w'))

    # LR-schedule horizon: the full planned run, or the --max_steps cap. Computed once
    # (the repo recomputes it per epoch from len(loader); the value is the same).
    n_items = len(train_ds) // (dist.get_world_size() if dist.is_initialized() else 1)
    iters_per_epoch = max(1, math.ceil(n_items / args.batch_size))
    total_iters = args.epochs * iters_per_epoch
    if args.max_steps:
        total_iters = min(total_iters, args.max_steps)

    Logger(f'[sft-counted] data={args.data_path} records={len(train_ds):,} '
           f'seq={args.max_seq_len} bs={args.batch_size} accum={args.accumulation_steps} '
           f'epochs={args.epochs} lr={args.learning_rate} device={args.device} '
           f'dtype={args.dtype} workers={args.num_workers} seed={args.seed} '
           f'iters/epoch={iters_per_epoch:,} schedule_horizon={total_iters:,} '
           f'max_steps={args.max_steps or "full"}')
    Logger(f'[sft-counted] metrics -> {args.metrics_path}')

    # ========== 8. train ==========
    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        seed_all(args.seed + epoch, device_type)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        if args.dump_batches and is_main_process():
            batches = list(batch_sampler)
            with open(args.dump_batches, 'w', encoding='utf-8') as f:
                json.dump(dict(seed=args.seed, epoch=epoch, batch_size=args.batch_size,
                               max_seq_len=args.max_seq_len, n_records=len(train_ds),
                               indices_per_step=batches), f)
            Logger(f'[sft-counted] dumped {len(batches)} batch index lists -> {args.dump_batches}')
            loader = DataLoader(train_ds, batch_sampler=batches, num_workers=args.num_workers,
                                pin_memory=(device_type == 'cuda'))
        else:
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers,
                                pin_memory=(device_type == 'cuda'))
        epoch_iters = len(loader)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: skipping first {start_step} steps')
        train_epoch(epoch, loader, epoch_iters, state, args, model, optimizer, scaler,
                    autocast_ctx, mw, lm_config, total_iters, args.max_steps)
        if args.max_steps and state.run_step >= args.max_steps:
            break

    # ========== 9. final save + summary ==========
    if is_main_process():
        model.eval()
        final_ckp = save_weights(model, args, lm_config)
        Logger(f'[sft-counted] final weight -> {final_ckp}')

    wall = time.time() - state.t_start
    summary = dict(
        steps=state.run_step,
        epochs=args.epochs,
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        batch_size=args.batch_size, max_seq_len=args.max_seq_len,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate, device=args.device, dtype=args.dtype,
        total_tokens=int(state.cum_tokens_total),
        supervised_tokens=int(state.cum_tokens_supervised),
        supervised_fraction=round(state.cum_tokens_supervised / max(state.cum_tokens_total, 1), 6),
        tokens_per_sec=round(state.cum_tokens_total / max(wall, 1e-6), 1),
        wall_sec=round(wall, 1),
        peak_alloc_MiB=(round(torch.cuda.max_memory_allocated() / 2 ** 20) if device_type == 'cuda' else 0),
        peak_reserved_MiB=peak_reserved_mib(device_type),
        data_path=args.data_path, dataset_records=len(train_ds),
        final_weight=final_ckp, metrics_path=args.metrics_path,
        seed=args.seed,
    )
    Logger('[sft-counted] SUMMARY ' + json.dumps(summary))
    with open(args.metrics_path.replace('.jsonl', '_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    mw.close()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
