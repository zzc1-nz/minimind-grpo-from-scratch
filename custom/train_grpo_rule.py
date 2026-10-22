"""Rule-verifiable GRPO trainer for MiniMind (Phase 3).

This is a *thin derivative* of the official `trainer/train_grpo.py`: the GRPO math
(group-normalised advantages, PPO clip, k3 KL estimator, token-level masking) and the
rollout path (`trainer/rollout_engine.py::TorchRolloutEngine`) are MiniMind's own code.
What is replaced:

  1. Reward: MiniMind's GRPO uses an internlm2-1.8B neural reward model (LLM judge).
     Here the reward is a pure function of the task spec -> 1.0 / 0.0, exactly as
     required for programmatically-verifiable alignment. See custom/verifiable_reward.py.
  2. Prompt construction: plain `<|im_start|>user...<|im_end|>` + assistant header,
     no <think> block (these are short-answer tasks, not reasoning traces).
  3. Instrumentation: every step writes sampled-token counts, reward, accuracy, KL,
     clip fraction, entropy and collapse flags to a JSONL metrics file, so the final
     token accounting comes from measured input_ids, not estimates.

Run from the repo trainer dir:
  python D:/llm-rl-lab/custom/train_grpo_rule.py --data_path D:/llm-rl-lab/data/grpo/train_grpo.jsonl ...
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
sys.path.insert(0, 'D:/llm-rl-lab/custom')

import datasets  # noqa: F401  # Windows pyarrow/torch DLL ordering workaround (repo issue #771)
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import Logger, setup_seed, get_lr, init_model, lm_checkpoint
from trainer.rollout_engine import TorchRolloutEngine
from verifiable_reward import reward as reward_fn

warnings_filter = __import__('warnings')
warnings_filter.filterwarnings('ignore')


# --------------------------------------------------------------------------- data
class VerifiablePromptDataset(Dataset):
    """Yields the chat-format prompt for a verifiable task; keeps the task dict."""

    def __init__(self, jsonl_path, tokenizer):
        self.tokenizer = tokenizer
        self.tasks = []
        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.tasks.append(json.loads(line))

    def __len__(self):
        return len(self.tasks)

    # The prompt MUST match the held-out evaluator's canonical `chat` mode
    # (eval/run_eval.py --prompt-mode chat == eval_llm.py --open_thinking 0):
    # the assistant header is followed by an already-CLOSED empty think block, so the
    # model's first generated token is the answer, not `</think>`. Using a different
    # prompt shape here would make the SFT-vs-GRPO comparison measure prompt format
    # instead of policy quality.
    ASSISTANT_HEADER = '<|im_start|>assistant\n<think>\n\n</think>\n\n'

    def build_prompt(self, task):
        return (f'<|im_start|>user\n{task["prompt"]}<|im_end|>\n'
                f'{self.ASSISTANT_HEADER}')

    def __getitem__(self, i):
        return self.build_prompt(self.tasks[i]), i


def collate(batch):
    prompts = [b[0] for b in batch]
    idxs = [b[1] for b in batch]
    return prompts, idxs


# --------------------------------------------------------------------------- metrics
class MetricsWriter:
    def __init__(self, path, mode='w'):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.path = path
        # mode='a' on resume: never truncate the evidence behind the reports.
        self.f = open(path, mode, encoding='utf-8', newline='\n')

    def write(self, rec):
        self.f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default=os.path.join(PROJECT_ROOT, 'data', 'grpo', 'train_grpo.jsonl'))
    p.add_argument('--from_weight', default='full_sft')
    p.add_argument('--save_weight', default='grpo_rule')
    p.add_argument('--save_dir', default=os.path.join(PROJECT_ROOT, 'src', 'minimind-master', 'out'))
    p.add_argument('--hidden_size', type=int, default=768)
    p.add_argument('--num_hidden_layers', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_generations', type=int, default=16)
    p.add_argument('--max_prompt_len', type=int, default=256)
    p.add_argument('--max_gen_len', type=int, default=48)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--learning_rate', type=float, default=5e-7)
    p.add_argument('--beta', type=float, default=0.01, help='KL penalty coefficient')
    p.add_argument('--epsilon', type=float, default=0.2, help='PPO clip epsilon')
    p.add_argument('--temperature', type=float, default=0.9)
    p.add_argument('--accumulation_steps', type=int, default=1)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--log_interval', type=int, default=1)
    p.add_argument('--save_interval', type=int, default=50)
    p.add_argument('--eval_interval', type=int, default=0, help='0 = disabled')
    p.add_argument('--metrics_path', default=os.path.join(PROJECT_ROOT, 'logs', 'grpo_metrics.jsonl'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--max_steps', type=int, default=0, help='0 = all')
    p.add_argument('--from_resume', type=int, default=0, choices=[0, 1],
                   help='1 = continue an interrupted stage from its _resume.pth')
    p.add_argument('--reward_mode', default='binary', choices=['binary', 'shaped'])
    p.add_argument('--lr_schedule', default='cosine', choices=['cosine', 'constant'])
    args = p.parse_args()

    setup_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device
    on_cuda = device.startswith('cuda') and torch.cuda.is_available()
    dtype = torch.bfloat16
    autocast_ctx = (torch.amp.autocast('cuda', dtype=dtype) if on_cuda
                    else __import__('contextlib').nullcontext())

    lm_config = MiniMindConfig(hidden_size=args.hidden_size,
                               num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_prompt_len + args.max_gen_len)

    model, tokenizer = init_model(lm_config, args.from_weight, save_dir=args.save_dir, device=device)
    ref_model, _ = init_model(lm_config, args.from_weight, save_dir=args.save_dir, device=device)
    ref_model = ref_model.eval().requires_grad_(False)

    rollout_engine = TorchRolloutEngine(model, tokenizer, device=device, autocast_ctx=autocast_ctx)
    ds = VerifiablePromptDataset(args.data_path, tokenizer)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                        num_workers=0, drop_last=True)
    iters = len(loader)
    if args.max_steps:
        iters = min(iters, args.max_steps)
    total_opt_steps = max(1, math.ceil(iters / args.accumulation_steps) * args.epochs)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    mw = MetricsWriter(args.metrics_path, mode=('a' if args.from_resume == 1 else 'w'))
    Logger(f'[GRPO-rule] tasks={len(ds)} iters/epoch={iters} group={args.num_generations} '
           f'bs={args.batch_size} gen_len={args.max_gen_len} beta={args.beta} lr={args.learning_rate}')
    Logger(f'[GRPO-rule] metrics -> {args.metrics_path}')

    # ---- resume -------------------------------------------------------------------
    # GRPO stages are short (37-150 steps) but a crash still loses the whole stage.
    # `lm_checkpoint` returns None when the file is missing, so a mismatched
    # --save_weight would silently restart the stage from step 1; make that a hard error.
    cum_sampled_tokens = 0
    cum_prompt_tokens = 0
    global_step = 0
    opt_step = 0
    if args.from_resume == 1:
        moe_sfx = '_moe' if lm_config.use_moe else ''
        # lm_checkpoint() resolves its dir relative to the PROCESS CWD, not to save_dir.
        # The launcher runs from <repo>/trainer, so the file lands in <repo>/checkpoints.
        resume_path = os.path.abspath(os.path.join('..', 'checkpoints',
                                                   f'{args.save_weight}_'
                                                   f'{lm_config.hidden_size}{moe_sfx}_resume.pth'))
        if not os.path.exists(resume_path):
            raise FileNotFoundError(
                f'--from_resume 1 was requested but no resume checkpoint exists at '
                f'{resume_path}. Refusing to restart the stage from scratch silently. '
                f'Check that --save_weight matches the interrupted run.')
        ckp = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints')
        if not ckp:
            raise RuntimeError(f'resume checkpoint {resume_path} unreadable')
        model.load_state_dict(ckp['model'])
        if 'optimizer' in ckp:
            optimizer.load_state_dict(ckp['optimizer'])
        global_step = int(ckp.get('step', 0))
        opt_step = int(ckp.get('opt_step', global_step))
        cum_sampled_tokens = int(ckp.get('cum_sampled_tokens', 0))
        cum_prompt_tokens = int(ckp.get('cum_prompt_tokens', 0))
        Logger(f'[GRPO-rule] RESUMED at step={global_step} sampled_tokens={cum_sampled_tokens:,}')
    t_start = time.time()
    recent_rewards = []

    for epoch in range(args.epochs):
        for step, (prompts, idxs) in enumerate(loader, start=1):
            # --max_steps is a GLOBAL step cap (it is also the LR-schedule horizon), so it
            # must be compared against global_step, not the per-epoch `step`. Comparing
            # the per-epoch counter let a resumed run overshoot the cap.
            if args.max_steps and global_step >= args.max_steps:
                break
            global_step += 1
            t_step = time.time()

            prompt_inputs = tokenizer(prompts, return_tensors='pt', padding=True,
                                      return_token_type_ids=False, padding_side='left',
                                      add_special_tokens=False).to(device)
            prompt_inputs['input_ids'] = prompt_inputs['input_ids'][:, -args.max_prompt_len:]
            prompt_inputs['attention_mask'] = prompt_inputs['attention_mask'][:, -args.max_prompt_len:]
            cum_prompt_tokens += int(prompt_inputs['attention_mask'].sum().item())

            roll = rollout_engine.rollout(
                prompt_ids=prompt_inputs['input_ids'],
                attention_mask=prompt_inputs['attention_mask'],
                num_generations=args.num_generations,
                max_new_tokens=args.max_gen_len,
                temperature=args.temperature,
            )
            outputs = roll.output_ids
            completion_ids = roll.completion_ids
            completions = roll.completions
            old_logps = roll.per_token_logps.to(device).detach()
            prompt_lens = roll.prompt_lens.to(device)

            # ---- measured token accounting (not estimated) ----
            completion_mask_roll = roll.completion_mask.to(device)
            sampled_tokens_step = int(completion_mask_roll.sum().item())
            cum_sampled_tokens += sampled_tokens_step

            full_mask = (outputs != tokenizer.pad_token_id).long()
            logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(
                completion_ids.size(1), device=device).unsqueeze(0)
            full_mask.scatter_(1, logp_pos + 1, completion_mask_roll.to(dtype=full_mask.dtype))

            # ---- rule-based reward, 1/0 ----
            flat_tasks = []
            for i in idxs:
                flat_tasks.extend([ds.tasks[int(i)]] * args.num_generations)
            rewards_list, oks, reasons = [], [], []
            for task, comp in zip(flat_tasks, completions):
                r, ok, why = reward_fn(task, comp, mode=args.reward_mode)
                rewards_list.append(r)
                oks.append(ok)
                reasons.append(why)
            rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
            acc = sum(oks) / len(oks)

            with autocast_ctx:
                res = model(outputs, attention_mask=full_mask)
                aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=device)
                logits = res.logits[:, :-1, :]
                per_tok = F.log_softmax(logits, dim=-1).gather(
                    2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
            with torch.no_grad():
                ref_logits = ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :]
                ref_tok = F.log_softmax(ref_logits, dim=-1).gather(
                    2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
                del ref_logits, logits

            grouped = rewards.view(-1, args.num_generations)
            mean_r = grouped.mean(dim=1).repeat_interleave(args.num_generations)
            std_r = grouped.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
            advantages = (rewards - mean_r) / (std_r + 1e-4)
            # group with zero variance (all right / all wrong) contributes no signal
            degenerate = (std_r < 1e-6).float().mean().item()

            comp_pad = completion_mask_roll.bool()
            is_eos = (completion_ids == tokenizer.eos_token_id) & comp_pad
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            cmask = ((torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
                      <= eos_idx.unsqueeze(1)) & comp_pad).int()

            kl_div = ref_tok - per_tok
            per_tok_kl = torch.exp(kl_div) - kl_div - 1
            ratio = torch.exp(per_tok - old_logps)
            clipped = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_tok_loss = -(torch.min(ratio * advantages.unsqueeze(1),
                                       clipped * advantages.unsqueeze(1)) - args.beta * per_tok_kl)
            policy_loss = ((per_tok_loss * cmask).sum(dim=1) / cmask.sum(dim=1).clamp(min=1)).mean()
            loss = (policy_loss + aux_loss) / args.accumulation_steps
            loss.backward()

            grad_norm = 0.0
            if global_step % args.accumulation_steps == 0 or step == iters:
                if args.grad_clip > 0:
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                if args.lr_schedule == 'cosine':
                    lr_now = get_lr(opt_step, total_opt_steps, args.learning_rate)
                    for g in optimizer.param_groups:
                        g['lr'] = lr_now
            lr_now = optimizer.param_groups[0]['lr']

            # ---- diagnostics ----
            with torch.no_grad():
                cm = cmask.bool()
                kl_mean = float((kl_div * cm).sum().item() / max(cm.sum().item(), 1))
                ent = float((-(per_tok * cm).sum().item()) / max(cm.sum().item(), 1))
                clip_frac = float(((ratio - 1).abs() > args.epsilon).float().mul(cm).sum().item()
                                  / max(cm.sum().item(), 1))
                resp_len = float(cmask.sum(dim=1).float().mean().item())
                adv_std = float(advantages.std().item())
            recent_rewards.append(float(rewards.mean().item()))
            if len(recent_rewards) > 20:
                recent_rewards.pop(0)

            rec = dict(
                step=global_step, epoch=epoch + 1, opt_step=opt_step,
                reward_mean=round(float(rewards.mean().item()), 4),
                reward_mean_20=round(sum(recent_rewards) / len(recent_rewards), 4),
                reward_std=round(float(rewards.std().item()), 4),
                acc=round(acc, 4),
                sampled_tokens=sampled_tokens_step,
                cum_sampled_tokens=cum_sampled_tokens,
                cum_prompt_tokens=cum_prompt_tokens,
                kl_mean=round(kl_mean, 5),
                entropy=round(ent, 4),
                clip_frac=round(clip_frac, 4),
                adv_std=round(adv_std, 4),
                degenerate_group_frac=round(degenerate, 3),
                resp_len_mean=round(resp_len, 2),
                grad_norm=round(grad_norm, 4),
                loss=round(float(policy_loss.item()), 5),
                lr=lr_now,
                sec=round(time.time() - t_step, 2),
                gen_tok_per_sec=round(sampled_tokens_step / max(time.time() - t_step, 1e-6), 1),
                collapsed=bool(ent < 0.05 or resp_len >= args.max_gen_len * 0.99),
            )
            mw.write(rec)
            if global_step % args.log_interval == 0:
                Logger(f"[GRPO-rule] step {global_step}/{iters*args.epochs} "
                       f"reward={rec['reward_mean']:.3f} (20avg {rec['reward_mean_20']:.3f}) "
                       f"acc={acc:.3f} kl={kl_mean:.4f} ent={ent:.3f} clip={clip_frac:.3f} "
                       f"len={resp_len:.1f} advstd={adv_std:.2f} gnorm={grad_norm:.2f} "
                       f"tok/s={rec['gen_tok_per_sec']:.0f} lr={lr_now:.2e}")

            if args.save_interval and global_step % args.save_interval == 0:
                moe = '_moe' if lm_config.use_moe else ''
                ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe}.pth'
                sd = model.state_dict()
                torch.save({k: v.half().cpu() for k, v in sd.items()}, ckp)
                del sd
                # persist the resume state too, so a crash mid-stage can be continued
                # with the same step / token position instead of restarting the stage
                lm_checkpoint(lm_config, weight=args.save_weight, model=model,
                              optimizer=optimizer, epoch=epoch, step=global_step,
                              save_dir='../checkpoints',
                              cum_sampled_tokens=int(cum_sampled_tokens),
                              cum_prompt_tokens=int(cum_prompt_tokens),
                              opt_step=int(opt_step))
                Logger(f'[GRPO-rule] saved {ckp} (step {global_step}, '
                       f'sampled_tokens {cum_sampled_tokens:,})')
                rollout_engine.update_policy(model)

            del prompt_inputs, outputs, completion_ids, per_tok, ref_tok, kl_div, per_tok_kl
            del ratio, clipped, per_tok_loss, cmask, comp_pad, advantages, rewards, loss

    # final save
    moe = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe}.pth'
    sd = model.state_dict()
    torch.save({k: v.half().cpu() for k, v in sd.items()}, ckp)
    del sd
    summary = dict(steps=global_step, opt_steps=opt_step,
                   cum_sampled_tokens=cum_sampled_tokens, cum_prompt_tokens=cum_prompt_tokens,
                   wall_sec=round(time.time() - t_start, 1), final_weight=ckp,
                   peak_alloc_MiB=round(torch.cuda.max_memory_allocated() / 2**20) if on_cuda else 0,
                   peak_reserved_MiB=round(torch.cuda.max_memory_reserved() / 2**20) if on_cuda else 0)
    Logger('[GRPO-rule] SUMMARY ' + json.dumps(summary))
    with open(args.metrics_path.replace('.jsonl', '_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    mw.close()


if __name__ == '__main__':
    main()
