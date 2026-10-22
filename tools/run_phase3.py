"""Phase-3 GRPO launcher: staged curriculum, healthchecks, and a summary.

Runs the difficulty curriculum built by `custom/build_grpo_curriculum.py` in
sequence, each stage resuming from the previous stage's weights:

    stage1_easy  -> grpo_healthcheck -> stage2_medium -> grpo_healthcheck
                 -> stage3_full      -> grpo_healthcheck -> summary

**The default is `--dry-run`: it prints the exact commands and executes nothing.**
`--execute` is required to actually launch the GPU training jobs. This is
deliberate: the GPU is occupied by the ~26 h pretraining job, and a launcher that
starts training by accident would cost the project a day.

Why a curriculum at all (measured evidence, not a guess)
--------------------------------------------------------
GRPO normalises advantages within a group: A_i = (r_i - mean(r)) / (std(r)+eps).
A group where every sample gets the same reward contributes exactly zero
gradient. Measured on this project:

  logs/grpo_rule_smoke.jsonl   600-task set, 3 steps: reward 0.000, adv_std 0.000,
                               degenerate_group_frac 1.0 for every step.
  logs/grpo_easy_probe.jsonl   200 easy `a+b, a,b in [0,4]` tasks, 30 steps:
                               reward 0.078 -> 0.169, adv_std ~1.0,
                               mean degenerate_group_frac 0.21.

So the model must first be put where it sometimes succeeds (stage1), then the
task distribution is widened once advantage is non-zero.

Recommended hyperparameters (each justified from a measurement in this repo)
---------------------------------------------------------------------------
  --num_generations 16  Decode is LATENCY-bound: 128 new tokens take ~2.4 s
                        regardless of batch size, so samples are nearly free
                        (group 8 = 430 tok/s, 16 = 950, 32 = 1,721; env_report
                        §7c). Group 16 doubles the chance of a mixed group vs
                        group 8 while costing ~0.3 s more per step.
  --max_gen_len 48      Answers here are 1-3 tokens (single-token ints 0..25);
                        32 is enough for the answer plus a short explanation, 48
                        leaves headroom for a `<|im_end|>` and prevents the
                        "hit the cap on every sample" degenerate-length mode
                        seen in the smoke run (resp_len_mean 32.0 == cap).
                        64+ only adds latency with no reward benefit.
  --beta 0.01           The repo default and the value used in every measured
                        GRPO run here; KL stayed small (mean kl_mean -0.037 to
                        0.033) at 0.01, so there is no evidence for a different
                        value.
  --learning_rate 5e-7  Repo default for GRPO. The policy is already SFT'd and
                        only needs nudging; the smoke runs used 5e-7..1e-6 with
                        no divergence.
  --batch_size 4        Prompts per optimisation step. group 16 x batch 4 = 64
                        rollouts/step. VRAM measured for exactly this regime:
                        group 8-16 at gen_len 32-64 costs ~2,270 MiB (env_report
                        §9c/9d), far below the 8,188 MiB card, so batch 4 fits
                        with the group size that actually helps.
  --temperature 0.9     Needed for group diversity: a greedy rollout makes every
                        sample of a group identical -> adv_std 0 always.
  --max_prompt_len 256  The longest task prompt in the 600-task set is well
                        under this; it is the repo default.
  --lr_schedule cosine  Decay over the stage's steps so later steps do not
                        overshoot a small policy change.

VRAM / time budget
------------------
  VRAM: ~2,270 MiB peak reserved for group 8-16, gen_len 32-64, batch 2-4
        (measured, logs/grpo_easy_probe_summary.json peak_reserved_MiB = 2270;
        logs/grpo_rule_smoke_summary.json = 1940). This leaves ~5.9 GB for the
        pretraining job, which is why GRPO must NOT run until pretraining is
        done anyway (the pretrain job itself reserves 5,288 MiB).
  Time: at 64 rollouts/step and ~950 gen tok/s, a 16-token answer set takes
        ~1.0-1.3 s/step for rollout plus ~0.3-0.8 s for the policy/ref forward
        pass. Measured 30-step probe: 23.2 s wall = 0.77 s/step at group 8,
        gen_len 24. Budget ~1.5 s/step at group 16 / gen_len 48:
          stage1  150 tasks / bs 4 = 38 iters  x 300 steps -> ~8 min
          stage2  350 tasks / bs 4 = 87 iters  x 600 steps -> ~15 min
          stage3  600 tasks / bs 4 = 150 iters x 900 steps -> ~23 min
        i.e. well under 1 h of GRPO after the GPU is free. The dominant Phase-3
        cost is SFT, not GRPO.

Usage
-----
    # print the commands (default; executes nothing)
    python tools/run_phase3.py

    # actually run (ONLY after the pretrain job has finished)
    python tools/run_phase3.py --execute

    # run from a specific SFT checkpoint, custom step counts
    python tools/run_phase3.py --execute --from-weight full_sft --stage1-steps 300
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
CUSTOM = os.path.join(ROOT, 'custom')
TRAINER_DIR = os.path.join(ROOT, 'src', 'minimind-master', 'trainer')
OUT_DIR = os.path.join(ROOT, 'src', 'minimind-master', 'out')
LOGS = os.path.join(ROOT, 'logs')
DATA = os.path.join(ROOT, 'data', 'grpo')
REPORTS = os.path.join(ROOT, 'reports')

TRAIN_SCRIPT = os.path.join(CUSTOM, 'train_grpo_rule.py')
HEALTHCHECK = os.path.join(TOOLS_DIR, 'grpo_healthcheck.py')

# One entry per curriculum stage. `target_steps` is the number of optimiser
# steps we want in the stage. IMPORTANT: train_grpo_rule.py counts `--max_steps`
# PER EPOCH (`for epoch ... for step ... if step > max_steps: break`, with
# `step` restarting at 1 each epoch), so the real knob is `--epochs`: the
# launcher converts target_steps into epochs = ceil(target / iters_per_epoch)
# and passes `--max_steps <iters_per_epoch>` (i.e. one full epoch) so the
# printed command is unambiguous and the achieved step count is iters*epochs.
STAGES = [
    dict(name='stage1_easy', data='stage1_easy.jsonl', target_steps=300,
         desc='easiest 150 tasks: single-step, |answer| small, no negation/JSON'),
    dict(name='stage2_medium', data='stage2_medium.jsonl', target_steps=600,
         desc='easy + medium 350 tasks: adds negation, JSON, 2-digit answers'),
    dict(name='stage3_full', data='stage3_full.jsonl', target_steps=900,
         desc='all 600 tasks, same difficulty order'),
]

HPARAMS = dict(
    num_generations=16,
    max_gen_len=48,
    max_prompt_len=256,
    beta=0.01,
    learning_rate=5e-7,
    batch_size=4,
    temperature=0.9,
    epsilon=0.2,
    accumulation_steps=1,
    grad_clip=1.0,
    lr_schedule='cosine',
    reward_mode='binary',
    epochs=1,
)

HPARAM_JUSTIFICATION = {
    'num_generations': ('decode is latency-bound (128 new tokens ~2.4 s at any batch '
                        'size; 430/950/1721 tok/s at group 8/16/32), so a bigger group '
                        'is nearly free and raises the chance of a mixed (non-degenerate) '
                        'group'),
    'max_gen_len': ('answers are 1-3 tokens; 32 already suffices, 48 leaves room for '
                    '<|im_end|> and avoids the every-sample-hits-the-cap mode'),
    'beta': 'repo default; measured mean kl_mean stayed small (-0.037..0.033) at 0.01',
    'learning_rate': 'repo default for GRPO on an already-SFT policy; smoke runs at 5e-7..1e-6 did not diverge',
    'batch_size': 'group 16 x batch 4 = 64 rollouts/step; measured ~2,270 MiB for this regime',
    'temperature': ('a greedy rollout makes every sample of a group identical -> adv_std '
                    'is always 0; 0.9 is needed for group diversity'),
    'max_prompt_len': 'longest task prompt is far below 256; repo default',
    'lr_schedule': 'cosine decay over the stage steps to avoid overshooting a small policy change',
}


def count_lines(path):
    if not os.path.exists(path):
        return None
    n = 0
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def build_stage_plan(args):
    """Return the list of stages with resolved paths, epochs and step counts."""
    plan = []
    for i, st in enumerate(STAGES):
        data_path = os.path.join(DATA, st['data'])
        n_tasks = count_lines(data_path)
        iters = (n_tasks // HPARAMS['batch_size']) if n_tasks else None
        target = getattr(args, f"{st['name'].split('_')[0]}_steps", st['target_steps'])
        if iters:
            # --max_steps is per epoch in the trainer; epochs deliver the steps.
            epochs = max(1, math.ceil(target / iters))
            max_steps = iters              # one full epoch per --max_steps window
            achieved = iters * epochs
        else:
            epochs, max_steps, achieved = 1, target, target
        metrics = os.path.join(LOGS, f'grpo_{st["name"]}.jsonl')
        save_weight = f'grpo_{st["name"]}'
        # Resume chain: stage1 from the SFT checkpoint, each later stage from
        # the previous stage's saved weight.
        from_weight = (args.from_weight if i == 0
                       else f'grpo_{STAGES[i - 1]["name"]}')
        plan.append(dict(
            name=st['name'], desc=st['desc'], data_path=data_path, n_tasks=n_tasks,
            target_steps=target, epochs=epochs, max_steps=max_steps,
            achieved_steps=achieved, iters=iters,
            metrics=metrics, save_weight=save_weight, from_weight=from_weight,
            healthcheck_json=os.path.join(LOGS, f'grpo_{st["name"]}_healthcheck.json'),
        ))
    return plan


def trainer_cmd(stage, args):
    """The exact command line for one stage (list of argv tokens)."""
    cmd = [sys.executable, TRAIN_SCRIPT,
           '--data_path', stage['data_path'],
           '--from_weight', stage['from_weight'],
           '--save_weight', stage['save_weight'],
           '--save_dir', OUT_DIR,
           '--hidden_size', str(args.hidden),
           '--num_hidden_layers', str(args.layers),
           '--batch_size', str(HPARAMS['batch_size']),
           '--num_generations', str(HPARAMS['num_generations']),
           '--max_prompt_len', str(HPARAMS['max_prompt_len']),
           '--max_gen_len', str(HPARAMS['max_gen_len']),
           '--epochs', str(stage['epochs']),
           '--learning_rate', str(HPARAMS['learning_rate']),
           '--beta', str(HPARAMS['beta']),
           '--epsilon', str(HPARAMS['epsilon']),
           '--temperature', str(HPARAMS['temperature']),
           '--accumulation_steps', str(HPARAMS['accumulation_steps']),
           '--grad_clip', str(HPARAMS['grad_clip']),
           '--lr_schedule', HPARAMS['lr_schedule'],
           '--reward_mode', HPARAMS['reward_mode'],
           '--max_steps', str(stage['max_steps']),
           '--save_interval', str(max(1, stage['achieved_steps'] // 3)),
           '--log_interval', '10',
           '--metrics_path', stage['metrics'],
           '--seed', str(args.seed),
           '--device', args.device]
    return cmd


def healthcheck_cmd(stage):
    return [sys.executable, HEALTHCHECK, stage['metrics'],
            '--json', stage['healthcheck_json'],
            '--min-steps', '20']


def fmt_cmd(cmd):
    """Shell-safe one-line rendering of an argv list."""
    if os.name == 'nt':
        return subprocess.list2cmdline(cmd)
    return ' '.join(shlex.quote(c) for c in cmd)


def print_plan(plan, args):
    print('=' * 78)
    print('PHASE 3 — rule-reward GRPO curriculum')
    print('=' * 78)
    print(f'project root : {ROOT}')
    print(f'python       : {sys.executable}')
    print(f'device       : {args.device}   (cuda = the GPU; cpu only for plumbing tests)')
    print(f'from_weight  : {args.from_weight}')
    print(f'out dir      : {OUT_DIR}')
    print(f'logs dir     : {LOGS}')
    print()
    print('STAGES (resume chain):')
    prev = args.from_weight
    for st in plan:
        iters = st['iters'] if st['iters'] is not None else 'n/a'
        print(f'  {st["name"]:<14} {st["n_tasks"] if st["n_tasks"] is not None else "n/a":>4} tasks'
              f'  {iters:>3} iters/epoch  --epochs {st["epochs"]:>2}'
              f'  -> {st["achieved_steps"]:>4} steps  from {prev}')
        print(f'{"":<16}{st["desc"]}')
        print(f'{"":<16}(target {st["target_steps"]} steps; --max_steps is PER EPOCH in '
              f'train_grpo_rule.py, so epochs deliver the count: {iters} x {st["epochs"]})')
        prev = st['save_weight']
    print()
    print('HYPERPARAMETERS (each justified by a measurement in this repo):')
    for key in ('num_generations', 'max_gen_len', 'beta', 'learning_rate', 'batch_size',
                'temperature', 'max_prompt_len', 'lr_schedule'):
        print(f'  {key:<16} {HPARAMS[key]!s:<10} {HPARAM_JUSTIFICATION[key]}')
    print()
    print('VRAM / TIME BUDGET:')
    print('  VRAM  ~2,270 MiB peak reserved measured for group 8-16 / gen_len 32-64 /')
    print('        batch 2-4 (logs/grpo_easy_probe_summary.json). The pretraining job')
    print('        reserves 5,288 MiB, so GRPO must run AFTER it finishes — do not')
    print('        launch concurrently.')
    print('  TIME  measured 0.77 s/step at group 8 / gen_len 24; ~1.5 s/step projected')
    print('        at group 16 / gen_len 48. Per-stage budget:')
    for st in plan:
        print(f'          {st["name"]:<14} {st["achieved_steps"]:>4} steps x 1.5 s = '
              f'{st["achieved_steps"] * 1.5 / 60:5.1f} min')
    total_steps = sum(s['achieved_steps'] for s in plan)
    print(f'          {"TOTAL":<14} {total_steps:>4} steps        = '
          f'{total_steps * 1.5 / 60:5.1f} min  (< 1 h; SFT dominates Phase 3 cost)')
    print()
    print('=' * 78)
    print('COMMANDS' + ('  (DRY RUN — nothing below is executed)' if not args.execute else
                        '  (--execute: these WILL run)'))
    print('=' * 78)
    for st in plan:
        print()
        print(f'# {st["name"]}: {st["desc"]}')
        print(fmt_cmd(trainer_cmd(st, args)))
        print()
        print(f'# healthcheck for {st["name"]}')
        print(fmt_cmd(healthcheck_cmd(st)))
    print()
    print('# final: SFT vs GRPO held-out comparison (after the real checkpoints exist)')
    print(fmt_cmd([sys.executable, os.path.join(TOOLS_DIR, 'compare_sft_grpo.py'),
                   '--sft', os.path.join(OUT_DIR, f'{args.sft_weight}_{args.hidden}.pth'),
                   '--grpo', os.path.join(OUT_DIR, f'grpo_stage3_full_{args.hidden}.pth'),
                   '--out', os.path.join(REPORTS, 'grpo_vs_sft.json'),
                   '--hidden', str(args.hidden), '--layers', str(args.layers),
                   '--max-new-tokens', str(args.max_new_tokens),
                   '--device', 'cuda']))
    print()
    print('=' * 78)


def run_stage(stage, args):
    """Run one training stage + its healthcheck. Returns a summary dict."""
    cmd = trainer_cmd(stage, args)
    t0 = time.time()
    print(f'\n[phase3] >>> {stage["name"]}')
    print(f'[phase3] {fmt_cmd(cmd)}')
    env = dict(os.environ)
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    proc = subprocess.run(cmd, cwd=TRAINER_DIR, env=env)
    train_sec = time.time() - t0
    rec = {'stage': stage['name'], 'train_rc': proc.returncode, 'train_sec': round(train_sec, 1),
           'metrics_path': stage['metrics']}
    if proc.returncode != 0:
        print(f'[phase3] !! {stage["name"]} trainer exited {proc.returncode}; '
              f'stopping the chain')
        rec['aborted'] = True
        return rec
    # healthcheck is a diagnostic: always exit 0, but capture the verdict
    hc = healthcheck_cmd(stage)
    print(f'[phase3] {fmt_cmd(hc)}')
    hproc = subprocess.run(hc, cwd=ROOT, env=env, capture_output=True, text=True,
                           encoding='utf-8', errors='replace')
    print(hproc.stdout or '')
    rec['healthcheck_rc'] = hproc.returncode
    verdict = None
    for line in (hproc.stdout or '').splitlines():
        if line.strip().startswith('VERDICT:'):
            verdict = line.split(':', 1)[1].strip()
    rec['verdict'] = verdict
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(
        description='Phase-3 staged GRPO launcher (dry-run by default)',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--execute', action='store_true',
                   help='actually run the GPU training commands (default: dry-run)')
    p.add_argument('--dry-run', action='store_true',
                   help='explicitly request dry-run (the default)')
    p.add_argument('--from-weight', default='full_sft',
                   help='SFT checkpoint name (without _<hidden>.pth) stage1 starts from')
    p.add_argument('--sft-weight', default='full_sft',
                   help='SFT checkpoint name used by the final comparison command')
    p.add_argument('--hidden', type=int, default=768)
    p.add_argument('--layers', type=int, default=8)
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                   help='device for the trainer (cpu only for plumbing tests)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-new-tokens', type=int, default=64,
                   help='max_new_tokens for the final comparison command')
    p.add_argument('--stage1-steps', type=int, default=STAGES[0]['target_steps'])
    p.add_argument('--stage2-steps', type=int, default=STAGES[1]['target_steps'])
    p.add_argument('--stage3-steps', type=int, default=STAGES[2]['target_steps'])
    p.add_argument('--summary-out', default=os.path.join(LOGS, 'phase3_run_summary.json'))
    p.add_argument('--stop-after', default=None, choices=['stage1_easy', 'stage2_medium'],
                   help='stop the chain after this stage (for a partial run)')
    args = p.parse_args(argv)

    if args.dry_run:
        args.execute = False
    plan = build_stage_plan(args)
    print_plan(plan, args)

    if not args.execute:
        print()
        print('DRY RUN complete: 0 training processes started, no CUDA context created.')
        print('Re-run with --execute once the pretraining job has finished.')
        return 0

    # ---- preflight: refuse to start on a busy GPU ----
    if args.device == 'cuda':
        try:
            import torch
            free_b, total_b = torch.cuda.mem_get_info()
            free_mib = free_b / (1024.0 * 1024.0)
            if free_mib < 5000:
                print(f'\n[phase3] REFUSING to start: only {free_mib:.0f} MiB free VRAM '
                      f'(need >= 5000; the pretraining job is probably still running). '
                      f'Wait for it to finish, or pass --device cpu for a plumbing test.')
                return 2
            print(f'[phase3] preflight: {free_mib:.0f} MiB free VRAM, proceeding')
        except Exception as exc:  # noqa: BLE001
            print(f'[phase3] preflight warning: could not read free VRAM ({exc})')

    missing = [s['data_path'] for s in plan if not os.path.exists(s['data_path'])]
    if missing:
        print(f'\n[phase3] REFUSING to start: missing curriculum files: {missing}')
        print('  build them first: python custom/build_grpo_curriculum.py')
        return 2

    results = []
    t_start = time.time()
    for stage in plan:
        rec = run_stage(stage, args)
        results.append(rec)
        if rec.get('aborted'):
            break
        if args.stop_after and stage['name'] == args.stop_after:
            print(f'[phase3] --stop-after {args.stop_after}: stopping the chain')
            break

    summary = {
        'created_at': __import__('datetime').datetime.now().astimezone().isoformat(
            timespec='seconds'),
        'device': args.device, 'seed': args.seed,
        'hidden': args.hidden, 'layers': args.layers,
        'hyperparameters': HPARAMS,
        'stages': results,
        'wall_sec_total': round(time.time() - t_start, 1),
        'aborted': any(r.get('aborted') for r in results),
    }
    os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)
    with open(args.summary_out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('\n' + '=' * 78)
    print('PHASE 3 SUMMARY')
    print('=' * 78)
    for r in results:
        print(f'  {r["stage"]:<14} rc={r["train_rc"]:<3} {r["train_sec"]:>7.1f} s  '
              f'verdict={r.get("verdict") or "n/a"}')
    print(f'  total wall {summary["wall_sec_total"]:.1f} s -> {args.summary_out}')
    if summary['aborted']:
        print('  ABORTED: a stage failed; see the log above.')
        return 1
    print('\nNext: run tools/compare_sft_grpo.py on the final checkpoints '
          '(the command is printed in the dry run above).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
