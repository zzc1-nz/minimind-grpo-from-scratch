"""Verify resume correctness without the GPU.

Builds a tiny corpus, trains a few steps, saves a checkpoint, then resumes and asserts:
  1. the LR at the resumed step equals the LR the uninterrupted schedule would have had
     (this is the bug that would silently reset cosine LR to its peak);
  2. the data cursor advanced (no re-training of already-consumed batches);
  3. the token counter continues (does not restart at 0);
  4. the checkpoint step written after resume is the GLOBAL step, not the per-run one.

CPU only. Uses hidden=64, 2 layers, seq=64 so it finishes in seconds.

Usage: python test_resume_correctness.py
"""
import json
import math
import os
import shutil
import subprocess
import sys

PY = r'D:\llm-rl-lab\.venv\Scripts\python.exe'
ROOT = r'D:\llm-rl-lab'
WORK = os.path.join(ROOT, 'logs', '_resume_test')
TRAINER_CWD = os.path.join(ROOT, 'src', 'minimind-master', 'trainer')
TRAIN = os.path.join(ROOT, 'custom', 'train_pretrain_packed.py')


def get_lr(step, total, lr=5e-4):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total)))


def read_metrics(p):
    rows = []
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    return rows


def run(cmd):
    r = subprocess.run(cmd, cwd=TRAINER_CWD, capture_output=True, text=True,
                      errors='replace')
    return r


def main():
    if os.path.exists(WORK):
        shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK, exist_ok=True)
    ckpt_dir = os.path.join(WORK, 'checkpoints')
    out_dir = os.path.join(WORK, 'out')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Tiny packed corpus: 256 blocks x 64 tokens -> 64 iters at bs=4.
    # Phase A is given only the first 80 blocks (20 iters) so it stops at step 20
    # naturally; phase B gets the whole file with --max_steps 60. This mirrors the real
    # situation (a run that ended before its planned budget and is then resumed).
    import numpy as np
    n_blocks, seq = 256, 64
    rng = np.random.default_rng(42)
    blocks = rng.integers(0, 6400, size=n_blocks * seq, dtype=np.uint16)
    binp = os.path.join(WORK, 'blocks.bin')
    blocks.tofile(binp)
    binp_a = os.path.join(WORK, 'blocks_a.bin')
    blocks[:80 * seq].tofile(binp_a)

    PHASE_A_MAX = 20
    PHASE_B_MAX = 60
    common = [PY, TRAIN,
              '--max_seq_len', str(seq),
              '--batch_size', '4', '--epochs', '1',
              '--learning_rate', '5e-4', '--log_interval', '5', '--save_interval', '20',
              '--num_workers', '0', '--use_compile', '0',
              '--hidden_size', '64', '--num_hidden_layers', '2', '--seed', '42',
              '--device', 'cpu', '--save_dir', out_dir, '--save_weight', 'rtest',
              '--dtype', 'float32']

    m1 = os.path.join(WORK, 'm1.jsonl')
    print(f'[test] phase A: train {PHASE_A_MAX} steps from scratch (80 blocks)')
    r = run(common + ['--data_path', binp_a, '--max_steps', str(PHASE_A_MAX),
                      '--metrics_path', m1])
    if r.returncode != 0:
        print('FAIL phase A:', r.stdout[-2000:], r.stderr[-2000:])
        return 1
    rows1 = read_metrics(m1)
    last1 = rows1[-1]
    print(f'[test] A last: step={last1["step"]} lr={last1["lr"]} tokens={last1["cum_tokens"]}')

    # the resume file is written to ../checkpoints relative to trainer cwd
    resume_src = os.path.join(ROOT, 'src', 'minimind-master', 'checkpoints', 'rtest_64_resume.pth')
    if not os.path.exists(resume_src):
        print(f'FAIL: resume file not written at {resume_src}')
        return 1
    # The trainer reads '../checkpoints' relative to trainer cwd -> move ours there
    print(f'[test] resume file: {resume_src} ({os.path.getsize(resume_src)/1e6:.1f} MB)')

    print(f'[test] phase B: resume and train up to {PHASE_B_MAX} steps')
    m2 = os.path.join(WORK, 'm2.jsonl')
    # append-mode metrics: start by copying phase A's history so we can check continuity
    shutil.copy2(m1, m2)
    r = run(common + ['--data_path', binp, '--max_steps', str(PHASE_B_MAX),
                      '--metrics_path', m2, '--from_resume', '1'])
    if r.returncode != 0:
        print('FAIL phase B:', r.stdout[-2500:], r.stderr[-2500:])
        return 1
    rows2 = read_metrics(m2)
    resumed = [x for x in rows2 if x['step'] > last1['step']]
    print(f'[test] B rows total={len(rows2)} new={len(resumed)}')
    if not resumed:
        print('FAIL: resume produced no new steps')
        return 1
    first2 = resumed[0]
    print(f'[test] B first new: step={first2["step"]} lr={first2["lr"]} tokens={first2["cum_tokens"]}')

    # ---- assertions -------------------------------------------------------
    # NB: metrics are logged every --log_interval (5) steps, so the first logged step
    # after a resume is last1['step'] + 5, not last1['step'] + 1. The counters must be
    # continuous in that sense.
    LOG_INTERVAL = 5
    total_iters = min(n_blocks // 4, PHASE_B_MAX) * 1
    expected_lr = get_lr(first2['step'], total_iters)
    checks = []

    # 1) LR continuity: resumed step must sit on the cosine curve, not at the peak
    lr_ok = abs(first2['lr'] - expected_lr) < 2e-6
    checks.append(('LR continuity (resumed step on cosine curve)', lr_ok,
                   f'got {first2["lr"]:.8f} expected {expected_lr:.8f} '
                   f'(peak would be {get_lr(1, total_iters):.8f})'))

    # 2) step counter advanced past the checkpoint step by exactly one log interval
    step_ok = first2['step'] == last1['step'] + LOG_INTERVAL
    checks.append(('step counter continues', step_ok,
                   f'last A={last1["step"]} first B={first2["step"]} '
                   f'(expected {last1["step"] + LOG_INTERVAL})'))

    # 3) token counter continues: tokens = step * bs * seq
    expected_tok = first2['step'] * 4 * seq
    tok_ok = first2['cum_tokens'] == expected_tok
    checks.append(('token counter continues', tok_ok,
                   f'first B={first2["cum_tokens"]} expected {expected_tok} '
                   f'(= step {first2["step"]} x 256); last A={last1["cum_tokens"]}'))

    # 4) saved resume file now records the GLOBAL step
    import torch
    d = torch.load(resume_src, map_location='cpu')
    saved_step = d['step']
    step_saved_ok = saved_step == first2['step'] or saved_step > last1['step']
    checks.append(('resume file records global step', step_saved_ok,
                   f'saved step={saved_step} (last A={last1["step"]}, last B logged={resumed[-1]["step"]})'))

    # 5) checkpoint filenames carry the GLOBAL step, monotonically, and never restart
    files = sorted(f for f in os.listdir(out_dir) if '_step' in f)
    nums = []
    for f in files:
        try:
            nums.append(int(f.split('_step')[1].split('.')[0]))
        except Exception:
            pass
    fn_ok = bool(nums) and nums == sorted(nums) and max(nums) >= first2['step'] \
        and len(set(nums)) == len(nums)
    checks.append(('checkpoint filenames use monotonic global step', fn_ok,
                   f'steps in filenames: {nums}'))

    print()
    ok_all = True
    for name, ok, detail in checks:
        print(f'  [{"PASS" if ok else "FAIL"}] {name}: {detail}')
        ok_all = ok_all and ok

    result = {'checks': [{'name': n, 'pass': bool(o), 'detail': d} for n, o, d in checks],
              'all_pass': bool(ok_all),
              'phaseA_last': last1, 'phaseB_first': first2}
    with open(os.path.join(ROOT, 'logs', 'resume_correctness_test.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print()
    print('RESULT:', 'ALL PASS' if ok_all else 'FAILURES PRESENT')
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
