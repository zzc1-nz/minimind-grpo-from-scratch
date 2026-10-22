"""Phase-1 -> Phase-5 unattended chain.

Waits for the pretraining run to finish (by polling `pretrain_metrics.jsonl` for the
final step and the summary file), then runs the remaining phases in order by delegating
to `run_pipeline.py`. Intended to be started detached so the whole project completes
without an operator.

Why a separate waiter: the pretraining process is itself detached, so this script cannot
just be a child of it. It polls a file, which also survives a session restart.

Usage (detached):
  python wait_then_pipeline.py --final-step 134817
"""
import argparse
import json
import os
import subprocess
import sys
import time

PY = r'D:\llm-rl-lab\.venv\Scripts\python.exe'
LOGS = r'D:\llm-rl-lab\logs'
METRICS = rf'{LOGS}\pretrain_metrics.jsonl'
SUMMARY = rf'{LOGS}\pretrain_metrics_summary.json'


def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    with open(rf'{LOGS}\chain.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def last_step():
    step = 0
    try:
        with open(METRICS, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    step = json.loads(line).get('step', step)
                except Exception:
                    pass
    except FileNotFoundError:
        return 0
    return step


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--final-step', type=int, default=134817)
    p.add_argument('--poll-sec', type=int, default=300)
    p.add_argument('--max-hours', type=float, default=40)
    p.add_argument('--start-phase', type=int, default=2)
    p.add_argument('--end-phase', type=int, default=5)
    a = p.parse_args()

    log('=' * 70)
    log(f'chain started: waiting for pretrain to reach step {a.final_step} '
        f'(currently {last_step()}), then phases {a.start_phase}..{a.end_phase}')
    t0 = time.time()
    while time.time() - t0 < a.max_hours * 3600:
        s = last_step()
        done = os.path.exists(SUMMARY)
        if s >= a.final_step or done:
            log(f'pretrain finished (step {s}, summary={done}); starting pipeline')
            break
        log(f'waiting... step {s}/{a.final_step} ({100.0*s/a.final_step:.1f}%)')
        time.sleep(a.poll_sec)
    else:
        log(f'chain timed out after {a.max_hours}h at step {last_step()}; NOT starting pipeline')
        return 1

    cmd = [PY, r'D:\llm-rl-lab\tools\run_pipeline.py',
           '--from', str(a.start_phase), '--to', str(a.end_phase)]
    log('launching: ' + ' '.join(cmd))
    rc = subprocess.call(cmd)
    log(f'pipeline exited with {rc}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
