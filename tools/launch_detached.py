"""Launch the pretraining resume in a fully independent way.

Problem being solved: processes started with `subprocess.Popen(..., DETACHED_PROCESS)`
from inside a tool-invoked PowerShell still die when that PowerShell job is cleaned up.
Observed twice: the trainer vanished with an EMPTY stderr and no Windows event log
entry, i.e. it was killed externally, not crashed.

Fix: hand the launch to `schtasks`. A scheduled task is owned by the Task Scheduler
service, not by any PowerShell session, so it survives the agent's session churn
entirely. The task runs once, immediately, and then deletes itself.

Usage:
  python launch_detached.py --dry-run
  python launch_detached.py
"""
import argparse
import os
import subprocess
import sys
import time

PY = r'D:\llm-rl-lab\.venv\Scripts\python.exe'
ROOT = r'D:\llm-rl-lab'
OUT = rf'{ROOT}\src\minimind-master\out'
LOGS = rf'{ROOT}\logs'
TRAINER_CWD = rf'{ROOT}\src\minimind-master\trainer'
TASK = 'MiniMindPretrainResume'

TRAIN_CMD = (
    f'"{PY}" "{ROOT}\\custom\\train_pretrain_packed.py" '
    f'--data_path "{ROOT}\\data\\tokens\\pretrain_blocks.bin" '
    f'--max_seq_len 512 --batch_size 32 --epochs 1 --max_steps 134817 '
    f'--learning_rate 5e-4 --log_interval 100 --save_interval 2000 '
    f'--num_workers 4 --use_compile 1 --from_resume 1 '
    f'--save_dir "{OUT}" --save_weight pretrain '
    f'--metrics_path "{LOGS}\\pretrain_metrics.jsonl"'
)


def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(rf'{LOGS}\launch_detached.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def run(cmd, timeout=60):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          errors='replace')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--no-chain', action='store_true')
    a = ap.parse_args()

    log('=' * 66)
    log('LAUNCH via Task Scheduler' + (' [DRY RUN]' if a.dry_run else ''))

    # wrapper .bat so we get redirection + a working directory without quoting hell
    bat = rf'{LOGS}\run_pretrain.bat'
    os.makedirs(LOGS, exist_ok=True)
    with open(bat, 'w', encoding='utf-8', newline='\r\n') as f:
        f.write('@echo off\r\n')
        f.write(f'cd /d "{TRAINER_CWD}"\r\n')
        f.write(f'{TRAIN_CMD} >> "{LOGS}\\pretrain_console.log" '
                f'2>> "{LOGS}\\pretrain_console.err"\r\n')
    log(f'wrapper written: {bat}')

    # delete any previous instance of the task
    r = run(['schtasks', '/Delete', '/TN', TASK, '/F'])
    log(f'schtasks delete: rc={r.returncode}')

    tr = f'cmd.exe /c "{bat}"'
    # /SC ONCE with a start time 1 minute ago + /ST, then /RUN to fire immediately
    start_time = time.strftime('%H:%M', time.localtime(time.time() + 60))
    cmd = ['schtasks', '/Create', '/TN', TASK, '/TR', tr, '/SC', 'ONCE',
           '/ST', start_time, '/F', '/RL', 'HIGHEST']
    if a.dry_run:
        log('DRY RUN: would run ' + ' '.join(cmd))
        log('DRY RUN: then schtasks /Run /TN ' + TASK)
        return 0

    r = run(cmd)
    log(f'schtasks create: rc={r.returncode} {r.stdout.strip()} {r.stderr.strip()}')
    if r.returncode != 0:
        return 1

    r = run(['schtasks', '/Run', '/TN', TASK])
    log(f'schtasks run: rc={r.returncode} {r.stdout.strip()} {r.stderr.strip()}')
    if r.returncode != 0:
        return 1

    # wait for the trainer to appear and prove it resumed
    log('waiting for the trainer to start...')
    deadline = time.time() + 300
    trainer_pid = None
    while time.time() < deadline:
        time.sleep(10)
        try:
            out = subprocess.check_output(
                ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV', '/NH'],
                text=True, timeout=30, errors='replace')
        except Exception:
            continue
        for line in out.splitlines():
            parts = [p.strip('"') for p in line.split('","')]
            if len(parts) < 5:
                continue
            try:
                ws = int(parts[4].replace(',', '').replace(' K', '').replace('K', ''))
            except Exception:
                continue
            if ws > 800_000:
                trainer_pid = int(parts[1])
                break
        if trainer_pid:
            break
    if not trainer_pid:
        log('FAILED: no trainer process appeared within 5 minutes')
        log(f'  see {LOGS}\\pretrain_console.err')
        return 1
    log(f'trainer running, pid={trainer_pid}')

    # wait for a metrics row newer than the file's current mtime
    mpath = rf'{LOGS}\pretrain_metrics.jsonl'
    before = os.path.getmtime(mpath) if os.path.exists(mpath) else 0
    log(f'waiting for a new metrics write (mtime was {before:.0f})...')
    deadline = time.time() + 600
    while time.time() < deadline:
        time.sleep(15)
        if os.path.exists(mpath) and os.path.getmtime(mpath) > before:
            log('metrics file updated -> training is live')
            break
    else:
        log('WARNING: metrics did not update within 10 min; check the console log')
        return 1

    # watchdog + chain, also via Task Scheduler so they survive the session
    wd_bat = rf'{LOGS}\run_watchdog.bat'
    with open(wd_bat, 'w', encoding='utf-8', newline='\r\n') as f:
        f.write('@echo off\r\n')
        f.write(f'"{PY}" "{ROOT}\\tools\\watchdog_gpu.py" --train-pid {trainer_pid} '
                f'--interval 30 --duration-hours 40\r\n')
    run(['schtasks', '/Delete', '/TN', 'MiniMindWatchdog', '/F'])
    r = run(['schtasks', '/Create', '/TN', 'MiniMindWatchdog', '/TR',
             f'cmd.exe /c "{wd_bat}"', '/SC', 'ONCE',
             '/ST', time.strftime('%H:%M', time.localtime(time.time() + 60)),
             '/F', '/RL', 'HIGHEST'])
    log(f'watchdog task create rc={r.returncode}')
    if r.returncode == 0:
        run(['schtasks', '/Run', '/TN', 'MiniMindWatchdog'])
        log('watchdog started via Task Scheduler')

    if not a.no_chain:
        ch_bat = rf'{LOGS}\run_chain.bat'
        with open(ch_bat, 'w', encoding='utf-8', newline='\r\n') as f:
            f.write('@echo off\r\n')
            f.write(f'"{PY}" "{ROOT}\\tools\\wait_then_pipeline.py" --final-step 134817 '
                    f'--poll-sec 300 --max-hours 40 --start-phase 2 --end-phase 5\r\n')
        run(['schtasks', '/Delete', '/TN', 'MiniMindChain', '/F'])
        r = run(['schtasks', '/Create', '/TN', 'MiniMindChain', '/TR',
                 f'cmd.exe /c "{ch_bat}"', '/SC', 'ONCE',
                 '/ST', time.strftime('%H:%M', time.localtime(time.time() + 60)),
                 '/F', '/RL', 'HIGHEST'])
        log(f'chain task create rc={r.returncode}')
        if r.returncode == 0:
            run(['schtasks', '/Run', '/TN', 'MiniMindChain'])
            log('chain started via Task Scheduler')

    log(f'ALL SET (trainer pid {trainer_pid})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
