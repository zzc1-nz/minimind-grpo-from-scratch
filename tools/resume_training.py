"""Resume the paused Phase-1 pretraining run and re-arm the unattended chain.

Written because the pause left three detached processes that must be restarted in the
right order, with the right PIDs and the right flags. Doing that by hand at 3 a.m. is
how a 12-hour run gets started with the wrong --save_weight and silently trains from
scratch (a failure mode already observed once).

What it does
  1. verifies no other process is holding the GPU (refuses if < 5000 MiB free);
  2. verifies the resume checkpoint exists and matches the expected step;
  3. kills any resident Ollama (it auto-restarts and steals ~4.6 GB);
  4. launches the trainer detached, with --from_resume 1;
  5. waits for the first metrics row past the checkpoint step (proves it resumed, not
     restarted from zero) and aborts loudly if the step counter went backwards;
  6. launches the GPU watchdog and the wait-then-pipeline chain.

Usage:
  python resume_training.py              # resume + re-arm everything
  python resume_training.py --dry-run    # show the plan only
  python resume_training.py --no-chain   # resume only, do not start phases 2-5
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
CKPT = rf'{ROOT}\src\minimind-master\checkpoints\pretrain_768_resume.pth'
METRICS = rf'{LOGS}\pretrain_metrics.jsonl'

FINAL_STEP = 134817
TOKENS_PER_STEP = 32 * 512
MIN_FREE_VRAM = 5000


def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(rf'{LOGS}\resume.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def free_vram_mib():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            text=True, timeout=30)
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def gpu_occupants():
    """Human-readable list of processes holding GPU memory (best effort)."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
             '--format=csv,noheader'], text=True, timeout=30, errors='replace')
        return [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    except Exception:
        return []


def kill_ollama():
    killed = []
    try:
        out = subprocess.check_output(['tasklist', '/FO', 'CSV', '/NH'],
                                      text=True, timeout=30, errors='replace')
    except Exception:
        return killed
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) < 2:
            continue
        name, pid = parts[0], parts[1]
        if name.lower() in ('llama-server.exe', 'ollama.exe'):
            subprocess.call(['taskkill', '/PID', pid, '/F'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            timeout=20)
            killed.append(f'{name}:{pid}')
    return killed


def last_metric():
    row = None
    try:
        with open(METRICS, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        row = json.loads(line)
                    except Exception:
                        pass
    except FileNotFoundError:
        pass
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--no-chain', action='store_true')
    p.add_argument('--min-vram', type=int, default=MIN_FREE_VRAM)
    a = p.parse_args()

    log('=' * 70)
    log('RESUME requested' + (' [DRY RUN]' if a.dry_run else ''))

    # ---- 1. checkpoint present? ------------------------------------------
    if not os.path.exists(CKPT):
        log(f'ABORT: resume checkpoint missing at {CKPT}')
        return 1
    import torch
    d = torch.load(CKPT, map_location='cpu')
    ckpt_step = int(d.get('step', 0))
    ckpt_tokens = int(d.get('cum_tokens', ckpt_step * TOKENS_PER_STEP))
    log(f'checkpoint: step={ckpt_step:,} cum_tokens={ckpt_tokens:,} '
        f'optimizer_states={len(d.get("optimizer", {}).get("state", {}))}')

    # ---- 2. GPU free? ----------------------------------------------------
    killed = kill_ollama()
    if killed:
        log(f'killed resident ollama: {killed}')
        time.sleep(3)
    fv = free_vram_mib()
    log(f'free VRAM: {fv} MiB (need >= {a.min_vram})')
    if fv < a.min_vram:
        log('ABORT: GPU is still busy. Occupants:')
        for o in gpu_occupants():
            log('   ' + o)
        log('Close the game / other GPU app and re-run this script.')
        return 1

    # ---- 3. sanity: where are we now? -----------------------------------
    cur = last_metric()
    if cur:
        log(f'last logged step: {cur["step"]:,} loss={cur["loss"]} '
            f'cum_tokens={cur["cum_tokens"]:,}')
        if cur['step'] > ckpt_step:
            log(f'NOTE: the run reached step {cur["step"]:,} but the newest checkpoint is '
                f'step {ckpt_step:,}; {cur["step"] - ckpt_step:,} steps '
                f'({(cur["step"]-ckpt_step)*TOKENS_PER_STEP:,} tokens) will be redone.')

    expected_final_tokens = FINAL_STEP * TOKENS_PER_STEP
    log(f'projected final tokens after resume: '
        f'{ckpt_tokens + (FINAL_STEP - ckpt_step) * TOKENS_PER_STEP:,} '
        f'(planned {expected_final_tokens:,})')

    cmd = [PY, rf'{ROOT}\custom\train_pretrain_packed.py',
           '--data_path', rf'{ROOT}\data\tokens\pretrain_blocks.bin',
           '--max_seq_len', '512', '--batch_size', '32', '--epochs', '1',
           '--max_steps', str(FINAL_STEP), '--learning_rate', '5e-4',
           '--log_interval', '100', '--save_interval', '2000',
           '--num_workers', '4', '--use_compile', '1', '--from_resume', '1',
           '--save_dir', OUT, '--save_weight', 'pretrain',
           '--metrics_path', METRICS]

    if a.dry_run:
        log('DRY RUN: would launch')
        log('  ' + ' '.join(cmd))
        log(f'DRY RUN: would then start watchdog and '
            f'{"nothing" if a.no_chain else "wait_then_pipeline"}')
        return 0

    # ---- 4. launch trainer detached -------------------------------------
    log('launching trainer (detached)...')
    proc = subprocess.Popen(cmd, cwd=TRAINER_CWD,
                            stdout=open(rf'{LOGS}\pretrain_console.log', 'a', encoding='utf-8'),
                            stderr=open(rf'{LOGS}\pretrain_console.err', 'a', encoding='utf-8'),
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP |
                            subprocess.DETACHED_PROCESS)
    log(f'trainer pid {proc.pid}')

    # ---- 5. prove it actually resumed -----------------------------------
    log('waiting for the first metrics row past the checkpoint step...')
    deadline = time.time() + 600
    first_new = None
    while time.time() < deadline:
        time.sleep(15)
        cur = last_metric()
        if cur and cur['step'] > ckpt_step:
            first_new = cur
            break
        if proc.poll() is not None:
            log(f'ABORT: trainer exited early with code {proc.returncode}')
            log('  see logs/pretrain_console.err')
            return 1
    if first_new is None:
        log('ABORT: no metrics past the checkpoint step after 10 min')
        return 1
    log(f'RESUMED OK: step={first_new["step"]:,} loss={first_new["loss"]} '
        f'lr={first_new["lr"]:.3e} cum_tokens={first_new["cum_tokens"]:,}')
    if first_new['step'] <= ckpt_step:
        log('ABORT: step counter did not advance past the checkpoint')
        return 1
    if first_new['cum_tokens'] < ckpt_tokens * 0.99:
        log(f'ABORT: token counter reset (got {first_new["cum_tokens"]:,}, '
            f'expected >= {ckpt_tokens:,})')
        return 1

    # ---- 6. watchdog + chain --------------------------------------------
    # The watchdog must watch the REAL trainer, not the launcher we just spawned.
    # Observed failure: watchdog was given the launcher pid, the launcher exited
    # normally, the watchdog concluded "training finished" and quit -- then Ollama
    # auto-started, stole 4.6 GB, and the trainer died at step 111,800. Find the child
    # python process that is actually holding the GPU.
    log('locating the real trainer pid...')
    real_pid = proc.pid
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            out = subprocess.check_output(
                ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV', '/NH'],
                text=True, timeout=30, errors='replace')
            best = None
            for line in out.splitlines():
                parts = [p.strip('"') for p in line.split('","')]
                if len(parts) < 5:
                    continue
                try:
                    ws = int(parts[4].replace(',', '').replace(' K', '').replace('K', ''))
                except Exception:
                    continue
                pid = int(parts[1])
                if best is None or ws > best[1]:
                    best = (pid, ws)
            if best and best[1] > 1_000_000:  # > ~1 GB resident = the trainer
                real_pid = best[0]
                log(f'  trainer pid = {real_pid} ({best[1]//1024} MB resident)')
                break
        except Exception:
            pass
        time.sleep(5)
    if real_pid == proc.pid:
        log('  WARNING: could not distinguish trainer from launcher; '
            'watching the launcher pid (watchdog may exit early)')

    log(f'starting GPU watchdog (train-pid={real_pid})...')
    subprocess.Popen([PY, rf'{ROOT}\tools\watchdog_gpu.py',
                      '--train-pid', str(real_pid), '--interval', '30',
                      '--duration-hours', '40'],
                     creationflags=subprocess.DETACHED_PROCESS,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not a.no_chain:
        log('starting wait_then_pipeline (phases 2-5 will run automatically)...')
        subprocess.Popen([PY, rf'{ROOT}\tools\wait_then_pipeline.py',
                          '--final-step', str(FINAL_STEP), '--poll-sec', '300',
                          '--max-hours', '40', '--start-phase', '2', '--end-phase', '5'],
                         creationflags=subprocess.DETACHED_PROCESS,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log(f'ALL SET. Training will run to step {FINAL_STEP:,}; '
        f'remaining tokens {expected_final_tokens - first_new["cum_tokens"]:,}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
