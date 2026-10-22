"""Watchdog: keep the GPU free for the training job.

On this machine a resident Ollama `llama-server` repeatedly auto-starts and grabs
~4.6 GB of the 8.2 GB card. The pretraining job reserves 5.3 GB, so when Ollama
reappears the next CUDA allocation fails and a 28-hour run dies.

This watchdog runs detached, polls every 30 s, and kills any `llama-server` /
`ollama` process. It only touches those two process names; it never kills python.
It stops by itself when the training process it was started for is gone.

Usage (detached):
  powershell -NoProfile -Command "python watchdog_gpu.py --train-pid 1234"
"""
import argparse
import os
import subprocess
import time

WATCH = {'llama-server.exe', 'ollama.exe'}


def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    with open(r'D:\llm-rl-lab\logs\watchdog.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def kill_ollama():
    killed = []
    try:
        out = subprocess.check_output(
            ['tasklist', '/FO', 'CSV', '/NH'], text=True, timeout=30,
            errors='replace')
    except Exception as e:
        log(f'tasklist failed: {e}')
        return killed
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) < 2:
            continue
        name, pid = parts[0], parts[1]
        if name.lower() in WATCH:
            try:
                subprocess.call(['taskkill', '/PID', pid, '/F'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                timeout=20)
                killed.append((name, pid))
            except Exception as e:
                log(f'taskkill {name} {pid} failed: {e}')
    return killed


def free_vram_mib():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            text=True, timeout=30)
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def pid_alive(pid):
    try:
        out = subprocess.check_output(['tasklist', '/FI', f'PID eq {pid}', '/NH'],
                                      text=True, timeout=30, errors='replace')
        return str(pid) in out
    except Exception:
        return True  # assume alive on error; better to keep watching


def any_trainer_alive():
    """True if ANY python process is burning GPU/CPU on training.

    Belt-and-braces: the pid we were given may be a launcher that exits normally while
    the real trainer lives on. Treating that as "training finished" made the watchdog
    quit and let Ollama steal the GPU (observed: trainer died at step 111,800).
    """
    try:
        out = subprocess.check_output(
            ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV', '/NH'],
            text=True, timeout=30, errors='replace')
        for line in out.splitlines():
            parts = [p.strip('"') for p in line.split('","')]
            if len(parts) < 5:
                continue
            try:
                ws = int(parts[4].replace(',', '').replace(' K', '').replace('K', ''))
            except Exception:
                continue
            if ws > 800_000:  # > ~800 MB resident: a real training process
                return True
    except Exception:
        return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-pid', type=int, default=0)
    p.add_argument('--interval', type=int, default=30)
    p.add_argument('--duration-hours', type=float, default=40)
    a = p.parse_args()

    log(f'watchdog start (train_pid={a.train_pid}, interval={a.interval}s, '
        f'max {a.duration_hours}h)')
    t0 = time.time()
    n_killed = 0
    while time.time() - t0 < a.duration_hours * 3600:
        killed = kill_ollama()
        if killed:
            n_killed += len(killed)
            log(f'killed {killed} (total {n_killed}); free VRAM now {free_vram_mib()} MiB')
        if a.train_pid and not pid_alive(a.train_pid) and not any_trainer_alive():
            log(f'training pid {a.train_pid} is gone AND no trainer process remains; '
                f'watchdog exiting (killed {n_killed} ollama processes total)')
            return
        if a.train_pid and not pid_alive(a.train_pid) and any_trainer_alive():
            log(f'watched pid {a.train_pid} exited but a trainer process still runs; '
                f'continuing to guard the GPU')
            a.train_pid = 0  # stop re-logging; fall back to the generic check
        time.sleep(a.interval)
    log(f'watchdog reached max duration; exiting (killed {n_killed})')


if __name__ == '__main__':
    main()
