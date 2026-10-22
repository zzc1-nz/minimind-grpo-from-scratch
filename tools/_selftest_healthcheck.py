"""Self-test for tools/grpo_healthcheck.py: exercise every verdict branch.

Builds synthetic metric files in a temp dir (never touches logs/), runs the
healthcheck as a subprocess, and asserts the verdict. Also asserts the tool
always exits 0, including on a missing file and an empty file.

Run: python tools/_selftest_healthcheck.py
"""
import json
import os
import subprocess
import sys
import tempfile

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
HC = os.path.join(HERE, 'grpo_healthcheck.py')
ROOT = os.path.dirname(HERE)


def mk(path, recs):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')


def base(step, **over):
    rec = dict(step=step, reward_mean=0.1, reward_mean_20=0.1, acc=0.1, adv_std=1.0,
               entropy=3.0, kl_mean=0.01, clip_frac=0.0, degenerate_group_frac=0.1,
               resp_len_mean=32.0, collapsed=False, sampled_tokens=512,
               cum_sampled_tokens=512 * step, sec=1.0)
    rec.update(over)
    return rec


def run(path, *extra):
    p = subprocess.run([PY, HC, path, '--quiet', *extra],
                       capture_output=True, text=True, encoding='utf-8')
    verdict = ''
    for line in p.stdout.splitlines():
        if line.startswith(path):
            verdict = line.split(': ', 1)[-1].split(' - ')[0]
    return p.returncode, verdict, p.stdout.strip()


CASES = {
    'sparse_allwrong': (
        [base(i, reward_mean=0.0, reward_mean_20=0.0, acc=0.0, adv_std=0.0,
              degenerate_group_frac=1.0, collapsed=True) for i in range(1, 31)],
        'SPARSE'),
    'sparse_flat_with_signal': (
        [base(i) for i in range(1, 31)],
        'SPARSE'),
    'degenerate_entropy_collapse': (
        [base(i, entropy=(3.0 if i <= 20 else 0.05)) for i in range(1, 31)],
        'DEGENERATE'),
    'degenerate_kl_runaway': (
        [base(i, kl_mean=0.9, clip_frac=0.5) for i in range(1, 31)],
        'DEGENERATE'),
    'degenerate_reward_falling': (
        [base(i, reward_mean=0.5 - 0.01 * i, reward_mean_20=0.5 - 0.01 * i,
              acc=0.5 - 0.01 * i) for i in range(1, 31)],
        'DEGENERATE'),
    'learning_rising': (
        [base(i, reward_mean=0.05 + 0.005 * i, reward_mean_20=0.05 + 0.005 * i,
              acc=0.05 + 0.005 * i) for i in range(1, 31)],
        'LEARNING'),
    'insufficient_3_steps': (
        [base(i, reward_mean=0.0, adv_std=0.0) for i in range(1, 4)],
        'INSUFFICIENT_DATA'),
}

REAL = {
    os.path.join(ROOT, 'logs', 'grpo_easy_probe.jsonl'): 'LEARNING',
    os.path.join(ROOT, 'logs', 'grpo_rule_smoke.jsonl'): 'INSUFFICIENT_DATA',
}

fails = 0
tmpdir = tempfile.mkdtemp(prefix='hc_selftest_')
print(f'temp dir: {tmpdir}')

for name, (recs, want) in CASES.items():
    path = os.path.join(tmpdir, name + '.jsonl')
    mk(path, recs)
    rc, got, out = run(path)
    ok = (got == want and rc == 0)
    fails += 0 if ok else 1
    print(f'{"ok  " if ok else "FAIL"} {name:32s} want={want:18s} got={got:18s} rc={rc}')

for path, want in REAL.items():
    if not os.path.exists(path):
        print(f'skip {path} (missing)')
        continue
    rc, got, out = run(path)
    ok = (got == want and rc == 0)
    fails += 0 if ok else 1
    print(f'{"ok  " if ok else "FAIL"} real {os.path.basename(path):27s} '
          f'want={want:18s} got={got:18s} rc={rc}')

# Edge cases: must exit 0, never crash.
edge = {
    'missing_file': os.path.join(tmpdir, 'does_not_exist.jsonl'),
    'empty_file': os.path.join(tmpdir, 'empty.jsonl'),
    'garbage_file': os.path.join(tmpdir, 'garbage.jsonl'),
    'partial_last_line': os.path.join(tmpdir, 'partial.jsonl'),
}
mk(edge['empty_file'], [])
with open(edge['garbage_file'], 'w', encoding='utf-8') as f:
    f.write('not json at all\n{"step": 1, "reward_mean": 0.1}\n')
with open(edge['partial_last_line'], 'w', encoding='utf-8') as f:
    for i in range(1, 26):
        f.write(json.dumps(base(i, reward_mean=0.05 + 0.005 * i,
                                reward_mean_20=0.05 + 0.005 * i,
                                acc=0.05 + 0.005 * i)) + '\n')
    f.write('{"step": 26, "reward_mean": 0.')   # partially flushed line

for name, path in edge.items():
    rc, got, out = run(path)
    ok = (rc == 0)
    fails += 0 if ok else 1
    print(f'{"ok  " if ok else "FAIL"} edge {name:27s} rc={rc} verdict={got or "(none)"}')

print()
print('HEALTHCHECK SELF-TEST', 'PASS' if fails == 0 else f'{fails} FAILURES')
sys.exit(0 if fails == 0 else 1)
