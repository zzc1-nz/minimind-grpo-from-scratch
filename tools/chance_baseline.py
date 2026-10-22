"""Compute the chance baseline of the held-out benchmark.

Why this matters: 24 tasks with mixed answer spaces means a *coin-flip* model already
scores ~15 % overall (24 % on `format`, 27 % on `logic`, because those are 2-3-way
choices). Reporting "SFT reached 4/24 = 16.7 %" without that context invites reading
noise as capability. This module makes the baseline a number in every report.

Usage:
  python chance_baseline.py                       # human-readable
  python chance_baseline.py --json                # machine-readable
  python chance_baseline.py --tasks other.jsonl
"""
import argparse
import json
import os
import sys

DEFAULT_TASKS = r'D:\llm-rl-lab\eval\heldout_tasks.jsonl'

# Answer-space sizes for open-ended checkers. These are deliberate, documented choices,
# not measurements: the model must produce a value the checker accepts, and the plausible
# value range is bounded by the task family.
#   exact_int  : integers the task can plausibly produce. Single-digit +/- gives -9..18
#                (28 values); "max of three in 0..30" gives 31; we use the generous 31
#                so the baseline is not understated.
#   exact_json : the numeric field is the same range as exact_int.
#   regex      : a fixed-format expression; treat as a 30-way space (conservative).
INT_SPACE = 31
JSON_SPACE = 31
REGEX_SPACE = 30


def space_size(task):
    c = task['checker']
    if c == 'choice':
        choices = task.get('choices')
        if not choices:
            return 3
        return len(choices)
    if c == 'label':
        labels = task.get('labels')
        return len(labels) if labels else 2
    if c == 'exact_json':
        return JSON_SPACE
    if c == 'regex':
        return REGEX_SPACE
    if c == 'exact_int':
        return INT_SPACE
    return 10


def analyse(tasks):
    fams = {}
    for t in tasks:
        fams.setdefault(t.get('family', 'unknown'), []).append(t)
    out = {'overall': {}, 'per_family': {}, 'per_task': []}
    total_exp = 0.0
    for t in tasks:
        p = 1.0 / space_size(t)
        total_exp += p
        out['per_task'].append({
            'id': t['id'], 'family': t.get('family'), 'checker': t['checker'],
            'answer_space': space_size(t), 'p_correct_by_chance': round(p, 5),
        })
    n = len(tasks)
    out['overall'] = {
        'tasks': n,
        'expected_correct_by_chance': round(total_exp, 3),
        'chance_accuracy': round(total_exp / n, 4) if n else 0.0,
        'chance_accuracy_pct': round(100.0 * total_exp / n, 1) if n else 0.0,
    }
    for fam, sub in sorted(fams.items()):
        e = sum(1.0 / space_size(t) for t in sub)
        out['per_family'][fam] = {
            'tasks': len(sub),
            'expected_correct_by_chance': round(e, 3),
            'chance_accuracy': round(e / len(sub), 4),
            'chance_accuracy_pct': round(100.0 * e / len(sub), 1),
        }
    # How many tasks must be correct before the result is distinguishable from chance?
    # Binomial(n, p) 95th percentile, computed exactly with integer arithmetic.
    from math import comb
    p = total_exp / n if n else 0.0
    thresh = None
    for k in range(0, n + 1):
        # P(X >= k) under Binomial(n, p)
        tail = sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))
        if tail <= 0.05:
            thresh = k
            break
    out['overall']['min_correct_for_significance_5pct'] = thresh
    out['overall']['interpretation'] = (
        f'A model scoring {thresh} or more of {n} is better than chance at the 5 % level; '
        f'scores below that are indistinguishable from guessing.' if thresh else
        'chance level is too high to reach significance with this many tasks'
    )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tasks', default=DEFAULT_TASKS)
    p.add_argument('--json', action='store_true')
    a = p.parse_args()
    with open(a.tasks, encoding='utf-8') as f:
        tasks = [json.loads(line) for line in f if line.strip()]
    res = analyse(tasks)
    res['tasks_file'] = os.path.abspath(a.tasks)
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    o = res['overall']
    print(f'held-out benchmark: {o["tasks"]} tasks  ({a.tasks})')
    print()
    print(f'{"family":<12}{"tasks":>7}{"chance acc":>13}')
    print('-' * 32)
    for fam, v in res['per_family'].items():
        print(f'{fam:<12}{v["tasks"]:>7}{v["chance_accuracy_pct"]:>12.1f}%')
    print('-' * 32)
    print(f'{"OVERALL":<12}{o["tasks"]:>7}{o["chance_accuracy_pct"]:>12.1f}%')
    print()
    print(f'expected correct by pure guessing: {o["expected_correct_by_chance"]} of {o["tasks"]}')
    print(f'minimum correct to beat chance at 5%: {o["min_correct_for_significance_5pct"]}')
    print()
    print('NOTE: format/logic are 2-3 way choices, so their chance level is high.')
    print('      A result like 4/24 (16.7%) is NOT evidence of learning on this benchmark.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
