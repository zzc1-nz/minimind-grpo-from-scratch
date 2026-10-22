"""Pre-flight feasibility check for a GRPO training set.

GRPO learns only from *within-group reward variance*: if all G samples of a prompt get
the same reward, the group-normalised advantage is exactly 0 and the step contributes no
gradient. Two things therefore have to hold before spending GPU hours:

  1. the answer space must be reachable -- the SFT seed set should have taught the output
     format and the answer vocabulary the GRPO tasks need;
  2. the answer prior must not be degenerate -- if one answer dominates, a random group
     of G samples is mostly identical and the variance collapses.

This script quantifies both from the data files alone (no model, no GPU).

Usage:
  python grpo_feasibility.py                     # default stage1 + stage3
  python grpo_feasibility.py --stage stage1_easy --group 16
"""
import argparse
import collections
import json
import math
import os
import sys

ROOT = r'D:\llm-rl-lab'
SEED = os.path.join(ROOT, 'data', 'sft', 'sft_seed.jsonl')
GRPO_DIR = os.path.join(ROOT, 'data', 'grpo')


def load(p):
    with open(p, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def binom_var_group(p, g):
    """Variance of the number of correct samples in a group of g, if each sample is
    correct independently with probability p. Collapses to 0 as p -> 0 or 1."""
    return g * p * (1 - p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='stage1_easy')
    ap.add_argument('--group', type=int, default=16)
    ap.add_argument('--seed-file', default=SEED)
    a = ap.parse_args()

    stage_path = os.path.join(GRPO_DIR, f'{a.stage}.jsonl')
    if not os.path.exists(stage_path):
        print(f'missing {stage_path}')
        return 2
    tasks = load(stage_path)
    seed = load(a.seed_file) if os.path.exists(a.seed_file) else []

    print('=' * 68)
    print(f'GRPO feasibility: {a.stage}  ({len(tasks)} tasks, group size {a.group})')
    print('=' * 68)

    # ---- 1. FORMAT reachability (the thing that actually gates GRPO) -------
    # Distinction that matters: SFT teaches the *format* (emit an integer / a JSON
    # object / a letter / a label); GRPO learns *which* value is right. An answer value
    # the seed set never contained (e.g. -5 or 21) is fine as long as the format is
    # reachable -- the policy can produce a digit and the reward will steer it. What
    # would truly block GRPO is a format the model was never taught to emit.
    print('\n[1] FORMAT reachability (can the model emit this shape at all?)')
    seed_checkers = {t.get('checker') for t in seed}
    stage_checkers = {t.get('checker') for t in tasks}
    stage_answers = collections.Counter(str(t.get('answer')).strip() for t in tasks)
    seed_answers = {str(t.get('answer')).strip() for t in seed}

    unseen_checkers = stage_checkers - seed_checkers
    if unseen_checkers:
        print(f'  checkers NOT exercised by SFT: {sorted(unseen_checkers)}  <-- BLOCKING')
        print('    the model has never produced this format, so every group starts')
        print('    all-wrong (reward 0, adv_std 0) until it stumbles on it by chance.')
    else:
        print(f'  checkers: all {len(stage_checkers)} stage checkers appear in the SFT '
              f'seed set  -> every required format was taught')

    # Answer-value novelty is reported but is NOT a failure: it is the intended gap
    # between SFT (easy, narrow) and GRPO (harder values, same format).
    novel = set(stage_answers) - seed_answers
    print(f'\n  answer VALUES not present in the SFT seed set: {len(novel)} '
          f'of {len(stage_answers)} distinct')
    if novel:
        by_checker = collections.Counter(
            t.get('checker') for t in tasks
            if str(t.get('answer')).strip() in novel)
        print(f'    by checker: {dict(by_checker)}')
        print('    -> NOT blocking: the format is reachable, only the value is new.')
        print('       This is the intended SFT(easy) -> GRPO(harder) gap.')

    # ---- 2. answer prior / group-variance collapse -------------------------
    # IMPORTANT: GRPO's group is G samples of the SAME prompt, so the variance that
    # matters is "how spread out is the model's output for this one question", not the
    # global frequency of an answer across the dataset. A dataset with 91 distinct
    # answers is *more* diverse per task, not less.
    #
    # What we can measure from data alone is the *answer space size* per task: a task
    # whose answer is 1-of-2 gives a group at most 2 distinct outcomes; a task whose
    # answer is an integer in a 31-wide space gives more room. This bounds the
    # achievable group diversity.
    print('\n[2] per-task answer space (bounds the group diversity GRPO can see)')
    total = len(tasks)
    top_answer, top_n = stage_answers.most_common(1)[0]
    top_share = top_n / total
    print(f'  global most-frequent answer: {top_answer!r} '
          f'{top_n}/{total} = {100*top_share:.1f}%  '
          f'(this is NOT the per-group variance)')

    def space_size(t):
        c = t.get('checker')
        if c == 'choice':
            return len(t.get('choices') or [1, 2, 3])
        if c == 'label':
            return len(t.get('labels') or [1, 2])
        if c == 'regex':
            return 30
        return 31  # exact_int / exact_json: a bounded integer range

    sizes = [space_size(t) for t in tasks]
    small = [s for s in sizes if s <= 3]
    print(f'  answer-space size per task : min={min(sizes)} '
          f'median={sorted(sizes)[len(sizes)//2]} max={max(sizes)}')
    print(f'  tasks with a <=3-wide answer space: {len(small)}/{total} '
          f'({100*len(small)/total:.0f}%)')
    print('    -> for those, a group of G can show at most 3 distinct outcomes;')
    print('       diversity comes from the model\'s uncertainty, not the label count.')
    # A pragmatic flag: if almost every task is 2-3 way, group diversity is structurally
    # capped and the effective signal is weaker. Not a blocker, a caution.
    structural_cap = len(small) / total
    if structural_cap > 0.8:
        print(f'  CAUTION: {100*structural_cap:.0f}% of tasks are 2-3-way choices; '
              f'consider a larger group size.')
    var = binom_var_group(top_share, a.group)
    print(f'  (global-prior variance for reference: {var:.2f}; not used as a gate)')

    # ---- 3. per-checker difficulty ----------------------------------------
    print('\n[3] per-checker answer distribution')
    byc = collections.defaultdict(collections.Counter)
    for t in tasks:
        byc[t.get('checker')][str(t.get('answer')).strip()] += 1
    for c, cnt in sorted(byc.items()):
        n = sum(cnt.values())
        top = cnt.most_common(1)[0]
        print(f'  {c:<12} n={n:<4} distinct={len(cnt):<3} '
              f'most common {top[0]!r} {100*top[1]/n:.0f}%')

    # ---- 4. verdict --------------------------------------------------------
    print('\n' + '-' * 68)
    if unseen_checkers:
        verdict = ('UNREACHABLE FORMAT: checkers ' + str(sorted(unseen_checkers)) +
                   ' were never taught by SFT; those tasks cannot bootstrap')
    else:
        verdict = ('FEASIBLE: every required format was taught by SFT; '
                   f'{100*structural_cap:.0f}% of tasks are 2-3-way choices '
                   f'(group size {a.group} recommended)')
    print('VERDICT:', verdict)
    out = {'stage': a.stage, 'group': a.group, 'tasks': total,
           'distinct_answers': len(stage_answers),
           'checkers': sorted(stage_checkers),
           'unseen_checkers': sorted(unseen_checkers),
           'novel_answer_values': sorted(novel),
           'top_answer': top_answer, 'top_answer_share': round(top_share, 4),
           'small_answer_space_frac': round(structural_cap, 3),
           'verdict': verdict}
    with open(os.path.join(ROOT, 'logs', f'grpo_feasibility_{a.stage}.json'), 'w',
              encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return 0 if verdict.startswith('FEASIBLE') else 1


if __name__ == '__main__':
    sys.exit(main())
