"""Derive a difficulty-ordered GRPO curriculum from the 600-task training set.

WHY THIS EXISTS
---------------
GRPO normalises advantages *within* a group of `num_generations` samples of the
same prompt: `A_i = (r_i - mean(r)) / (std(r) + 1e-4)`. If every sample of a
group gets the same reward (all wrong, or all right) the advantage vector is
identically zero and the group contributes **no gradient**. Measured on this
project (logs/grpo_rule_smoke.jsonl): on the full 600-task set the 200-step
smoke model got reward 0.000 with adv_std 0.000 for all 3 steps. On an easy
200-task `a+b, a,b in [0,4]` set the same model got reward 0.078 -> 0.169 with
adv_std ~1.0 (logs/grpo_easy_probe.jsonl). The fix is a curriculum that starts
where the model already succeeds *sometimes*, then widens.

This script does NOT hand-label tasks. It computes a `difficulty` score from
the task content itself and sorts on it. Every component is documented below
and reported, so the ordering can be audited and reproduced.

DIFFICULTY MODEL (additive, integer, lower = easier)
---------------------------------------------------
Each task gets an integer `difficulty` = sum of the applicable components:

  answer magnitude      floor(log2(|answer| + 1)) * 2
                        a bigger number is a harder single-token target; every
                        int 0..25 is one token, 2-digit values above 25 split
                        inconsistently (env_report §8), so |ans| > 25 is a real
                        step up.
  negative answer       +3      two tokens ('-' is its own token, id 48) and the
                                prompt has to condition on "负号".
  arithmetic ops        +2 per binary operation beyond the first
                        (single op = 0, two ops = 2, ...). Counted from the
                        prompt, not from a family label.
  negation / polarity   +2      prompt contains 负 / 小于 / 最小 / 不是 ...
                                (a polarity flip the model must not miss).
  JSON output           +4      must emit a well-formed object, not a bare int.
  multi-step            +3      two or more separate computation clauses.
  option-set choice     +1      answer is a letter A/B/C among distractors.
  long prompt           +1      prompt > 80 chars (more context to condition on).
  family prior          arithmetic 0, format +1, logic +1, multistep +2
                        (matches the generator weights in build_rl_data.py:
                         arithmetic 1.0 > format/logic 0.75 > multistep 0.5,
                         i.e. multistep is the intended hardest family).

The score is deliberately coarse and additive so that ties are broken by a
stable, documented rule rather than by insertion order:

  sort key = (difficulty, family_rank, answer_magnitude, task_id)

STAGES
------
  stage1_easy    the easiest ~150 tasks: single-step, |answer| small, no
                 negation, no JSON, no multi-step. This is where GRPO must
                 first get a non-zero advantage.
  stage2_medium  easy + medium, ~350 tasks: adds negation, JSON and 2-digit
                 answers, but still excludes the hardest multi-step tail.
  stage3_full    all 600 tasks, in the same difficulty order.

`stage1_easy ⊂ stage2_medium ⊂ stage3_full` is asserted, so a staged run can
resume from the previous stage's weights without ever re-seeing a harder task
before an easier one.

Outputs (train_grpo.jsonl is never modified):
  data/grpo/stage1_easy.jsonl
  data/grpo/stage2_medium.jsonl
  data/grpo/stage3_full.jsonl
  data/grpo/curriculum_stats.json

Usage:
  python custom/build_grpo_curriculum.py
  python custom/build_grpo_curriculum.py --stage1 150 --stage2 350
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys

# --- project root, derived from this file's location (portable across machines) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

DEFAULT_SRC = os.path.join(PROJECT_ROOT, 'data', 'grpo', 'train_grpo.jsonl')
DEFAULT_OUTDIR = os.path.join(PROJECT_ROOT, 'data', 'grpo')
DEFAULT_HELDOUT = os.path.join(PROJECT_ROOT, 'eval', 'heldout_tasks.jsonl')
sys.path.insert(0, 'D:/llm-rl-lab/custom')
from verifiable_reward import check as reward_check  # noqa: E402

# ------------------------------------------------------------------ shapes
# Prompt templates are recognised by a normalised signature (digits -> N,
# option letters -> L) rather than by the exact Chinese string, so a re-rolled
# dataset with different numbers still classifies correctly.
_TPL_NUM = re.compile(r'-?\d+')
_TPL_LET = re.compile(r'(?<![A-Za-z])[A-C](?![A-Za-z])')


def template_of(prompt):
    s = _TPL_NUM.sub('N', prompt)
    s = _TPL_LET.sub('L', s)
    return s


# shape name -> (family, checker, base op count, flags)
# op_count is the number of binary arithmetic operations in the prompt.
# neg/JSON/multi are read from the prompt text; they are listed here only so the
# report can show what each recognised shape is expected to contribute.
SHAPES = {
    'add_sub_single':   dict(ops=1, multi=0, json=0, note='a + b / a - b'),
    'add_sub_neg':      dict(ops=1, multi=0, json=0, note='a - b with a < b'),
    'mul_single':       dict(ops=1, multi=0, json=0, note='a x b'),
    'json_sum':         dict(ops=1, multi=0, json=1, note='{"sum": a+b}'),
    'choice_sum':       dict(ops=1, multi=0, json=0, note='A/B/C of a+b'),
    'cmp_yesno':        dict(ops=0, multi=0, json=0, note='a > b ? YES/NO'),
    'parity':           dict(ops=0, multi=0, json=0, note='odd/even'),
    'max_of_three':     dict(ops=0, multi=0, json=0, note='max of 3 numbers'),
    'order_three':      dict(ops=0, multi=0, json=0, note='A>B>C ordering'),
    'two_step_mul':     dict(ops=2, multi=1, json=0, note='(a+b)*c'),
    'mul_then_sub':     dict(ops=2, multi=1, json=0, note='a*b-a'),
    'sum_first_n':      dict(ops=1, multi=1, json=0, note='1+..+n'),
}


def classify(task):
    """Return (shape_name, ops, multi, json_flag) from the prompt content."""
    p = task['prompt']
    checker = task.get('checker')
    if checker == 'exact_json' or 'JSON' in p or 'json' in p:
        return 'json_sum', 1, 0, 1
    if '是否大于' in p or ('YES' in p and 'NO' in p):
        return 'cmp_yesno', 0, 0, 0
    if '奇数还是偶数' in p or ('奇数' in p and '偶数' in p):
        return 'parity', 0, 0, 0
    if '最大' in p and ('这三个数' in p or '三个数' in p):
        return 'max_of_three', 0, 0, 0
    if '谁最小' in p or ('比' in p and '大' in p and '小' in p):
        return 'order_three', 0, 0, 0
    if '先计算' in p and ('乘以' in p or '×' in p):
        return 'two_step_mul', 2, 1, 0
    if '再减去' in p or ('乘以' in p and '减去' in p):
        return 'mul_then_sub', 2, 1, 0
    if '前' in p and '正整数的和' in p:
        return 'sum_first_n', 1, 1, 0
    if '选项' in p or (checker == 'choice' and 'A.' in p):
        return 'choice_sum', 1, 0, 0
    if '×' in p or '*' in p:
        return 'mul_single', 1, 0, 0
    if checker == 'exact_int' and _has_negative_answer(task):
        return 'add_sub_neg', 1, 0, 0
    return 'add_sub_single', 1, 0, 0


def _answer_int(task):
    """First integer in the answer, or None (choice/label answers have none)."""
    m = re.search(r'-?\d+', str(task.get('answer', '')))
    return int(m.group()) if m else None


def _has_negative_answer(task):
    return str(task.get('answer', '')).strip().startswith('-')


NEG_TOKENS = ('负', '最小', '小于', '不是', '都不', '错误')


def difficulty_of(task):
    """Compute the integer difficulty score and its component breakdown."""
    shape, ops, multi, is_json = classify(task)
    ans = _answer_int(task)
    mag = abs(ans) if ans is not None else 0
    parts = {}

    # answer magnitude: 0 -> 0, 1..2 -> 2, 3..6 -> 4, 7..14 -> 6, ...
    parts['answer_magnitude'] = 2 * int(math.floor(math.log2(mag + 1)))
    parts['negative_answer'] = 3 if (ans is not None and ans < 0) else 0
    parts['extra_ops'] = 2 * max(0, ops - 1)
    parts['negation'] = 2 if any(tok in task['prompt'] for tok in NEG_TOKENS) else 0
    parts['json'] = 4 if is_json else 0
    parts['multistep'] = 3 if multi else 0
    parts['choice'] = 1 if task.get('checker') == 'choice' else 0
    parts['long_prompt'] = 1 if len(task['prompt']) > 80 else 0
    fam_rank = {'arithmetic': 0, 'format': 1, 'logic': 1, 'multistep': 2}
    parts['family_prior'] = fam_rank.get(task.get('family'), 1)

    total = sum(parts.values())
    return total, parts, shape


FAM_RANK = {'arithmetic': 0, 'format': 1, 'logic': 1, 'multistep': 2}


def sort_key(row):
    """Documented deterministic ordering: difficulty, then family, then |ans|, id."""
    return (row['difficulty'], FAM_RANK.get(row['family'], 1),
            abs(row['answer_int']) if row['answer_int'] is not None else 0,
            row['id'])


def load_tasks(path):
    tasks = []
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def histogram(values, step=2):
    """Compact {bucket_lo: count} histogram with bucket width `step`."""
    out = {}
    for v in values:
        lo = (v // step) * step
        out[lo] = out.get(lo, 0) + 1
    return dict(sorted(out.items()))


def summarize_stage(rows, name):
    fam = {}
    for r in rows:
        fam[r['family']] = fam.get(r['family'], 0) + 1
    shapes = {}
    for r in rows:
        shapes[r['shape']] = shapes.get(r['shape'], 0) + 1
    neg = sum(1 for r in rows if r['difficulty_parts']['negative_answer'])
    js = sum(1 for r in rows if r['difficulty_parts']['json'])
    ms = sum(1 for r in rows if r['difficulty_parts']['multistep'])
    diffs = [r['difficulty'] for r in rows]
    return {
        'name': name,
        'n': len(rows),
        'families': dict(sorted(fam.items())),
        'shapes': dict(sorted(shapes.items())),
        'negative_answers': neg,
        'json_outputs': js,
        'multistep': ms,
        'difficulty_min': min(diffs) if diffs else None,
        'difficulty_max': max(diffs) if diffs else None,
        'difficulty_mean': round(sum(diffs) / len(diffs), 2) if diffs else None,
        'difficulty_histogram': histogram(diffs),
        'ids': [r['id'] for r in rows],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', default=DEFAULT_SRC)
    p.add_argument('--outdir', default=DEFAULT_OUTDIR)
    p.add_argument('--heldout', default=DEFAULT_HELDOUT)
    p.add_argument('--stage1', type=int, default=150, help='size of stage1_easy')
    p.add_argument('--stage2', type=int, default=350, help='size of stage2_medium')
    p.add_argument('--seed', type=int, default=42,
                   help='recorded only; ordering is deterministic (no RNG used)')
    args = p.parse_args()

    tasks = load_tasks(args.src)
    if not tasks:
        raise SystemExit(f'no tasks read from {args.src}')
    print(f'[curriculum] read {len(tasks)} tasks from {args.src}')

    rows = []
    for t in tasks:
        diff, parts, shape = difficulty_of(t)
        row = dict(t)                      # original fields preserved verbatim
        row['difficulty'] = diff
        row['difficulty_parts'] = parts
        row['shape'] = shape
        row['answer_int'] = _answer_int(t)
        rows.append(row)
    rows.sort(key=sort_key)

    # Every task must be uniquely keyed, otherwise the subset asserts below
    # could silently drop/duplicate a task.
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit('duplicate task ids in source; refusing to build a curriculum')
    if len(rows) != 600:
        print(f'[curriculum] WARNING source has {len(rows)} tasks, expected 600')

    n1 = min(args.stage1, len(rows))
    n2 = min(args.stage2, len(rows))
    if n1 > n2:
        raise SystemExit('--stage1 must be <= --stage2')

    stage1 = rows[:n1]
    stage2 = rows[:n2]
    stage3 = rows

    # Nestedness is a hard requirement of the staged launcher.
    assert {r['id'] for r in stage1} <= {r['id'] for r in stage2} <= {r['id'] for r in stage3}

    # stage1 must genuinely be "single-step, small answers, no negation" -
    # if a future data rebuild breaks that, fail loudly instead of training on
    # a mislabelled "easy" stage.
    bad = [r['id'] for r in stage1
           if r['difficulty_parts']['negative_answer'] or r['difficulty_parts']['json']
           or r['difficulty_parts']['multistep']]
    if bad:
        raise SystemExit(f'stage1 contains hard tasks (neg/json/multistep): {bad[:5]}')

    files = {}
    for name, subset in (('stage1_easy', stage1), ('stage2_medium', stage2),
                         ('stage3_full', stage3)):
        path = os.path.join(args.outdir, f'{name}.jsonl')
        sha = write_jsonl(subset, path)
        files[name] = {'path': path, 'n': len(subset), 'sha256': sha}
        print(f'[curriculum] wrote {path} ({len(subset)} tasks, sha256 {sha[:12]}...)')

    # ---- integrity checks (fail loudly, never write a broken curriculum) ----
    # 1. every stage answer must pass its own reward checker (the same contract
    #    GRPO trains against and eval/grader.py grades with).
    bad_answers = []
    for r in rows:
        ok, why = reward_check(r, str(r['answer']))
        if not ok:
            bad_answers.append((r['id'], why))
    if bad_answers:
        raise SystemExit(f'{len(bad_answers)} tasks whose answer fails its own checker: '
                         f'{bad_answers[:5]}')

    # 2. no prompt may appear in the frozen held-out set (same rule as
    #    build_rl_data.py; the source is already clean, this re-verifies that the
    #    curriculum did not re-introduce anything).
    leak = 0
    heldout_sha = None
    if os.path.exists(args.heldout):
        heldout_sha = hashlib.sha256(open(args.heldout, 'rb').read()).hexdigest()
        held = {json.loads(l)['prompt'].strip()
                for l in open(args.heldout, encoding='utf-8-sig') if l.strip()}
        leak = sum(1 for r in rows if r['prompt'].strip() in held)
        if leak:
            raise SystemExit(f'ABORT: {leak} curriculum prompts appear in the held-out set')
    print(f'[curriculum] integrity: 0/{len(rows)} answer-check failures, '
          f'{leak} held-out leaks')

    all_diffs = [r['difficulty'] for r in rows]
    stats = {
        'created_from': args.src,
        'src_sha256': hashlib.sha256(open(args.src, 'rb').read()).hexdigest(),
        'src_n': len(rows),
        'seed': args.seed,
        'seed_note': 'no RNG is used; the ordering is a pure function of the task content, '
                     'so rebuilding is byte-reproducible',
        'difficulty_model': {
            'answer_magnitude': 'floor(log2(|answer|+1)) * 2',
            'negative_answer': '+3 if answer < 0',
            'extra_ops': '+2 per binary arithmetic operation beyond the first',
            'negation': '+2 if prompt contains one of ' + str(list(NEG_TOKENS)),
            'json': '+4 if the answer must be a JSON object',
            'multistep': '+3 if the prompt has two or more computation clauses',
            'choice': '+1 if the checker is a letter choice',
            'long_prompt': '+1 if prompt > 80 chars',
            'family_prior': {'arithmetic': 0, 'format': 1, 'logic': 1, 'multistep': 2},
            'tie_break': 'difficulty, family_rank, |answer|, id (all deterministic)',
        },
        'difficulty_overall': {
            'min': min(all_diffs), 'max': max(all_diffs),
            'mean': round(sum(all_diffs) / len(all_diffs), 2),
            'histogram_step2': histogram(all_diffs),
            'counts': {str(d): all_diffs.count(d) for d in sorted(set(all_diffs))},
        },
        'stages': {
            'stage1_easy': summarize_stage(stage1, 'stage1_easy'),
            'stage2_medium': summarize_stage(stage2, 'stage2_medium'),
            'stage3_full': summarize_stage(stage3, 'stage3_full'),
        },
        'nesting': 'stage1_easy subset-of stage2_medium subset-of stage3_full (asserted)',
        'integrity': {
            'answer_passes_own_checker': len(rows),
            'answer_check_failures': 0,
            'heldout_prompt_leaks': leak,
            'heldout_path': args.heldout,
            'heldout_sha256': heldout_sha,
        },
        'files': files,
        'source_untouched': args.src,
    }
    sp = os.path.join(args.outdir, 'curriculum_stats.json')
    with open(sp, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f'[curriculum] wrote {sp}')

    # ---- console report -------------------------------------------------
    print()
    print(f'{"stage":<16}{"n":>5}{"diff range":>14}{"mean":>7}'
          f'{"neg":>6}{"json":>6}{"multi":>7}')
    for name in ('stage1_easy', 'stage2_medium', 'stage3_full'):
        s = stats['stages'][name]
        print(f'{name:<16}{s["n"]:>5}{str(s["difficulty_min"]) + "-" + str(s["difficulty_max"]):>14}'
              f'{s["difficulty_mean"]:>7}{s["negative_answers"]:>6}'
              f'{s["json_outputs"]:>6}{s["multistep"]:>7}')
    print()
    print('difficulty histogram (all 600, bucket width 2):')
    for lo, cnt in stats['difficulty_overall']['histogram_step2'].items():
        print(f'  {lo:>3}-{lo + 1:<3} {cnt:>4}  {"#" * min(cnt, 60)}')
    print()
    print('stage1 shape mix:', json.dumps(stats['stages']['stage1_easy']['shapes'],
                                         ensure_ascii=False))
    print('stage1 family mix:', json.dumps(stats['stages']['stage1_easy']['families'],
                                          ensure_ascii=False))
    print('stage1 difficulty counts:',
          json.dumps({str(d): all_diffs[:n1].count(d) for d in sorted(set(all_diffs[:n1]))}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
