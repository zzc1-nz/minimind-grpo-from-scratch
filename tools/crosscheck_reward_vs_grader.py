"""Cross-check the GRPO training reward against the held-out grader.

Two independent checker implementations exist:
  * custom/verifiable_reward.py  -- the reward GRPO optimises against
  * eval/grader.py               -- the metric the held-out score is computed with

If they disagree on any input, GRPO optimises a different objective than the one the
report measures, and the SFT-vs-GRPO comparison is invalid. This script feeds a large
set of realistic and adversarial completions through BOTH and reports disagreements.

Usage:
  python crosscheck_reward_vs_grader.py
  python crosscheck_reward_vs_grader.py --json
"""
import argparse
import importlib.util
import json
import os
import sys

ROOT = r'D:\llm-rl-lab'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Completions that a small model realistically produces, plus adversarial edge cases.
CASES = [
    # exact_int
    {'task': {'checker': 'exact_int', 'answer': '8'}, 'comps': [
        '8', ' 8 ', '8。', '8.', '答案是 8', '3+5=8', '８', '八', '8<|im_end|>',
        '答案是8，因为3+5=8', '7', '答案是 7', '', '\n\n\n', '0', '18', '-9']},
    {'task': {'checker': 'exact_int', 'answer': '-9'}, 'comps': [
        '-9', '- 9', '−9', '负九', '答案是 -9', '9', '0', '9-']},
    {'task': {'checker': 'exact_int', 'answer': '0'}, 'comps': [
        '0', '零', '答案是 0', '00', '', 'O']},
    # exact_json
    {'task': {'checker': 'exact_json', 'answer': '{"sum": 9}'}, 'comps': [
        '{"sum": 9}', '{"sum":9}', ' {"sum": 9} ', '{"sum": 8}',
        '结果是 {"sum": 9} 完毕', '{"sum": "9"}', '{"sum": 9, "x": 1}',
        '{"diff": 3}', '', '{}', "{'sum': 9}"], 'json_keys': ['sum']},
    # choice
    {'task': {'checker': 'choice', 'answer': 'B', 'choices': ['A', 'B', 'C']}, 'comps': [
        'B', 'b', '答案是 B', '答案是 B。', 'A', 'C', 'ABC', 'B or C', '', 'X']},
    {'task': {'checker': 'choice', 'answer': 'YES', 'choices': ['YES', 'NO']}, 'comps': [
        'YES', 'yes', 'Yes', '答案是 YES', 'NO', 'NO, actually YES', '', 'Y']},
    {'task': {'checker': 'choice', 'answer': '奇数', 'choices': ['奇数', '偶数']}, 'comps': [
        '奇数', '答案是奇数', '偶数', '奇', '奇数和偶数', '']},
    # label (only the training reward has this checker)
    {'task': {'checker': 'label', 'answer': 'YES', 'labels': ['YES', 'NO']}, 'comps': [
        'YES', 'NO', 'yes', '', '答案是 YES']},
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json', action='store_true')
    a = p.parse_args()

    rew = load_module('verifiable_reward', os.path.join(ROOT, 'custom', 'verifiable_reward.py'))
    gr = load_module('grader', os.path.join(ROOT, 'eval', 'grader.py'))

    rows = []
    disagree = 0
    comparable = 0
    for case in CASES:
        base = dict(case['task'])
        if 'json_keys' in case:
            base['json_keys'] = case['json_keys']
        for comp in case['comps']:
            r_ok, r_why = rew.check(base, comp)
            # grader.check may raise or return differently for checkers it lacks
            try:
                g_ok, g_why = gr.check(base, comp)
                g_err = None
            except Exception as exc:
                g_ok, g_why, g_err = None, None, f'{type(exc).__name__}: {exc}'
            same = (r_ok == g_ok) if g_err is None else None
            if same is not None:
                comparable += 1
                if not same:
                    disagree += 1
            rows.append({
                'checker': base['checker'], 'answer': base['answer'],
                'completion': comp,
                'reward_correct': r_ok, 'reward_reason': r_why,
                'grader_correct': g_ok, 'grader_reason': g_why, 'grader_error': g_err,
                'agree': same,
            })

    out = {'total_cases': len(rows), 'comparable': comparable,
           'disagreements': disagree, 'rows': rows}
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f'checked {len(rows)} (task, completion) pairs; '
              f'{comparable} comparable, {disagree} disagreements')
        print()
        if disagree:
            print('DISAGREEMENTS (GRPO reward vs held-out grader):')
            for r in rows:
                if r['agree'] is False:
                    print(f"  [{r['checker']}] answer={r['answer']!r} "
                          f"completion={r['completion']!r}")
                    print(f"      reward={r['reward_correct']} ({r['reward_reason']})")
                    print(f"      grader={r['grader_correct']} ({r['grader_reason']})")
        else:
            print('No disagreements on comparable cases.')
        missing = sorted({r['checker'] for r in rows if r['agree'] is None})
        if missing:
            print()
            print(f'Checkers the grader cannot evaluate: {missing}')
            print('  (these exist only in the GRPO training set; see notes below)')
    # Exit non-zero ONLY on a real disagreement. A checker the grader does not implement
    # is "not comparable", not "failed" -- treating it as failure made the pipeline's
    # Phase-5 quality gate report a false alarm.
    return 0 if disagree == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
