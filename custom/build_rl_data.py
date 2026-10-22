"""Build the verifiable-RL datasets for this project.

Produces three files, all with an explicit `checker` contract understood by
`custom/verifiable_reward.py` and by the held-out grader:

  D:/llm-rl-lab/data/grpo/train_grpo.jsonl   GRPO policy-training prompts   (seed 42)
  D:/llm-rl-lab/data/sft/sft_aux.jsonl       SFT "format seeding" prompts  (seed 1337)
  D:/llm-rl-lab/data/grpo/train_grpo_stats.json

HARD RULE: neither file may contain a prompt that appears in the held-out set
(D:/llm-rl-lab/eval/heldout_tasks.jsonl). Overlap is checked at build time and the
build aborts if any prompt collides.

Usage: python build_rl_data.py --per-family 200
"""
import argparse
import hashlib
import json
import os
import random

p = argparse.ArgumentParser()
p.add_argument('--per-family', type=int, default=200)
p.add_argument('--outdir', default=os.path.join(PROJECT_ROOT, 'data'))
p.add_argument('--heldout', default=os.path.join(PROJECT_ROOT, 'eval', 'heldout_tasks.jsonl'))
args = p.parse_args()

ARITH_Q = '请只回复一个数字，不要写任何解释或计算过程。'


def gen_arithmetic(rng):
    kind = rng.random()
    if kind < 0.60:
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        op = rng.choice(['+', '-'])
        ans = a + b if op == '+' else a - b
    elif kind < 0.85:
        a, b = rng.randint(0, 20), rng.randint(0, 20)
        op = rng.choice(['+', '-'])
        ans = a + b if op == '+' else a - b
    else:
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        op = '*'
        ans = a * b
    opstr = {'+': '+', '-': '-', '*': '×'}[op]
    tail = ARITH_Q if ans >= 0 else '请只回复一个数字，若为负数请带上负号。不要写任何解释。'
    return dict(prompt=f'{a} {opstr} {b} 等于多少？{tail}', answer=str(ans),
                checker='exact_int')


def gen_format(rng):
    kind = rng.random()
    if kind < 0.40:
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        ans = a + b
        return dict(prompt=('请输出一个 JSON 对象，且只包含键 "sum"，值为 %d+%d 的结果。'
                            '不要输出任何其他文字。例如：{"sum": 3}' % (a, b)),
                    answer=json.dumps({'sum': ans}, separators=(',', ':')),
                    checker='exact_json')
    if kind < 0.70:
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        return dict(prompt=f'{a} 是否大于 {b}？只回答 YES 或 NO，不要输出其他任何内容。',
                    answer='YES' if a > b else 'NO',
                    checker='label', labels=['YES', 'NO'])
    a, b = rng.randint(0, 9), rng.randint(0, 9)
    correct = a + b
    choices = [correct, correct + 1, max(0, correct - 1)]
    rng.shuffle(choices)
    idx = choices.index(correct)
    return dict(prompt=(f'{a} + {b} = ? 请从下面三个选项中选择一个，只回答对应的字母（A、B 或 C）：'
                        f'A. {choices[0]}  B. {choices[1]}  C. {choices[2]}'),
                answer='ABC'[idx], checker='choice', choices=['A', 'B', 'C'])


def gen_logic(rng):
    kind = rng.random()
    if kind < 0.35:
        names = ['A', 'B', 'C']
        order = rng.sample(range(3), 3)
        vals = {names[i]: order[i] for i in range(3)}
        srt = sorted(names, key=lambda n: -vals[n])
        return dict(prompt=(f'{srt[0]} 比 {srt[1]} 大，{srt[1]} 比 {srt[2]} 大。'
                            f'谁最小？只回答 {srt[0]}、{srt[1]} 或 {srt[2]} 中的一个字母。'),
                    answer=srt[2], checker='choice', choices=[srt[0], srt[1], srt[2]])
    if kind < 0.70:
        n = rng.randint(1, 30)
        return dict(prompt=f'{n} 是奇数还是偶数？只回答"奇数"或"偶数"。',
                    answer='奇数' if n % 2 else '偶数',
                    checker='label', labels=['奇数', '偶数'])
    nums = [rng.randint(0, 30) for _ in range(3)]
    return dict(prompt=(f'这三个数中最大的是哪个：{nums[0]}, {nums[1]}, {nums[2]}？'
                        '请只回复那个数字，不要写任何解释。'),
                answer=str(max(nums)), checker='exact_int')


def gen_multistep(rng):
    kind = rng.random()
    if kind < 0.40:
        a, b, c = rng.randint(1, 9), rng.randint(1, 9), rng.randint(2, 5)
        return dict(prompt=f'先计算 {a} + {b}，再把结果乘以 {c}，最终结果是多少？请只回复一个数字。',
                    answer=str((a + b) * c), checker='exact_int')
    if kind < 0.70:
        n = rng.randint(2, 12)
        return dict(prompt=f'前 {n} 个正整数的和是多少？请只回复一个数字。',
                    answer=str(n * (n + 1) // 2), checker='exact_int')
    a, b = rng.randint(2, 9), rng.randint(2, 9)
    return dict(prompt=f'{a} 乘以 {b} 再减去 {a}，结果是多少？请只回复一个数字。',
                answer=str(a * b - a), checker='exact_int')


GENS = {'arithmetic': gen_arithmetic, 'format': gen_format,
        'logic': gen_logic, 'multistep': gen_multistep}
WEIGHTS = {'arithmetic': 1.0, 'format': 0.75, 'logic': 0.75, 'multistep': 0.5}

heldout_prompts = set()
if os.path.exists(args.heldout):
    with open(args.heldout, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                heldout_prompts.add(json.loads(line)['prompt'].strip())
    print(f'[build] loaded {len(heldout_prompts)} held-out prompts to exclude')
else:
    print(f'[build] WARNING held-out file not found at {args.heldout}; no exclusion applied')


def build(seed, per_family, tag):
    rng = random.Random(seed)
    records, seen_prompts = [], set()
    for fam, gen in GENS.items():
        n = int(per_family * WEIGHTS[fam])
        made, tries = 0, 0
        while made < n and tries < n * 100:
            tries += 1
            rec = gen(rng)
            key = rec['prompt'].strip()
            if key in seen_prompts or key in heldout_prompts:
                continue
            seen_prompts.add(key)
            rec['id'] = f'{tag}_{fam}_{made:04d}'
            rec['family'] = fam
            records.append(rec)
            made += 1
        if made < n:
            print(f'[build] WARNING {fam}: only generated {made}/{n}')
    rng.shuffle(records)
    return records


def write_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    h = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    return h


grpo = build(seed=42, per_family=args.per_family, tag='grpo')
sft_aux = build(seed=1337, per_family=max(40, args.per_family // 5), tag='sftaux')

grpo_path = os.path.join(args.outdir, 'grpo', 'train_grpo.jsonl')
sft_path = os.path.join(args.outdir, 'sft', 'sft_aux.jsonl')
h1 = write_jsonl(grpo, grpo_path)
h2 = write_jsonl(sft_aux, sft_path)

# cross-contamination check
g_prompts = {r['prompt'].strip() for r in grpo}
s_prompts = {r['prompt'].strip() for r in sft_aux}
leak_h = len((g_prompts | s_prompts) & heldout_prompts)
print(f'[build] held-out leaks: {leak_h} (must be 0)')

stats = {
    'grpo': {'path': grpo_path, 'n': len(grpo), 'sha256': h1, 'seed': 42,
             'per_family': {f: sum(1 for r in grpo if r['family'] == f) for f in GENS}},
    'sft_aux': {'path': sft_path, 'n': len(sft_aux), 'sha256': h2, 'seed': 1337,
                'per_family': {f: sum(1 for r in sft_aux if r['family'] == f) for f in GENS}},
    'heldout_overlap': leak_h,
    'heldout_sha256': hashlib.sha256(open(args.heldout, 'rb').read()).hexdigest() if os.path.exists(args.heldout) else None,
}
sp = os.path.join(args.outdir, 'grpo', 'train_grpo_stats.json')
with open(sp, 'w', encoding='utf-8') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print(json.dumps(stats, ensure_ascii=False, indent=2))
if leak_h:
    raise SystemExit('ABORT: held-out leakage detected')
