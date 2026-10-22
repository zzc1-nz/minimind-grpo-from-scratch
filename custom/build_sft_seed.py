#!/usr/bin/env python
"""Build the Phase-2 "format seeding" set: `data/sft/sft_seed.jsonl`.

Purpose: teach the SFT model the SHORT-ANSWER OUTPUT FORMAT (one number / one JSON
object / one letter / one Chinese label) on deliberately narrow, easy instances, so that
  * the held-out baseline is not 0/24 for a *format* reason, and
  * Phase-3 GRPO does not start from all-wrong groups (adv_std = 0 -> no gradient),
while leaving the held-out families' harder end UNSEEN so GRPO has headroom to close it
(the "knows the format but cannot reason" regime the spec asks for).

Allowed shapes ONLY (Lead's spec, seed=1337, ~2,000 records):
  add        a + b,            a,b in [0,4]        -> exact_int   (answers 0..8)
  cmp        a > b ?           a,b in [0,9], a!=b  -> choice YES/NO
  json_sum   {"sum": a+b}      a,b in [0,4]        -> exact_json
  choice_sum A/B/C of a+b      a,b in [0,4]        -> choice A/B/C
  parity     n odd/even        n in [1,20]         -> label 奇数/偶数
  max3       max of 3 numbers  n in [0,20]         -> exact_int

Deliberately EXCLUDED (these stay unseen = GRPO's headroom):
  * multi-step (a+b)*c / sum-of-first-n / count-to-n   (held-out `multistep` family)
  * subtraction at all (the held-out arithmetic family includes 7-2, 8-8, 0-9, 4-9;
    two of those answers are negative, which the Lead explicitly forbids here)
  * any a,b > 9 arithmetic, any sequence / A>B>C ordering task

Each shape is realised with MANY phrasings so the model learns the format rather than
memorising instances.

Checker contract: identical to `custom/verifiable_reward.py` (exact_int / exact_json /
choice / label), so the same file is scored identically by the GRPO training reward and
by `eval/grader.py`. Every generated record is validated with
`verifiable_reward.check()` before it is written.

Held-out leak safety (aborts if violated):
  1. exact prompt string equality against `eval/heldout_tasks.jsonl`;
  2. prompt substring containment in either direction;
  3. arithmetic INSTANCE identity: (op, a, b) extracted from the held-out prompts with
     a regex, e.g. `3 + 5` -> ('add', 3, 5); our add/json_sum/choice_sum instances are
     compared against that set (regardless of the answer format);
  4. parity instance identity: held-out prompts that mention 奇数/偶数 contribute
     ('parity', n) for every integer n in the prompt;
  5. max-of-three instance identity: held-out prompts that mention 最大 contribute
     ('max3', sorted triple) for the integers in the prompt.

Usage:
  D:\\llm-rl-lab\\.venv\\Scripts\\python.exe D:\\llm-rl-lab\\custom\\build_sft_seed.py
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys

# --- project root, derived from this file's location (portable across machines) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

CUSTOM = os.path.join(PROJECT_ROOT, 'custom')
REPO = os.path.join(PROJECT_ROOT, 'src', 'minimind-master')
sys.path.insert(0, CUSTOM)

from verifiable_reward import check as vr_check   # noqa: E402

# --------------------------------------------------------------------------- phrasings
# Every phrasing asks for exactly the same output format; only the wording changes, so
# the model cannot key on one sentence.
ADD_PHRASINGS = [
    '{a} + {b} 等于多少？请只回复一个数字，不要写任何解释或计算过程。',
    '计算 {a} + {b}，只回答结果数字。',
    '{a} 加 {b} 是多少？请直接给出数字。',
    '请算一下 {a} + {b} = ?，只输出一个数字。',
    '{a} + {b} 的结果是什么？只回复数字，不要解释。',
    '求 {a} 与 {b} 的和。请只回复一个数字。',
    '算术题：{a} + {b}。只回答最终数字。',
    '{a} 和 {b} 相加等于几？只回复数字。',
    '计算 {a}+{b} 的值（只输出数字）。',
    '请问 {a} + {b} 的和是？请只给一个数字。',
    '把 {a} 和 {b} 加起来，结果是？只回复数字。',
    '{a} + {b} = 多少？请只输出数字，不要过程。',
]
CMP_PHRASINGS = [
    '{a} 是否大于 {b}？请只回答 YES 或 NO 这两个词中的一个，不要输出其他内容。',
    '比较 {a} 和 {b}：{a} > {b} 成立吗？只回答 YES 或 NO。',
    '{a} 比 {b} 大吗？请只回复 YES 或 NO。',
    '判断：{a} > {b}。只回答 YES 或 NO，不要解释。',
    '{a} 与 {b} 相比，{a} 更大吗？只回答 YES 或 NO。',
    '{a} > {b} 是否正确？请只回答 YES 或 NO。',
    '请判断 {a} 是否大于 {b}，只输出 YES 或 NO。',
    '{a} 大于 {b} 吗？请只给 YES 或 NO 一个词。',
]
JSON_PHRASINGS = [
    '请计算 {a} + {b} 的结果，并只输出一个 JSON 对象，键名为 sum，例如：{{"sum": 7}}。不要输出任何其他内容。',
    '计算 {a} + {b}，用 JSON 回答，只包含键 sum（例如 {{"sum": 7}}）。',
    '{a} + {b} 等于几？请只输出形如 {{"sum": 7}} 的 JSON 对象。',
    '请给出 {a} + {b} 的结果，格式为 JSON：{{"sum": 数值}}，不要输出别的。',
    '求 {a} + {b}，并且只输出一个 JSON 对象，键 sum（示例 {{"sum": 7}}）。',
    '{a} + {b} = ? 请用 JSON 对象回答，键名为 sum，例如 {{"sum": 7}}。',
    '把 {a} + {b} 的结果写成 JSON：只含 sum 键，例如 {{"sum": 7}}。',
    '请只输出 {{"sum": N}} 形式的 JSON，其中 N 是 {a} + {b} 的结果。',
]
CHOICE_PHRASINGS = [
    '{a} + {b} = ? 请从下面三个选项中选择一个，只回答对应字母（A、B 或 C）。A. {o1}  B. {o2}  C. {o3}',
    '{a} + {b} 的结果是下面哪个？A. {o1}  B. {o2}  C. {o3}。请只回答字母 A、B 或 C。',
    '下面哪个选项等于 {a} + {b}？A. {o1}  B. {o2}  C. {o3}。只回答一个字母。',
    '{a} 加 {b} 等于多少？A. {o1}  B. {o2}  C. {o3}。请只回复 A、B 或 C。',
    '选择正确的结果：{a} + {b} = A. {o1}  B. {o2}  C. {o3}。只输出字母。',
    '{a} + {b} 的正确选项是？A. {o1}  B. {o2}  C. {o3}。请只回答字母。',
    '单选题：{a} + {b} = ? A. {o1}  B. {o2}  C. {o3}。只回答 A、B 或 C。',
    '在 A/B/C 中选出 {a} + {b} 的结果：A. {o1}  B. {o2}  C. {o3}。只回复字母。',
]
PARITY_PHRASINGS = [
    '数字 {n} 是奇数还是偶数？请只回答“奇数”或“偶数”这两个词中的一个。',
    '{n} 是奇数还是偶数？只回答“奇数”或“偶数”。',
    '请判断 {n} 的奇偶性，只输出“奇数”或“偶数”。',
    '整数 {n} 是奇数吗？请只回答“奇数”或“偶数”。',
    '{n} 属于奇数还是偶数？只回复一个词：奇数 或 偶数。',
    '请只用一个词回答：{n} 是奇数还是偶数？',
    '判断数字 {n} 是奇数或偶数，只输出“奇数”或“偶数”。',
    '{n} 这个数是奇数还是偶数？只回答对应的中文词。',
    '数字 {n}：奇数还是偶数？只输出“奇数”或“偶数”其中一个。',
    '请问 {n} 是奇数还是偶数？请只回复“奇数”或“偶数”。',
]
MAX3_PHRASINGS = [
    '三个数 {n1}、{n2}、{n3} 中最大的是哪一个？请只回复这个数字。',
    '{n1}、{n2}、{n3} 里最大的数是？只回复一个数字。',
    '请找出 {n1}、{n2}、{n3} 中的最大值，只输出数字。',
    '{n1}、{n2}、{n3} 三个数中最大的是？只回答这个数。',
    '在 {n1}、{n2}、{n3} 中选出最大的一个，只回复数字。',
    '求 {n1}、{n2}、{n3} 的最大值（只输出数字）。',
    '哪一个数最大：{n1}、{n2}、{n3}？请只回复该数字。',
    '从 {n1}、{n2}、{n3} 中挑出最大数，只回答数字。',
    '比较 {n1}、{n2}、{n3}，最大的那个是多少？只回复数字。',
    '{n1}、{n2}、{n3} 中谁最大？请只输出这个数字。',
]


def heldout_instance_keys(heldout):
    """Extract arithmetic/parity/max3 instance identities from the frozen held-out set."""
    add, sub, parity, max3 = set(), set(), set(), set()
    for t in heldout:
        p = t['prompt']
        for m in re.finditer(r'(\d+)\s*\+\s*(\d+)', p):
            add.add((int(m.group(1)), int(m.group(2))))
        for m in re.finditer(r'(\d+)\s*-\s*(\d+)', p):
            sub.add((int(m.group(1)), int(m.group(2))))
        ints = [int(x) for x in re.findall(r'\d+', p)]
        if '奇数' in p or '偶数' in p:
            for n in ints:
                parity.add(n)
        if '最大' in p:
            if len(ints) >= 3:
                max3.add(tuple(sorted(ints[:3])))
    return dict(add=add, sub=sub, parity=parity, max3=max3)


def build_tasks(seed, target_total, heldout_keys):
    """Generate the seed tasks: shape instances x phrasings, quota-balanced, seeded."""
    rng = random.Random(seed)
    tasks = []

    # ---- instances per shape ----
    add_inst = [(a, b) for a in range(5) for b in range(5)]
    add_inst = [x for x in add_inst if x not in heldout_keys['add']]
    cmp_inst = [(a, b) for a in range(10) for b in range(10) if a != b]
    rng.shuffle(cmp_inst)
    cmp_inst = cmp_inst[:45]
    json_inst = [x for x in add_inst]
    choice_inst = [x for x in add_inst]
    parity_inst = [n for n in range(1, 21) if n not in heldout_keys['parity']]
    triples = []
    seen = set()
    while len(triples) < 25:
        tri = tuple(sorted(rng.sample(range(0, 21), 3)))
        if tri in seen or tri in heldout_keys['max3']:
            continue
        seen.add(tri)
        triples.append(tri)

    quotas = dict(add=400, cmp=400, json_sum=300, choice_sum=300, parity=300, max3=300)
    scale = target_total / sum(quotas.values())
    quotas = {k: max(1, int(round(v * scale))) for k, v in quotas.items()}

    def emit(shape, family, prompt, answer, checker, **extra):
        task = dict(id=f'seed_{shape}_{len(tasks):04d}', shape=shape, family=family,
                    prompt=prompt, answer=str(answer), checker=checker, **extra)
        tasks.append(task)

    # ---- add ----
    ph = ADD_PHRASINGS[:]
    rng.shuffle(ph)
    k = 0
    while sum(1 for t in tasks if t['shape'] == 'add') < quotas['add']:
        a, b = add_inst[k % len(add_inst)]
        p = ph[(k // len(add_inst)) % len(ph)]
        emit('add', 'arithmetic', p.format(a=a, b=b), a + b, 'exact_int')
        k += 1

    # ---- cmp (YES/NO) ----
    ph = CMP_PHRASINGS[:]
    rng.shuffle(ph)
    k = 0
    while sum(1 for t in tasks if t['shape'] == 'cmp') < quotas['cmp']:
        a, b = cmp_inst[k % len(cmp_inst)]
        p = ph[(k // len(cmp_inst)) % len(ph)]
        emit('cmp', 'logic', p.format(a=a, b=b), 'YES' if a > b else 'NO', 'choice',
             choices=['YES', 'NO'])
        k += 1

    # ---- json_sum ----
    ph = JSON_PHRASINGS[:]
    rng.shuffle(ph)
    k = 0
    while sum(1 for t in tasks if t['shape'] == 'json_sum') < quotas['json_sum']:
        a, b = json_inst[k % len(json_inst)]
        p = ph[(k // len(json_inst)) % len(ph)]
        ans = json.dumps({'sum': a + b}, separators=(',', ':'))
        emit('json_sum', 'format', p.format(a=a, b=b), ans, 'exact_json', json_keys=['sum'])
        k += 1

    # ---- choice_sum (A/B/C) ----
    ph = CHOICE_PHRASINGS[:]
    rng.shuffle(ph)
    k = 0
    while sum(1 for t in tasks if t['shape'] == 'choice_sum') < quotas['choice_sum']:
        a, b = choice_inst[k % len(choice_inst)]
        correct = a + b
        wrong = [x for x in range(0, 9) if x != correct]
        rng.shuffle(wrong)
        opts = [correct, wrong[0], wrong[1]]
        rng.shuffle(opts)
        letter = 'ABC'[opts.index(correct)]
        p = ph[(k // len(choice_inst)) % len(ph)]
        emit('choice_sum', 'format', p.format(a=a, b=b, o1=opts[0], o2=opts[1], o3=opts[2]),
             letter, 'choice', choices=['A', 'B', 'C'])
        k += 1

    # ---- parity ----
    ph = PARITY_PHRASINGS[:]
    rng.shuffle(ph)
    k = 0
    while sum(1 for t in tasks if t['shape'] == 'parity') < quotas['parity']:
        n = parity_inst[k % len(parity_inst)]
        p = ph[(k // len(parity_inst)) % len(ph)]
        emit('parity', 'logic', p.format(n=n), '奇数' if n % 2 else '偶数', 'label',
             labels=['奇数', '偶数'])
        k += 1

    # ---- max3 ----
    ph = MAX3_PHRASINGS[:]
    rng.shuffle(ph)
    k = 0
    while sum(1 for t in tasks if t['shape'] == 'max3') < quotas['max3']:
        n1, n2, n3 = triples[k % len(triples)]
        p = ph[(k // len(triples)) % len(ph)]
        emit('max3', 'logic', p.format(n1=n1, n2=n2, n3=n3), max(n1, n2, n3), 'exact_int')
        k += 1

    rng.shuffle(tasks)
    return tasks


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description='Build the SFT format-seeding set (seed 1337)')
    p.add_argument('--heldout_path', default=os.path.join(PROJECT_ROOT, 'eval', 'heldout_tasks.jsonl'))
    p.add_argument('--out_path', default=os.path.join(PROJECT_ROOT, 'data', 'sft', 'sft_seed.jsonl'))
    p.add_argument('--meta_path', default=os.path.join(PROJECT_ROOT, 'data', 'sft', 'sft_seed_meta.json'))
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--total', type=int, default=2000)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or '.', exist_ok=True)
    heldout = [json.loads(l) for l in open(args.heldout_path, encoding='utf-8') if l.strip()]
    hk = heldout_instance_keys(heldout)
    print(f'[seed] held-out tasks={len(heldout)} | excluded instances: '
          f'add={sorted(hk["add"])} sub={sorted(hk["sub"])} parity={sorted(hk["parity"])} '
          f'max3={sorted(hk["max3"])}')

    tasks = build_tasks(args.seed, args.total, hk)

    # ---- self-check: every answer must pass its own checker (same contract as GRPO) ----
    bad = []
    for t in tasks:
        ok, reason = vr_check(t, t['answer'])
        if not ok:
            bad.append((t['id'], t['answer'], reason))
    if bad:
        for b in bad[:10]:
            print('  BAD', b)
        raise SystemExit(f'FATAL: {len(bad)} seed tasks fail their own checker')

    # ---- leak checks ----
    held_prompts = {t['prompt'] for t in heldout}
    n_exact = sum(1 for t in tasks if t['prompt'] in held_prompts)
    n_substr = 0
    for t in tasks:
        for hp in held_prompts:
            if hp in t['prompt'] or t['prompt'] in hp:
                n_substr += 1
                break
    n_inst = 0
    inst_hits = []
    for t in tasks:
        s = t['shape']
        if s == 'add' or s == 'json_sum' or s == 'choice_sum':
            m = re.search(r'(\d+)\s*\+\s*(\d+)', t['prompt'])
            if m and (int(m.group(1)), int(m.group(2))) in hk['add']:
                n_inst += 1
                inst_hits.append((t['id'], 'add', m.group(0)))
        elif s == 'parity':
            m = re.search(r'(\d+)', t['prompt'])
            if m and int(m.group(1)) in hk['parity']:
                n_inst += 1
                inst_hits.append((t['id'], 'parity', m.group(0)))
        elif s == 'max3':
            ints = tuple(sorted(int(x) for x in re.findall(r'\d+', t['prompt'])))
            if ints in hk['max3']:
                n_inst += 1
                inst_hits.append((t['id'], 'max3', ints))
    print(f'[seed] leak checks: exact_prompt={n_exact} substring={n_substr} '
          f'instance={n_inst}')
    if n_exact or n_substr or n_inst:
        for h in inst_hits[:10]:
            print('  INSTANCE HIT', h)
        raise SystemExit('FATAL: seed set leaks the held-out benchmark')

    # ---- write ----
    tmp = args.out_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')
    os.replace(tmp, args.out_path)
    with open(args.out_path, 'rb') as f:
        head = f.read(3)
    if head == b'\xef\xbb\xbf':
        raise SystemExit('FATAL: output starts with a UTF-8 BOM')

    shapes = {}
    for t in tasks:
        shapes.setdefault(t['shape'], dict(count=0, family=t['family'],
                                           answers=set(), prompts=set()))
        shapes[t['shape']]['count'] += 1
        shapes[t['shape']]['answers'].add(t['answer'])
        shapes[t['shape']]['prompts'].add(t['prompt'])
    meta = dict(created_at=__import__('time').strftime('%Y-%m-%dT%H:%M:%S%z'),
                out_path=args.out_path, out_sha256=sha256_file(args.out_path),
                out_bytes=os.path.getsize(args.out_path), records=len(tasks),
                seed=args.seed, target_total=args.total, bom=False,
                shapes={k: dict(count=v['count'], family=v['family'],
                                distinct_answers=len(v['answers']),
                                distinct_prompts=len(v['prompts']))
                        for k, v in sorted(shapes.items())},
                excluded_shapes=['sub', 'multistep', 'sequence', 'ordering',
                                 'any a,b > 9 arithmetic'],
                leak_check=dict(exact_prompt=n_exact, substring=n_substr, instance=n_inst,
                                heldout_instance_keys={k: sorted(map(list, v)) if k == 'max3'
                                                       else sorted(v) for k, v in hk.items()}),
                checker_self_test='all answers pass custom/verifiable_reward.check()',
                )
    with open(args.meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f'[seed] wrote {args.out_path} records={len(tasks)} sha256={meta["out_sha256"]}')
    for k, v in sorted(shapes.items()):
        print(f'  {k:11s} family={v["family"]:10s} count={v["count"]:4d} '
              f'distinct_answers={len(v["answers"]):3d} distinct_prompts={len(v["prompts"]):4d}')
    print(f'[seed] meta -> {args.meta_path}')


if __name__ == '__main__':
    main()
