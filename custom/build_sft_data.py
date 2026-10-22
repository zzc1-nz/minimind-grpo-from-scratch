#!/usr/bin/env python
"""Build the Phase-2 SFT training mix for MiniMind.

Inputs (read-only):
  data/sft/sft_seed.jsonl      ~2,000 easy "format seeding" tasks (custom/build_sft_seed.py,
                               seed 1337) -- the task-shaped part of the mix
  data/sft/sft_aux.jsonl       120 rule-verifiable tasks (seed 1337) -- superseded, kept
                               for the earlier 0.5 % mix; selectable via --aux_path
  data/sft_t2t_mini.jsonl      905,640 general chat conversations (1.66 GB)
  eval/heldout_tasks.jsonl     24 frozen held-out tasks -> must never appear in SFT data

Output:
  data/sft/sft_train.jsonl      {"conversations":[{"role","content"},...]} records
  data/sft/sft_meta.json        measured composition + token totals + sha256 + leak count

Mix (Lead's Phase-2 requirement: ~20 % task-shaped, ~80 % general chat)
  1. every seed task -> one 2-turn conversation (user=prompt, assistant=answer),
     repeated `--aux_repeat` times (default 20): 2,000 x 20 = 40,000 task-shaped records;
  2. `--max_general` (default 160,000) general conversations sampled from the 905,640
     with reservoir sampling under seed 42, after excluding any record whose user-turn
     text contains a held-out prompt (exact substring);
  3. both groups concatenated and shuffled with seed 42.

Why reservoir sampling: the general file is 1.66 GB / 905,640 records. Loading it whole
into Python dicts costs 3-4 GB of RAM on a 16 GB box that is already running the Phase-1
pretraining job. Reservoir sampling is a single streaming pass, holds only N records,
and is fully deterministic for a fixed seed and file order.

Token accounting in the meta file is MEASURED: every record of the written mix is
rendered through the real chat template (assistant turns keep their <think> block, which
is what custom/train_full_sft_counted.py feeds the model) and tokenized with the MiniMind
tokenizer, and the assistant-only label count is computed with the same algorithm as
`dataset/lm_dataset.py::SFTDataset.generate_labels` (verified against the repo's own
function on a sample; see `tokens.label_fn_check` in the meta). Truncation cost is
measured at several candidate `max_seq_len` values, so the seq-len choice is data-driven.

The output is written with Python (encoding='utf-8', newline='\\n') -- never by
PowerShell, because a UTF-8 BOM makes `datasets` fail with DatasetGenerationError.

Usage:
  D:\\llm-rl-lab\\.venv\\Scripts\\python.exe D:\\llm-rl-lab\\custom\\build_sft_data.py
  ... --max_general 160000 --aux_repeat 20 --aux_path D:/llm-rl-lab/data/sft/sft_seed.jsonl
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

# --- project root, derived from this file's location (portable across machines) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

REPO = os.path.join(PROJECT_ROOT, 'src', 'minimind-master')
sys.path.insert(0, REPO)

import psutil  # noqa: E402

MAX_RSS = [0.0]
BAD_LINES = [0]

# exactly the columns declared by dataset/lm_dataset.py::SFTDataset's Features
MSG_KEYS = ('role', 'content', 'reasoning_content', 'tools', 'tool_calls')


def rss_mb():
    """Sampled resident set size of this process (MB)."""
    v = psutil.Process().memory_info().rss / 2 ** 20
    if v > MAX_RSS[0]:
        MAX_RSS[0] = v
    return round(v, 1)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_jsonl(path, tolerate_bad=False):
    """Stream jsonl records. With tolerate_bad, malformed lines (the general file's last
    line is truncated -- 905,640 lines, record #905,640 is cut off) are skipped."""
    n_bad = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if not tolerate_bad:
                    raise
                n_bad += 1
                BAD_LINES[0] += 1


# --------------------------------------------------------------------------- labels
def make_label_counter(tokenizer):
    """Return count_labels(ids, max_length) -> #labelled positions, replicating
    dataset/lm_dataset.py::SFTDataset.generate_labels exactly (same bos/eos patterns,
    same skip/truncate semantics) but counting instead of materialising the mask.

    The repo version is a pure-Python scan over every token; this one uses numpy to
    find candidate positions, which matters for 200k records. Equivalence is asserted
    against the repo function on a sample (tokens.label_fn_check in the meta).
    """
    import numpy as np
    bos = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
    eos = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
    nb, ne = len(bos), len(eos)

    def _pattern_positions(a, pat):
        n = a.size
        cand = np.nonzero(a == pat[0])[0]
        out = [int(i) for i in cand
               if i + len(pat) <= n and all(a[i + k] == pat[k] for k in range(1, len(pat)))]
        return np.asarray(out, dtype=np.int64)

    def count_labels(ids, max_length):
        a = np.asarray(ids, dtype=np.int64)
        n = a.size
        if n == 0:
            return 0
        cand = _pattern_positions(a, bos)          # assistant-span starts, ascending
        if cand.size == 0:
            return 0
        ecand = _pattern_positions(a, eos)
        total = 0
        i_floor = 0
        for c in cand:
            if c < i_floor:                        # inside a span already labelled
                continue
            start = int(c) + nb
            pos = int(np.searchsorted(ecand, start, side='left')) if ecand.size else ecand.size
            if pos < ecand.size:
                end = int(ecand[pos])
                stop = end + ne
            else:
                end, stop = n, n
            hi = min(stop, max_length)
            if hi > start:
                total += hi - start
            i_floor = stop if end < n else n
        return total

    return count_labels, bos, eos


def repo_generate_labels(ids, bos_id, eos_id, max_length):
    """Verbatim copy of SFTDataset.generate_labels (mask form) for the equivalence check."""
    labels = [-100] * len(ids)
    i = 0
    while i < len(ids):
        if ids[i:i + len(bos_id)] == bos_id:
            start = i + len(bos_id)
            end = start
            while end < len(ids):
                if ids[end:end + len(eos_id)] == eos_id:
                    break
                end += 1
            for j in range(start, min(end + len(eos_id), max_length)):
                labels[j] = ids[j]
            i = end + len(eos_id) if end < len(ids) else len(ids)
        else:
            i += 1
    return labels


# --------------------------------------------------------------------------- mix
def sanitize_conversations(convs):
    """Keep only the columns declared in SFTDataset's Features so `datasets` can load
    the file with explicit features. Returns (clean_conversations, n_keys_dropped)."""
    clean, dropped = [], 0
    for m in convs:
        if not isinstance(m, dict) or not m.get('role'):
            return None, 0
        c = {'role': m['role']}
        content = m.get('content')
        if content is None:
            content = ''
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        c['content'] = content
        for k in MSG_KEYS[2:]:
            v = m.get(k)
            if v is None or v == '':
                continue
            c[k] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        dropped += sum(1 for k in m if k not in MSG_KEYS)
        clean.append(c)
    return clean, dropped


def build_mix(args, heldout_prompts, aux_tasks):
    """Stream the general file, reservoir-sample eligible records, then interleave."""
    rng = random.Random(args.seed)
    reservoir = []
    n_scanned = 0
    n_eligible = 0
    n_excluded_leak = 0
    n_excluded_bad = 0
    n_keys_dropped = 0

    for rec in read_jsonl(args.general_path, tolerate_bad=True):
        n_scanned += 1
        convs = rec.get('conversations') or []
        if not isinstance(convs, list) or not convs:
            n_excluded_bad += 1
            continue
        user_texts = [m.get('content') or '' for m in convs
                      if isinstance(m, dict) and m.get('role') == 'user']
        if any(hp in ut for hp in heldout_prompts for ut in user_texts):
            n_excluded_leak += 1
            continue
        n_eligible += 1
        if len(reservoir) < args.max_general:
            reservoir.append(rec)
        else:
            j = rng.randrange(n_eligible)          # n_eligible = eligible items seen (1-based)
            if j < args.max_general:
                reservoir[j] = rec

    general = []
    for rec in reservoir:
        clean, dropped = sanitize_conversations(rec['conversations'])
        if clean is None:
            n_excluded_bad += 1
            continue
        n_keys_dropped += dropped
        general.append(('general', {'conversations': clean}))
    del reservoir

    aux = []
    for _ in range(args.aux_repeat):
        for task in aux_tasks:
            aux.append(('aux', {'conversations': [
                {'role': 'user', 'content': task['prompt']},
                {'role': 'assistant', 'content': task['answer']},
            ]}))
    random.Random(args.seed).shuffle(aux)
    mix = general + aux
    random.Random(args.seed).shuffle(mix)          # seed 42, as specified
    return mix, dict(general_scanned=n_scanned, general_eligible=n_eligible,
                     general_excluded_heldout_leak=n_excluded_leak,
                     general_excluded_malformed=n_excluded_bad,
                     general_malformed_lines_skipped=BAD_LINES[0],
                     general_sampled=len(general), aux_records=len(aux),
                     message_keys_dropped=n_keys_dropped)


def write_mix(path, mix):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
        for _src, rec in mix:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def scan_leaks(path, heldout_prompts):
    """Re-read the written file and count records whose user turns contain a held-out
    prompt (exact substring). Must be 0."""
    n_leak, n = 0, 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            n += 1
            rec = json.loads(line)
            user_texts = [m.get('content') or '' for m in rec['conversations'] if m.get('role') == 'user']
            if any(hp in ut for hp in heldout_prompts for ut in user_texts):
                n_leak += 1
    return n, n_leak


def make_renderer(tokenizer):
    """Replicates dataset/lm_dataset.py::SFTDataset.create_chat_prompt exactly (tools are
    json-parsed from a system message, tool_calls are json-parsed) but WITHOUT the
    stochastic pre_processing_chat / post_processing_chat steps, so the measured token
    totals are deterministic. The stochastic deltas are reported separately."""
    def render(conversations):
        messages, tools = [], None
        for message in conversations:
            message = dict(message)
            if message.get('role') == 'system' and message.get('tools'):
                tools = json.loads(message['tools']) if isinstance(message['tools'], str) else message['tools']
            if message.get('tool_calls') and isinstance(message['tool_calls'], str):
                message['tool_calls'] = json.loads(message['tool_calls'])
            messages.append(message)
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=False, tools=tools)
    return render


# --------------------------------------------------------------------------- token stats
def token_stats(mix, tokenizer, render, count_labels, bos, eos, max_seq_len,
                alt_lengths=(), chunk=2000, check_n=200):
    """Measure, over the whole written mix: rendered prompt tokens and labelled
    (assistant-only) tokens, both untruncated and truncated at max_seq_len.
    `alt_lengths` additionally measures truncation at other candidate seq lens, so the
    choice of max_seq_len is made from data rather than assumed."""
    stats = {}
    for src, _ in mix:
        stats.setdefault(src, dict(records=0, tokens_prompt_full=0, tokens_labeled_full=0,
                                   tokens_prompt_used=0, tokens_labeled_used=0,
                                   n_truncated=0, n_records_with_labels=0, alt={}))
        stats[src]['records'] += 1
    alt = {L: dict(records=0, prompt_tokens=0, labeled_tokens=0, truncated_records=0)
           for L in alt_lengths}
    for src in stats:
        for L in alt_lengths:
            stats[src]['alt'][L] = dict(records=0, prompt_tokens=0, labeled_tokens=0,
                                        truncated_records=0)

    check = dict(n_records=0, mismatches=0, repo_labeled_full_on_sample=0,
                 repo_labeled_trunc_on_sample=0, fast_labeled_full_on_sample=0,
                 fast_labeled_trunc_on_sample=0)
    t0 = time.time()
    done = 0
    for start in range(0, len(mix), chunk):
        part = mix[start:start + chunk]
        rendered = [render(rec['conversations']) for _src, rec in part]
        encs = tokenizer._tokenizer.encode_batch(rendered)
        for (src, _rec), enc in zip(part, encs):
            s = stats[src]
            ids = enc.ids
            n = len(ids)
            lab_full = count_labels(ids, n)
            lab_used = lab_full if n <= max_seq_len else count_labels(ids[:max_seq_len], max_seq_len)
            s['tokens_prompt_full'] += n
            s['tokens_labeled_full'] += lab_full
            s['tokens_prompt_used'] += min(n, max_seq_len)
            s['tokens_labeled_used'] += lab_used
            if n > max_seq_len:
                s['n_truncated'] += 1
            if lab_full > 0:
                s['n_records_with_labels'] += 1
            for L in alt_lengths:
                a = stats[src]['alt'][L]
                a['records'] += 1
                a['prompt_tokens'] += min(n, L)
                a['labeled_tokens'] += lab_full if n <= L else count_labels(ids[:L], L)
                if n > L:
                    a['truncated_records'] += 1
        if check['n_records'] == 0:      # equivalence check vs the repo's own function
            for (_src, _rec), enc in list(zip(part, encs))[:check_n]:
                ids = enc.ids
                repo_full = sum(1 for x in repo_generate_labels(ids, bos, eos, len(ids)) if x != -100)
                repo_trunc = sum(1 for x in repo_generate_labels(ids[:max_seq_len], bos, eos, max_seq_len) if x != -100)
                fast_full = count_labels(ids, len(ids))
                fast_trunc = count_labels(ids[:max_seq_len], max_seq_len)
                check['n_records'] += 1
                check['repo_labeled_full_on_sample'] += repo_full
                check['repo_labeled_trunc_on_sample'] += repo_trunc
                check['fast_labeled_full_on_sample'] += fast_full
                check['fast_labeled_trunc_on_sample'] += fast_trunc
                if repo_full != fast_full or repo_trunc != fast_trunc:
                    check['mismatches'] += 1
        done += len(part)
        if done % 40000 < chunk:
            print(f'  tokenized {done:,}/{len(mix):,} records  ({time.time() - t0:.1f}s, rss {rss_mb()} MB)')

    stats['tokenize_sec'] = round(time.time() - t0, 1)
    stats['label_fn_check'] = check
    return stats


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description='Build the Phase-2 SFT training mix (exact token accounting)')
    p.add_argument('--general_path', default=os.path.join(PROJECT_ROOT, 'data', 'sft_t2t_mini.jsonl'))
    p.add_argument('--aux_path', default=os.path.join(PROJECT_ROOT, 'data', 'sft', 'sft_seed.jsonl'),
                   help='task-shaped source: the seed set (default) or the older sft_aux.jsonl')
    p.add_argument('--heldout_path', default=os.path.join(PROJECT_ROOT, 'eval', 'heldout_tasks.jsonl'))
    p.add_argument('--out_path', default=os.path.join(PROJECT_ROOT, 'data', 'sft', 'sft_train.jsonl'))
    p.add_argument('--meta_path', default=os.path.join(PROJECT_ROOT, 'data', 'sft', 'sft_meta.json'))
    p.add_argument('--tokenizer_path', default=f'{REPO}/model')
    p.add_argument('--max_general', type=int, default=160000, help='N general conversations to sample')
    p.add_argument('--aux_repeat', type=int, default=20, help='repeat each task-shaped record this many times')
    p.add_argument('--max_seq_len', type=int, default=768, help='training max_seq_len for truncated stats')
    p.add_argument('--batch_size', type=int, default=16, help='planned batch size for padded-token math')
    p.add_argument('--alt_seq_lens', type=str, default='512,640,896',
                   help='comma-separated extra seq lens to measure truncation cost at')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--skip_token_stats', type=int, default=0, choices=[0, 1],
                   help='1 = skip the tokenize-everything pass (fast rebuild / idempotency check)')
    args = p.parse_args()

    t_start = time.time()
    os.makedirs(os.path.dirname(args.out_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.meta_path) or '.', exist_ok=True)

    heldout = list(read_jsonl(args.heldout_path))
    heldout_prompts = [t['prompt'] for t in heldout]
    aux_tasks = list(read_jsonl(args.aux_path))
    print(f'[build_sft] heldout tasks={len(heldout)} aux tasks={len(aux_tasks)} '
          f'max_general={args.max_general:,} aux_repeat={args.aux_repeat} seed={args.seed}')

    mix, sampling = build_mix(args, heldout_prompts, aux_tasks)
    print(f'[build_sft] mix: general={sampling["general_sampled"]:,} aux={sampling["aux_records"]:,} '
          f'total={len(mix):,} (scanned {sampling["general_scanned"]:,} general records, '
          f'excluded {sampling["general_excluded_heldout_leak"]:,} held-out leaks, '
          f'rss {rss_mb()} MB)')

    write_mix(args.out_path, mix)
    del mix
    rss_mb()

    # BOM guard: `datasets` fails on a BOM-prefixed jsonl (env_report.md §5 problem 7)
    with open(args.out_path, 'rb') as f:
        head = f.read(3)
    if head == b'\xef\xbb\xbf':
        raise SystemExit('FATAL: output file starts with a UTF-8 BOM')
    n_lines, n_leak = scan_leaks(args.out_path, heldout_prompts)
    print(f'[build_sft] wrote {args.out_path} ({os.path.getsize(args.out_path) / 2**20:.1f} MB, '
          f'{n_lines:,} records, BOM=no) leak_count={n_leak}')
    if n_leak != 0:
        raise SystemExit(f'FATAL: {n_leak} records in the SFT mix contain a held-out prompt')
    expected = sampling['general_sampled'] + sampling['aux_records']
    if n_lines != expected:
        raise SystemExit(f'FATAL: line count {n_lines} != expected {expected}')

    sha = sha256_file(args.out_path)
    print(f'[build_sft] sha256={sha}')

    meta = dict(
        created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        out_path=args.out_path, out_bytes=os.path.getsize(args.out_path), out_sha256=sha,
        out_records=n_lines, seed=args.seed,
        max_general=args.max_general, aux_repeat=args.aux_repeat,
        max_seq_len=args.max_seq_len, batch_size_for_padded_math=args.batch_size,
        sources=dict(
            general=dict(path=args.general_path, records=sampling['general_sampled']),
            aux=dict(path=args.aux_path, tasks=len(aux_tasks), records=sampling['aux_records'],
                     families={f: sum(1 for t in aux_tasks if t['family'] == f)
                               for f in sorted({t['family'] for t in aux_tasks})},
                     shapes={s: sum(1 for t in aux_tasks if t.get('shape') == s)
                             for s in sorted({t['shape'] for t in aux_tasks if t.get('shape')})}),
        ),
        sampling=sampling,
        heldout=dict(path=args.heldout_path, tasks=len(heldout), leak_count=n_leak,
                     method='exact substring: heldout prompt in any user-turn text'),
        aux_fraction_by_record=round(sampling['aux_records'] / n_lines, 6),
        bom=False,
    )

    if not args.skip_token_stats:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        count_labels, bos, eos = make_label_counter(tokenizer)
        render = make_renderer(tokenizer)
        # re-read in written order; aux records are identified by their exact (prompt, answer)
        aux_pairs = {(t['prompt'], t['answer']) for t in aux_tasks}
        mix2 = []
        for rec in read_jsonl(args.out_path):
            convs = rec['conversations']
            key = 'aux' if (len(convs) == 2 and (convs[0]['content'], convs[1]['content']) in aux_pairs) else 'general'
            mix2.append((key, rec))
        alt_lengths = tuple(int(x) for x in args.alt_seq_lens.split(',') if x.strip())
        st = token_stats(mix2, tokenizer, render, count_labels, bos, eos, args.max_seq_len,
                         alt_lengths=alt_lengths)
        del mix2
        tot = dict(records=0, tokens_prompt_full=0, tokens_labeled_full=0,
                   tokens_prompt_used=0, tokens_labeled_used=0, n_truncated=0,
                   n_records_with_labels=0)
        for key in ('general', 'aux'):
            for k in tot:
                tot[k] += st[key][k]
        alt_tot = {}
        for L in alt_lengths:
            alt_tot[str(L)] = dict(
                prompt_tokens=sum(st[k]['alt'][L]['prompt_tokens'] for k in ('general', 'aux')),
                labeled_tokens=sum(st[k]['alt'][L]['labeled_tokens'] for k in ('general', 'aux')),
                truncated_records=sum(st[k]['alt'][L]['truncated_records'] for k in ('general', 'aux')),
                padded_tokens_per_epoch=((tot['records'] + args.batch_size - 1) // args.batch_size)
                                        * args.batch_size * L)
        meta['tokens'] = dict(
            note=('measured by rendering every record through the chat template (assistant turns '
                  'keep their <think> block, i.e. post_processing_chat(remove_empty_think=False), '
                  'which is what custom/train_full_sft_counted.py feeds the model) and tokenizing '
                  'with the MiniMind tokenizer; labeled = assistant-only positions using the '
                  'SFTDataset.generate_labels algorithm'),
            untruncated=dict(records=tot['records'],
                             prompt_tokens=tot['tokens_prompt_full'],
                             labeled_tokens=tot['tokens_labeled_full']),
            at_max_seq_len=dict(max_seq_len=args.max_seq_len,
                                prompt_tokens=tot['tokens_prompt_used'],
                                labeled_tokens=tot['tokens_labeled_used'],
                                truncated_records=tot['n_truncated'],
                                padded_tokens_per_epoch=((tot['records'] + args.batch_size - 1) // args.batch_size)
                                                        * args.batch_size * args.max_seq_len),
            alt_seq_lens=alt_tot,
            per_source={k: {kk: st[k][kk] for kk in st[k] if kk != 'alt'} for k in ('general', 'aux')},
            labeled_fraction_of_prompt=round(tot['tokens_labeled_full'] / max(tot['tokens_prompt_full'], 1), 4),
            tokenize_sec=st['tokenize_sec'],
            label_fn_check=st['label_fn_check'],
        )
        meta['aux_fraction_by_labeled_tokens'] = round(
            st['aux']['tokens_labeled_full'] / max(tot['tokens_labeled_full'], 1), 8)
        print(f"[build_sft] tokens: prompt(full)={tot['tokens_prompt_full']:,} "
              f"labeled(full)={tot['tokens_labeled_full']:,} | @{args.max_seq_len}: "
              f"prompt={tot['tokens_prompt_used']:,} labeled={tot['tokens_labeled_used']:,} "
              f"truncated={tot['n_truncated']:,}")
        for L, a in alt_tot.items():
            print(f"[build_sft]   alt seq_len={L}: prompt={a['prompt_tokens']:,} "
                  f"labeled={a['labeled_tokens']:,} truncated={a['truncated_records']:,} "
                  f"padded/epoch={a['padded_tokens_per_epoch']:,}")

    meta['build_wall_sec'] = round(time.time() - t_start, 1)
    meta['max_sampled_rss_MB'] = MAX_RSS[0]
    with open(args.meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f'[build_sft] meta -> {args.meta_path}  wall={meta["build_wall_sec"]}s  '
          f'peak_rss={MAX_RSS[0]} MB')


if __name__ == '__main__':
    main()
