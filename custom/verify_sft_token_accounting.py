#!/usr/bin/env python
"""Independent verification of `train_full_sft_counted.py`'s token accounting.

Nothing here trusts the trainer: it re-reads the metrics JSONL that the trainer wrote,
then re-tokenizes the very same dataset records itself (same `dataset.lm_dataset.SFTDataset`,
same RNG state, same index order recorded by `--dump_batches`) and recomputes

    tokens_total_step      = input_ids.numel()            = batch * max_seq_len
    tokens_supervised_step = (labels != -100).sum()

from the returned tensors. It then compares per-step and cumulative values against what

# --- project root, derived from this file's location (portable across machines) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
the trainer logged.

It also checks the hard invariants:
  * cum_tokens_total is strictly monotonic and equals steps * batch * max_seq_len
    (SFTDataset pads every sample to max_length, so this must hold exactly);
  * cum_tokens_supervised is monotonic, strictly less than cum_tokens_total;
  * every logged loss is finite;
  * the summary JSON totals agree with the last metrics row.

Usage:
  python custom/verify_sft_token_accounting.py \
      --metrics logs/sft_val/cpu_val_metrics.jsonl \
      --batches logs/sft_val/cpu_val_batches.json \
      --data_path logs/sft_val/sft_slice200.jsonl \
      --max_seq_len 128 --hidden_size 128 --num_hidden_layers 2 \
      --seed 42 --out logs/sft_val/cpu_val_reconciliation.json
"""
import argparse
import json
import os
import sys

REPO = os.path.join(PROJECT_ROOT, 'src', 'minimind-master')
sys.path.insert(0, REPO)

import random

import numpy as np
import torch

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import SFTDataset, pre_processing_chat, post_processing_chat
from transformers import AutoTokenizer


class SFTDatasetKeepThink(SFTDataset):
    """Same class the trainer uses: repo __getitem__ with remove_empty_think=False."""

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt, remove_empty_think=False)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def seed_all(seed):
    """Same seeding order as the trainer at the moment it touches the dataset."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metrics', required=True)
    p.add_argument('--batches', required=True)
    p.add_argument('--data_path', required=True)
    p.add_argument('--max_seq_len', type=int, required=True)
    p.add_argument('--hidden_size', type=int, default=128)
    p.add_argument('--num_hidden_layers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tokenizer_path', default=f'{REPO}/model')
    p.add_argument('--out', default='')
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.metrics, encoding='utf-8') if l.strip()]
    dump = json.load(open(args.batches, encoding='utf-8'))
    batches = dump['indices_per_step']
    bs = dump['batch_size']

    # ---- replicate the trainer's state: seed -> model -> dataset, then index in order ----
    seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    _ = MiniMindForCausalLM(lm_config)          # same construction order as init_model()
    ds = SFTDatasetKeepThink(args.data_path, tokenizer, max_length=args.max_seq_len)

    results = []
    mismatches = 0
    by_step = {}
    for step_no, batch in enumerate(batches, start=1):
        tot = 0
        sup = 0
        for idx in batch:
            input_ids, labels = ds[idx]
            tot += int(input_ids.numel())
            sup += int((labels != -100).sum().item())
        by_step[step_no] = (tot, sup)

    # `--log_interval` may skip steps, so accumulate over EVERY batch up to the last
    # logged step and only compare at the steps that were actually logged.
    logged = {row['step']: row for row in rows}
    last_step = max(logged)
    cum_tot = cum_sup = 0
    for step_no in range(1, last_step + 1):
        tot, sup = by_step[step_no]
        cum_tot += tot
        cum_sup += sup
        row = logged.get(step_no)
        if row is None:
            continue
        ok_tot = (row['cum_tokens_total'] == cum_tot)
        ok_sup = (row['cum_tokens_supervised'] == cum_sup)
        if not (ok_tot and ok_sup):
            mismatches += 1
        results.append(dict(
            step=step_no,
            trainer_cum_tokens_total=row['cum_tokens_total'],
            manual_cum_tokens_total=cum_tot,
            trainer_cum_tokens_supervised=row['cum_tokens_supervised'],
            manual_cum_tokens_supervised=cum_sup,
            manual_step_tokens_total=tot, manual_step_tokens_supervised=sup,
            match_total=ok_tot, match_supervised=ok_sup,
        ))

    # ---- invariants ----
    steps = last_step
    expected_total = steps * bs * args.max_seq_len
    totals = [r['cum_tokens_total'] for r in rows]
    sups = [r['cum_tokens_supervised'] for r in rows]
    losses = [r['loss'] for r in rows]
    checks = dict(
        steps_logged=steps,
        expected_total_tokens=expected_total,
        trainer_total_tokens=totals[-1],
        total_equals_steps_batch_seq=(totals[-1] == expected_total),
        cum_total_strictly_increasing=all(b > a for a, b in zip(totals, totals[1:])),
        cum_supervised_strictly_increasing=all(b > a for a, b in zip(sups, sups[1:])),
        supervised_less_than_total=all(s < t for s, t in zip(sups, totals)),
        all_losses_finite=all(np.isfinite(x) for x in losses),
        loss_first=losses[0], loss_last=losses[-1],
        loss_min=min(losses), loss_max=max(losses),
        supervised_fraction=round(sups[-1] / totals[-1], 6),
        per_step_match=all(r['match_total'] and r['match_supervised'] for r in results),
        mismatches=mismatches,
        records_in_dataset=len(ds),
        batches_dumped=len(batches),
    )

    summary_path = args.metrics.replace('.jsonl', '_summary.json')
    if os.path.exists(summary_path):
        summ = json.load(open(summary_path, encoding='utf-8'))
        checks['summary_total_tokens'] = summ['total_tokens']
        checks['summary_supervised_tokens'] = summ['supervised_tokens']
        checks['summary_matches_metrics'] = (summ['total_tokens'] == totals[-1]
                                             and summ['supervised_tokens'] == sups[-1])
        checks['summary_wall_sec'] = summ['wall_sec']
        checks['summary_peak_reserved_MiB'] = summ['peak_reserved_MiB']

    report = dict(metrics=args.metrics, batches=args.batches, data_path=args.data_path,
                  max_seq_len=args.max_seq_len, batch_size=bs, checks=checks,
                  per_step=results)
    print(json.dumps({k: v for k, v in checks.items()}, indent=2))
    for r in results:
        print('step %d: total trainer=%d manual=%d %s | supervised trainer=%d manual=%d %s' % (
            r['step'], r['trainer_cum_tokens_total'], r['manual_cum_tokens_total'],
            'OK' if r['match_total'] else 'MISMATCH',
            r['trainer_cum_tokens_supervised'], r['manual_cum_tokens_supervised'],
            'OK' if r['match_supervised'] else 'MISMATCH'))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print('report ->', args.out)
    ok = (checks['per_step_match'] and checks['total_equals_steps_batch_seq']
          and checks['cum_total_strictly_increasing'] and checks['cum_supervised_strictly_increasing']
          and checks['supervised_less_than_total'] and checks['all_losses_finite']
          and checks.get('summary_matches_metrics', True))
    print('RECONCILIATION', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
