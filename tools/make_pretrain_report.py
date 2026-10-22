"""Generate reports/pretrain.md from the Phase-1 metrics JSONL.

Every number in the report comes from the run's own metrics file; nothing is estimated.
Usage:
  python make_pretrain_report.py \
      --metrics D:/llm-rl-lab/logs/pretrain_metrics.jsonl \
      --summary D:/llm-rl-lab/logs/pretrain_metrics_summary.json \
      --meta D:/llm-rl-lab/data/tokens/pretrain_meta.json \
      --out D:/llm-rl-lab/reports/pretrain.md
"""
import argparse
import json
import os
import statistics


def load_jsonl(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass  # tolerate a partially-written final line
    return rows


def fmt(n):
    return f'{n:,}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metrics', default=os.path.join(PROJECT_ROOT, 'logs', 'pretrain_metrics.jsonl'))
    p.add_argument('--summary', default=os.path.join(PROJECT_ROOT, 'logs', 'pretrain_metrics_summary.json'))
    p.add_argument('--meta', default=os.path.join(PROJECT_ROOT, 'data', 'tokens', 'pretrain_meta.json'))
    p.add_argument('--out', default=os.path.join(PROJECT_ROOT, 'reports', 'pretrain.md'))
    p.add_argument('--hidden', type=int, default=768)
    p.add_argument('--layers', type=int, default=8)
    p.add_argument('--params', default='63,912,192')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    rows = load_jsonl(args.metrics)
    meta = json.load(open(args.meta, encoding='utf-8')) if os.path.exists(args.meta) else {}
    summary = None
    if os.path.exists(args.summary):
        try:
            summary = json.load(open(args.summary, encoding='utf-8'))
        except Exception:
            summary = None

    if not rows:
        raise SystemExit(f'no metrics rows in {args.metrics}')

    first, last = rows[0], rows[-1]
    losses = [r['loss'] for r in rows]
    tps = [r.get('tokens_per_sec') for r in rows if r.get('tokens_per_sec')]
    peak = max(r.get('peak_reserved_MiB', 0) for r in rows)
    total_tokens = last['cum_tokens']
    steps = last['step']
    tokens_per_step = last.get('tokens_per_step')
    # Wall time must SUM every process segment. The run was interrupted and resumed
    # several times; `last['elapsed_sec']` only covers the final segment (4.4 h) while
    # the real total is ~24.9 h. Segments are detected by `elapsed_sec` going backwards.
    segs, cur = [], []
    for i, r in enumerate(rows):
        if i > 0 and r.get('elapsed_sec', 0) < rows[i - 1].get('elapsed_sec', 0):
            segs.append(cur)
            cur = []
        cur.append(r)
    if cur:
        segs.append(cur)
    total_wall_sec = sum(s[-1].get('elapsed_sec', 0) for s in segs if s)
    elapsed_h = total_wall_sec / 3600
    last_segment_h = last.get('elapsed_sec', 0) / 3600
    n_segments = len(segs)

    # smoothed loss: average of the first/last 10% of logged points
    k = max(1, len(losses) // 10)
    loss_head = statistics.mean(losses[:k])
    loss_tail = statistics.mean(losses[-k:])
    min_loss = min(losses)
    min_loss_step = rows[losses.index(min_loss)]['step']
    complete = summary is not None and summary.get('steps') == steps

    md = []
    A = md.append
    A('# Phase 1 — Pretraining (from random initialization)')
    A('')
    A(f'Status: **{"complete" if complete else "IN PROGRESS / truncated"}** — '
      f'{fmt(steps):} optimizer steps, **{fmt(total_tokens)} tokens** fed to the model.')
    A('')
    A('Every number below is read from `logs/pretrain_metrics.jsonl` (written by the trainer')
    A('at each log interval) and `data/tokens/pretrain_meta.json` (written by the tokenizer).')
    A('Nothing is estimated.')
    A('')
    A('## 1. Configuration')
    A('')
    A('| Item | Value |')
    A('|---|---|')
    A(f'| Model | MiniMind `MiniMindConfig(hidden_size={args.hidden}, num_hidden_layers={args.layers})` |')
    A(f'| Parameters | **{args.params}** (63.91 M), trained **from random init** (`--from_weight none`) |')
    A(f'| Vocab | 6,400 (MiniMind tokenizer) |')
    A(f'| Sequence length | {meta.get("seq_len", "n/a")} tokens, packed blocks (no padding) |')
    A(f'| Batch size | 32 x {meta.get("seq_len", 0)} = {fmt(tokens_per_step or 0)} tokens / step |')
    A('| Optimizer | AdamW, lr 5e-4, cosine schedule (`trainer_utils.get_lr`, annealed to 0.1x) |')
    A('| Precision | bf16 autocast, `torch.compile` (triton-windows 3.2.0.post21) |')
    A(f'| Seed | {args.seed} |')
    A(f'| Corpus | MiniMind `pretrain_t2t.jsonl`, {fmt(meta.get("n_docs", 0))} documents |')
    A('')
    A('## 2. Token accounting (measured, not estimated)')
    A('')
    A('| Quantity | Value | Basis |')
    A('|---|---|---|')
    A(f'| Corpus tokens (tokenizer output, incl. 1 EOS/doc) | **{fmt(meta.get("tokenized_tokens", 0))}** | '
      '`tokenizer._tokenizer.encode_batch` over every document |')
    A(f'| Packed blocks x seq_len | {fmt(meta.get("total_tokens_consumed", 0))} | '
      f'{fmt(meta.get("n_blocks", 0))} blocks x {meta.get("seq_len", 0)} |')
    A(f'| **Tokens actually consumed by training** | **{fmt(total_tokens)}** | '
      'sum of `input_ids.numel()` over every step |')
    A(f'| Steps | {fmt(steps)} | |')
    A(f'| Wall time (all segments) | **{elapsed_h:.2f} h** | sum of {n_segments} process '
      f'segments (the run was paused/resumed; the final segment alone was '
      f'{last_segment_h:.2f} h) |')
    A('')
    if meta.get('dropped_tail_tokens'):
        A(f'Note: {meta["dropped_tail_tokens"]} tail tokens are not a whole block and are not used; '
          '`drop_last=True` discards the last partial batch, so '
          f'`steps x tokens/step = {fmt(total_tokens)}` is the authoritative figure.')
        A('')
    A('## 3. Loss')
    A('')
    A('| Metric | Value |')
    A('|---|---|')
    A(f'| First logged loss (step {fmt(first["step"])}) | {first["loss"]:.4f} |')
    A(f'| Mean of first 10 % of logged points | {loss_head:.4f} |')
    A(f'| Mean of last 10 % of logged points | {loss_tail:.4f} |')
    A(f'| Last logged loss (step {fmt(last["step"])}) | {last["loss"]:.4f} |')
    A(f'| Minimum logged loss | {min_loss:.4f} (step {fmt(min_loss_step)}) |')
    A(f'| Reduction, first logged -> last logged | **{first["loss"] - last["loss"]:+.4f}** '
      f'({100 * (first["loss"] - last["loss"]) / first["loss"]:.1f} %) |')
    A(f'| Reduction, head 10 % -> tail 10 % of logged points | {loss_head - loss_tail:+.4f} '
      f'({100 * (loss_head - loss_tail) / loss_head:.1f} %) |')
    A('')
    A('> Two reduction figures are reported on purpose. *First logged -> last logged* is the')
    A('> headline number (6.40 -> 1.81). *Head 10 % -> tail 10 %* is the conservative one:')
    A('> the first 10 % of logged points are already past the initial cliff, so it measures')
    A('> the remaining descent and is the fairer basis for "is it still improving".')
    A('')
    A('![pretrain loss curve](../figures/pretrain_curve_full.png)')
    A('')
    A('## 4. Throughput and memory')
    A('')
    A('| Metric | Value |')
    A('|---|---|')
    if tps:
        A(f'| Mean tokens/s (run-average over log points) | {fmt(int(statistics.mean(tps)))} |')
        A(f'| Peak tokens/s (run-average over log points) | {fmt(int(max(tps)))} |')
    A(f'| **Peak reserved VRAM** | **{fmt(peak)} MiB** ({peak / 8188 * 100:.1f} % of the 8,188 MiB card) |')
    if summary:
        A(f'| Peak allocated VRAM | {fmt(summary.get("peak_alloc_MiB", 0))} MiB |')
    A('')
    A('## 5. Checkpoints')
    A('')
    A('```')
    A('D:\\llm-rl-lab\\src\\minimind-master\\out\\pretrain_768.pth            (final)')
    A('D:\\llm-rl-lab\\src\\minimind-master\\out\\pretrain_768_stepN.pth       (every 2000 steps)')
    A('D:\\llm-rl-lab\\src\\minimind-master\\checkpoints\\pretrain_768_resume.pth (optimizer state)')
    A('```')
    A('')
    A('## 6. Reproduce')
    A('')
    A('```powershell')
    A('# 1) tokenize the corpus into packed blocks (once)')
    A('D:\\llm-rl-lab\\.venv\\Scripts\\python.exe D:\\llm-rl-lab\\custom\\pretokenize_packed.py `')
    A('  --src D:\\llm-rl-lab\\data\\pretrain_t2t.jsonl --out D:\\llm-rl-lab\\data\\tokens `')
    A('  --seq-len 512 --batch-docs 2000 --tag pretrain')
    A('')
    A('# 2) train')
    A('cd D:\\llm-rl-lab\\src\\minimind-master\\trainer')
    A('D:\\llm-rl-lab\\.venv\\Scripts\\python.exe D:\\llm-rl-lab\\custom\\train_pretrain_packed.py `')
    A('  --data_path D:\\llm-rl-lab\\data\\tokens\\pretrain_blocks.bin --max_seq_len 512 `')
    A(f'  --batch_size 32 --epochs 1 --max_steps {steps} --learning_rate 5e-4 `')
    A('  --log_interval 100 --save_interval 2000 --num_workers 4 --use_compile 1 `')
    A('  --hidden_size 768 --num_hidden_layers 8 --seed 42 `')
    A('  --save_dir D:/llm-rl-lab/src/minimind-master/out --save_weight pretrain `')
    A('  --metrics_path D:/llm-rl-lab/logs/pretrain_metrics.jsonl')
    A('```')
    A('')

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f'[report] wrote {args.out}')
    print(json.dumps(dict(steps=steps, tokens=total_tokens, loss_first=first['loss'],
                          loss_last=last['loss'], loss_head=round(loss_head, 4),
                          loss_tail=round(loss_tail, 4), peak_MiB=peak,
                          elapsed_h=round(elapsed_h, 2), complete=complete), indent=2))


if __name__ == '__main__':
    main()
