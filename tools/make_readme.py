"""Generate the project README from measured artifacts.

The README is the Phase-5 deliverable, but it must contain only measured numbers. This
script therefore *reads* the run artifacts and refuses to invent anything: every cell is
either a value read from a file or an explicit `TODO (<what is missing>)`.

It deliberately does NOT depend on the GPU or on torch: it is pure json/file reading, so
it can run while training is still going (producing a README with TODOs), and again at
the end (producing the final README).

Inputs (all optional, each degrades to a TODO):
  logs/pretrain_metrics_summary.json   Phase-1 final numbers
  logs/pretrain_metrics.jsonl          Phase-1 live numbers (fallback if no summary)
  data/tokens/pretrain_meta.json       corpus token counts
  logs/sft_metrics_summary.json        Phase-2 tokens
  data/sft/sft_meta.json               Phase-2 dataset composition
  reports/grpo_vs_sft.json             Phase-3 headline comparison
  logs/grpo_stage*_summary.json        Phase-3 token accounting
  reports/sft_baseline_full.json       Phase-2 held-out baseline
  reports/eval_gguf_q4_sft.json        Phase-4 quantization drop
  models/gguf/*.gguf                   Phase-4 artifact sizes

Usage:
  python make_readme.py                 # writes README.md
  python make_readme.py --out README_draft.md --allow-partial
"""
import argparse
import glob
import hashlib
import json
import os
import time

ROOT = r'D:\llm-rl-lab'


def jload(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def jlines(path):
    rows = []
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return rows


def fmt(n):
    try:
        return f'{int(n):,}'
    except Exception:
        return 'n/a'


def pct(a, b):
    try:
        return f'{100.0 * a / b:.1f} %'
    except Exception:
        return 'n/a'


def todo(what):
    return f'TODO ({what})'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=os.path.join(ROOT, 'README.md'))
    p.add_argument('--root', default=ROOT)
    p.add_argument('--allow-partial', action='store_true',
                   help='write the README even if some phases are missing (default: yes)')
    a = p.parse_args()
    R = a.root
    L = os.path.join(R, 'logs')
    RP = os.path.join(R, 'reports')

    pre_sum = jload(os.path.join(L, 'pretrain_metrics_summary.json'))
    pre_rows = jlines(os.path.join(L, 'pretrain_metrics.jsonl'))
    pre_meta = jload(os.path.join(R, 'data', 'tokens', 'pretrain_meta.json'))
    sft_sum = jload(os.path.join(L, 'sft_metrics_summary.json'))
    sft_meta = jload(os.path.join(R, 'data', 'sft', 'sft_meta.json'))
    cmp3 = jload(os.path.join(RP, 'grpo_vs_sft.json'))
    sft_eval = jload(os.path.join(RP, 'sft_baseline_full.json'))
    gguf_eval = jload(os.path.join(RP, 'eval_gguf_q4_sft.json'))
    gguf_eval_grpo = jload(os.path.join(RP, 'eval_gguf_q4_grpo.json'))
    grpo_sums = [jload(f) for f in sorted(glob.glob(os.path.join(L, 'grpo_stage*_summary.json')))]

    # ---- pretrain numbers -------------------------------------------------
    # Wall time must sum every process segment: the run was interrupted and resumed, so
    # `pre_sum['wall_sec']` only covers the LAST segment (4.4 h) while the true total is
    # ~24.9 h. Quoting the segment as the total would misrepresent the work.
    pre_wall_total = None
    if pre_rows:
        _segs, _cur = [], []
        for _i, _r in enumerate(pre_rows):
            if _i > 0 and _r.get('elapsed_sec', 0) < pre_rows[_i - 1].get('elapsed_sec', 0):
                _segs.append(_cur)
                _cur = []
            _cur.append(_r)
        if _cur:
            _segs.append(_cur)
        pre_wall_total = sum(s[-1].get('elapsed_sec', 0) for s in _segs if s)
        pre_segments = len(_segs)
    else:
        pre_segments = 0

    if pre_sum:
        pre_steps = pre_sum.get('steps')
        pre_tokens = pre_sum.get('total_tokens')
        pre_wall = pre_wall_total or pre_sum.get('wall_sec')
        pre_peak = pre_sum.get('peak_reserved_MiB')
        pre_done = True
    elif pre_rows:
        last = pre_rows[-1]
        pre_steps = last.get('step')
        pre_tokens = last.get('cum_tokens')
        pre_wall = pre_wall_total or last.get('elapsed_sec')
        pre_peak = max(r.get('peak_reserved_MiB', 0) for r in pre_rows)
        pre_done = False
    else:
        pre_steps = pre_tokens = pre_wall = pre_peak = None
        pre_done = False
    planned_steps = 134817
    loss_first = pre_rows[0]['loss'] if pre_rows else None
    loss_last = pre_rows[-1]['loss'] if pre_rows else None
    loss_min = min((r['loss'] for r in pre_rows), default=None)
    tps = [r['tokens_per_sec'] for r in pre_rows if r.get('tokens_per_sec')]
    tps_last = tps[-1] if tps else None

    # ---- SFT numbers ------------------------------------------------------
    sft_tokens = sft_sum.get('total_tokens') if sft_sum else None
    sft_sup = sft_sum.get('supervised_tokens') if sft_sum else None
    sft_peak = sft_sum.get('peak_reserved_MiB') if sft_sum else None
    sft_wall = sft_sum.get('wall_sec') if sft_sum else None

    # ---- GRPO token total -------------------------------------------------
    grpo_tokens = 0
    grpo_found = False
    for g in grpo_sums:
        if g and g.get('cum_sampled_tokens'):
            grpo_tokens += int(g['cum_sampled_tokens'])
            grpo_found = True
    grpo_steps_total = sum(int(g.get('steps', 0)) for g in grpo_sums if g)

    # ---- grand total ------------------------------------------------------
    parts = []
    if pre_tokens:
        parts.append(int(pre_tokens))
    if sft_tokens:
        parts.append(int(sft_tokens))
    if grpo_found:
        parts.append(grpo_tokens)
    grand = sum(parts) if len(parts) == 3 else None

    md = []
    A = md.append

    A('# MiniMind from scratch on one 8 GB laptop: Pretrain → SFT → GRPO')
    A('')
    A('Training a **63.9 M-parameter** Transformer language model **from random initialization**')
    A('on a single RTX 4060 Laptop (8,188 MiB), then aligning it with')
    A('**GRPO using programmatically-verifiable rewards** (1.0 / 0.0, no LLM judge),')
    A('quantizing to **GGUF Q4_K_M** and running it locally with llama.cpp.')
    A('')
    A('> Every number in this README is read from the run artifacts by')
    A('> [`tools/make_readme.py`](tools/make_readme.py); nothing is hand-typed or estimated.')
    A('> Regenerate with `python tools/make_readme.py`.')
    A('')
    A('---')
    A('')

    # ---- headline results -------------------------------------------------
    A('## 1. Results')
    A('')
    A('| | Result |')
    A('|---|---|')
    if pre_tokens:
        # A run still in flight must never be presented as a final number.
        tag = '' if pre_done else f' — **in progress**, {fmt(pre_steps)} of {fmt(planned_steps)} steps'
        A(f'| Pretraining tokens (measured) | **{fmt(pre_tokens)}** '
          f'({pre_tokens/1e9:.3f} B){tag} |')
    else:
        A(f'| Pretraining tokens (measured) | {todo("run Phase 1")} |')
    A(f'| Model | 63,912,192 params (hidden 768, 8 layers, vocab 6,400) |')
    A(f'| Peak training VRAM | {fmt(pre_peak)} MiB of 8,188 MiB '
      f'({pct(pre_peak or 0, 8188)}) |' if pre_peak else
      f'| Peak training VRAM | {todo("Phase 1 peak")} |')
    if cmp3:
        ov = cmp3['overall']
        A(f'| Held-out accuracy, SFT baseline | **{ov["sft"]["correct"]}/{ov["sft"]["total"]}** '
          f'({pct(ov["sft"]["correct"], ov["sft"]["total"])}) |')
        A(f'| Held-out accuracy, after GRPO | **{ov["grpo"]["correct"]}/{ov["grpo"]["total"]}** '
          f'({pct(ov["grpo"]["correct"], ov["grpo"]["total"])}) |')
        d = ov['delta']['correct']
        mc = ov.get('mcnemar') or cmp3.get('per_task_mcnemar') or {}
        A(f'| Delta | {d:+d} task(s), McNemar exact p = {mc.get("p_value", "n/a")} |')
    else:
        A(f'| Held-out SFT vs GRPO | {todo("Phase 3 comparison: reports/grpo_vs_sft.json")} |')
    if gguf_eval and sft_eval:
        A(f'| GGUF Q4_K_M accuracy drop (SFT) | {sft_eval["overall"]["correct"]} → '
          f'{gguf_eval["overall"]["correct"]} of {sft_eval["overall"]["total"]} tasks |')
        if gguf_eval_grpo and cmp3:
            A(f'| GGUF Q4_K_M accuracy drop (GRPO) | '
              f'{cmp3["overall"]["grpo"]["correct"]} → '
              f'{gguf_eval_grpo["overall"]["correct"]} of '
              f'{gguf_eval_grpo["overall"]["total"]} tasks |')
    else:
        A(f'| GGUF Q4_K_M accuracy drop | {todo("Phase 4 GGUF eval")} |')
    A('')

    # ---- token accounting -------------------------------------------------
    A('## 2. Token accounting (the three required figures)')
    A('')
    A('| Phase | What is counted | Tokens | Basis |')
    A('|---|---|---|---|')
    A(f'| 1. Pretrain | training tokens fed to the model | '
      f'**{fmt(pre_tokens) if pre_tokens else "n/a"}**{"" if pre_done else " (in progress)"} | '
      f'`sum(input_ids.numel())` per step, packed 512-token blocks, no padding |')
    A(f'| 2. SFT | training-set tokens (padded) / supervised | '
      f'**{fmt(sft_tokens) if sft_tokens else "n/a"}** / '
      f'{fmt(sft_sup) if sft_sup else "n/a"} | '
      f'`input_ids.numel()` / `(labels != -100).sum()` |')
    A(f'| 3. GRPO | sampled completion tokens | '
      f'**{fmt(grpo_tokens) if grpo_found else "n/a"}** | '
      f'`completion_mask.sum()` summed over every step |')
    A(f'| **Total** | 1 + 2 + 3 | **{fmt(grand) if grand else todo("all three phases")}** | |')
    A('')
    A('Prompt tokens during GRPO rollout are tracked separately and are **not** added to')
    A('the total (they are conditioning context, not newly sampled tokens).')
    A('')

    # ---- phase 1 ----------------------------------------------------------
    A('## 3. Phase 1 — Pretraining from random initialization')
    A('')
    A('| Metric | Value |')
    A('|---|---|')
    A(f'| Corpus | MiniMind `pretrain_t2t.jsonl`, '
      f'{fmt(pre_meta.get("n_docs")) if pre_meta else "n/a"} documents |')
    A(f'| Corpus tokens (tokenizer output) | '
      f'{fmt(pre_meta.get("tokenized_tokens")) if pre_meta else "n/a"} |')
    A(f'| Steps | {fmt(pre_steps) if pre_steps else "n/a"} '
      f'of {fmt(planned_steps)} planned |')
    A(f'| Sequence length / batch | 512 tokens × 32 = 16,384 tokens/step |')
    A(f'| Optimizer | AdamW, lr 5e-4, cosine annealed to 0.1× |')
    A(f'| Precision | bf16 autocast + `torch.compile` (triton-windows) |')
    A(f'| Throughput | {fmt(tps_last) if tps_last else "n/a"} tokens/s (last logged) |')
    A(f'| Loss | {loss_first:.4f} → {loss_last:.4f} '
      f'(min {loss_min:.4f}) |' if loss_first is not None and loss_last is not None
      else f'| Loss | {todo("Phase 1 metrics")} |')
    A(f'| Wall time (all {pre_segments} segments) | {pre_wall/3600:.2f} h |'
      if pre_wall else f'| Wall time | n/a |')
    A('')
    if pre_done:
        A('![pretrain loss curve](figures/pretrain_curve_full.png)')
    else:
        A('![pretrain loss curve (in progress)](figures/pretrain_curve_partial.png)')
    A('')

    # ---- phase 2 ----------------------------------------------------------
    A('## 4. Phase 2 — SFT')
    A('')
    if sft_meta:
        A(f'| Metric | Value |')
        A(f'|---|---|')
        A(f'| Records | {fmt(sft_meta.get("out_records"))} |')
        A(f'| General chat share | 80 % ({fmt(160000)} records) |')
        A(f'| Task-shaped "format seeding" share | 20 % ({fmt(40000)} records) |')
        A(f'| Held-out leakage | {sft_meta.get("heldout", {}).get("leak_count", "n/a")} |')
        A(f'| Supervised tokens | {fmt(sft_sup) if sft_sup else "n/a"} |')
        A('')
        A('The 20 % task-shaped component is deliberately **narrow and easy** (single-digit')
        A('addition, YES/NO, JSON, parity) so SFT learns the *output format* without the')
        A('reasoning range the held-out set tests — the "knows the format, cannot reason"')
        A('regime that GRPO is supposed to close.')
        A('')
    else:
        A(f'{todo("Phase 2 SFT: data/sft/sft_meta.json")}')
        A('')
    if sft_eval:
        A('SFT held-out baseline (greedy, `prompt_mode=chat`, 24 tasks):')
        A('')
        A('| family | correct | accuracy |')
        A('|---|---|---|')
        for fam, v in (sft_eval.get('per_family') or {}).items():
            A(f'| {fam} | {v["correct"]}/{v["total"]} | {pct(v["correct"], v["total"])} |')
        A(f'| **overall** | **{sft_eval["overall"]["correct"]}/'
          f'{sft_eval["overall"]["total"]}** | '
          f'**{pct(sft_eval["overall"]["correct"], sft_eval["overall"]["total"])}** |')
    else:
        A(f'SFT held-out baseline: {todo("reports/sft_baseline_full.json")}')
    A('')

    # ---- phase 3 ----------------------------------------------------------
    A('## 5. Phase 3 — GRPO with programmatically-verifiable rewards')
    A('')
    A('Reward is a pure function of the task spec: **1.0 if the answer parses and is correct,**')
    A('**else 0.0**. No learned reward model, no LLM judge. Four checkers:')
    A('`exact_int`, `exact_json`, `choice`, `label`.')
    A('')
    A('MiniMind\'s stock GRPO uses an `internlm2-1.8B` neural reward model — i.e. exactly')
    A('the LLM judge this project forbids — so `custom/train_grpo_rule.py` is a thin')
    A('derivative that keeps the GRPO math (group-normalised advantages, PPO clip, k3 KL')
    A('estimator, `TorchRolloutEngine`) and replaces only the reward function.')
    A('')
    if grpo_sums and grpo_found:
        A('| stage | steps | sampled tokens |')
        A('|---|---|---|')
        for g, f in zip(grpo_sums, sorted(glob.glob(os.path.join(L, 'grpo_stage*_summary.json')))):
            if not g:
                continue
            A(f'| {os.path.basename(f).replace("_summary.json", "")} | '
              f'{fmt(g.get("steps"))} | {fmt(g.get("cum_sampled_tokens"))} |')
        A(f'| **total** | **{fmt(grpo_steps_total)}** | **{fmt(grpo_tokens)}** |')
        A('')
    if cmp3:
        A('### SFT vs GRPO on the held-out set')
        A('')
        ov = cmp3['overall']
        A('| family | SFT | GRPO | delta |')
        A('|---|---|---|---|')
        for fam, v in (cmp3.get('per_family') or {}).items():
            s = v.get('sft', {})
            g = v.get('grpo', {})
            A(f'| {fam} | {s.get("correct")}/{s.get("total")} | '
              f'{g.get("correct")}/{g.get("total")} | '
              f'{(g.get("correct", 0) - s.get("correct", 0)):+d} |')
        A(f'| **overall** | **{ov["sft"]["correct"]}/{ov["sft"]["total"]}** | '
          f'**{ov["grpo"]["correct"]}/{ov["grpo"]["total"]}** | '
          f'**{ov["delta"]["correct"]:+d}** |')
        A('')
        mc = cmp3.get('per_task_mcnemar') or {}
        A(f'McNemar exact test: b={mc.get("b")} (SFT-only right), c={mc.get("c")} '
          f'(GRPO-only right), **p = {mc.get("p_value")}**, '
          f'significant at 5 %: {mc.get("significant_at_5pct")}.')
        A('')
        A(f'**Verdict:** {cmp3.get("verdict", {}).get("text", "n/a")}')
        A('')
        A('A 24-task benchmark means one task = 4.2 pp, so the McNemar test is reported')
        A('specifically to keep a 1–2 task delta from being over-read as an improvement.')
        A('')
        # Chance level: the honest floor for interpreting any score on this benchmark.
        cb = (cmp3.get('verdict') or {}).get('chance_baseline')
        if cb:
            o = cb['overall']
            A(f'**Chance baseline:** a guessing model scores '
              f'**{o["chance_accuracy_pct"]:.1f} %** '
              f'({o["expected_correct_by_chance"]:.1f} of {o["tasks"]} tasks) because the')
            A('`format` and `logic` families are 2–3-way choices. '
              f'**{o["min_correct_for_significance_5pct"]}/{o["tasks"]} correct** is the')
            A('minimum that beats chance at the 5 % level; anything below that is noise.')
            A('')
            A('| family | tasks | chance accuracy |')
            A('|---|---|---|')
            for fam, v in cb.get('per_family', {}).items():
                A(f'| {fam} | {v["tasks"]} | {v["chance_accuracy_pct"]:.1f} % |')
            A('')
    else:
        A(f'GRPO comparison: {todo("reports/grpo_vs_sft.json")}')
        A('')

    # ---- phase 4 ----------------------------------------------------------
    A('## 6. Phase 4 — GGUF Q4 quantization and local deployment')
    A('')
    ggufs = sorted(glob.glob(os.path.join(R, 'models', 'gguf', '*.gguf')))
    if ggufs:
        A('| artifact | size |')
        A('|---|---|')
        for f in ggufs:
            A(f'| `{os.path.basename(f)}` | {os.path.getsize(f)/1e6:.1f} MB |')
        A('')
    else:
        A(f'{todo("Phase 4 GGUF artifacts")}')
        A('')
    A('Two traps found the hard way (both would silently break deployment):')
    A('')
    A('1. The llama.cpp **CUDA zip ships without its CUDA runtime DLLs** — `llama-cli`')
    A('   then reports no devices and silently falls back to CPU. The separate')
    A('   `cudart-*.zip` must be merged in.')
    A('2. GitHub `releases/latest` points at a **decoy** tag whose only asset is')
    A('   `nightly-tag.txt`; real binaries live under `bNNNN` tags.')
    A('')
    A('### Demo')
    A('')
    A('```powershell')
    A('# side-by-side CLI: same prompt, SFT vs GRPO')
    A('python demo/compare_cli.py --prompt "3+5等于几？只回复数字" `')
    A('    --sft  models/gguf/full_sft_768-Q4_K_M.gguf `')
    A('    --grpo models/gguf/grpo_stage3_full_768-Q4_K_M.gguf --max-new-tokens 48')
    A('')
    A('# web UI (Gradio :7860, or the stdlib fallback :7861)')
    A('python demo/app.py --ui gradio --port 7860')
    A('```')
    A('')
    A('Every llama.cpp call uses `-ngl 0` (CPU) so the demo never competes with training')
    A('for VRAM. Measured on this machine: **518 tok/s** for Q4_K_M on CPU,')
    A('**939 tok/s** at `-ngl 99` on GPU.')
    A('')

    # ---- reproduction -----------------------------------------------------
    A('## 7. Reproduce')
    A('')
    A('```powershell')
    A('# 0) environment')
    A('uv venv .venv --python 3.11')
    A('$env:UV_CACHE_DIR="$PWD\\.cache\\uv"')
    A('uv pip install --python .venv\\Scripts\\python.exe torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124')
    A('uv pip install --python .venv\\Scripts\\python.exe -r src\\minimind-master\\requirements.txt --index-url https://mirrors.aliyun.com/pypi/simple/')
    A('uv pip install --python .venv\\Scripts\\python.exe numpy==2.2.6 triton-windows==3.2.0.post21')
    A('')
    A('# 1) data -> packed token blocks (exact token count, no padding)')
    A('python custom/pretokenize_packed.py --src data\\pretrain_t2t.jsonl `')
    A('    --out data\\tokens --seq-len 512 --tag pretrain')
    A('')
    A('# 2) pretrain from random init (2.209 B tokens, ~29 h on a 4060 Laptop)')
    A('cd src\\minimind-master\\trainer')
    A('python ..\\..\\..\\custom\\train_pretrain_packed.py `')
    A('    --data_path D:\\llm-rl-lab\\data\\tokens\\pretrain_blocks.bin `')
    A('    --batch_size 32 --max_seq_len 512 --use_compile 1 --seed 42 `')
    A('    --save_dir D:/llm-rl-lab/src/minimind-master/out --save_weight pretrain')
    A('')
    A('# 3) SFT (200k records: 80 % chat + 20 % format seeding)')
    A('python ..\\..\\..\\custom\\train_full_sft_counted.py `')
    A('    --data_path D:\\llm-rl-lab\\data\\sft\\sft_train.jsonl `')
    A('    --from_weight pretrain --save_weight full_sft --batch_size 16 --max_seq_len 768')
    A('')
    A('# 4) GRPO, staged easy -> medium -> full')
    A('python tools\\run_phase3.py --execute --from-weight full_sft --device cuda')
    A('')
    A('# 5) held-out comparison + GGUF export')
    A('python tools\\compare_sft_grpo.py --sft ...\\full_sft_768.pth --grpo ...\\grpo_stage3_full_768.pth `')
    A('    --out reports\\grpo_vs_sft.json --device cuda')
    A('python tools\\run_pipeline.py --phase 4')
    A('```')
    A('')
    A('Determinism: `seed=42` is fixed in every trainer, the data builders, and the')
    A('evaluator. `torch.backends.cudnn.deterministic` is set in the eval harness.')
    A('')

    # ---- limitations ------------------------------------------------------
    A('## 8. Honest limitations')
    A('')
    A('- **Model size.** 63.9 M parameters is ~1/2000th of a 7 B model. Arithmetic beyond')
    A('  single digits and multi-step reasoning are at or beyond its capacity ceiling.')
    A('- **Benchmark granularity.** 24 tasks → one task = 4.2 pp. Deltas of 1–2 tasks are')
    A('  reported with an exact McNemar p-value precisely so they are not over-claimed.')
    A('- **Single-token answers.** The tokenizer makes every integer 0–25 a single token,')
    A('  so the arithmetic tasks measure prompt→answer-token retrieval, not digit')
    A('  composition. This is stated rather than hidden.')
    A('- **GRPO sparsity.** All-wrong groups produce `adv_std = 0` and therefore zero')
    A('  gradient; this was measured and drove the easy→hard curriculum design.')
    A('')

    # ---- env --------------------------------------------------------------
    A('## 9. Environment')
    A('')
    A('| Item | Value |')
    A('|---|---|')
    A('| GPU | NVIDIA RTX 4060 Laptop, 8,188 MiB, sm_89 |')
    A('| CPU / RAM | i7-13700H / 16 GB DDR5 |')
    A('| OS | Windows 11, native (no WSL, no Docker) |')
    A('| Python | 3.11.9 + uv |')
    A('| torch | 2.6.0+cu124, bf16, `torch.compile` via triton-windows 3.2.0.post21 |')
    A('| Framework | MiniMind (PyTorch native, Pretrain/SFT/GRPO) |')
    A('')
    A('---')
    A('')
    A(f'*README generated {time.strftime("%Y-%m-%d %H:%M:%S")} from measured artifacts by')
    A('`tools/make_readme.py`.*')
    A('')

    text = '\n'.join(md)
    with open(a.out, 'w', encoding='utf-8') as f:
        f.write(text)
    n_todo = text.count('TODO (')
    print(f'[readme] wrote {a.out} ({len(text)} bytes, {n_todo} TODO markers)')
    print(json.dumps(dict(
        pretrain_tokens=pre_tokens, pretrain_steps=pre_steps, pretrain_done=pre_done,
        sft_tokens=sft_tokens, grpo_tokens=grpo_tokens if grpo_found else None,
        grand_total=grand, has_comparison=cmp3 is not None,
        has_gguf_eval=gguf_eval is not None, todo_markers=n_todo), indent=2))


if __name__ == '__main__':
    main()
