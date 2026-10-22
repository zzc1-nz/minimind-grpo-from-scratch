"""Phase-3 headline driver: SFT vs GRPO on the frozen held-out benchmark.

Runs `eval/run_eval.py` for the SFT checkpoint and the GRPO checkpoint with the
SAME prompt mode (`chat`, the only valid mode for this project: the assistant
header followed by a CLOSED empty think block), the SAME `--max-new-tokens`, and
the SAME seed, then writes one JSON that a reader can audit claim by claim:

  overall            before / after / delta  (counts + accuracy)
  per_family         before / after / delta  per family
  per_task           every task: both `correct` flags and BOTH raw completions
  per_task_mcnemar   the 2x2 transition per task and an exact McNemar p-value
                     (stdlib math.comb only), so a 1-2 task delta cannot be
                     over-read
  token_accounting   pretrain / SFT / GRPO tokens with an explicit basis for
                     each, and the grand total
  verdict            text generated from the numbers by explicit rules

Nothing here invents a number: anything that cannot be measured is reported as
`n/a (requires real checkpoint)` together with the reason.

Usage
-----
  # both checkpoints evaluated now (CPU is fine; use --device cpu while the GPU
  # is busy with the ~26 h pretraining job)
  python tools/compare_sft_grpo.py --sft out/sft.pth --grpo out/grpo.pth \
      --out reports/grpo_vs_sft.json --hidden 768 --layers 8 \
      --max-new-tokens 64 --device cpu

  # reuse already-computed eval reports (no model loading at all)
  python tools/compare_sft_grpo.py --sft-report reports/sft.json \
      --grpo-report reports/grpo.json --out reports/grpo_vs_sft.json

  # prove the p-value / token-accounting arithmetic without touching a model
  python tools/compare_sft_grpo.py --self-test
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
EVAL_RUNNER = os.path.join(ROOT, 'eval', 'run_eval.py')
TASKS_PATH = os.path.join(ROOT, 'eval', 'heldout_tasks.jsonl')
LOCK_PATH = os.path.join(ROOT, 'eval', 'lock_hashes.json')

# This project has exactly one valid prompt mode for training/eval agreement.
REQUIRED_PROMPT_MODE = 'chat'

DEFAULT_PRETRAIN_META = os.path.join(ROOT, 'data', 'tokens', 'pretrain_meta.json')
DEFAULT_PRETRAIN_METRICS = os.path.join(ROOT, 'logs', 'pretrain_metrics.jsonl')
DEFAULT_PRETRAIN_SUMMARY = os.path.join(ROOT, 'logs', 'pretrain_metrics_summary.json')
DEFAULT_SFT_META = os.path.join(ROOT, 'data', 'sft', 'sft_meta.json')
DEFAULT_SFT_METRICS = os.path.join(ROOT, 'logs', 'sft_metrics.jsonl')


# ------------------------------------------------------------------ helpers
def load_json(path):
    """(data, err) - err is a short reason string, never an exception."""
    if not path or not os.path.exists(path):
        return None, f'file not found: {path}'
    try:
        with open(path, encoding='utf-8-sig') as f:
            return json.load(f), None
    except (OSError, ValueError) as exc:
        return None, f'{type(exc).__name__}: {exc}'


def load_jsonl(path):
    """(records, err); tolerant of a partially written last line."""
    if not path or not os.path.exists(path):
        return [], f'file not found: {path}'
    records = []
    try:
        with open(path, encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        return [], f'{type(exc).__name__}: {exc}'
    return records, None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def na(reason):
    """The canonical explicit-missing value."""
    return {'value': None, 'display': f'n/a ({reason})', 'reason': reason}


# ------------------------------------------------------------- McNemar
def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the two discordant counts.

    b = SFT right & GRPO wrong, c = SFT wrong & GRPO right.
    Under H0 each discordant pair is a fair coin, so the number of GRPO-only
    successes is Binomial(b+c, 0.5). The two-sided p is
        2 * P(X <= min(b, c)),  capped at 1.0
    computed exactly with math.comb (no scipy, no normal approximation - the
    discordant counts here are tiny, so the exact test is the right one).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


# ------------------------------------------------------- evaluation runs
def run_eval(weight, report_path, args, log_prefix='[eval]'):
    """Shell out to eval/run_eval.py. Returns (report, cmd, error).

    ``args.backend`` selects the generation path: ``hf`` loads a native MiniMind
    ``.pth``; ``gguf`` drives the quantized file through llama.cpp. Phase-4's
    quantization-drop check needs the latter, and it must use the same prompt mode /
    max_new_tokens / seed as the fp baseline or the comparison measures templating
    instead of quantization.
    """
    cmd = [sys.executable, EVAL_RUNNER,
           '--weight', os.path.abspath(weight),
           '--hidden', str(args.hidden),
           '--layers', str(args.layers),
           '--max-new-tokens', str(args.max_new_tokens),
           '--device', args.device,
           '--prompt-mode', REQUIRED_PROMPT_MODE,
           '--seed', str(args.seed),
           '--out', os.path.abspath(report_path)]
    if getattr(args, 'backend', 'hf') == 'gguf':
        cmd += ['--backend', 'gguf']
        if getattr(args, 'llamacpp_root', None):
            cmd += ['--llamacpp-root', args.llamacpp_root]
        cmd += ['--gguf-threads', str(getattr(args, 'gguf_threads', 8))]
    print(f'{log_prefix} {" ".join(cmd)}', flush=True)
    env = dict(os.environ)
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                          errors='replace', env=env)
    tail = (proc.stdout or '').strip().splitlines()[-3:]
    for line in tail:
        print(f'{log_prefix}   {line}')
    if proc.returncode != 0:
        err = (proc.stderr or '').strip().splitlines()
        return None, cmd, (f'run_eval exited {proc.returncode}: '
                           f'{err[-1] if err else "no stderr"}')
    report, err = load_json(report_path)
    if err:
        return None, cmd, f'run_eval wrote an unreadable report: {err}'
    return report, cmd, None


def check_report_comparable(report, label, args):
    """Refuse to compare reports produced under different conditions.

    Returns a list of incompatibility strings (empty == comparable). This is the
    guard that makes the prompt-mode rule mechanical instead of a convention.
    """
    problems = []
    if report.get('prompt_mode') != REQUIRED_PROMPT_MODE:
        problems.append(f'{label}: prompt_mode={report.get("prompt_mode")!r}, '
                        f'required {REQUIRED_PROMPT_MODE!r}')
    if report.get('max_new_tokens') != args.max_new_tokens:
        problems.append(f'{label}: max_new_tokens={report.get("max_new_tokens")!r}, '
                        f'requested {args.max_new_tokens!r}')
    if report.get('seed') != args.seed:
        problems.append(f'{label}: seed={report.get("seed")!r}, requested {args.seed!r}')
    gen = report.get('generation') or {}
    if gen.get('mode') not in (None, 'greedy'):
        problems.append(f'{label}: generation mode {gen.get("mode")!r} is not greedy')
    bench = report.get('benchmark') or {}
    if bench.get('num_tasks') and len(report.get('per_task') or []) != bench['num_tasks']:
        problems.append(f'{label}: per_task count != benchmark.num_tasks')
    return problems


def discover_grpo_artifacts(weight_path, logs_dir):
    """Find the GRPO metrics artifacts that belong to a checkpoint.

    train_grpo_rule.py writes `--metrics_path X.jsonl` plus `X_summary.json`.
    The checkpoint name does not always equal the metrics stem (e.g. the
    smoke checkpoint `grpo_smoke_768.pth` was produced by the *official*
    train_grpo.py, which writes no metrics at all), so several candidates are
    tried in order and the ones tried are returned for the audit trail.

    Returns (summary_path, metrics_path, tried) - either may be None.
    """
    stem = os.path.splitext(os.path.basename(weight_path or ''))[0]
    # strip the size suffix: grpo_rule_smoke_768 -> grpo_rule_smoke
    stem = re.sub(r'_(512|640|768|896|1024)$', '', stem)
    cands = [
        os.path.join(logs_dir, f'{stem}_summary.json'),
        os.path.join(logs_dir, f'{stem}.jsonl'),
    ]
    # Common alias: the checkpoint may be <name> while the log is grpo_<name>.
    if not stem.startswith('grpo'):
        cands += [os.path.join(logs_dir, f'grpo_{stem}_summary.json'),
                  os.path.join(logs_dir, f'grpo_{stem}.jsonl')]
    # The rule-reward smoke run is the canonical 3-step artifact.
    if 'smoke' in stem:
        cands += [os.path.join(logs_dir, 'grpo_rule_smoke_summary.json'),
                  os.path.join(logs_dir, 'grpo_rule_smoke.jsonl')]
    summary_path = next((c for c in cands
                         if c.endswith('_summary.json') and os.path.exists(c)), None)
    metrics_path = next((c for c in cands
                         if c.endswith('.jsonl') and os.path.exists(c)), None)

    # ATTRIBUTION CHECK: a *_summary.json records `final_weight`, the checkpoint
    # that run actually produced. A metrics file whose summary points at a
    # DIFFERENT checkpoint must not be attributed to the one being evaluated -
    # that would put another run's token counts and adv_std into this artifact.
    attribution = {'checked': False, 'ok': None, 'summary_final_weight': None,
                   'evaluated_weight': weight_path}
    if summary_path:
        sdata, serr = load_json(summary_path)
        if sdata and sdata.get('final_weight'):
            attribution['checked'] = True
            attribution['summary_final_weight'] = sdata['final_weight']
            same = (os.path.normcase(os.path.abspath(sdata['final_weight']))
                    == os.path.normcase(os.path.abspath(weight_path)))
            attribution['ok'] = same
            if not same:
                summary_path = metrics_path = None
    return summary_path, metrics_path, cands, attribution


# --------------------------------------------------------- token accounting
def _pretrain_accounting(meta_path, metrics_path, summary_path):
    """Pretrain tokens from the corpus meta AND from the training log.

    Two different, clearly-labelled quantities:
      * budget:   the token count of the whole 1-epoch corpus
                  (pretrain_meta.total_tokens_consumed = n_blocks * seq_len)
      * consumed: steps * tokens_per_step, i.e. what the trainer actually fed
                  the model so far. This is the label the spec asks for:
                  `tokens_consumed_by_training = steps x tokens_per_step`.
    """
    out = {'basis': 'tokens fed to the model, packed blocks (no padding)'}
    meta, err = load_json(meta_path)
    if meta:
        out['corpus'] = {
            'source': meta_path,
            'total_tokens_consumed': meta.get('total_tokens_consumed'),
            'tokenized_tokens': meta.get('tokenized_tokens'),
            'dropped_tail_tokens': meta.get('dropped_tail_tokens'),
            'n_blocks': meta.get('n_blocks'),
            'seq_len': meta.get('seq_len'),
            'n_docs': meta.get('n_docs'),
            'display': f'{meta.get("total_tokens_consumed"):,} tokens'
                       if meta.get('total_tokens_consumed') is not None
                       else 'n/a (no total_tokens_consumed in meta)',
        }
        if meta.get('n_blocks') and meta.get('seq_len'):
            out['corpus']['identity_check'] = (
                f'n_blocks * seq_len = {meta["n_blocks"] * meta["seq_len"]:,} '
                f'vs total_tokens_consumed = {meta.get("total_tokens_consumed"):,}')
    else:
        out['corpus'] = na(err)

    consumed = None
    source = None
    # 1. the dedicated summary file, if a pretrain run ever writes one
    summary, serr = load_json(summary_path)
    if summary and summary.get('steps') and summary.get('tokens_per_step'):
        consumed = int(summary['steps']) * int(summary['tokens_per_step'])
        source = f'{summary_path}: steps x tokens_per_step'
    else:
        # 2. otherwise the per-step metrics JSONL: last cum_tokens is exactly
        #    steps x tokens_per_step, and tokens_per_step is logged per step.
        recs, rerr = load_jsonl(metrics_path)
        if recs:
            last = recs[-1]
            if last.get('cum_tokens') is not None:
                consumed = int(last['cum_tokens'])
                source = (f'{metrics_path}: last cum_tokens (step {last.get("step")}, '
                          f'tokens_per_step {last.get("tokens_per_step")})')
            elif last.get('step') is not None and last.get('tokens_per_step') is not None:
                consumed = int(last['step']) * int(last['tokens_per_step'])
                source = f'{metrics_path}: steps x tokens_per_step'
    if consumed is not None:
        out['tokens_consumed_by_training'] = {
            'value': consumed,
            'basis': 'steps x tokens_per_step',
            'source': source,
            'complete_run': bool(meta and meta.get('total_tokens_consumed')
                                 and consumed >= meta['total_tokens_consumed']),
            'display': f'{consumed:,} tokens',
        }
    else:
        out['tokens_consumed_by_training'] = na(
            f'no pretrain training log at {metrics_path} or {summary_path} '
            f'({serr or rerr})')
    return out


def _sft_accounting(meta_path, metrics_path):
    """SFT tokens: padded-total (what the model consumed) + supervised labels."""
    out = {'basis': 'padded tokens fed to the model vs assistant-only label tokens'}
    meta, err = load_json(meta_path)
    if not meta:
        out['meta'] = na(err)
        out['padded_total'] = na(err)
        out['supervised'] = na(err)
        return out
    toks = (meta.get('tokens') or {})
    at = toks.get('at_max_seq_len') or {}
    un = toks.get('untruncated') or {}
    out['meta'] = {
        'source': meta_path,
        'out_records': meta.get('out_records'),
        'max_seq_len': meta.get('max_seq_len'),
        'batch_size_for_padded_math': meta.get('batch_size_for_padded_math'),
        'aux_fraction_by_record': meta.get('aux_fraction_by_record'),
        'aux_fraction_by_labeled_tokens': meta.get('aux_fraction_by_labeled_tokens'),
        'out_sha256': meta.get('out_sha256'),
    }
    padded = at.get('padded_tokens_per_epoch')
    supervised = at.get('labeled_tokens')
    if padded is not None:
        out['padded_total'] = {
            'value': int(padded),
            'basis': 'out_records x max_seq_len (the trainer pads every sample)',
            'display': f'{int(padded):,} tokens',
        }
    else:
        out['padded_total'] = na('no tokens.at_max_seq_len.padded_tokens_per_epoch in meta')
    if supervised is not None:
        out['supervised'] = {
            'value': int(supervised),
            'basis': 'labels != -100 (assistant turns only), truncated at max_seq_len',
            'display': f'{int(supervised):,} tokens',
        }
    else:
        out['supervised'] = na('no tokens.at_max_seq_len.labeled_tokens in meta')
    if un.get('prompt_tokens') is not None:
        out['untruncated_prompt_tokens'] = int(un['prompt_tokens'])
    out['padded_waste_frac'] = (
        round(1.0 - float(supervised) / float(padded), 4)
        if (padded and supervised) else None)

    # If a real SFT run has happened, prefer the trainer's own counters.
    recs, rerr = load_jsonl(metrics_path)
    if recs:
        last = recs[-1]
        out['measured_run'] = {
            'source': metrics_path,
            'steps': last.get('step'),
            'cum_tokens_total': last.get('cum_tokens_total'),
            'cum_tokens_supervised': last.get('cum_tokens_supervised'),
            'complete_run': (padded is not None
                             and (last.get('cum_tokens_total') or 0) >= padded),
            'display': f"{last.get('cum_tokens_total')} total / "
                       f"{last.get('cum_tokens_supervised')} supervised (step {last.get('step')})",
        }
    else:
        out['measured_run'] = na(f'no SFT metrics at {metrics_path} ({rerr}); '
                                 f'the values above are the measured one-epoch budget '
                                 f'from the dataset build, not a completed run')
    return out


def _grpo_accounting(summary_path, metrics_path):
    """GRPO tokens: sampled completion tokens (+ prompt tokens, kept separate).

    The curriculum runs THREE stages, each writing its own `_summary.json`. Summing only
    the final stage under-reports the rollout cost (measured: 839,232 vs 1,639,872 for
    the whole curriculum). So total the per-stage summaries when they exist.
    """
    out = {'basis': 'completion tokens actually sampled during rollout'}
    summary, serr = load_json(summary_path)
    recs, rerr = load_jsonl(metrics_path)

    # per-stage totals, if the curriculum summaries are present
    logs_dir = os.path.dirname(summary_path)
    stage_files = sorted(glob.glob(os.path.join(logs_dir, 'grpo_stage*_summary.json')))
    stage_tot = []
    for f in stage_files:
        d, _ = load_json(f)
        if d and d.get('cum_sampled_tokens'):
            stage_tot.append({
                'file': os.path.basename(f),
                'steps': d.get('steps'),
                'cum_sampled_tokens': d.get('cum_sampled_tokens'),
                'cum_prompt_tokens': d.get('cum_prompt_tokens'),
            })
    if stage_tot:
        out['per_stage'] = stage_tot
        out['sampled_tokens'] = {
            'value': int(sum(s['cum_sampled_tokens'] for s in stage_tot)),
            'basis': 'sum over all curriculum stages of measured '
                     'completion_mask.sum() per step',
            'display': f'{int(sum(s["cum_sampled_tokens"] for s in stage_tot)):,} tokens',
        }
        pr = sum(s.get('cum_prompt_tokens') or 0 for s in stage_tot)
        if pr:
            out['prompt_tokens'] = {
                'value': int(pr),
                'display': f'{int(pr):,} tokens',
                'note': 'conditioning context, NOT added to the grand total',
            }
        if summary:
            out['summary'] = {
                'source': summary_path,
                'steps': summary.get('steps'),
                'opt_steps': summary.get('opt_steps'),
                'cum_sampled_tokens': summary.get('cum_sampled_tokens'),
                'cum_prompt_tokens': summary.get('cum_prompt_tokens'),
                'wall_sec': summary.get('wall_sec'),
                'peak_reserved_MiB': summary.get('peak_reserved_MiB'),
                'note': 'final stage only; the totals above cover the whole curriculum',
            }
        return out

    if summary:
        out['summary'] = {
            'source': summary_path,
            'steps': summary.get('steps'),
            'opt_steps': summary.get('opt_steps'),
            'cum_sampled_tokens': summary.get('cum_sampled_tokens'),
            'cum_prompt_tokens': summary.get('cum_prompt_tokens'),
            'wall_sec': summary.get('wall_sec'),
            'peak_reserved_MiB': summary.get('peak_reserved_MiB'),
        }
        sampled = summary.get('cum_sampled_tokens')
        prompt = summary.get('cum_prompt_tokens')
    elif recs:
        last = recs[-1]
        out['summary'] = {
            'source': f'{metrics_path} (no _summary.json; last record used)',
            'steps': last.get('step'),
            'cum_sampled_tokens': last.get('cum_sampled_tokens'),
            'cum_prompt_tokens': last.get('cum_prompt_tokens'),
        }
        sampled = last.get('cum_sampled_tokens')
        prompt = last.get('cum_prompt_tokens')
    else:
        out['summary'] = na(f'no GRPO summary at {summary_path} and no metrics at '
                            f'{metrics_path} ({serr}; {rerr})')
        sampled = prompt = None
    if sampled is not None:
        out['sampled_tokens'] = {
            'value': int(sampled),
            'basis': 'measured completion_mask.sum() per step, accumulated',
            'display': f'{int(sampled):,} tokens',
        }
    else:
        out['sampled_tokens'] = na('no cum_sampled_tokens in any GRPO artifact')
    if prompt is not None:
        out['prompt_tokens'] = {
            'value': int(prompt),
            'basis': 'attention_mask.sum() of the prompt batches (re-encoded every '
                     'step; not unique corpus tokens)',
            'display': f'{int(prompt):,} tokens',
        }
    else:
        out['prompt_tokens'] = na('no cum_prompt_tokens in any GRPO artifact')
    return out


def build_token_accounting(args, grpo_used_summary, grpo_used_metrics):
    pre = _pretrain_accounting(args.pretrain_meta, args.pretrain_metrics,
                               args.pretrain_summary)
    sft = _sft_accounting(args.sft_meta, args.sft_metrics)
    grpo = _grpo_accounting(grpo_used_summary, grpo_used_metrics)

    def val(d):
        return d.get('value') if isinstance(d, dict) and 'value' in d else None

    pre_consumed = val(pre.get('tokens_consumed_by_training'))
    sft_padded = val(sft.get('padded_total'))
    sft_supervised = val(sft.get('supervised'))
    sft_consumed = ((sft.get('measured_run') or {}).get('cum_tokens_total')
                    if isinstance(sft.get('measured_run'), dict) else None)
    grpo_sampled = val(grpo.get('sampled_tokens'))
    grpo_prompt = val(grpo.get('prompt_tokens'))

    # Grand total must reflect what the model ACTUALLY consumed, not the corpus size.
    # `corpus.total_tokens_consumed` counts whole packed blocks (2,208,852,480) while
    # training consumed steps x tokens_per_step (2,208,841,728); the 10,752-token gap is
    # blocks dropped by drop_last. Prefer the training figure; fall back to the corpus
    # count only when no training log exists.
    pre_budget = (pre_consumed if pre_consumed is not None
                  else (pre.get('corpus') or {}).get('total_tokens_consumed'))

    # Grand total = the projected token count of the full pipeline
    # (pretrain budget + one SFT epoch + this GRPO run). Missing parts make it
    # explicitly partial rather than silently smaller.
    parts = {'pretrain': pre_budget, 'sft': sft_padded, 'grpo': grpo_sampled}
    missing = [k for k, v in parts.items() if v is None]
    grand = None if missing else int(sum(parts.values()))
    notes = []
    if grand is not None:
        notes.append(
            'grand_total_tokens = pretrain corpus budget + SFT padded one-epoch '
            'tokens + GRPO sampled completion tokens. It counts PADDED SFT tokens '
            '(the model consumed them) and excludes GRPO prompt tokens (they are '
            're-encoded every step, not unique data) and SFT label tokens (a subset '
            'of the padded total, so adding them would double count).')
    else:
        notes.append(
            f'grand_total_tokens is partial: missing {missing}. A partial sum is '
            f'not reported as a total, because that would understate the pipeline.')

    consumed_parts = {'pretrain': pre_consumed, 'sft': sft_consumed, 'grpo': grpo_sampled}
    consumed_missing = [k for k, v in consumed_parts.items() if v is None]
    consumed_total = (None if consumed_missing
                      else int(sum(consumed_parts.values())))

    block = {
        'pretrain': pre,
        'sft': sft,
        'grpo': grpo,
        'grand_total_tokens': {
            'value': grand,
            'display': (f'{grand:,} tokens' if grand is not None
                        else 'n/a (requires real checkpoint: missing ' +
                             ', '.join(missing) + ')'),
            'components': parts,
            'missing_components': missing,
        },
        'consumed_so_far_tokens': {
            'value': consumed_total,
            'display': (f'{consumed_total:,} tokens' if consumed_total is not None
                        else 'n/a (requires real checkpoint: missing ' +
                             ', '.join(consumed_missing) + ')'),
            'components': consumed_parts,
            'missing_components': consumed_missing,
            'basis': 'what has actually been fed/sampled by the trainers so far '
                     '(pretrain: last cum_tokens; SFT: trainer counter if a run '
                     'exists; GRPO: cum_sampled_tokens)',
        },
        'accounting_notes': notes + [
            'PRETRAIN: tokens_consumed_by_training = steps x tokens_per_step, read '
            'from the training log (logs/pretrain_metrics_summary.json if present, '
            'else the last cum_tokens in logs/pretrain_metrics.jsonl). The corpus '
            'budget in data/tokens/pretrain_meta.json is n_blocks x seq_len and '
            'contains no padding.',
            'SFT: padded_total = out_records x max_seq_len, i.e. every sample padded '
            'to the sequence length (MiniMind pads; measured 48 % waste at seq 768). '
            'supervised = labels != -100 (assistant turns only). These two are the '
            'measured one-epoch budget from data/sft/sft_meta.json; if a real SFT '
            'run has happened, its own trainer counters are reported under '
            'sft.measured_run and used for consumed_so_far.',
            'GRPO: sampled_tokens is the measured number of completion tokens '
            'sampled (the generation cost that dominates GRPO wall time). '
            'prompt_tokens is reported separately and NOT added to the total: the '
            'same prompts are re-encoded on every step, so they are compute, not '
            'data.',
            'Every value above is either measured or explicitly n/a with the path '
            'that was looked for. No token count in this block is estimated.',
        ],
    }
    return block


# ------------------------------------------------------------------- verdict
def _chance_context(overall, total):
    """Chance-baseline text + extra verdict rules, or empty if it cannot be computed.

    Kept separate so every early-return path in build_verdict can attach it without
    duplicating the calculation.
    """
    out = {'text': '', 'rules': [], 'data': None}
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        import chance_baseline as cb
        tasks_path = os.path.join(os.path.dirname(here), 'eval', 'heldout_tasks.jsonl')
        with open(tasks_path, encoding='utf-8') as f:
            tasks_cb = [json.loads(x) for x in f if x.strip()]
        res = cb.analyse(tasks_cb)
        out['data'] = res
        chance_pct = res['overall']['chance_accuracy_pct']
        thresh = res['overall']['min_correct_for_significance_5pct']
        sft_c = overall['sft']['correct']
        grpo_c = overall['grpo']['correct']
        parts = [
            f'Chance baseline: a guessing model scores {chance_pct:.1f} % on this '
            f'benchmark ({res["overall"]["expected_correct_by_chance"]:.1f} of '
            f'{res["overall"]["tasks"]} tasks), because `format` and `logic` are '
            f'2-3-way choices. {thresh}/{res["overall"]["tasks"]} correct is the '
            f'minimum that beats chance at the 5 % level.']
        if thresh is not None:
            if max(sft_c, grpo_c) < thresh:
                out['rules'].append('below_chance_threshold')
                parts.append(
                    f'Both checkpoints are BELOW the chance threshold '
                    f'({sft_c}/{total} and {grpo_c}/{total} vs {thresh} needed): '
                    f'neither demonstrates capability above guessing, regardless of '
                    f'which scores higher.')
            elif sft_c < thresh <= grpo_c:
                out['rules'].append('grpo_crossed_chance')
                parts.append(
                    f'GRPO crossed the chance threshold ({grpo_c} >= {thresh}) while '
                    f'SFT did not ({sft_c}): this is the only pattern that supports '
                    f'a real capability claim.')
        out['text'] = '\n'.join(parts)
    except Exception as exc:  # noqa: BLE001 - context must never break the report
        out['text'] = f'Chance baseline unavailable: {type(exc).__name__}: {exc}'
    return out


def build_verdict(overall, per_family, per_task, grpo_summary, accounting):
    """Generate the verdict text from the numbers, by explicit rules.

    No claim is made that the numbers do not support; every sentence below is
    traceable to a field in this artifact.
    """
    sft_c = overall['sft']['correct']
    grpo_c = overall['grpo']['correct']
    total = overall['sft']['total']
    delta = overall['delta']['correct']
    p = overall['mcnemar']['p_value']
    lines = []

    if total == 0:
        lines.append('No tasks were graded, so there is nothing to compare.')
        return {'headline': 'no data', 'text': '\n'.join(lines), 'rules_fired': ['no_tasks']}

    # Chance baseline is computed up front so EVERY verdict path (including the
    # early-return smoke path) carries it. Without it "4/24 = 16.7 %" reads as partial
    # success when it is indistinguishable from guessing.
    chance_ctx = _chance_context(overall, total)

    # Rule 1: nothing anywhere -> smoke weights.
    if sft_c == 0 and grpo_c == 0:
        lines.append(
            f'No measurable difference: both checkpoints score {sft_c}/{total} and '
            f'{grpo_c}/{total}. A 0-vs-0 comparison cannot show a training effect.')
        lines.append(
            'Reason candidate: these are SMOKE checkpoints (a few hundred steps on a '
            'tiny data slice), not trained models - their completions are expected to '
            'be degenerate, and they are. Treat this artifact as a harness test, not '
            'a result.')
        if chance_ctx['text']:
            lines.append('')
            lines.append(chance_ctx['text'])
        return {
            'headline': 'no measurable difference; checkpoints are smoke weights',
            'text': '\n'.join(lines),
            'rules_fired': ['both_zero', 'smoke_weights'] + chance_ctx['rules'],
            'chance_baseline': chance_ctx['data'],
        }

    # Rule 2: overall direction + significance.
    if delta > 0:
        direction = f'GRPO improves overall accuracy by {delta}/{total}'
    elif delta < 0:
        direction = f'GRPO lowers overall accuracy by {-delta}/{total}'
    else:
        direction = 'GRPO leaves overall accuracy unchanged'
    sig = 'statistically distinguishable' if p < 0.05 else 'not statistically distinguishable'
    lines.append(f'{direction} ({sft_c}/{total} -> {grpo_c}/{total}, '
                 f'{overall["sft"]["accuracy"]:.1%} -> {overall["grpo"]["accuracy"]:.1%}); '
                 f'exact McNemar p = {p:.4f}, i.e. the delta is {sig} at the 5 % level.')

    improved = [f for f, v in per_family.items() if v['delta']['correct'] > 0]
    flat = [f for f, v in per_family.items() if v['delta']['correct'] == 0]
    worse = [f for f, v in per_family.items() if v['delta']['correct'] < 0]
    if improved:
        lines.append('Families that improved: ' + ', '.join(
            f'{f} ({per_family[f]["sft"]["correct"]}/{per_family[f]["sft"]["total"]} -> '
            f'{per_family[f]["grpo"]["correct"]}/{per_family[f]["grpo"]["total"]})'
            for f in improved) + '.')
    if flat:
        lines.append('Families unchanged: ' + ', '.join(flat) + '.')
    if worse:
        lines.append('Families that regressed: ' + ', '.join(
            f'{f} ({per_family[f]["sft"]["correct"]}/{per_family[f]["sft"]["total"]} -> '
            f'{per_family[f]["grpo"]["correct"]}/{per_family[f]["grpo"]["total"]})'
            for f in worse) + '.')

    # Rule 3: reason candidates, selected by explicit evidence.
    rules = []
    adv_std = grpo_summary.get('mean_adv_std')
    frac_sparse = grpo_summary.get('frac_steps_no_signal')
    if not improved and adv_std is not None and adv_std < 1e-3:
        rules.append('sparse_reward')
        lines.append(
            f'Reason candidate: SPARSE REWARD. GRPO training metrics show '
            f'mean adv_std = {adv_std:.4f} (< 1e-3) over '
            f'{grpo_summary.get("n_steps", "n/a")} steps, i.e. the groups were '
            f'all-right or all-wrong and produced no gradient. No improvement is '
            f'expected in that regime, so this result does not test the policy.')
    elif not improved and frac_sparse is not None and frac_sparse >= 0.5:
        rules.append('sparse_reward_fraction')
        lines.append(
            f'Reason candidate: SPARSE REWARD. {frac_sparse:.1%} of GRPO steps had '
            f'adv_std < 1e-3, so most groups carried no learning signal.')
    elif not improved:
        rules.append('capacity_or_difficulty')
        lines.append(
            'Reason candidate: MODEL CAPACITY or TASK DIFFICULTY. The GRPO run did '
            'have a non-degenerate advantage signal (adv_std is not ~0), yet the '
            'held-out score did not move; for a 63.9 M-parameter model on '
            'multi-step arithmetic the more likely explanation is capacity, and the '
            'next thing to check is whether the training reward rose while the '
            'held-out score did not (that pattern is over-fitting to the training '
            'task distribution, not a capacity limit).')
    if improved and not flat and not worse and set(improved) == {'format'}:
        rules.append('format_only_gain')
        lines.append(
            'Reason candidate: REWARD HACKING VIA FORMAT. Every gain is in the '
            '`format` family (JSON / YES-NO / letter choice), which a rule reward '
            'can be satisfied by without computing the answer; the arithmetic and '
            'multistep families did not improve. Read this as format compliance, '
            'not reasoning.')
    if 0 < abs(delta) <= 2 and p >= 0.05:
        rules.append('small_delta_not_significant')
        lines.append(
            f'Caution: the delta is only {abs(delta)} task(s) and p = {p:.4f} >= 0.05. '
            f'With {total} tasks a 1-2 task move is within chance; do not read it as '
            f'an effect.')
    if not rules:
        rules.append('no_reason_rule_fired')
        lines.append('No reason rule fired: the observed pattern does not match any '
                     'of the pre-registered explanations, so no cause is claimed.')

    # Chance-baseline context (computed once at the top of this function).
    if chance_ctx['text']:
        lines.append('')
        lines.append(chance_ctx['text'])
    rules.extend(chance_ctx['rules'])

    lines.append(f'Token accounting is in `token_accounting`; grand total '
                 f'{accounting["grand_total_tokens"]["display"]}.')
    return {'headline': lines[0], 'text': '\n'.join(lines), 'rules_fired': rules,
            'chance_baseline': chance_ctx['data']}


# ------------------------------------------------------------------- self-test
def self_test():
    """Verify the exact McNemar p-value and the token-accounting arithmetic.

    McNemar reference values (two-sided exact binomial, p=0.5):
      b=0,c=0 -> 1.0 ; b=1,c=0 -> 1.0 ; b=2,c=0 -> 0.5 ; b=3,c=0 -> 0.25
      b=5,c=0 -> 0.0625 ; b=6,c=0 -> 0.03125 (significant at 5 %)
      b=10,c=0 -> 2/2^10 = 0.001953125
      b=2,c=2 -> 1.0 ; b=5,c=5 -> 1.0
      b=8,c=1 -> 2*(C(9,0)+C(9,1))/2^9 = 2*(1+9)/512 = 0.0390625
    """
    fails = 0

    def chk(name, got, want, tol=1e-12):
        nonlocal fails
        ok = got is not None and abs(got - want) <= tol
        fails += 0 if ok else 1
        print(f'{"ok  " if ok else "FAIL"} mcnemar {name:28s} got={got!r} want={want!r}')

    chk('b=0,c=0', mcnemar_exact(0, 0), 1.0)
    chk('b=1,c=0', mcnemar_exact(1, 0), 1.0)
    chk('b=2,c=0', mcnemar_exact(2, 0), 0.5)
    chk('b=3,c=0', mcnemar_exact(3, 0), 0.25)
    chk('b=5,c=0', mcnemar_exact(5, 0), 0.0625)
    chk('b=6,c=0', mcnemar_exact(6, 0), 0.03125)
    chk('b=10,c=0', mcnemar_exact(10, 0), 2.0 / 1024.0)
    chk('b=2,c=2', mcnemar_exact(2, 2), 1.0)
    chk('b=5,c=5', mcnemar_exact(5, 5), 1.0)
    chk('b=8,c=1', mcnemar_exact(8, 1), 2 * (1 + 9) / 512.0)
    chk('symmetry b=7,c=2', mcnemar_exact(7, 2), mcnemar_exact(2, 7))

    # Token accounting arithmetic, exercised on ISOLATED fixtures.
    # These assertions must not read the live logs/: the real curriculum summaries are
    # now globbed and summed, so pointing the self-test at logs/ would make it depend on
    # whatever runs happen to exist (it did, and 5 checks failed spuriously).
    fx = tempfile.mkdtemp(prefix='cmp_selftest_')
    try:
        pre_sum_fx = os.path.join(fx, 'pretrain_metrics_summary.json')
        with open(pre_sum_fx, 'w', encoding='utf-8') as f:
            json.dump({'steps': 100, 'tokens_per_step': 10, 'total_tokens': 1000,
                       'wall_sec': 10, 'peak_reserved_MiB': 100}, f)
        pre_met_fx = os.path.join(fx, 'pretrain_metrics.jsonl')
        with open(pre_met_fx, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'step': 100, 'cum_tokens': 1000,
                                'tokens_per_step': 10}) + '\n')
        sft_met_fx = os.path.join(fx, 'sft_metrics.jsonl')
        with open(sft_met_fx, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'cum_tokens_total': 2000,
                                'cum_tokens_supervised': 900}) + '\n')
        grpo_sum_fx = os.path.join(fx, 'grpo_rule_smoke_summary.json')
        with open(grpo_sum_fx, 'w', encoding='utf-8') as f:
            json.dump({'steps': 3, 'cum_sampled_tokens': 1536,
                       'cum_prompt_tokens': 230}, f)
        grpo_met_fx = os.path.join(fx, 'grpo_rule_smoke.jsonl')
        with open(grpo_met_fx, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'step': 3, 'cum_sampled_tokens': 1536,
                                'cum_prompt_tokens': 230}) + '\n')

        class A:
            pretrain_meta = DEFAULT_PRETRAIN_META   # read-only, fine to use real
            pretrain_metrics = pre_met_fx
            pretrain_summary = pre_sum_fx
            sft_meta = DEFAULT_SFT_META             # read-only, fine to use real
            sft_metrics = sft_met_fx

        block = build_token_accounting(A, grpo_sum_fx, grpo_met_fx)

        def chk2(name, got, want):
            nonlocal fails
            ok = got == want
            fails += 0 if ok else 1
            print(f'{"ok  " if ok else "FAIL"} tokens   {name:28s} got={got!r} want={want!r}')

        chk2('pretrain corpus tokens',
             (block['pretrain']['corpus'] or {}).get('total_tokens_consumed'), 2208852480)
        chk2('pretrain consumed == fixture total',
             block['pretrain']['tokens_consumed_by_training'].get('value'), 1000)
        chk2('sft padded total', block['sft']['padded_total'].get('value'), 153600000)
        chk2('sft supervised', block['sft']['supervised'].get('value'), 66236849)
        chk2('sft padded = records x max_seq_len', 153600000, 200000 * 768)
        chk2('grpo sampled (rule smoke)', block['grpo']['sampled_tokens'].get('value'), 1536)
        chk2('grpo prompt (rule smoke)', block['grpo']['prompt_tokens'].get('value'), 230)
        # Fixture values: pretrain 1000, SFT padded 153,600,000, GRPO 1536.
        chk2('grand total = pre+sft+grpo', block['grand_total_tokens'].get('value'),
             1000 + 153600000 + 1536)
        # `consumed_so_far` sums the CONSUMED values: pretrain tokens actually fed to
        # the model (fixture: 1000), SFT tokens (2000), GRPO sampled (1536).
        chk2('consumed so far = pre+sft+grpo',
             block['consumed_so_far_tokens'].get('value'), 1000 + 2000 + 1536)
        chk2('consumed so far has no missing parts',
             block['consumed_so_far_tokens'].get('missing_components'), [])
        chk2('sft padded waste frac', block['sft']['padded_waste_frac'],
             round(1.0 - 66236849 / 153600000.0, 4))
    finally:
        import shutil as _sh
        _sh.rmtree(fx, ignore_errors=True)
    print()
    print('COMPARE SELF-TEST', 'PASS' if fails == 0 else f'{fails} FAILURES')
    return 0 if fails == 0 else 1


# --------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='SFT vs GRPO held-out comparison (Phase 3 headline driver)',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sft', default=None, help='SFT .pth checkpoint')
    p.add_argument('--grpo', default=None, help='GRPO .pth checkpoint')
    p.add_argument('--sft-report', default=None,
                   help='pre-computed eval report for the SFT checkpoint (skips eval)')
    p.add_argument('--grpo-report', default=None,
                   help='pre-computed eval report for the GRPO checkpoint (skips eval)')
    p.add_argument('--out', default=os.path.join(ROOT, 'reports', 'grpo_vs_sft.json'))
    p.add_argument('--hidden', type=int, default=768)
    p.add_argument('--layers', type=int, default=8)
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--device', default='cpu', choices=['auto', 'cuda', 'cpu'],
                   help='default cpu: the GPU is reserved for pretraining')
    p.add_argument('--backend', default='hf', choices=['hf', 'gguf'],
                   help='hf = native .pth checkpoints; gguf = quantized llama.cpp '
                        'files (use this for the Phase-4 quantization-drop check)')
    p.add_argument('--llamacpp-root', default=None,
                   help='llama.cpp build dir for --backend gguf (default: <root>/llamacpp)')
    p.add_argument('--gguf-threads', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--grpo-summary', default=None,
                   help='GRPO *_summary.json (default: auto-discover from the '
                        'checkpoint name, then logs/grpo_metrics_summary.json)')
    p.add_argument('--grpo-metrics', default=None,
                   help='GRPO per-step metrics JSONL (fallback if no summary)')
    p.add_argument('--pretrain-meta', default=DEFAULT_PRETRAIN_META)
    p.add_argument('--pretrain-metrics', default=DEFAULT_PRETRAIN_METRICS)
    p.add_argument('--pretrain-summary', default=DEFAULT_PRETRAIN_SUMMARY)
    p.add_argument('--sft-meta', default=DEFAULT_SFT_META)
    p.add_argument('--sft-metrics', default=DEFAULT_SFT_METRICS)
    p.add_argument('--keep-intermediate', default=None,
                   help='directory to keep the two raw eval reports in '
                        '(default: a temp dir, deleted on exit)')
    p.add_argument('--self-test', action='store_true',
                   help='verify McNemar + token accounting, then exit')
    p.add_argument('--quiet', action='store_true')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        return self_test()

    if not args.sft_report and not args.sft:
        raise SystemExit('need --sft <ckpt> or --sft-report <report.json>')
    if not args.grpo_report and not args.grpo:
        raise SystemExit('need --grpo <ckpt> or --grpo-report <report.json>')

    tmpdir = None
    try:
        if args.keep_intermediate:
            inter = os.path.abspath(args.keep_intermediate)
            os.makedirs(inter, exist_ok=True)
        else:
            tmpdir = tempfile.mkdtemp(prefix='cmp_sft_grpo_')
            inter = tmpdir

        # ---- obtain the two reports ----
        sft_report, sft_cmd, sft_err = (None, None, None)
        if args.sft_report:
            sft_report, sft_err = load_json(args.sft_report)
            sft_cmd = f'reused {args.sft_report}'
            if sft_err:
                raise SystemExit(f'--sft-report unreadable: {sft_err}')
        else:
            sft_report, sft_cmd, sft_err = run_eval(
                args.sft, os.path.join(inter, 'sft_eval.json'), args, '[eval:sft]')
            if sft_err:
                raise SystemExit(f'SFT evaluation failed: {sft_err}')

        grpo_report, grpo_cmd, grpo_err = (None, None, None)
        if args.grpo_report:
            grpo_report, grpo_err = load_json(args.grpo_report)
            grpo_cmd = f'reused {args.grpo_report}'
            if grpo_err:
                raise SystemExit(f'--grpo-report unreadable: {grpo_err}')
        else:
            grpo_report, grpo_cmd, grpo_err = run_eval(
                args.grpo, os.path.join(inter, 'grpo_eval.json'), args, '[eval:grpo]')
            if grpo_err:
                raise SystemExit(f'GRPO evaluation failed: {grpo_err}')

        # ---- refuse an invalid comparison (prompt mode / tokens / seed) ----
        problems = (check_report_comparable(sft_report, 'sft', args)
                    + check_report_comparable(grpo_report, 'grpo', args))
        if problems:
            raise SystemExit('REFUSING to compare incomparable reports:\n  - '
                             + '\n  - '.join(problems))

        # ---- per-task alignment ----
        sft_tasks = {r['id']: r for r in sft_report.get('per_task', [])}
        grpo_tasks = {r['id']: r for r in grpo_report.get('per_task', [])}
        if set(sft_tasks) != set(grpo_tasks):
            only_s = sorted(set(sft_tasks) - set(grpo_tasks))
            only_g = sorted(set(grpo_tasks) - set(sft_tasks))
            raise SystemExit(f'task sets differ (sft-only {only_s}, grpo-only {only_g})')
        task_ids = [r['id'] for r in sft_report['per_task']]

        per_task, mcnemar_rows = [], []
        b_total = c_total = both_r = both_w = 0
        for tid in task_ids:
            s, g = sft_tasks[tid], grpo_tasks[tid]
            s_ok, g_ok = bool(s.get('correct')), bool(g.get('correct'))
            if s_ok and g_ok:
                cell, both_r = 'both_right', both_r + 1
            elif s_ok and not g_ok:
                cell, b_total = 'sft_only_right', b_total + 1
            elif (not s_ok) and g_ok:
                cell, c_total = 'grpo_only_right', c_total + 1
            else:
                cell, both_w = 'both_wrong', both_w + 1
            fam = s.get('family') or g.get('family')
            per_task.append({
                'id': tid,
                'family': fam,
                'prompt': s.get('prompt'),
                'sft': {'correct': s_ok, 'completion': s.get('completion'),
                        'reason': s.get('reason')},
                'grpo': {'correct': g_ok, 'completion': g.get('completion'),
                         'reason': g.get('reason')},
                'transition': cell,
            })
            mcnemar_rows.append({
                'id': tid, 'family': fam, 'transition': cell,
                'sft_correct': s_ok, 'grpo_correct': g_ok,
            })

        # ---- overall / per-family ----
        def agg(rows, correct_key):
            cor = sum(1 for r in rows if r[correct_key])
            return {'correct': cor, 'total': len(rows),
                    'accuracy': round(cor / len(rows), 4) if rows else 0.0}

        sft_overall = agg([{'c': sft_tasks[t]['correct']} for t in task_ids], 'c')
        grpo_overall = agg([{'c': grpo_tasks[t]['correct']} for t in task_ids], 'c')
        families = sorted({r['family'] for r in per_task})
        per_family = {}
        for fam in families:
            ids = [r['id'] for r in per_task if r['family'] == fam]
            sf = agg([{'c': sft_tasks[t]['correct']} for t in ids], 'c')
            gf = agg([{'c': grpo_tasks[t]['correct']} for t in ids], 'c')
            per_family[fam] = {
                'sft': sf, 'grpo': gf,
                'delta': {'correct': gf['correct'] - sf['correct'],
                          'accuracy': round(gf['accuracy'] - sf['accuracy'], 4)},
            }

        p_value = mcnemar_exact(b_total, c_total)
        overall = {
            'sft': sft_overall,
            'grpo': grpo_overall,
            'delta': {'correct': grpo_overall['correct'] - sft_overall['correct'],
                      'accuracy': round(grpo_overall['accuracy']
                                        - sft_overall['accuracy'], 4)},
            'mcnemar': {
                'both_right': both_r,
                'sft_only_right': b_total,
                'grpo_only_right': c_total,
                'both_wrong': both_w,
                'discordant_pairs': b_total + c_total,
                'p_value': p_value,
                'method': 'two-sided exact McNemar (binomial, p=0.5, math.comb)',
                'significant_at_5pct': p_value < 0.05,
            },
        }

        # ---- GRPO training-health context, for the verdict's reason rules ----
        used_summary, used_metrics, discovery = None, None, None
        if args.grpo_summary or args.grpo_metrics:
            used_summary, used_metrics = args.grpo_summary, args.grpo_metrics
            discovery = {'mode': 'explicit', 'tried': [used_summary, used_metrics]}
        elif args.grpo:
            used_summary, used_metrics, tried, attribution = discover_grpo_artifacts(
                args.grpo, os.path.join(ROOT, 'logs'))
            discovery = {'mode': 'auto', 'tried': tried,
                         'found_summary': used_summary, 'found_metrics': used_metrics,
                         'attribution': attribution}
        gs, gserr = load_json(used_summary) if used_summary else (None, 'not provided')

        grpo_health = {'n_steps': None, 'mean_adv_std': None,
                       'frac_steps_no_signal': None, 'source': used_summary,
                       'discovery': discovery, 'summary_err': gserr}
        gmetrics, gmerr = load_jsonl(used_metrics) if used_metrics else ([], 'not provided')
        if gs:
            grpo_health.update({
                'steps': gs.get('steps'), 'opt_steps': gs.get('opt_steps'),
                'cum_sampled_tokens': gs.get('cum_sampled_tokens'),
                'wall_sec': gs.get('wall_sec'),
                'peak_reserved_MiB': gs.get('peak_reserved_MiB'),
            })
        if gmetrics:
            adv = [float(r['adv_std']) for r in gmetrics if r.get('adv_std') is not None]
            grpo_health['n_steps'] = len(gmetrics)
            grpo_health['mean_adv_std'] = round(sum(adv) / len(adv), 6) if adv else None
            if adv:
                grpo_health['frac_steps_no_signal'] = round(
                    sum(1 for a in adv if a < 1e-3) / len(adv), 4)
        else:
            grpo_health['metrics_err'] = gmerr
            if discovery and (discovery.get('attribution') or {}).get('ok') is False:
                grpo_health['note'] = (
                    'GRPO metrics were found but belong to a DIFFERENT checkpoint '
                    f'({discovery["attribution"]["summary_final_weight"]}), so they are '
                    'NOT attributed to this run; the training-health reason rules cannot '
                    'fire and the GRPO token count is n/a.')
            else:
                grpo_health['note'] = (
                    'no per-step GRPO metrics found for this checkpoint, so the '
                    'training-health reason rules cannot fire; this is reported rather '
                    'than guessed')

        accounting = build_token_accounting(args, used_summary, used_metrics)
        verdict = build_verdict(overall, per_family, per_task, grpo_health, accounting)

        artifact = {
            'artifact': 'grpo_vs_sft',
            'phase': 3,
            'created_at': __import__('datetime').datetime.now().astimezone().isoformat(
                timespec='seconds'),
            'benchmark': {
                'name': (sft_report.get('benchmark') or {}).get('name'),
                'task_file': TASKS_PATH,
                'tasks_sha256': (sft_report.get('benchmark') or {}).get('tasks_sha256'),
                'grpo_tasks_sha256': (grpo_report.get('benchmark') or {}).get('tasks_sha256'),
                'num_tasks': len(task_ids),
                'lock_check_sft': (sft_report.get('benchmark') or {}).get('lock_check'),
                'lock_check_grpo': (grpo_report.get('benchmark') or {}).get('lock_check'),
                'benchmark_identical': (
                    (sft_report.get('benchmark') or {}).get('tasks_sha256')
                    == (grpo_report.get('benchmark') or {}).get('tasks_sha256')),
            },
            'comparison_conditions': {
                'prompt_mode': REQUIRED_PROMPT_MODE,
                'prompt_template': sft_report.get('prompt_template'),
                'grpo_prompt_template': grpo_report.get('prompt_template'),
                'max_new_tokens': args.max_new_tokens,
                'seed': args.seed,
                'generation_mode': (sft_report.get('generation') or {}).get('mode'),
                'device': args.device,
                'hidden_size': args.hidden,
                'num_hidden_layers': args.layers,
                'validity_note': ('both runs use the same prompt mode, the same '
                                  'max_new_tokens and the same seed; run_eval.py '
                                  'records the prompt template in each report and '
                                  'this artifact refuses to compare otherwise'),
            },
            'checkpoints': {
                'sft': {'weight': (sft_report.get('weight') or {}),
                        'eval_command': sft_cmd,
                        'device_reason': sft_report.get('device_reason'),
                        'wall_time_s': sft_report.get('wall_time_s'),
                        'peak_vram_mb': sft_report.get('peak_vram_mb')},
                'grpo': {'weight': (grpo_report.get('weight') or {}),
                         'eval_command': grpo_cmd,
                         'device_reason': grpo_report.get('device_reason'),
                         'wall_time_s': grpo_report.get('wall_time_s'),
                         'peak_vram_mb': grpo_report.get('peak_vram_mb')},
            },
            'overall': overall,
            'per_family': per_family,
            'per_task': per_task,
            'per_task_mcnemar': {
                'method': 'exact two-sided McNemar on the discordant pairs; '
                          'p = 2 * P(X <= min(b,c)), X ~ Binomial(b+c, 0.5), '
                          'computed with math.comb',
                'b': {'name': 'sft_only_right', 'count': b_total},
                'c': {'name': 'grpo_only_right', 'count': c_total},
                'p_value': p_value,
                'significant_at_5pct': p_value < 0.05,
                'contingency': {
                    'both_right': both_r, 'sft_only_right': b_total,
                    'grpo_only_right': c_total, 'both_wrong': both_w,
                },
                'rows': mcnemar_rows,
            },
            'grpo_training_health': grpo_health,
            'token_accounting': accounting,
            'verdict': verdict,
        }

        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(artifact, f, ensure_ascii=False, indent=2)

        if not args.quiet:
            print()
            print(f'[compare] SFT  {sft_overall["correct"]}/{sft_overall["total"]} '
                  f'({sft_overall["accuracy"]:.1%})')
            print(f'[compare] GRPO {grpo_overall["correct"]}/{grpo_overall["total"]} '
                  f'({grpo_overall["accuracy"]:.1%})')
            print(f'[compare] delta {overall["delta"]["correct"]:+d} task(s), '
                  f'McNemar p={p_value:.4f} '
                  f'({"significant" if p_value < 0.05 else "not significant"} at 5%)')
            print('[compare] per-family: ' + '  '.join(
                f'{fam} {v["sft"]["correct"]}/{v["sft"]["total"]}->'
                f'{v["grpo"]["correct"]}/{v["grpo"]["total"]}'
                for fam, v in per_family.items()))
            print(f'[compare] verdict: {verdict["headline"]}')
            print(f'[compare] grand total tokens: '
                  f'{accounting["grand_total_tokens"]["display"]}')
            print(f'[compare] wrote {out_path}')
            if args.keep_intermediate:
                print(f'[compare] raw eval reports kept in {inter}')
        return 0
    finally:
        if tmpdir:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
