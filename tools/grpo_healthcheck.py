"""GRPO learning-health diagnostic.

Reads the per-step JSONL metrics that `custom/train_grpo_rule.py` writes and
answers, with the real numbers from that file, the questions the Phase-3 spec
demands:

  1. Is the reward actually rising?          -> OLS trend on `reward_mean` plus a
                                                first-20 vs last-20 comparison of
                                                `reward_mean_20` (the trainer's
                                                own 20-step rolling mean).
  2. What fraction of steps had no signal?   -> fraction of steps with
                                                `adv_std < 1e-3` (the sparsity
                                                detector) and the mean
                                                `degenerate_group_frac`.
  3. Is the policy collapsing?               -> mean/tail `entropy`, its trend,
                                                mean `kl_mean`, mean `clip_frac`,
                                                and the count of `collapsed`
                                                flags. The `collapsed` flag in
                                                `train_grpo_rule.py` is
                                                `entropy < 0.05 OR
                                                resp_len >= 0.99*max_gen_len`,
                                                so in practice it means "hit
                                                max_gen_len" - it is NOT a
                                                policy-collapse detector. This
                                                tool says so explicitly and
                                                reports the two arms separately
                                                where the data allows.
  4. Is accuracy rising, and does it track reward? -> OLS trend on `acc`, plus
                                                Pearson r between `acc` and
                                                `reward_mean` (both are the same
                                                quantity under a binary reward,
                                                so r should be ~1.0; a low r
                                                means the two counters disagree,
                                                i.e. a bug or a shaped reward).
  5. A one-line verdict:
        LEARNING                  reward has a positive, non-trivial trend and
                                  enough non-degenerate groups to explain it
        SPARSE                    reward is flat AND most groups are degenerate
                                  (all-wrong / all-right) -> no gradient
        DEGENERATE                reward is falling, or entropy has collapsed,
                                  or KL is running away
        INSUFFICIENT_DATA         too few steps (< --min-steps, default 20) to
                                  say anything

Exit code is ALWAYS 0: this is a diagnostic, not a gate. It prints a compact
table plus the verdict line, and with --json writes the full analysis.

Usage:
  python tools/grpo_healthcheck.py logs/grpo_easy_probe.jsonl
  python tools/grpo_healthcheck.py logs/grpo_rule_smoke.jsonl --min-steps 20
  python tools/grpo_healthcheck.py logs/grpo_metrics.jsonl --json logs/hc.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys

# Thresholds. Every one is a named constant so the verdict rule below can be
# read against the numbers that produced it.
SPARSE_ADV_STD = 1e-3        # adv_std below this == no learning signal in the group
MIN_STEPS = 20               # fewer steps than this -> INSUFFICIENT_DATA
MIN_REWARD_GAIN = 0.02       # first-20 vs last-20 reward_mean_20 difference worth calling "rising"
MIN_SIGNAL_FRAC = 0.5        # < 50 % of steps with signal == "sparse dominates"
ENTROPY_COLLAPSE_FRAC = 0.1  # tail entropy < 10 % of head entropy == collapsing
ENTROPY_ABS_FLOOR = 0.05     # trainer's own entropy-collapse threshold (< 0.05)
TAIL = 20                    # window for "first 20 vs last 20" reward comparisons
TAIL_FRAC = 0.25             # trend windows use the last max(5, n*TAIL_FRAC) steps
MIN_TAIL = 5
KL_RUNAWAY = 0.5             # mean per-token KL above this is a runaway policy


# --------------------------------------------------------------------- io
def read_jsonl(path):
    """Tolerant JSONL reader: skips blanks / unparsable / a partially flushed
    last line, and reports what it skipped. Never raises on a live file."""
    stats = {'path': path, 'exists': False, 'bytes': 0, 'lines': 0,
             'records': 0, 'blank': 0, 'skipped_partial': 0,
             'skipped_invalid': 0, 'err': None}
    try:
        with open(path, 'rb') as f:
            raw = f.read()
    except OSError as exc:
        stats['err'] = f'{type(exc).__name__}: {exc}'
        return [], stats
    stats['exists'] = True
    stats['bytes'] = len(raw)
    text = raw.decode('utf-8-sig', errors='replace')
    lines = text.splitlines(keepends=True)
    stats['lines'] = len(lines)
    records = []
    for idx, line in enumerate(lines):
        s = line.strip()
        if not s:
            stats['blank'] += 1
            continue
        try:
            rec = json.loads(s)
        except json.JSONDecodeError:
            is_last = idx == len(lines) - 1
            if is_last and not line.endswith(('\n', '\r')):
                stats['skipped_partial'] += 1
            else:
                stats['skipped_invalid'] += 1
            continue
        if isinstance(rec, dict):
            records.append(rec)
        else:
            stats['skipped_invalid'] += 1
    stats['records'] = len(records)
    return records, stats


def col(records, key):
    """Column as floats with None for missing/unparsable (never a fake 0)."""
    out = []
    for r in records:
        v = r.get(key)
        if v is None:
            out.append(None)
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(None)
    return out


# ------------------------------------------------------------------- math
def ols_slope(xs, ys):
    """Ordinary least squares slope of y on x, with the Pearson r.

    Returns (slope, intercept, r) or (None, None, None) when undefined
    (< 2 points, or zero variance in x).
    """
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 2:
        return None, None, None
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    sxx = sum((p[0] - mx) ** 2 for p in pairs)
    syy = sum((p[1] - my) ** 2 for p in pairs)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    if sxx == 0:
        return None, None, None
    slope = sxy / sxx
    intercept = my - slope * mx
    r = (sxy / math.sqrt(sxx * syy)) if syy > 0 else None
    return slope, intercept, r


def pearson(xs, ys):
    _, _, r = ols_slope(xs, ys)
    return r


def mean(vals):
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def window_mean(vals, start, end):
    """Mean of vals[start:end] ignoring None; None when the window is empty."""
    return mean(vals[start:end])


def count_below(vals, threshold):
    vals = [v for v in vals if v is not None]
    if not vals:
        return 0, 0, None
    n = sum(1 for v in vals if v < threshold)
    return n, len(vals), n / len(vals)


def fmt(v, digits=4, dash='n/a'):
    if v is None:
        return dash
    try:
        if isinstance(v, float) and not math.isfinite(v):
            return dash
        return f'{float(v):.{digits}f}'
    except (TypeError, ValueError):
        return dash


def fmt_pct(v, digits=1, dash='n/a'):
    if v is None:
        return dash
    return f'{100.0 * float(v):.{digits}f}%'


# --------------------------------------------------------------- analysis
def analyze(records, stats, args):
    n = len(records)
    steps = col(records, 'step')
    reward = col(records, 'reward_mean')
    reward20 = col(records, 'reward_mean_20')
    acc = col(records, 'acc')
    adv_std = col(records, 'adv_std')
    entropy = col(records, 'entropy')
    kl = col(records, 'kl_mean')
    clip = col(records, 'clip_frac')
    degen = col(records, 'degenerate_group_frac')
    resp_len = col(records, 'resp_len_mean')
    collapsed = [bool(r.get('collapsed')) for r in records]

    res = {'file': stats, 'n_steps': n}
    seen_steps = [s for s in steps if s is not None]
    res['steps_seen'] = ([seen_steps[0], seen_steps[-1]] if seen_steps else [])

    if n == 0:
        res['verdict'] = 'INSUFFICIENT_DATA'
        res['verdict_rule'] = 'no parsable metric records in the file'
        return res

    # ---- 1. reward trend -------------------------------------------------
    slope, intercept, r = ols_slope(steps, reward)
    head20 = window_mean(reward20, 0, TAIL)
    tail20 = window_mean(reward20, max(0, n - TAIL), n)
    head_reward = window_mean(reward, 0, min(TAIL, n))
    tail_reward = window_mean(reward, max(0, n - TAIL), n)
    gain20 = (tail20 - head20) if (head20 is not None and tail20 is not None) else None
    res['reward'] = {
        'first': reward[0] if reward else None,
        'last': reward[-1] if reward else None,
        'mean': mean(reward),
        'max': max([v for v in reward if v is not None], default=None),
        'head20_reward_mean_20': head20,
        'tail20_reward_mean_20': tail20,
        'gain20': gain20,
        'head20_reward_mean': head_reward,
        'tail20_reward_mean': tail_reward,
        'ols_slope_per_step': slope,
        'ols_intercept': intercept,
        'ols_r': r,
        'rising': (gain20 is not None and gain20 >= MIN_REWARD_GAIN and slope is not None
                   and slope > 0),
    }

    # ---- 2. sparsity -----------------------------------------------------
    n_sparse, n_adv, frac_sparse = count_below(adv_std, SPARSE_ADV_STD)
    res['sparsity'] = {
        'adv_std_threshold': SPARSE_ADV_STD,
        'steps_adv_std_below_threshold': n_sparse,
        'steps_with_adv_std': n_adv,
        'frac_steps_no_signal': frac_sparse,
        'mean_adv_std': mean(adv_std),
        'mean_degenerate_group_frac': mean(degen),
        'max_degenerate_group_frac': max([v for v in degen if v is not None], default=None),
        'sparse_dominates': (frac_sparse is not None and frac_sparse >= MIN_SIGNAL_FRAC),
    }

    # ---- 3. collapse / KL / clip ----------------------------------------
    # Trend windows are proportional (last 25 %, min 5 steps) so a short run is
    # not dominated by a fixed 20-step window that covers most of it; a policy
    # collapse is also accepted on the trainer's own absolute entropy floor.
    tw = max(MIN_TAIL, int(round(n * TAIL_FRAC)))
    head_ent = window_mean(entropy, 0, min(tw, n))
    tail_ent = window_mean(entropy, max(0, n - tw), n)
    ent_slope, _, ent_r = ols_slope(steps, entropy)
    ent_ratio = (tail_ent / head_ent) if (head_ent not in (None, 0) and tail_ent is not None) else None
    # Any step below the trainer's absolute floor is direct evidence of collapse
    # regardless of window placement.
    n_ent_floor = sum(1 for r in records
                      if r.get('entropy') is not None
                      and float(r['entropy']) < ENTROPY_ABS_FLOOR)
    n_collapsed = sum(1 for c in collapsed if c)
    # The flag in train_grpo_rule.py is an OR:
    #   collapsed = (entropy < 0.05) OR (resp_len_mean >= 0.99 * max_gen_len)
    # so a flagged step whose entropy is >= 0.05 can only have been flagged by
    # the LENGTH arm. That makes the arm split exact without knowing
    # max_gen_len. The cap itself is reported as the observed max resp_len_mean
    # (resp_len_mean can never exceed max_gen_len, so if the cap was ever hit the
    # observed max IS the cap; if it was never hit this is a lower bound).
    rl_vals = [v for v in resp_len if v is not None]
    rl_max = max(rl_vals) if rl_vals else None
    n_ent_arm = sum(1 for r in records
                    if r.get('entropy') is not None and float(r['entropy']) < 0.05)
    n_len_arm = 0
    for rec, flag in zip(records, collapsed):
        if not flag:
            continue
        ent = rec.get('entropy')
        if ent is not None and float(ent) < 0.05:
            continue          # explained by the entropy arm
        n_len_arm += 1        # flagged with healthy entropy -> length arm
    res['stability'] = {
        'mean_entropy': mean(entropy),
        'trend_window': tw,
        'head_entropy': head_ent,
        'tail_entropy': tail_ent,
        'entropy_slope_per_step': ent_slope,
        'entropy_ols_r': ent_r,
        'tail_over_head_entropy': ent_ratio,
        'steps_entropy_below_floor': n_ent_floor,
        'entropy_collapsing': ((ent_ratio is not None and ent_ratio < ENTROPY_COLLAPSE_FRAC)
                               or n_ent_floor > 0),
        'mean_kl_mean': mean(kl),
        'max_abs_kl_mean': max([abs(v) for v in kl if v is not None], default=None),
        'kl_runaway': (mean(kl) is not None and mean(kl) > KL_RUNAWAY),
        'mean_clip_frac': mean(clip),
        'max_clip_frac': max([v for v in clip if v is not None], default=None),
        'steps_flagged_collapsed': n_collapsed,
        'frac_steps_flagged_collapsed': (n_collapsed / n) if n else None,
        'collapsed_flag_note': ('train_grpo_rule.py sets collapsed = (entropy < 0.05) OR '
                                '(resp_len_mean >= 0.99 * max_gen_len); the length arm means '
                                '"hit max_gen_len", NOT "policy collapsed". Arms counted '
                                'separately below.'),
        'collapsed_steps_entropy_arm': n_ent_arm,
        'collapsed_steps_length_arm': n_len_arm,
        'max_resp_len_mean': rl_max,
    }

    # ---- 4. accuracy -----------------------------------------------------
    acc_slope, _, acc_r = ols_slope(steps, acc)
    head_acc = window_mean(acc, 0, min(TAIL, n))
    tail_acc = window_mean(acc, max(0, n - TAIL), n)
    acc_reward_r = pearson(acc, reward)
    res['accuracy'] = {
        'first': acc[0] if acc else None,
        'last': acc[-1] if acc else None,
        'mean': mean(acc),
        'max': max([v for v in acc if v is not None], default=None),
        'head20_acc': head_acc,
        'tail20_acc': tail_acc,
        'gain20': (tail_acc - head_acc) if (head_acc is not None and tail_acc is not None) else None,
        'ols_slope_per_step': acc_slope,
        'rising': (tail_acc is not None and head_acc is not None and tail_acc > head_acc
                   and acc_slope is not None and acc_slope > 0),
        'pearson_acc_vs_reward': acc_reward_r,
        'tracks_reward': (acc_reward_r is not None and acc_reward_r > 0.9),
    }

    # ---- 5. token / cost facts (reported, not judged) --------------------
    sampled = col(records, 'sampled_tokens')
    cum_sampled = col(records, 'cum_sampled_tokens')
    sec = col(records, 'sec')
    res['tokens'] = {
        'cum_sampled_tokens_last': (cum_sampled[-1] if cum_sampled else None),
        'sampled_tokens_per_step_mean': mean(sampled),
        'wall_sec_sum': sum(v for v in sec if v is not None) if any(v is not None for v in sec) else None,
    }

    # ---- verdict ---------------------------------------------------------
    # Order matters: insufficient data first, then DEGENERATE (a collapsing or
    # diverging policy is never "LEARNING"), then SPARSE, then LEARNING/flat.
    if n < args.min_steps:
        verdict = 'INSUFFICIENT_DATA'
        rule = (f'n_steps={n} < --min-steps={args.min_steps}: not enough steps to '
                f'distinguish a trend from noise')
    elif res['stability']['entropy_collapsing'] or res['stability']['kl_runaway']:
        verdict = 'DEGENERATE'
        bits = []
        if res['stability']['entropy_collapsing']:
            if res['stability']['steps_entropy_below_floor']:
                bits.append(f'{res["stability"]["steps_entropy_below_floor"]} steps have '
                            f'entropy < {ENTROPY_ABS_FLOOR}')
            if (ent_ratio is not None and ent_ratio < ENTROPY_COLLAPSE_FRAC):
                bits.append(f'tail/head entropy = {fmt(ent_ratio, 3)} < '
                            f'{ENTROPY_COLLAPSE_FRAC}')
        if res['stability']['kl_runaway']:
            bits.append(f'mean kl_mean = {fmt(mean(kl))} > {KL_RUNAWAY}')
        rule = ' OR '.join(bits) + ' -> the policy is collapsing or diverging'
    elif res['sparsity']['sparse_dominates']:
        verdict = 'SPARSE'
        rule = (f'{fmt_pct(frac_sparse)} of steps have adv_std < {SPARSE_ADV_STD} '
                f'(>= {fmt_pct(MIN_SIGNAL_FRAC)} required) and reward gain20 = '
                f'{fmt(gain20)}: all-wrong/all-right groups dominate -> no gradient')
    elif (gain20 is not None and gain20 <= -MIN_REWARD_GAIN) and slope is not None and slope < 0:
        verdict = 'DEGENERATE'
        rule = (f'reward_mean_20 falls by {fmt(gain20)} over the run '
                f'(< -{MIN_REWARD_GAIN}) with OLS slope {fmt(slope, 6)}/step')
    elif res['reward']['rising']:
        verdict = 'LEARNING'
        rule = (f'reward_mean_20 gain20 = {fmt(gain20)} >= {MIN_REWARD_GAIN} with OLS '
                f'slope {fmt(slope, 6)}/step > 0, and only '
                f'{fmt_pct(frac_sparse)} of steps are signal-free '
                f'(< {fmt_pct(MIN_SIGNAL_FRAC)})')
    else:
        verdict = 'SPARSE'
        rule = (f'reward_mean_20 gain20 = {fmt(gain20)} is below the {MIN_REWARD_GAIN} '
                f'threshold and {fmt_pct(frac_sparse)} of steps have adv_std < '
                f'{SPARSE_ADV_STD} -> flat reward with too little group variance')
    res['verdict'] = verdict
    res['verdict_rule'] = rule
    res['thresholds'] = {
        'sparse_adv_std': SPARSE_ADV_STD,
        'min_steps': args.min_steps,
        'min_reward_gain': MIN_REWARD_GAIN,
        'min_signal_frac': MIN_SIGNAL_FRAC,
        'entropy_collapse_frac': ENTROPY_COLLAPSE_FRAC,
        'kl_runaway': KL_RUNAWAY,
        'tail_window': TAIL,
    }
    return res


# ------------------------------------------------------------------ print
def print_report(res):
    st = res['file']
    n = res['n_steps']
    print(f'GRPO healthcheck  {st["path"]}')
    print(f'  file: {st["bytes"]} B, {st["lines"]} lines, {n} metric records'
          + (f', {st["blank"]} blank' if st['blank'] else '')
          + (f', {st["skipped_invalid"]} invalid' if st['skipped_invalid'] else '')
          + (f', {st["skipped_partial"]} partial-last-line' if st['skipped_partial'] else '')
          + (f'  ERROR={st["err"]}' if st['err'] else ''))
    if n == 0:
        print(f'\n  VERDICT: {res["verdict"]}')
        print(f'  rule: {res["verdict_rule"]}')
        return

    r = res['reward']
    s = res['sparsity']
    k = res['stability']
    a = res['accuracy']

    def row(label, value, note=''):
        print(f'  {label:<34}{value:<22}{note}')

    print()
    print(f'  {"metric":<34}{"value":<22}note')
    print(f'  {"-" * 34:<34}{"-" * 22:<22}{"-" * 30}')
    row('steps', f'{n}', f'first step {res["steps_seen"][0] if res["steps_seen"] else "n/a"}, '
                         f'last {res["steps_seen"][1] if len(res["steps_seen"]) > 1 else "n/a"}')
    row('reward_mean first -> last', f'{fmt(r["first"])} -> {fmt(r["last"])}',
        f'mean {fmt(r["mean"])}, max {fmt(r["max"])}')
    row('reward_mean_20 head20 -> tail20', f'{fmt(r["head20_reward_mean_20"])} -> '
        f'{fmt(r["tail20_reward_mean_20"])}', f'gain20 {fmt(r["gain20"])}')
    row('reward OLS slope / step', f'{fmt(r["ols_slope_per_step"], 8)}',
        f'r={fmt(r["ols_r"], 3)}  rising={r["rising"]}')
    row('acc head20 -> tail20', f'{fmt(a["head20_acc"])} -> {fmt(a["tail20_acc"])}',
        f'gain20 {fmt(a["gain20"])}')
    row('acc OLS slope / step', f'{fmt(a["ols_slope_per_step"], 8)}',
        f'r={fmt(a["pearson_acc_vs_reward"], 3)} vs reward  rising={a["rising"]}')
    row(f'steps adv_std < {SPARSE_ADV_STD}', f'{s["steps_adv_std_below_threshold"]}/'
        f'{s["steps_with_adv_std"]}', f'= {fmt_pct(s["frac_steps_no_signal"])}  '
                                     f'(no learning signal)')
    row('mean adv_std', fmt(s['mean_adv_std']), f'max degenerate_group_frac '
                                               f'{fmt(s["max_degenerate_group_frac"])}')
    row('mean degenerate_group_frac', fmt(s['mean_degenerate_group_frac']),
        'all-wrong / all-right groups')
    row('mean entropy', fmt(k['mean_entropy']),
        f'head {fmt(k["head_entropy"])} -> tail {fmt(k["tail_entropy"])} '
        f'(window {k["trend_window"]})')
    row('entropy trend', fmt(k['entropy_slope_per_step'], 8),
        f'tail/head {fmt(k["tail_over_head_entropy"], 3)}  collapsing='
        f'{k["entropy_collapsing"]}  steps<{ENTROPY_ABS_FLOOR}: '
        f'{k["steps_entropy_below_floor"]}')
    row('mean kl_mean', fmt(k['mean_kl_mean']), f'max |kl| {fmt(k["max_abs_kl_mean"])}')
    row('mean clip_frac', fmt(k['mean_clip_frac']), f'max {fmt(k["max_clip_frac"])}')
    row('steps flagged collapsed', f'{k["steps_flagged_collapsed"]}/{n}',
        f'= {fmt_pct(k["frac_steps_flagged_collapsed"])}')
    row('  .. entropy arm (<0.05)', f'{k["collapsed_steps_entropy_arm"]}',
        'real entropy collapse')
    row('  .. length arm (resp_len cap)', f'{k["collapsed_steps_length_arm"]}',
        f'max resp_len_mean {fmt(k["max_resp_len_mean"], 1)} -> "hit max_gen_len", '
        f'NOT collapse')
    t = res['tokens']
    row('cum_sampled_tokens (last)', f'{t["cum_sampled_tokens_last"]:,.0f}'
        if t['cum_sampled_tokens_last'] is not None else 'n/a',
        f'{fmt(t["sampled_tokens_per_step_mean"], 1)} sampled tok/step')
    print()
    print(f'  VERDICT: {res["verdict"]}')
    print(f'  rule: {res["verdict_rule"]}')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('metrics', nargs='?',
                   default=os.path.join(PROJECT_ROOT, 'logs', 'grpo_metrics.jsonl'),
                   help='per-step GRPO metrics JSONL')
    p.add_argument('--json', default=None, help='write the full analysis to this JSON path')
    p.add_argument('--min-steps', type=int, default=MIN_STEPS,
                   help='fewer steps than this -> INSUFFICIENT_DATA')
    p.add_argument('--quiet', action='store_true', help='verdict line only')
    args = p.parse_args(argv)

    records, stats = read_jsonl(args.metrics)
    res = analyze(records, stats, args)
    if args.quiet:
        print(f'{args.metrics}: {res["verdict"]} - {res["verdict_rule"]}')
    else:
        print_report(res)
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        if not args.quiet:
            print(f'\n  analysis -> {args.json}')
    return 0  # a diagnostic never fails the shell


if __name__ == '__main__':
    sys.exit(main())
