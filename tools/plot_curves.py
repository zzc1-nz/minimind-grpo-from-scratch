#!/usr/bin/env python
"""plot_curves.py -- turn a training metrics JSONL into a report-ready PNG.

    python plot_curves.py --metrics <jsonl> --out <png> \
        [--kind pretrain|grpo|auto] [--title T]

Kinds
  pretrain : 2 stacked panels
             (a) loss vs cumulative tokens, x axis in **B tokens** (cum_tokens)
             (b) tokens/sec vs step, with the peak-reserved-VRAM value annotated
  grpo     : 3 stacked panels
             (a) reward_mean + reward_mean_20 vs step
             (b) acc + adv_std vs step
             (c) KL + entropy vs step
             every step where "collapsed" is true is shaded
  auto     : infer the kind from the record keys (pretrain if cum_tokens +
             tokens_per_sec are present, grpo if reward_mean + kl_mean are).

Robustness (all covered by tools/selfcheck_tools.py)
  * a JSONL that is *still being appended to*: an unterminated final line that
    does not parse is counted as "skipped_partial" and dropped;
  * empty file / header-less file / no usable rows -> clear message, exit 0,
    no traceback, and NO PNG is written;
  * fewer than 2 usable points -> same;
  * a missing metrics file -> clear message, exit 2 (a real error, not "no data");
  * missing individual metrics -> gaps in the line, never plotted as zeros.

No CUDA, no torch, no model loading. matplotlib Agg backend only. Pure
numpy/matplotlib/stdlib; matplotlib does not import torch.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must happen before pyplot import; never opens a window

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from metrics_lib import fmt_int, read_jsonl, series

DEFAULT_DPI = 150
DEFAULT_SIZE = (1600, 900)  # px at dpi=150 -> figsize (10.667, 6.0) inches

# Colours (fixed so repeated runs are pixel-identical)
C_LOSS = "#1f4e79"
C_MA = "#c00000"
C_RAW = "#9fb6cd"
C_TPS = "#1f4e79"
C_REWARD = "#1f4e79"
C_REWARD20 = "#c00000"
C_ACC = "#1f4e79"
C_ADV = "#2e7d32"
C_KL = "#1f4e79"
C_ENT = "#c00000"
C_COLLAPSED = "#d62728"


# ------------------------------------------------------------------ utilities


def _clean(ys):
    """Replace non-finite floats with None so matplotlib draws gaps, not zeros."""
    out = []
    for y in ys:
        if y is None or (isinstance(y, float) and not math.isfinite(y)):
            out.append(None)
        else:
            out.append(float(y))
    return out


def _moving_average(ys, window):
    """Trailing window mean that ignores None. Returns a same-length list."""
    window = max(1, int(window))
    out = []
    queue = []
    for y in ys:
        if y is not None:
            queue.append(y)
            if len(queue) > window:
                queue.pop(0)
        out.append(sum(queue) / len(queue) if queue else None)
    return out


def _decimate(xs, ys, target):
    """Keep at most `target` evenly spaced points (always the last one).

    A 134,817-step run must not produce a 200 MB PNG, but the endpoint must
    still be visible, so the stride is derived from n and the tail is kept.
    """
    n = len(xs)
    if n <= target or target <= 1:
        return xs, ys
    stride = int(math.ceil(n / float(target)))
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return [xs[i] for i in idx], [ys[i] for i in idx]


def _ma_window(n_points):
    """Window that yields ~200 smoothed points regardless of run length."""
    return max(1, int(round(n_points / 200.0)))


def _ranges(steps):
    """[1,2,3,7,9] -> [(1,3),(7,7),(9,9)] for merged shaded spans."""
    out = []
    for s in steps:
        if out and s == out[-1][1] + 1:
            out[-1] = (out[-1][0], s)
        else:
            out.append((s, s))
    return out


def _token_axis_formatter(ax, xs):
    """Tick labels for a B-token axis with enough digits for the actual ticks.

    A full run spans 0..2.2 B while the first few hundred steps only reach
    0.02 B, so a fixed format either prints '0.0, 0.0, 0.0' (short runs) or
    rounds 0.25 to '0.2' (long runs). Choose a "nice" tick step, then size the
    format to that step so every printed label is exact.
    """
    import math as _math

    from matplotlib.ticker import FuncFormatter, MultipleLocator

    vals = [x for x in xs if x is not None]
    if not vals:
        return
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 0:
        return
    # nice step: 1/2/5 x 10^k, aiming at <= 8 major ticks
    raw = span / 8.0
    mag = 10 ** _math.floor(_math.log10(raw))
    for mult in (1, 2, 5, 10):
        step = mult * mag
        if step >= raw:
            break
    decimals = max(0, -int(_math.floor(_math.log10(step))))
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:.{decimals}f}"))


def _style(ax, ylabel, xlabel=None, title=None):
    ax.set_facecolor("white")
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.8)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_ylabel(ylabel, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, loc="left", fontweight="bold")


def _header(fig, title, notes):
    fig.suptitle(title, fontsize=15, fontweight="bold", x=0.008, ha="left", y=0.985)
    if notes:
        fig.text(0.008, 0.955, "\n".join(notes), fontsize=9, color="#404040",
                 ha="left", va="top", family="monospace")


def _finish(fig, out, dpi):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Fixed metadata so the PNG bytes are reproducible run-to-run.
    meta = {"Software": "llm-rl-lab plot_curves.py", "Creation Time": None}
    fig.savefig(out, dpi=dpi, facecolor="white", metadata=meta)
    plt.close(fig)
    return out


def _empty_result(msg, out, exit_code=0):
    """Report 'no figure' without a traceback. Never writes a misleading PNG."""
    print(f"[plot_curves] {msg}", file=sys.stderr)
    if out and Path(out).exists():
        print(f"[plot_curves] existing file {out} is STALE (not overwritten)", file=sys.stderr)
    return exit_code


def _infer_kind(records):
    keys = set()
    for rec in records[:50]:
        keys.update(rec.keys())
    if {"reward_mean", "kl_mean", "acc"} <= keys:
        return "grpo"
    if {"loss", "tokens_per_sec"} <= keys and ({"cum_tokens"} & keys):
        return "pretrain"
    if "reward_mean" in keys:
        return "grpo"
    if "loss" in keys:
        return "pretrain"
    return None


def _stats_note(stats, kind):
    bits = [f"source: {Path(stats['path']).name}",
            f"{stats['records']:,} rows / {stats['lines']:,} lines"]
    if stats["skipped_partial"]:
        bits.append(f"skipped {stats['skipped_partial']} partial trailing line (file still being written)")
    if stats["skipped_invalid"]:
        lnos = ",".join(str(x) for x in stats["invalid_linenos"][:8])
        bits.append(f"skipped {stats['skipped_invalid']} unparsable line(s): {lnos}")
    if stats["blank"]:
        bits.append(f"{stats['blank']} blank line(s)")
    return " | ".join(bits)


# --------------------------------------------------------------------- panels


def _pretrain_panels(ax_loss, ax_tps, records, total_steps=None):
    steps = series(records, "step", int)
    loss = _clean(series(records, "loss"))
    tps = _clean(series(records, "tokens_per_sec"))
    cum = series(records, "cum_tokens", float)

    # ---- x axis for the loss curve: cumulative tokens, in B tokens
    if any(v is not None for v in cum):
        xs_tok = [None if v is None else v / 1e9 for v in cum]
        xlabel = "cumulative tokens (B)"
        cum_source = "cum_tokens"
    else:
        # documented fallback if the field is ever absent
        per_step = series(records, "tokens_per_step", float)
        xs_tok = []
        for s, p in zip(steps, per_step):
            xs_tok.append(None if s is None else (s * (p or 0)) / 1e9)
        xlabel = "cumulative tokens (B, from step x tokens_per_step)"
        cum_source = "step*tokens_per_step"

    pairs = [(x, y) for x, y in zip(xs_tok, loss) if x is not None and y is not None]
    if pairs:
        px = [p[0] for p in pairs]
        py = [p[1] for p in pairs]
        window = _ma_window(len(py))
        ma = _moving_average(py, window)
        if window > 1:
            # A short run has no smoothing to do: drawing "raw" and "moving avg"
            # would be the same line twice, so only the raw trace is shown.
            dx, dy = _decimate(px, py, 4000)
            ax_loss.plot(dx, dy, color=C_RAW, linewidth=0.7, alpha=0.55,
                         label=f"raw ({len(py):,} pts)")
            maxx, may = _decimate(px, ma, 1500)
            ax_loss.plot(maxx, may, color=C_MA, linewidth=2.0,
                         label=f"moving avg (window {window} pts)")
        else:
            maxx, may = _decimate(px, py, 1500)
            ax_loss.plot(maxx, may, color=C_MA, linewidth=1.6, marker="o",
                         markersize=3, label=f"loss ({len(py):,} pts, no downsampling)")
        ax_loss.set_xlim(left=min(px), right=max(px) if max(px) > min(px) else min(px) + 1e-9)
        first_loss, last_loss = py[0], py[-1]
        # Both labels live in the top-right corner, which is empty for a
        # descending loss curve; a white box keeps them legible over the grid.
        ax_loss.text(0.995, 0.95,
                     f"first  {first_loss:.4f}\nlast   {last_loss:.4f}",
                     transform=ax_loss.transAxes, fontsize=9, color="#404040",
                     ha="right", va="top", family="monospace",
                     bbox=dict(facecolor="white", edgecolor="#d0d0d0",
                               boxstyle="round,pad=0.3", alpha=0.9))
        ax_loss.legend(loc="center right", fontsize=9, frameon=False)
        _token_axis_formatter(ax_loss, px)
    _style(ax_loss, "training loss", xlabel,
           f"(a) loss vs cumulative tokens  [{cum_source}]")

    # ---- tokens/sec vs step
    st = [s for s in steps]
    tp = tps
    pairs = [(x, y) for x, y in zip(st, tp) if x is not None and y is not None]
    if pairs:
        px = [p[0] for p in pairs]
        py = [p[1] for p in pairs]
        window = _ma_window(len(py))
        ma = _moving_average(py, window)
        if window > 1:
            dx, dy = _decimate(px, py, 4000)
            ax_tps.plot(dx, dy, color=C_RAW, linewidth=0.7, alpha=0.55,
                        label=f"raw ({len(py):,} pts)")
            maxx, may = _decimate(px, ma, 1500)
            ax_tps.plot(maxx, may, color=C_TPS, linewidth=2.0,
                        label=f"moving avg (window {window} pts)")
        else:
            maxx, may = _decimate(px, py, 1500)
            ax_tps.plot(maxx, may, color=C_TPS, linewidth=1.6, marker="o",
                        markersize=3, label=f"tokens/sec ({len(py):,} pts)")
        ax_tps.set_ylim(0, max(py) * 1.32)
        ax_tps.set_xlim(left=min(px), right=max(px) if max(px) > min(px) else min(px) + 1e-9)
        ax_tps.text(0.995, 0.97, f"last {py[-1]:,.0f} tok/s", transform=ax_tps.transAxes,
                    fontsize=9, color="#404040", ha="right", va="top")
        # legend in the upper-left, the VRAM line is annotated at the left edge
        # of the data, so put the legend above it
        ax_tps.legend(loc="upper left", fontsize=9, frameon=False)

    peak = series(records, "peak_reserved_MiB", float)
    peak_vals = [p for p in peak if p is not None]
    if peak_vals:
        peak_v = max(peak_vals)
        ax_tps.axhline(peak_v, color="#7f7f7f", linestyle=":", linewidth=1.2)
        ax_tps.text(0.995, peak_v, f"peak reserved VRAM {peak_v:,.0f} MiB ",
                    transform=ax_tps.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=9, color="#404040")
    xlabel = "step"
    if total_steps and steps and steps[-1]:
        xlabel = f"step (of {total_steps:,} planned)"
    _style(ax_tps, "tokens / sec", xlabel, "(b) throughput vs step")


def _grpo_panels(axs, records, total_steps=None):
    steps = series(records, "step", int)
    st = [s for s in steps]

    def plot_pair(ax, key_a, key_b, lab_a, lab_b, col_a, col_b, panel_title, ylabel,
                  right_ylabel=None):
        """Plot two series. When their scales differ by >4x, use a twin y axis
        instead of squashing the smaller one onto the x axis (kl_mean ~ -0.01
        vs entropy ~ 2.9 would otherwise be a flat line at zero)."""
        a = _clean(series(records, key_a))
        b = _clean(series(records, key_b)) if key_b else None
        va = [v for v in a if v is not None]
        vb = [v for v in b if v is not None] if b else []
        drawn = False
        twin = False
        if va and vb and right_ylabel:
            sa = max(abs(v) for v in va) or 1.0
            sb = max(abs(v) for v in vb) or 1.0
            twin = max(sa, sb) / max(min(sa, sb), 1e-12) > 4.0
        ax_b = ax.twinx() if twin else ax
        if va:
            ax.plot(st, a, color=col_a, linewidth=1.6, marker="o", markersize=2.5,
                    label=lab_a)
            drawn = True
        if vb:
            ax_b.plot(st, b, color=col_b, linewidth=1.8, linestyle="--", label=lab_b)
            drawn = True
        if drawn:
            h1, l1 = ax.get_legend_handles_labels()
            if twin:
                h2, l2 = ax_b.get_legend_handles_labels()
                h1, l1 = h1 + h2, l1 + l2
            ax.legend(h1, l1, loc="best", fontsize=9, frameon=False)
        _style(ax, ylabel, None, panel_title)
        if twin:
            ax_b.set_ylabel(right_ylabel, fontsize=11)
            ax_b.grid(False)
            for spine in ("top", "left"):
                ax_b.spines[spine].set_visible(False)
            # colour the twin axis label to match its series
            ax_b.yaxis.label.set_color(col_b)
            ax_b.tick_params(axis="y", colors=col_b, labelsize=9)
        return twin

    plot_pair(axs[0], "reward_mean", "reward_mean_20",
              "reward_mean", "reward_mean_20 (20-step avg)",
              C_REWARD, C_REWARD20, "(a) reward vs step", "reward")
    plot_pair(axs[1], "acc", "adv_std",
              "acc", "adv_std", C_ACC, C_ADV, "(b) accuracy and advantage std", "value")
    twin = plot_pair(axs[2], "kl_mean", "entropy",
                     "kl_mean", "entropy", C_KL, C_ENT,
                     "(c) KL and entropy", "kl_mean", right_ylabel="entropy")
    if twin:
        axs[2].set_ylabel("kl_mean", fontsize=11, color=C_KL)
        axs[2].tick_params(axis="y", colors=C_KL, labelsize=9)
    axs[2].set_xlabel("step (of %s planned)" % f"{total_steps:,}" if total_steps else "step",
                      fontsize=11)

    # ---- collapsed steps: merged shaded spans on all three panels
    collapsed = [s for s, c in zip(st, series(records, "collapsed", bool)) if c and s is not None]
    if collapsed:
        for lo, hi in _ranges(collapsed):
            for ax in axs:
                ax.axvspan(lo - 0.5, hi + 0.5, color=C_COLLAPSED, alpha=0.10,
                           linewidth=0, zorder=0)
        # one shared legend entry for the shading, appended to panel (a)
        proxy = Patch(facecolor=C_COLLAPSED, alpha=0.18,
                      label=f"collapsed ({len(collapsed)} step(s))")
        handles, labels = axs[0].get_legend_handles_labels()
        handles.append(proxy)
        labels.append(proxy.get_label())
        axs[0].legend(handles, labels, loc="best", fontsize=9, frameon=False)
    for ax in axs[:-1]:
        ax.set_xlim(left=min(st) - 1 if st else None)
    return collapsed


# ----------------------------------------------------------------------- main


def build_parser():
    p = argparse.ArgumentParser(
        description="Plot MiniMind training metrics JSONL to a PNG (no CUDA).")
    p.add_argument("--metrics", required=True, help="metrics JSONL path")
    p.add_argument("--out", required=True, help="output PNG path")
    p.add_argument("--kind", default="auto", choices=["auto", "pretrain", "grpo"])
    p.add_argument("--title", default=None, help="figure title")
    p.add_argument("--note", action="append", default=[],
                   help="extra subtitle line (repeatable)")
    p.add_argument("--total-steps", type=int, default=None,
                   help="planned total steps; enables the 'partial: N of M' label")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    p.add_argument("--width", type=int, default=DEFAULT_SIZE[0])
    p.add_argument("--height", type=int, default=DEFAULT_SIZE[1])
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    records, stats = read_jsonl(args.metrics)
    if stats["err"]:
        return _empty_result(f"cannot read metrics: {stats['err']}", args.out, exit_code=2)

    # Say it on stderr too, not only in the figure subtitle: a caller piping
    # this into a build log must see that data was dropped.
    if stats["skipped_partial"]:
        print(f"[plot_curves] WARNING: skipped {stats['skipped_partial']} partially "
              f"written trailing line(s) -- {args.metrics} is still being appended to",
              file=sys.stderr)
    if stats["skipped_invalid"]:
        print(f"[plot_curves] WARNING: skipped {stats['skipped_invalid']} unparsable "
              f"line(s) at {stats['invalid_linenos'][:8]}", file=sys.stderr)

    if not records:
        extra = ""
        if stats["skipped_partial"]:
            extra = " (only a partially written trailing line was present)"
        return _empty_result(
            f"no usable records in {args.metrics} -- {stats['lines']} line(s), "
            f"{stats['bytes']} byte(s){extra}; no figure written", args.out, exit_code=0)

    kind = args.kind
    if kind == "auto":
        kind = _infer_kind(records)
        if kind is None:
            return _empty_result(
                "cannot infer kind from record keys "
                f"({sorted(records[0].keys())}); pass --kind pretrain|grpo", args.out, exit_code=0)
    print(f"[plot_curves] kind={kind} (requested {args.kind})")

    # Minimum data: 2 usable points for the primary series of the kind.
    primary = "loss" if kind == "pretrain" else "reward_mean"
    usable = [v for v in series(records, primary) if v is not None]
    if len(usable) < 2:
        return _empty_result(
            f"only {len(usable)} usable '{primary}' point(s) in {args.metrics} "
            f"(need >= 2); no figure written", args.out, exit_code=0)

    dpi = args.dpi
    figsize = (args.width / float(dpi), args.height / float(dpi))

    title = args.title
    if title is None:
        title = ("Phase-1 pretrain loss / throughput" if kind == "pretrain"
                 else "GRPO training curves")

    notes = []
    if args.total_steps:
        n_steps = max([s for s in series(records, "step", int) if s is not None] or [0])
        if n_steps and n_steps < args.total_steps:
            notes.append(f"PARTIAL: {n_steps:,} steps of {args.total_steps:,} planned")
        else:
            notes.append(f"{n_steps:,} steps of {args.total_steps:,} planned")
    notes.extend(args.note)
    notes.append(_stats_note(stats, kind))

    if kind == "pretrain":
        fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=dpi)
        _header(fig, title, notes)
        _pretrain_panels(axes[0], axes[1], records, args.total_steps)
    else:
        fig, axes = plt.subplots(3, 1, figsize=figsize, dpi=dpi)
        _header(fig, title, notes)
        collapsed = _grpo_panels(axes, records, args.total_steps)
        if collapsed:
            print(f"[plot_curves] collapsed steps: {len(collapsed)} "
                  f"of {len(records)} ({', '.join(str(c) for c in collapsed[:20])}"
                  f"{', ...' if len(collapsed) > 20 else ''})")

    fig.subplots_adjust(top=0.86, bottom=0.075, left=0.075,
                        right=0.93 if kind == "grpo" else 0.985, hspace=0.42)
    out = _finish(fig, args.out, dpi)
    print(f"[plot_curves] wrote {out} ({out.stat().st_size:,} bytes, "
          f"{args.width}x{args.height} px @ {dpi} dpi)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
