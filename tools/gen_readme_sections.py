#!/usr/bin/env python
"""gen_readme_sections.py -- emit the *data-driven* parts of README.md.

    python gen_readme_sections.py [--out readme_sections.md]
                                 [--eval REPORT.json ...] [--figures-dir figures]

This deliberately does NOT write prose. It prints markdown fragments that the
lead pastes into README.md, each wrapped in explicit BEGIN/END markers so a
re-run can be diffed and the block boundaries are obvious:

    <!-- BEGIN GENERATED: metrics_table (tools/gen_readme_sections.py) -->
    ...
    <!-- END GENERATED: metrics_table -->

Sections
  metrics_table      : the headline numbers of the production pretrain run
                       (steps, tokens, throughput, peak VRAM, loss at the first
                       and last logged step) + the GRPO probes.
  eval_table         : the SFT-vs-GRPO held-out comparison (delegates to
                       make_tables.py so there is exactly one implementation).
  token_accounting   : pretrain / SFT / GRPO token totals.
  figures            : markdown image references for whatever PNGs exist.

Anything not yet available prints an explicit `TODO: ...` marker naming the
missing file -- never a fabricated number. Every real number is read from the
files at run time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from metrics_lib import (fmt_int, fmt_mib, fmt_num, fmt_pct, fmt_secs, fmt_sig,
                         fmt_tokens, last_value, load_json, md_table, project_root,
                         read_jsonl, rel)

import make_tables

PLANNED_PRETRAIN_STEPS = 134817  # 4,314,165 blocks / 32 per step; see pretrain_meta.json


# ------------------------------------------------------------------- fragments


def marker(name: str, tool="tools/gen_readme_sections.py") -> tuple:
    return (f"<!-- BEGIN GENERATED: {name} ({tool}) -->",
            f"<!-- END GENERATED: {name} -->")


def wrap(name: str, body: str) -> str:
    begin, end = marker(name)
    return "\n".join([begin, body.rstrip("\n"), end])


def todo(what, path=None):
    if path:
        return f"TODO: {what} -- missing `{path}`"
    return f"TODO: {what}"


# --------------------------------------------------------------- 1. metrics


def section_metrics(root: Path, total_steps=PLANNED_PRETRAIN_STEPS):
    """Headline metrics table for the production runs, straight from the files."""
    logs = root / "logs"
    metrics = logs / "pretrain_metrics.jsonl"
    summary_p = logs / "pretrain_metrics_summary.json"

    recs, stats = read_jsonl(metrics)
    relm = rel(metrics, root)
    rels = rel(summary_p, root)

    lines = ["### Training metrics", ""]

    if stats["err"]:
        lines += [todo("Phase-1 pretrain metrics", relm), ""]
    elif not recs:
        lines += [todo(f"Phase-1 pretrain metrics (file has {stats['lines']} line(s), "
                       f"no parsable records)", relm), ""]
    else:
        summary, serr = load_json(summary_p)
        first, last = recs[0], recs[-1]
        last_step = last.get("step")
        planned = total_steps
        if summary and summary.get("steps"):
            planned = int(summary["steps"])
        partial = last_step is not None and planned and last_step < planned

        peak = max([r["peak_reserved_MiB"] for r in recs if r.get("peak_reserved_MiB") is not None],
                   default=None)
        tps_vals = [r["tokens_per_sec"] for r in recs if r.get("tokens_per_sec") is not None]
        tps_last = tps_vals[-1] if tps_vals else None
        # Steady-state throughput: mean over the most recent logged points. The
        # window is named in the row label because a growing run has few points.
        steady_n = min(10, len(tps_vals))
        steady = tps_vals[-steady_n:] if steady_n else []
        tps_steady = sum(steady) / len(steady) if steady else None

        rows = [
            ["status",
             (f"**partial -- {fmt_int(last_step)} of {fmt_int(planned)} steps "
              f"({fmt_pct((last_step or 0) / planned if planned else None)})**"
              if partial else f"complete ({fmt_int(last_step)} steps)")],
            ["logged points", f"{fmt_int(stats['records'])} (from `{relm}`)"],
            ["cumulative tokens", fmt_tokens(last.get("cum_tokens"))],
            ["tokens / step", fmt_int(last.get("tokens_per_step"))],
            ["throughput (last logged)", f"{fmt_int(tps_last)} tok/s" if tps_last else "n/a"],
            [f"throughput (mean of last {steady_n} points)",
             f"{fmt_int(tps_steady)} tok/s" if tps_steady else "n/a"],
            ["peak reserved VRAM", fmt_mib(peak)],
            ["loss (first logged step)",
             f"{fmt_num(first.get('loss'), 4)} @ step {fmt_int(first.get('step'))}"],
            ["loss (last logged step)",
             f"{fmt_num(last.get('loss'), 4)} @ step {fmt_int(last.get('step'))}"],
            ["elapsed (last logged)", fmt_secs(last.get("elapsed_sec"))],
            ["ETA (last logged)", f"{fmt_num(last.get('eta_hours'), 2)} h"
             if last.get("eta_hours") is not None else "n/a"],
        ]
        if summary:
            rows += [
                ["run wall time (summary)", fmt_secs(summary.get("wall_sec"))],
                ["final tokens (summary)", fmt_tokens(summary.get("total_tokens"))],
            ]
        else:
            rows += [["run wall time (summary)",
                      todo("Phase-1 summary is only written when the run ends", rels)]]
        lines += [md_table(["metric", "value"], rows), ""]

        if stats["skipped_partial"]:
            lines += [f"_Note: {stats['skipped_partial']} partially written trailing "
                      f"line(s) were skipped; the file is still being appended to._", ""]

    # ---- GRPO probes
    grpo_rows = []
    for name in ("grpo_rule_smoke", "grpo_easy_probe"):
        gp = logs / f"{name}.jsonl"
        sp = logs / f"{name}_summary.json"
        grecs, gstats = read_jsonl(gp)
        gsum, gerr = load_json(sp)
        if gstats["err"] and gerr:
            grpo_rows.append([name, todo("GRPO probe metrics", rel(gp, root)),
                              "n/a", "n/a", "n/a", "n/a"])
            continue
        n = gstats["records"]
        coll = sum(1 for r in grecs if r.get("collapsed"))
        rows = [r for r in grecs if r.get("reward_mean") is not None]
        first_rw = rows[0]["reward_mean"] if rows else None
        last_rw = rows[-1]["reward_mean"] if rows else None
        acc_last = last_value(grecs, "acc")
        adv_last = last_value(grecs, "adv_std")
        vram = gsum.get("peak_reserved_MiB") if gsum else None
        grpo_rows.append([
            name,
            f"{fmt_int(n)} steps",
            f"{fmt_num(first_rw, 4)} -> {fmt_num(last_rw, 4)}",
            f"acc {fmt_num(acc_last, 3)}, adv_std {fmt_num(adv_last, 3)}",
            f"{fmt_int(coll)} / {fmt_int(n)}",
            fmt_mib(vram) if vram is not None else todo("GRPO summary", rel(sp, root)),
        ])
    lines += ["#### GRPO rule-reward probes", "",
              md_table(["run", "steps", "reward_mean (first -> last)", "final accuracy",
                        "collapsed steps", "peak reserved VRAM"], grpo_rows), ""]
    return wrap("metrics_table", "\n".join(lines))


# -------------------------------------------------------------------- 2. eval


def section_eval(root: Path, eval_reports):
    lines = ["### Held-out eval comparison", ""]
    if not eval_reports:
        # Fall back to whatever eval reports exist on disk, newest first.
        cand = []
        for p in sorted((root / "reports").glob("*.json")):
            data, err = load_json(p)
            if err or not isinstance(data, dict):
                continue
            if "overall" in data and "per_family" in data:
                cand.append(str(p))
        if len(cand) >= 2:
            eval_reports = cand

    if not eval_reports or len(eval_reports) < 2:
        lines.append(todo("eval comparison needs >= 2 held-out eval reports "
                          "(pass --eval <a.json> <b.json>)"))
        lines.append("")
        if eval_reports:
            lines.append("Only 1 report supplied:")
            lines.append("")
        found = [p for p in sorted((root / "reports").glob("*.json"))]
        lines += ["Held-out eval reports currently on disk:", ""]
        any_found = False
        for p in found:
            data, err = load_json(p)
            if err or not isinstance(data, dict) or "overall" not in data:
                continue
            any_found = True
            lines.append(
                f"- `{rel(p, root)}` -- {data.get('prompt_mode')} / "
                f"max_new_tokens={data.get('max_new_tokens')} / "
                f"overall {fmt_pct((data.get('overall') or {}).get('accuracy'))} "
                f"({fmt_int((data.get('overall') or {}).get('correct'))}"
                f"/{fmt_int((data.get('overall') or {}).get('total'))})")
        if not any_found:
            lines.append("- none")
        lines.append("")
        return wrap("eval_table", "\n".join(lines))

    body = make_tables.table_eval_comparison(root, eval_reports)
    # make_tables emits its own "## Eval comparison" heading; reuse the body only.
    body = body.split("\n", 1)[1].lstrip("\n") if body.startswith("## ") else body
    lines.append(body)
    lines.append("")
    return wrap("eval_table", "\n".join(lines))


# ------------------------------------------------------------------ 3. tokens


def section_tokens(root: Path, sft_summary=None, grpo_summaries=None):
    logs = root / "logs"
    if grpo_summaries is None:
        grpo_summaries = sorted(str(p) for p in logs.glob("grpo*_summary.json"))
    body = make_tables.table_token_accounting(
        root, logs / "pretrain_metrics_summary.json", grpo_summaries, sft_summary)
    body = body.split("\n", 1)[1].lstrip("\n") if body.startswith("## ") else body
    return wrap("token_accounting", body)


# ----------------------------------------------------------------- 4. figures


# (filename, caption, which tool call produces it)
EXPECTED_FIGURES = [
    ("pretrain_curve_partial.png",
     "Phase-1 pretrain (partial): loss vs cumulative tokens and throughput vs step. "
     "Source: `logs/pretrain_metrics.jsonl`.",
     "python tools/plot_curves.py --metrics logs/pretrain_metrics.jsonl "
     "--out figures/pretrain_curve_partial.png --kind pretrain --total-steps 134817"),
    ("grpo_probe_curves.png",
     "GRPO rule-reward probe: reward / accuracy / KL-entropy per step. Shaded steps "
     "are `collapsed=true`. Source: `logs/grpo_easy_probe.jsonl`.",
     "python tools/plot_curves.py --metrics logs/grpo_easy_probe.jsonl "
     "--out figures/grpo_probe_curves.png --kind grpo --total-steps 30"),
    ("pretrain_curve_full.png",
     "Phase-1 pretrain (complete run). Generated once the 28 h run finishes and "
     "`logs/pretrain_metrics_summary.json` exists.",
     "python tools/plot_curves.py --metrics logs/pretrain_metrics.jsonl "
     "--out figures/pretrain_curve_full.png --kind pretrain --total-steps 134817"),
]


def _png_info(path: Path):
    """(width, height) from the PNG IHDR header, or None. Stdlib only."""
    try:
        with open(path, "rb") as f:
            head = f.read(33)
    except OSError:
        return None
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    if head[12:16] != b"IHDR":
        return None
    w = int.from_bytes(head[16:20], "big")
    h = int.from_bytes(head[20:24], "big")
    return w, h


def section_figures(root: Path):
    figs = root / "figures"
    lines = ["### Figures", ""]
    missing = []
    for name, caption, cmd in EXPECTED_FIGURES:
        p = figs / name
        if not p.exists():
            missing.append(name)
            lines += [f"<!-- {name}: TODO (not generated yet) -->",
                      f"**{name}** -- {todo('figure not generated', rel(p, root))}", ""]
            continue
        dims = _png_info(p)
        size = p.stat().st_size
        dim_txt = f"{dims[0]}x{dims[1]} px" if dims else "dimensions unreadable"
        lines += [f"![{caption}]({rel(p, root)})", "",
                  f"*{caption}* ({dim_txt}, {size:,} bytes)", "",
                  f"<sub>regenerate: `{cmd}`</sub>", ""]
    if missing:
        lines += ["_Missing figures (TODO): " + ", ".join(f"`{m}`" for m in missing) + "_", ""]
    return wrap("figures", "\n".join(lines))


# ------------------------------------------------------------------------ main


def build_parser():
    p = argparse.ArgumentParser(
        description="Emit the data-driven README fragments (no CUDA, no prose).")
    p.add_argument("--out", default=None, help="write markdown here (default: stdout)")
    p.add_argument("--root", default=None, help="project root (default: parent of tools/)")
    p.add_argument("--eval", nargs="+", default=None, metavar="REPORT.json",
                   help="2+ held-out eval reports to compare")
    p.add_argument("--sft-summary", default=None, help="SFT summary JSON (optional)")
    p.add_argument("--grpo-summary", nargs="+", default=None,
                   help="GRPO summary JSONs (default: all logs/grpo*_summary.json)")
    p.add_argument("--total-pretrain-steps", type=int, default=PLANNED_PRETRAIN_STEPS)
    p.add_argument("--section", action="append",
                   choices=["metrics", "eval", "tokens", "figures"],
                   help="emit only these sections (default: all)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = project_root(args.root)
    sections = set(args.section) if args.section else {"metrics", "eval", "tokens", "figures"}

    blocks = []
    if "metrics" in sections:
        blocks.append(section_metrics(root, args.total_pretrain_steps))
    if "eval" in sections:
        blocks.append(section_eval(root, args.eval))
    if "tokens" in sections:
        blocks.append(section_tokens(root, args.sft_summary, args.grpo_summary))
    if "figures" in sections:
        blocks.append(section_figures(root))

    text = "\n\n".join(blocks) + "\n"
    if args.out:
        outp = Path(args.out)
        if not outp.is_absolute():
            outp = root / outp
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(text, encoding="utf-8", newline="\n")
        print(f"[gen_readme_sections] wrote {outp} ({len(text):,} chars)")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
