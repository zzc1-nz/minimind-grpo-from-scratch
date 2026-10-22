#!/usr/bin/env python
"""make_tables.py -- build the README / report tables from the raw project files.

    python make_tables.py [--out tables.md] [--runs] [--tokens]
                          [--eval REPORT1.json REPORT2.json ...]

Sections
  training runs    : one row per run found in logs/*_summary.json (plus the
                     canonical-but-missing Phase-1 summary, which shows as an
                     explicit n/a row). Steps / tokens / wall time / peak VRAM
                     come from the summary; the run config comes from the
                     matching logs/vram_train_*.json argv; the final loss comes
                     from the last record of the sibling *.jsonl metrics file.
  eval comparison  : overall + per-family accuracy side by side for 2+ held-out
                     eval reports. Reports whose prompt_mode / max_new_tokens
                     differ are still tabulated but preceded by a loud warning
                     (a mismatch is itself a finding -- it must not be silent).
  token accounting : pretrain / SFT / GRPO token totals from the summary files
                     (GRPO uses cum_sampled_tokens) plus a known-components sum.

Every number is read from a file. Anything absent prints as
"n/a (file not found: <path>)" or "n/a (<reason>)"; nothing is invented.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from metrics_lib import (fmt_int, fmt_mib, fmt_num, fmt_pct, fmt_secs, fmt_sig,
                        fmt_tokens, last_value, load_json, md_table, na_cell,
                        project_root, read_jsonl, rel, warn_block)

# --------------------------------------------------------------------- helpers

# Files this project is expected to produce. Missing ones get an explicit row
# instead of silently vanishing from the table.
EXPECTED_SUMMARIES = [
    # (summary path relative to root, phase label, metrics jsonl relative to root)
    ("logs/pretrain_metrics_summary.json", "Phase 1 -- pretrain (production)",
     "logs/pretrain_metrics.jsonl"),
]

# Order the auto-discovered runs so the table reads chronologically.
PHASE_ORDER = {
    "Phase 0": 0,
    "Phase 1": 1,
    "Phase 2": 2,
    "Phase 3": 3,
    "Phase 4": 4,
}


def phase_of(name: str) -> str:
    """Infer the project phase from a run name (filename stem)."""
    n = name.lower()
    if n.startswith("pretrain_metrics") or n == "pretrain":
        return "Phase 1"
    if n.startswith("sft"):
        return "Phase 2"
    if n.startswith("grpo"):
        return "Phase 3"
    return "Phase 0"


def parse_argv(argv):
    """--flag value / --flag (bool) -> dict. Tolerant of a missing value."""
    out = {}
    i = 0
    while i < len(argv):
        tok = str(argv[i])
        if tok.startswith("--"):
            key = tok.lstrip("-")
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            if nxt is not None and not str(nxt).startswith("--"):
                out[key] = nxt
                i += 2
                continue
            out[key] = True
            i += 1
            continue
        i += 1
    return out


def find_run_metadata(logs_dir: Path, save_weight, script_hint=None):
    """Find the newest logs/vram_train_*.json whose --save_weight matches.

    Returns (metadata_dict, path) or (None, None). The wrapper records the exact
    argv and measured peak VRAM of each run, which is the only place the run
    config is recorded.
    """
    if not save_weight:
        return None, None
    best, best_path = None, None
    for p in sorted(Path(logs_dir).glob("vram_train_*.json"), key=lambda x: x.stat().st_mtime):
        data, err = load_json(p)
        if err or not isinstance(data, dict):
            continue
        flags = parse_argv(data.get("argv") or [])
        sw = flags.get("save_weight")
        if not isinstance(sw, str) or sw != save_weight:
            continue
        if script_hint and data.get("script") and script_hint not in str(data["script"]):
            continue
        best, best_path = data, p  # keep the latest mtime match
    return best, best_path


def config_from_flags(flags):
    """Compact, factual one-line config description from a run argv."""
    if not flags:
        return None
    bits = []
    if "hidden_size" in flags or "num_hidden_layers" in flags:
        bits.append(f"h{flags.get('hidden_size', '?')}/L{flags.get('num_hidden_layers', '?')}")
    if "batch_size" in flags:
        bits.append(f"bs{flags['batch_size']}")
    if "max_seq_len" in flags:
        bits.append(f"seq{flags['max_seq_len']}")
    if "num_generations" in flags:
        bits.append(f"group{flags['num_generations']}")
    if "max_gen_len" in flags:
        bits.append(f"gen{flags['max_gen_len']}")
    if "learning_rate" in flags:
        bits.append(f"lr{flags['learning_rate']}")
    if str(flags.get("use_compile")) in ("1", "True", "true"):
        bits.append("compile")
    if "max_steps" in flags:
        bits.append(f"max_steps{flags['max_steps']}")
    if "epochs" in flags:
        bits.append(f"epochs{flags['epochs']}")
    return ", ".join(bits) if bits else None


def run_name_from_summary(summary, path: Path):
    """A human run name: the final_weight stem, else the summary file stem."""
    fw = summary.get("final_weight")
    if isinstance(fw, str) and fw:
        return Path(fw).name.replace(".pth", "")
    return Path(path).stem.replace("_summary", "")


def sibling_metrics_path(summary_path: Path) -> Path:
    """logs/foo_summary.json -> logs/foo.jsonl (that is the writer's convention)."""
    name = Path(summary_path).name
    stem = name[:-len("_summary.json")] if name.endswith("_summary.json") else Path(name).stem
    return Path(summary_path).with_name(stem + ".jsonl")


def read_summary_row(path: Path, root: Path, phase_hint=None):
    """Build one training-run row from a summary JSON + its sibling JSONL."""
    summary, err = load_json(path)
    relp = rel(path, root)
    if not err and not isinstance(summary, dict):
        # Parsed but not an object (e.g. a JSON list): treat as unusable rather
        # than crashing on summary.get().
        summary, err = None, f"unexpected JSON type: {type(summary).__name__}"
    if err:
        return {
            "run": Path(path).stem.replace("_summary", ""),
            "phase": phase_hint or phase_of(path.stem),
            "steps": na_cell(relp, err),
            "tokens_per_step": "n/a",
            "config": "n/a",
            "tokens": "n/a",
            "wall": "n/a",
            "vram": "n/a",
            "final_loss": "n/a",
            "source": f"`{relp}` (missing)",
            "_sort": (PHASE_ORDER.get((phase_hint or phase_of(path.stem)).split(" --")[0], 9),
                      Path(path).name),
        }

    name = run_name_from_summary(summary, path)
    phase = phase_hint or phase_of(name)
    steps = summary.get("steps")
    epochs = summary.get("epochs")
    steps_cell = fmt_int(steps)
    if steps is not None and epochs not in (None, 1):
        steps_cell += f" ({fmt_int(epochs)} epochs)"

    # config: match the wrapper metadata by save_weight
    flags = None
    meta, meta_path = find_run_metadata(root / "logs", name.split("_")[0] if "_" in name else name)
    if meta is None:
        # try the full stem first (e.g. save_weight "grpo_easy_probe"), then the
        # stem with the hidden-size suffix removed (e.g. "pretrain_768")
        for cand in (name, name.rsplit("_", 1)[0] if "_" in name else None):
            if not cand:
                continue
            meta, meta_path = find_run_metadata(root / "logs", cand)
            if meta is not None:
                break
    if meta is not None:
        flags = parse_argv(meta.get("argv") or [])
    config = config_from_flags(flags) or "n/a (no run argv recorded)"

    # tokens: pretrain/SFT summaries use total_tokens, GRPO uses cum_sampled_tokens
    tokens = summary.get("total_tokens")
    tokens_note = ""
    if tokens is None:
        tokens = summary.get("cum_sampled_tokens")
        if tokens is not None:
            prompt_tok = summary.get("cum_prompt_tokens")
            tokens_note = "sampled" + (f" + {fmt_int(prompt_tok)} prompt" if prompt_tok is not None else "")
    tokens_cell = fmt_tokens(tokens) if tokens is not None else "n/a (not in summary)"

    # peak VRAM: reserved is the number the reports quote
    vram = summary.get("peak_reserved_MiB")
    vram_cell = fmt_mib(vram) if vram is not None else "n/a"
    alloc = summary.get("peak_alloc_MiB")
    if alloc is not None and vram is not None:
        vram_cell += f" (alloc {int(alloc):,})"

    # final loss from the last record of the sibling JSONL
    jpath = sibling_metrics_path(path)
    recs, stats = read_jsonl(jpath)
    final_loss = last_value(recs, "loss") if recs else None
    loss_cell = fmt_num(final_loss, 4) if final_loss is not None else "n/a (no metrics jsonl)"
    if recs:
        last_step = last_value(recs, "step")
        if last_step is not None and steps is not None and int(last_step) < int(steps):
            loss_cell += f" (at step {int(last_step):,})"

    source = f"`{relp}`"
    if meta_path is not None:
        source += f" + `{rel(meta_path, root)}`"
    source += f" + `{rel(jpath, root)}`"

    return {
        "run": name,
        "phase": phase,
        "steps": steps_cell,
        "tokens_per_step": fmt_int(summary.get("tokens_per_step")),
        "config": config,
        "tokens": tokens_cell + (f" ({tokens_note})" if tokens_note else ""),
        "wall": fmt_secs(summary.get("wall_sec")),
        "vram": vram_cell,
        "final_loss": loss_cell,
        "source": source,
        "_sort": (PHASE_ORDER.get(phase.split(" --")[0], 9), Path(path).name),
    }


def live_pretrain_row(root: Path, metrics_path: Path, total_steps, summary_path: Path):
    """Row for an in-progress run: the summary does not exist yet, the JSONL does.

    This is explicitly labelled as live JSONL so nobody mistakes a partial run
    for a finished one.
    """
    recs, stats = read_jsonl(metrics_path)
    if not recs:
        return None
    last = recs[-1]
    step = last.get("step")
    tokens = last.get("cum_tokens")
    wall = last.get("elapsed_sec")
    vram = last.get("peak_reserved_MiB")
    loss = last.get("loss")
    tps = last.get("tokens_per_step")

    steps_cell = fmt_int(step)
    if total_steps and step is not None:
        steps_cell += f" of {fmt_int(total_steps)} planned ({fmt_pct(step / total_steps)} complete)"
    note = f" (live JSONL, {stats['records']:,} logged points"
    if stats["skipped_partial"]:
        note += ", 1 partial trailing line skipped"
    note += ")"

    return {
        "run": "pretrain (in progress)",
        "phase": "Phase 1 -- pretrain (production)",
        "steps": steps_cell,
        "tokens_per_step": fmt_int(tps),
        "config": "n/a (no run argv recorded; measured tokens/step in previous column)",
        "tokens": fmt_tokens(tokens) + note,
        "wall": fmt_secs(wall),
        "vram": fmt_mib(vram),
        "final_loss": fmt_num(loss, 4),
        "source": f"`{rel(metrics_path, root)}` (summary not yet written: `{rel(summary_path, root)}`)",
        "_sort": (1, "pretrain_metrics.jsonl"),
    }


# ------------------------------------------------------------ training runs


def table_training_runs(root: Path, extra_summaries=(), total_pretrain_steps=None):
    """Emit the 'training runs' markdown table (+ a footnote line)."""
    logs = root / "logs"
    rows = []
    seen = set()

    # 1. canonical expected runs (explicit n/a row when the summary is missing)
    for relpath, phase_label, metrics_rel in EXPECTED_SUMMARIES:
        spath = root / relpath
        if spath.exists():
            row = read_summary_row(spath, root, phase_label)
            seen.add(spath.name)
            rows.append(row)
        else:
            live = live_pretrain_row(root, root / metrics_rel, total_pretrain_steps, spath)
            if live is not None:
                rows.append(live)
            else:
                rows.append(read_summary_row(spath, root, phase_label))

    # 2. every other summary that exists on disk
    for spath in sorted(logs.glob("*_summary.json")):
        if spath.name in seen:
            continue
        if spath.name.startswith("pretrain_metrics_summary"):
            continue  # handled as the canonical row above
        rows.append(read_summary_row(spath, root))
        seen.add(spath.name)

    # 3. explicitly requested extras (--summary)
    for extra in extra_summaries:
        p = Path(extra)
        if not p.is_absolute():
            p = root / p
        rows.append(read_summary_row(p, root))

    rows.sort(key=lambda r: r["_sort"])
    headers = ["run", "phase", "steps", "tokens/step", "config (from run argv)",
               "tokens", "wall time", "peak VRAM (reserved)", "final loss", "source"]
    body = [[r["run"], r["phase"], r["steps"], r["tokens_per_step"], r["config"],
             r["tokens"], r["wall"], r["vram"], r["final_loss"], r["source"]] for r in rows]

    out = ["## Training runs", "",
           "All values read from `logs/*_summary.json` (steps, tokens, wall, VRAM), "
           "`logs/vram_train_*.json` (run argv) and the last record of the sibling "
           "`logs/*.jsonl` (final loss).", "",
           md_table(headers, body)]
    grpo = [r["run"] for r in rows if str(r["phase"]).startswith("Phase 3")]
    if grpo:
        out += ["", "Notes: for GRPO runs the `tokens` column is "
                    "`cum_sampled_tokens` (completion tokens actually sampled, "
                    "not counting prompt tokens) and `final loss` is the last "
                    "logged PPO policy loss, not a cross-entropy loss. "
                    f"Rows affected: {', '.join(f'`{g}`' for g in grpo)}."]
    return "\n".join(out)


# ------------------------------------------------------------ eval comparison


def load_eval_report(path: Path, root: Path):
    data, err = load_json(path)
    relp = rel(path, root)
    if err:
        return None, err, relp
    if not isinstance(data, dict):
        return None, f"not a JSON object ({type(data).__name__})", relp
    return data, None, relp


def eval_label(report, path: Path):
    w = (report.get("weight") or {}).get("path") or report.get("model_path") or path.name
    stem = Path(str(w)).name.replace(".pth", "")
    mode = report.get("prompt_mode") or (report.get("generation") or {}).get("prompt_mode")
    mnt = report.get("max_new_tokens") or (report.get("generation") or {}).get("max_new_tokens")
    bits = [stem]
    if mode:
        bits.append(str(mode))
    if mnt is not None:
        bits.append(f"{mnt} tok")
    return f"{bits[0]} [{', '.join(bits[1:])}]" if len(bits) > 1 else bits[0]


def _acc_cell(node):
    """{'correct':0,'total':24,'accuracy':0.0} -> '0.0% (0/24)'."""
    if not isinstance(node, dict):
        return "n/a"
    acc = node.get("accuracy")
    correct, total = node.get("correct"), node.get("total")
    pct = fmt_pct(acc)
    if correct is not None and total is not None:
        return f"{pct} ({fmt_int(correct)}/{fmt_int(total)})"
    return pct


def _pp_delta(a, b):
    """Difference in percentage points between two accuracy fractions."""
    try:
        if a is None or b is None:
            return "n/a"
        return f"{100.0 * (float(b) - float(a)):+.1f} pp"
    except (TypeError, ValueError):
        return "n/a"


def table_eval_comparison(root: Path, paths):
    """Emit the eval comparison table + a loud warning when reports are mismatched."""
    if not paths:
        return "\n".join([
            "## Eval comparison", "",
            "n/a (no eval reports given: pass `--eval <report1.json> <report2.json> ...`)",
        ])

    reports, labels, errs, rels = [], [], [], []
    for raw in paths:
        p = Path(raw)
        if not p.is_absolute():
            p = root / p
        data, err, relp = load_eval_report(p, root)
        rels.append(relp)
        if err:
            errs.append((relp, err))
            reports.append(None)
            labels.append(Path(raw).stem)
        else:
            reports.append(data)
            labels.append(eval_label(data, p))

    ok = [(r, l, rp) for r, l, rp in zip(reports, labels, rels) if r is not None]
    out = ["## Eval comparison"]

    if errs:
        out += ["", "Missing or unreadable reports:"]
        out += [f"- `{rp}`: {e}" for rp, e in errs]
    if len(ok) < 2:
        out += ["", f"n/a (need >= 2 readable eval reports, got {len(ok)})"]
        return "\n".join(out)

    # Disambiguate identical labels (same report passed twice, or two runs of one
    # checkpoint) so no two columns end up indistinguishable.
    seen_labels = {}
    uniq = []
    for r, l, rp in ok:
        seen_labels[l] = seen_labels.get(l, 0) + 1
        if seen_labels[l] > 1:
            l = f"{l} #{seen_labels[l]}"
        uniq.append((r, l, rp))
    ok = uniq

    # ---- comparability checks (a mismatch is a finding, so warn loudly)
    warnings = []
    modes = [r.get("prompt_mode") or (r.get("generation") or {}).get("prompt_mode")
             for r, _l, _p in ok]
    mnts = [r.get("max_new_tokens") if r.get("max_new_tokens") is not None
            else (r.get("generation") or {}).get("max_new_tokens")
            for r, _l, _p in ok]
    sha = [(r.get("benchmark") or {}).get("tasks_sha256") for r, _l, _p in ok]
    gen_modes = [(r.get("generation") or {}).get("mode") for r, _l, _p in ok]

    def _diff_note(values, keyname, why):
        if len({str(v) for v in values}) <= 1:
            return None
        pairs = ", ".join(f"`{l}` = {v if v is not None else 'n/a (absent)'}"
                          for (_r, l, _p), v in zip(ok, values))
        return f"`{keyname}` differs between reports: {pairs}. {why}"

    warnings.append(_diff_note(modes, "prompt_mode",
                               "Accuracy is NOT comparable across prompt formats."))
    warnings.append(_diff_note(mnts, "max_new_tokens",
                               "A shorter budget can truncate thinking models and "
                               "produce false negatives."))
    warnings.append(_diff_note(sha, "benchmark.tasks_sha256",
                               "The reports were not run on the same frozen task file."))
    warnings.append(_diff_note([g for g in gen_modes if g is not None], "generation.mode",
                               "Greedy and sampled numbers are not interchangeable."))
    warnings = [w for w in warnings if w]
    if warnings:
        out.append(warn_block(warnings))

    # ---- overall + per-family accuracy, side by side
    families = []
    for r, _l, _p in ok:
        for fam in (r.get("per_family") or {}):
            if fam not in families:
                families.append(fam)
    families.sort()

    headers = ["metric"] + labels_of(ok)
    rows = [["**overall**"] + [_acc_cell(r.get("overall")) for r, _l, _p in ok]]
    for fam in families:
        rows.append([fam] + [_acc_cell((r.get("per_family") or {}).get(fam)) for r, _l, _p in ok])

    # delta column only makes sense for exactly two reports
    if len(ok) == 2:
        a, b = ok[0][0], ok[1][0]
        headers.append(f"delta ({labels_of(ok)[1]} - {labels_of(ok)[0]})")
        da = _pp_delta((a.get("overall") or {}).get("accuracy"),
                       (b.get("overall") or {}).get("accuracy"))
        rows[0].append(da)
        for i, fam in enumerate(families, start=1):
            rows[i].append(_pp_delta(((a.get("per_family") or {}).get(fam) or {}).get("accuracy"),
                                     ((b.get("per_family") or {}).get(fam) or {}).get("accuracy")))

    out += ["", md_table(headers, rows)]

    # ---- provenance rows that make the comparison auditable
    prov = [["benchmark", *[str((r.get("benchmark") or {}).get("name", "n/a")) for r, _l, _p in ok]],
            ["tasks sha256 (first 12)", *[str(v)[:12] if v else "n/a" for v in sha]],
            ["prompt_mode", *[str(v) if v is not None else "n/a" for v in modes]],
            ["max_new_tokens", *[str(v) if v is not None else "n/a" for v in mnts]],
            ["decoding", *[str(v) if v is not None else "n/a" for v in gen_modes]],
            ["tasks", *[fmt_int((r.get("overall") or {}).get("total")) for r, _l, _p in ok]],
            ["device", *[str(r.get("device", "n/a")) for r, _l, _p in ok]],
            ["wall time (s)", *[fmt_num(r.get("wall_time_s"), 1) for r, _l, _p in ok]],
            ["peak VRAM (MB)", *[fmt_num(r.get("peak_vram_mb"), 1) for r, _l, _p in ok]],
            ["report", *[f"`{p}`" for _r, _l, p in ok]]]
    out += ["", md_table(["provenance", *labels_of(ok)], prov)]
    return "\n".join(out)


def labels_of(ok):
    return [l for _r, l, _p in ok]


# ---------------------------------------------------------- token accounting


def _tok_from_summary(path: Path, key: str):
    data, err = load_json(path)
    if err:
        return None, err
    if not isinstance(data, dict):
        return None, f"unexpected JSON type: {type(data).__name__}"
    val = data.get(key)
    return (int(val) if val is not None else None), None


def table_token_accounting(root: Path, pretrain_summary: Path, grpo_summaries,
                           sft_summary=None):
    """Sum pretrain / SFT / GRPO tokens from the summary files."""
    rows = []
    known = 0
    missing = []

    # ---- Phase 1 pretrain
    psum = pretrain_summary
    p_tokens, p_err = (None, None)
    if psum.exists():
        p_tokens, p_err = _tok_from_summary(psum, "total_tokens")
    else:
        p_err = f"file not found: {rel(psum, root)}"
        recs, _st = read_jsonl(Path(str(psum).replace("_summary.json", ".jsonl")))
        if recs:
            p_tokens = last_value(recs, "cum_tokens")
            if p_tokens is not None:
                p_tokens = int(p_tokens)
                p_err = None
    if p_tokens is not None:
        known += p_tokens
        rows.append(["Phase 1 -- pretrain", "training tokens (1 epoch over the packed corpus)",
                     fmt_tokens(p_tokens),
                     f"`{rel(psum, root)}`" if psum.exists()
                     else f"live `{rel(Path(str(psum).replace('_summary.json', '.jsonl')), root)}` "
                          f"(summary not yet written; value = last cum_tokens)"])
    else:
        missing.append("Phase 1 pretrain")
        rows.append(["Phase 1 -- pretrain", "training tokens (1 epoch over the packed corpus)",
                     na_cell(rel(psum, root), p_err), f"`{rel(psum, root)}`"])

    # ---- Phase 2 SFT
    if sft_summary is not None:
        sp = Path(sft_summary)
        if not sp.is_absolute():
            sp = root / sp
        if sp.exists():
            s_tokens, s_err = _tok_from_summary(sp, "total_tokens")
            if s_tokens is None:
                s_tokens, s_err = _tok_from_summary(sp, "cum_sampled_tokens")
            if s_tokens is not None:
                known += s_tokens
                rows.append(["Phase 2 -- SFT", "supervised tokens (loss on response tokens)",
                             fmt_tokens(s_tokens), f"`{rel(sp, root)}`"])
            else:
                missing.append("Phase 2 SFT")
                rows.append(["Phase 2 -- SFT", "supervised tokens", na_cell(rel(sp, root), s_err),
                             f"`{rel(sp, root)}`"])
        else:
            missing.append("Phase 2 SFT")
            rows.append(["Phase 2 -- SFT", "supervised tokens", na_cell(rel(sp, root)),
                         f"`{rel(sp, root)}`"])
    else:
        missing.append("Phase 2 SFT")
        rows.append(["Phase 2 -- SFT", "supervised tokens",
                     "n/a (no SFT summary file supplied: pass --sft-summary <path>)",
                     "n/a"])

    # ---- Phase 3 GRPO (cum_sampled_tokens)
    g_tokens = 0
    g_known = False
    detail = []
    for raw in grpo_summaries:
        gp = Path(raw)
        if not gp.is_absolute():
            gp = root / gp
        if not gp.exists():
            detail.append(f"`{rel(gp, root)}` n/a (file not found)")
            continue
        data, err = load_json(gp)
        if err:
            detail.append(f"`{rel(gp, root)}` n/a ({err})")
            continue
        if not isinstance(data, dict):
            detail.append(f"`{rel(gp, root)}` n/a (unexpected JSON type: "
                          f"{type(data).__name__})")
            continue
        sampled = data.get("cum_sampled_tokens")
        prompt = data.get("cum_prompt_tokens")
        steps = data.get("steps")
        if sampled is not None:
            g_tokens += int(sampled)
            g_known = True
        detail.append(f"`{Path(raw).stem.replace('_summary', '')}`: "
                      f"{fmt_tokens(sampled)} sampled"
                      + (f" + {fmt_tokens(prompt)} prompt" if prompt is not None else "")
                      + (f" over {fmt_int(steps)} steps" if steps is not None else ""))
    if g_known:
        known += g_tokens
        rows.append(["Phase 3 -- GRPO", "sampled completion tokens (cum_sampled_tokens)",
                     fmt_tokens(g_tokens), "; ".join(detail) if detail else "n/a"])
    else:
        missing.append("Phase 3 GRPO")
        rows.append(["Phase 3 -- GRPO", "sampled completion tokens (cum_sampled_tokens)",
                     "n/a (no GRPO summary found)", "; ".join(detail) if detail else "n/a"])

    total_label = ("**TOTAL (known components)**" if missing
                   else "**TOTAL**")
    total_cell = f"**{fmt_tokens(known)}**"
    if missing:
        total_cell += f" -- excludes {', '.join(missing)}"
    rows.append([total_label, "sum of the rows above", total_cell, ""])

    out = ["## Token accounting", "",
           "Training tokens per phase. GRPO counts `cum_sampled_tokens` (completion "
           "tokens actually sampled by the rollout engine; prompt tokens are listed "
           "separately and are not added to the total).", "",
           md_table(["phase", "what is counted", "tokens", "source / detail"], rows)]
    return "\n".join(out)


# ------------------------------------------------------------------------ main


def build_parser():
    p = argparse.ArgumentParser(
        description="Build markdown tables from MiniMind logs/summaries (no CUDA).")
    p.add_argument("--out", default=None, help="write markdown here (default: stdout)")
    p.add_argument("--root", default=None, help="project root (default: parent of tools/)")
    p.add_argument("--eval", nargs="+", default=None, metavar="REPORT.json",
                   help="2+ held-out eval report JSONs to compare")
    p.add_argument("--summary", nargs="+", default=None,
                   help="extra training summary JSON(s) to include in the runs table")
    p.add_argument("--sft-summary", default=None,
                   help="SFT summary JSON for token accounting (optional)")
    p.add_argument("--grpo-summary", nargs="+", default=None,
                   help="GRPO summary JSONs (default: all logs/grpo*_summary.json)")
    p.add_argument("--pretrain-summary", default="logs/pretrain_metrics_summary.json")
    p.add_argument("--total-pretrain-steps", type=int, default=134817,
                   help="planned Phase-1 steps, for the 'N of M planned' label")
    p.add_argument("--section", action="append",
                   choices=["runs", "eval", "tokens"],
                   help="emit only these sections (default: all)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = project_root(args.root)
    sections = set(args.section) if args.section else {"runs", "eval", "tokens"}

    blocks = []
    if "runs" in sections:
        blocks.append(table_training_runs(root, args.summary or [],
                                          args.total_pretrain_steps))
    if "eval" in sections:
        blocks.append(table_eval_comparison(root, args.eval))
    if "tokens" in sections:
        grpo = args.grpo_summary
        if grpo is None:
            grpo = sorted(str(p) for p in (root / "logs").glob("grpo*_summary.json"))
        blocks.append(table_token_accounting(
            root, root / args.pretrain_summary, grpo, args.sft_summary))

    text = "\n\n".join(blocks) + "\n"
    if args.out:
        outp = Path(args.out)
        if not outp.is_absolute():
            outp = root / outp
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(text, encoding="utf-8", newline="\n")
        print(f"[make_tables] wrote {outp} ({len(text):,} chars)")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
