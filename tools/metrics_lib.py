"""Shared helpers for the D:/llm-rl-lab reporting tools.

Pure standard library (json / math / pathlib / re). Deliberately imports neither
torch nor matplotlib, so it is safe to use while the GPU is busy training.

Everything that reads a file is defensive: a metrics JSONL that is still being
appended to, a missing file, an empty file, a BOM, CRLF line endings and
unparsable lines are all reported through an explicit stats/err value instead of
raising. Callers are expected to render "n/a (file not found: ...)" rather than
crash.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# D:/llm-rl-lab/tools/metrics_lib.py -> D:/llm-rl-lab
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_root(override=None) -> Path:
    """Project root: --root override, else the directory above tools/."""
    if override:
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT


# --------------------------------------------------------------------------- io


def read_jsonl(path):
    """Read a JSONL file tolerantly.

    Returns (records, stats). `records` is a list of dicts. `stats` always
    exists and explains exactly what was skipped:

        exists, bytes, lines, records, blank,
        skipped_partial   -- last line has no newline and is not valid JSON
                             (a file that is still being appended to)
        skipped_invalid   -- any other unparsable line
        invalid_linenos   -- 1-based line numbers of skipped_invalid lines
        err               -- set when the path does not exist / cannot be read

    A last line without a trailing newline that *does* parse is kept: a finished
    writer may simply not emit the final newline.
    """
    p = Path(path)
    stats = {
        "path": str(p),
        "exists": False,
        "bytes": 0,
        "lines": 0,
        "records": 0,
        "blank": 0,
        "skipped_partial": 0,
        "skipped_invalid": 0,
        "invalid_linenos": [],
        "err": None,
    }
    if not p.exists():
        stats["err"] = f"file not found: {p}"
        return [], stats
    try:
        raw = p.read_bytes()
    except OSError as exc:  # locked / permission / device error
        stats["err"] = f"{type(exc).__name__}: {exc}"
        return [], stats

    stats["exists"] = True
    stats["bytes"] = len(raw)
    # utf-8-sig strips a BOM if one slipped in (PowerShell Set-Content adds one).
    text = raw.decode("utf-8-sig", errors="replace")
    lines = text.splitlines(keepends=True)
    stats["lines"] = len(lines)

    records = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            stats["blank"] += 1
            continue
        try:
            rec = json.loads(stripped)
        except json.JSONDecodeError:
            is_last = idx == len(lines) - 1
            if is_last and not line.endswith(("\n", "\r")):
                # A writer appends "json + \n" in one call; a line without its
                # newline that does not parse is a partially flushed record.
                stats["skipped_partial"] += 1
            else:
                stats["skipped_invalid"] += 1
                stats["invalid_linenos"].append(idx + 1)
            continue
        if isinstance(rec, dict):
            records.append(rec)
        else:
            stats["skipped_invalid"] += 1
            stats["invalid_linenos"].append(idx + 1)

    stats["records"] = len(records)
    return records, stats


def load_json(path):
    """Load a JSON file. Returns (data, err); data is None when err is set."""
    p = Path(path)
    if not p.exists():
        return None, f"file not found: {p}"
    try:
        return json.loads(p.read_text(encoding="utf-8-sig")), None
    except (OSError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------- format


def fmt_int(value, dash="n/a"):
    """12345 -> '12,345'. None / non-finite / non-numeric -> 'n/a'."""
    if value is None:
        return dash
    try:
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, float) and not math.isfinite(value):
            return dash
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return dash


def fmt_num(value, digits=4, dash="n/a", unit=""):
    """Fixed-decimal number, 'n/a' when absent. Never invents a value."""
    if value is None:
        return dash
    try:
        if isinstance(value, float) and not math.isfinite(value):
            return dash
        return f"{float(value):.{digits}f}{unit}"
    except (TypeError, ValueError):
        return dash


def fmt_sig(value, digits=4, dash="n/a"):
    """Compact significant-figure number (good for lr / kl / loss)."""
    if value is None:
        return dash
    try:
        v = float(value)
        if not math.isfinite(v):
            return dash
        if v != 0 and abs(v) < 1e-3:
            return f"{v:.2e}"
        return f"{v:.{digits}f}"
    except (TypeError, ValueError):
        return dash


def fmt_pct(frac, digits=1, dash="n/a"):
    """0.25 -> '25.0%'. `frac` is a fraction, not a percentage."""
    if frac is None:
        return dash
    try:
        v = float(frac)
        if not math.isfinite(v):
            return dash
        return f"{100.0 * v:.{digits}f}%"
    except (TypeError, ValueError):
        return dash


def fmt_secs(value, digits=1, dash="n/a"):
    if value is None:
        return dash
    try:
        v = float(value)
        if not math.isfinite(v):
            return dash
        return f"{v:,.{digits}f} s"
    except (TypeError, ValueError):
        return dash


def fmt_mib(value, dash="n/a"):
    if value is None:
        return dash
    try:
        v = float(value)
        if not math.isfinite(v):
            return dash
        return f"{int(round(v)):,} MiB"
    except (TypeError, ValueError):
        return dash


def fmt_tokens(value, dash="n/a"):
    """Exact token count with thousands separators (never rounded to B/M)."""
    return fmt_int(value, dash=dash)


def na_cell(path, err=None):
    """The canonical explicit-missing cell."""
    if err:
        return f"n/a ({err})"
    return f"n/a (file not found: {path})"


def rel(path, root):
    """Path relative to the project root when possible (for markdown links)."""
    p = Path(path)
    try:
        return p.resolve().relative_to(Path(root).resolve()).as_posix()
    except (ValueError, OSError):
        return str(p)


# ------------------------------------------------------------------------ md


def md_table(headers, rows, align=None):
    """Render a GitHub markdown table. Pipes inside cells are escaped."""
    align = align or ["---"] * len(headers)

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").strip()

    out = ["| " + " | ".join(cell(h) for h in headers) + " |",
           "| " + " | ".join(align) + " |"]
    for row in rows:
        out.append("| " + " | ".join(cell(c) for c in row) + " |")
    return "\n".join(out)


def warn_block(lines):
    """A loud, unmistakable markdown blockquote warning."""
    if not lines:
        return ""
    body = ["", "> [!WARNING]", "> **These results are not directly comparable.**"]
    body += [f"> - {ln}" for ln in lines]
    body.append("")
    return "\n".join(body)


# ---------------------------------------------------------------------- series


def series(records, key, cast=float):
    """Extract a column as a list; missing / null / unparsable values become None.

    Returning None (not 0.0) matters: a missing metric must never be plotted as
    a real zero, it must leave a gap.
    """
    out = []
    for rec in records:
        val = rec.get(key)
        if val is None:
            out.append(None)
            continue
        try:
            out.append(cast(val))
        except (TypeError, ValueError):
            out.append(None)
    return out


def last_value(records, key):
    """Last non-null value of a column, or None."""
    for rec in reversed(records):
        val = rec.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def first_present(records, keys):
    """First key from `keys` that appears in any record (schema probing)."""
    for rec in records:
        for key in keys:
            if key in rec:
                return key
    return None
