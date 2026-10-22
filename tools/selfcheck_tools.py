#!/usr/bin/env python
"""selfcheck_tools.py -- regression tests for plot_curves / make_tables /
gen_readme_sections, including every documented edge case.

    python selfcheck_tools.py [-v]

Exit code 0 = all checks passed. Nothing here touches CUDA, torch or the GPU.
Fixtures are written to tools/_selftest/ (safe to delete; not a deliverable).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable
FIXTURES = HERE / "_selftest"

PLOT = HERE / "plot_curves.py"
TABLES = HERE / "make_tables.py"
README = HERE / "gen_readme_sections.py"

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   [{detail}]" if detail and not cond else ""))
    return bool(cond)


def run(args, cwd=HERE):
    """Run a tool, return (rc, stdout, stderr)."""
    proc = subprocess.run([PY] + [str(a) for a in args], cwd=str(cwd),
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def png_size(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        head = f.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return "not-a-png"
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def make_mismatch_report(base_report: Path):
    """Derive a deliberately non-comparable eval report from the real one.

    Different prompt_mode + max_new_tokens + different fake accuracies, so the
    comparability guard has something real to complain about. Written to the
    fixtures dir, never to reports/.
    """
    import copy

    base = json.loads(Path(base_report).read_text(encoding="utf-8"))
    alt = copy.deepcopy(base)
    alt["prompt_mode"] = "chat_think"
    alt["max_new_tokens"] = 128
    alt.setdefault("generation", {})["prompt_mode"] = "chat_think"
    alt["generation"]["max_new_tokens"] = 128
    alt["weight"] = {"path": "D:/llm-rl-lab/src/minimind-master/out/grpo_easy_probe_768.pth"}
    alt["overall"] = {"correct": 3, "total": 24, "accuracy": 3 / 24}
    for fam in ("arithmetic", "format", "logic", "multistep"):
        node = alt.get("per_family", {}).get(fam)
        if isinstance(node, dict) and node.get("total"):
            node["correct"] = min(node["total"], node["correct"] + 1)
            node["accuracy"] = node["correct"] / node["total"]
    p = fixture("eval_mismatch.json", json.dumps(alt, indent=2, ensure_ascii=False))
    return p


def fixture(name, content, binary=False):
    FIXTURES.mkdir(parents=True, exist_ok=True)
    p = FIXTURES / name
    if binary:
        p.write_bytes(content)
    else:
        p.write_text(content, encoding="utf-8", newline="\n")
    return p


# ---------------------------------------------------------------------------


def test_plot_edge_cases():
    print("\n[plot_curves] edge cases")

    # 1. truncated final line (file still being written)
    good = [{"step": i, "loss": 7.0 - i * 0.01, "cum_tokens": i * 16384,
             "tokens_per_sec": 20000 + i, "peak_reserved_MiB": 5288} for i in range(1, 11)]
    body = "".join(json.dumps(r) + "\n" for r in good)
    truncated = fixture("truncated.jsonl", body + '{"step": 11, "loss": 5.1, "cum_tok')
    out = FIXTURES / "truncated.png"
    out.unlink(missing_ok=True)
    rc, so, se = run([PLOT, "--metrics", truncated, "--out", out, "--kind", "pretrain"])
    check("truncated last line -> exit 0", rc == 0, f"rc={rc} stderr={se[-300:]}")
    check("truncated last line -> PNG written", png_size(out) == (1600, 900), str(png_size(out)))
    check("truncated last line -> reported as partial in stderr",
          "partial" in se.lower(), se[-300:])

    # 2. empty file
    empty = fixture("empty.jsonl", "")
    out2 = FIXTURES / "empty.png"
    out2.unlink(missing_ok=True)
    rc, so, se = run([PLOT, "--metrics", empty, "--out", out2])
    check("empty file -> exit 0", rc == 0, f"rc={rc}")
    check("empty file -> clear message, no traceback",
          "no usable records" in se and "Traceback" not in se, se[-300:])
    check("empty file -> no PNG written", not out2.exists(), str(out2))

    # 3. one point only
    one = fixture("one.jsonl", json.dumps(good[0]) + "\n")
    out3 = FIXTURES / "one.png"
    out3.unlink(missing_ok=True)
    rc, so, se = run([PLOT, "--metrics", one, "--out", out3, "--kind", "pretrain"])
    check("1 point -> exit 0", rc == 0, f"rc={rc}")
    check("1 point -> explicit message, no traceback",
          "need >= 2" in se and "Traceback" not in se, se[-300:])
    check("1 point -> no PNG written", not out3.exists(), str(out3))

    # 4. missing file -> hard error (exit 2) but still no traceback
    rc, so, se = run([PLOT, "--metrics", FIXTURES / "does_not_exist.jsonl",
                      "--out", FIXTURES / "nope.png"])
    check("missing metrics -> exit 2", rc == 2, f"rc={rc}")
    check("missing metrics -> no traceback", "Traceback" not in se, se[-300:])

    # 5. grpo kind, collapsed marking
    grpo = fixture("grpo.jsonl", "".join(
        json.dumps({"step": i, "reward_mean": 0.05 + 0.01 * i,
                    "reward_mean_20": 0.05 + 0.005 * i, "acc": 0.01 * i,
                    "adv_std": 1.0, "kl_mean": -0.01 * i, "entropy": 2.9 - 0.01 * i,
                    "collapsed": (i % 4 == 0)}) + "\n" for i in range(1, 13)))
    out5 = FIXTURES / "grpo.png"
    rc, so, se = run([PLOT, "--metrics", grpo, "--out", out5, "--kind", "grpo"])
    check("grpo kind -> PNG written", rc == 0 and png_size(out5) == (1600, 900),
          f"rc={rc} {png_size(out5)}")
    check("grpo kind -> collapsed steps reported", "collapsed steps" in so, so[-300:])

    # 6. auto inference
    rc, so, se = run([PLOT, "--metrics", grpo, "--out", FIXTURES / "auto.png"])
    check("auto infers grpo from keys", "kind=grpo" in so, so[-200:])
    rc, so, se = run([PLOT, "--metrics", truncated, "--out", FIXTURES / "auto2.png"])
    check("auto infers pretrain from keys", "kind=pretrain" in so, so[-200:])

    # 7. BOM + CRLF (PowerShell-written file)
    bom = fixture("bom.jsonl",
                  ("\ufeff" + "\r\n".join(json.dumps(r) for r in good[:5]) + "\r\n").encode("utf-8"),
                  binary=True)
    rc, so, se = run([PLOT, "--metrics", bom, "--out", FIXTURES / "bom.png", "--kind", "pretrain"])
    check("UTF-8 BOM + CRLF -> parsed", rc == 0 and png_size(FIXTURES / "bom.png") == (1600, 900),
          f"rc={rc}")

    # 8. invalid middle line is skipped, rest still plots
    bad = fixture("badline.jsonl",
                  json.dumps(good[0]) + "\nnot json at all\n" + json.dumps(good[1]) + "\n")
    rc, so, se = run([PLOT, "--metrics", bad, "--out", FIXTURES / "badline.png", "--kind", "pretrain"])
    check("invalid middle line -> still plots", rc == 0 and png_size(FIXTURES / "badline.png") is not None,
          f"rc={rc}")
    check("invalid middle line -> reported in stderr", "unparsable" in se, se[-300:])

    # 9. missing individual metric -> gap, no fabricated zeros
    gap = fixture("gap.jsonl",
                  json.dumps({"step": 1, "loss": 7.0, "cum_tokens": 16384,
                              "tokens_per_sec": 20000}) + "\n"
                  + json.dumps({"step": 2, "cum_tokens": 32768, "tokens_per_sec": 21000}) + "\n"
                  + json.dumps({"step": 3, "loss": 6.8, "cum_tokens": 49152,
                                "tokens_per_sec": 22000}) + "\n")
    rc, so, se = run([PLOT, "--metrics", gap, "--out", FIXTURES / "gap.png", "--kind", "pretrain"])
    check("missing loss at one step -> still plots", rc == 0, f"rc={rc}")

    # 10. downsample: a 134k-step file must not produce a huge PNG
    big = FIXTURES / "big.jsonl"
    with open(big, "w", encoding="utf-8", newline="\n") as f:
        for i in range(1, 134818, 50):
            f.write(json.dumps({"step": i, "loss": 7.0 - 4.0 * i / 134817,
                                "cum_tokens": i * 16384, "tokens_per_sec": 22000,
                                "peak_reserved_MiB": 5288}) + "\n")
    outbig = FIXTURES / "big.png"
    rc, so, se = run([PLOT, "--metrics", big, "--out", outbig, "--kind", "pretrain",
                      "--total-steps", 134817])
    n = sum(1 for _ in open(big, encoding="utf-8"))
    size = outbig.stat().st_size if outbig.exists() else 0
    check(f"{n:,}-point run -> PNG < 1 MB (downsampled)", rc == 0 and size < 1_000_000,
          f"rc={rc} bytes={size:,}")


def test_tables_edge_cases():
    print("\n[make_tables] edge cases")

    # 1. default run over the real project files
    rc, so, se = run([TABLES])
    check("real project -> exit 0", rc == 0, f"rc={rc} {se[-300:]}")
    for want in ("## Training runs", "## Token accounting"):
        check(f"real project -> has {want!r}", want in so)

    # 2. missing eval reports -> explicit n/a, no crash
    rc, so, se = run([TABLES, "--section", "eval", "--eval",
                      FIXTURES / "no1.json", FIXTURES / "no2.json"])
    check("missing eval reports -> exit 0", rc == 0, f"rc={rc}")
    check("missing eval reports -> explicit n/a cells",
          "n/a" in so and "file not found" in so, so[-400:])

    # 3. single eval report -> refuses to compare, says why
    real = ROOT / "reports" / "sft_baseline_smoke.json"
    rc, so, se = run([TABLES, "--section", "eval", "--eval", real])
    check("1 eval report -> exit 0 + 'need >= 2'", rc == 0 and "need >= 2" in so, so[-300:])

    # 4. mismatched prompt_mode / max_new_tokens -> loud warning + table still shown
    mism = make_mismatch_report(real)
    rc, so, se = run([TABLES, "--section", "eval", "--eval", real, mism])
    check("mismatched prompt_mode -> exit 0", rc == 0, f"rc={rc}")
    check("mismatched prompt_mode -> loud WARNING block", "[!WARNING]" in so, so[:400])
    check("mismatched prompt_mode -> names both values",
          "chat_think" in so and "max_new_tokens" in so, so[:800])
    check("mismatched prompt_mode -> table still shown", "**overall**" in so)
    check("mismatched prompt_mode -> shows a delta column", "delta (" in so)

    # 5. missing training summary -> n/a row, not a crash
    rc, so, se = run([TABLES, "--section", "runs", "--summary", FIXTURES / "nope_summary.json"])
    check("missing summary -> exit 0", rc == 0, f"rc={rc}")
    check("missing summary -> explicit n/a row", "file not found" in so, so[-500:])

    # 6. GRPO tokens use cum_sampled_tokens
    rc, so, se = run([TABLES, "--section", "tokens"])
    check("token accounting -> GRPO row present", "Phase 3 -- GRPO" in so)
    check("token accounting -> GRPO counts sampled tokens",
          "cum_sampled_tokens" in so and "sampled" in so)

    # 7. malformed summary JSON (not an object) does not crash
    bad = fixture("bad_summary.json", "[1,2,3]")
    rc, so, se = run([TABLES, "--section", "runs", "--summary", bad])
    check("malformed summary -> exit 0", rc == 0, f"rc={rc} {se[-300:]}")
    check("malformed summary -> no traceback", "Traceback" not in se, se[-300:])

    # 8. --out writes a file
    outp = FIXTURES / "tables.md"
    outp.unlink(missing_ok=True)
    rc, so, se = run([TABLES, "--out", outp])
    check("--out writes markdown", rc == 0 and outp.exists() and outp.stat().st_size > 500,
          f"rc={rc}")

    # 9. idempotent: two runs produce byte-identical output
    rc, first, _ = run([TABLES])
    rc, second, _ = run([TABLES])
    check("make_tables is idempotent", first == second)


def test_readme_edge_cases():
    print("\n[gen_readme_sections] edge cases")

    rc, so, se = run([README])
    check("real project -> exit 0", rc == 0, f"rc={rc} {se[-300:]}")
    for name in ("metrics_table", "eval_table", "token_accounting", "figures"):
        check(f"emits BEGIN/END marker for {name}",
              f"BEGIN GENERATED: {name}" in so and f"END GENERATED: {name}" in so)
    check("emits real figure references", "![Phase-1 pretrain" in so)
    check("marks the not-yet-generated figure with TODO", "TODO" in so)
    check("does not fabricate an eval comparison",
          ">= 2 held-out eval reports" in so or "**overall**" in so)

    # no placeholder numbers: a TODO must be a TODO, not a fake 0
    check("no literal 'n/a' without a reason string",
          "n/a ()" not in so and "n/a (file not found: )" not in so)

    # idempotent
    rc, first, _ = run([README])
    rc, second, _ = run([README])
    check("gen_readme_sections is idempotent", first == second)

    # --out writes
    outp = FIXTURES / "readme_sections.md"
    outp.unlink(missing_ok=True)
    rc, so, se = run([README, "--out", outp])
    check("--out writes markdown", rc == 0 and outp.exists() and outp.stat().st_size > 500,
          f"rc={rc}")


def test_figures():
    print("\n[figures] deliverables")
    figs = ROOT / "figures"
    for name in ("pretrain_curve_partial.png", "grpo_probe_curves.png"):
        p = figs / name
        size = p.stat().st_size if p.exists() else 0
        dims = png_size(p)
        check(f"{name} exists and > 10 KB", p.exists() and size > 10_000,
              f"bytes={size:,}")
        check(f"{name} is a valid 1600x900 PNG", dims == (1600, 900), str(dims))

    # PIL round-trip when available (optional, matplotlib.image is the fallback)
    try:
        from PIL import Image
        for name in ("pretrain_curve_partial.png", "grpo_probe_curves.png"):
            p = figs / name
            if not p.exists():
                continue
            with Image.open(p) as im:
                im.load()
                check(f"{name} opens in PIL ({im.format} {im.size[0]}x{im.size[1]})",
                      im.format == "PNG" and im.size == (1600, 900), str(im.size))
    except ImportError:
        print("  SKIP  PIL not available (stdlib IHDR check already passed)")


def main():
    verbose = "-v" in sys.argv
    FIXTURES.mkdir(parents=True, exist_ok=True)
    print(f"python: {PY}")
    print(f"project root: {ROOT}")

    test_plot_edge_cases()
    test_tables_edge_cases()
    test_readme_edge_cases()
    test_figures()

    passed = sum(1 for _n, ok, _d in results if ok)
    total = len(results)
    print(f"\n{'=' * 70}\n{passed}/{total} checks passed")
    failed = [(n, d) for n, ok, d in results if not ok]
    if failed:
        print("\nFAILURES:")
        for n, d in failed:
            print(f"  - {n}: {d}")
        return 1
    if not verbose:
        print("(run with -v for the full check list)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
