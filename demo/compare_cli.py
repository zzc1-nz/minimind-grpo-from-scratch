#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase-4 minimal demo CLI: same prompt, SFT vs GRPO, side by side, on CPU.

Example::

    python compare_cli.py --prompt "3+5等于几？只回复数字" ^
        --sft D:\\llm-rl-lab\\src\\minimind-master\\out\\sft_smoke_768.pth ^
        --grpo D:\\llm-rl-lab\\src\\minimind-master\\out\\grpo_smoke_768.pth

``--sft`` / ``--grpo`` accept a native MiniMind ``.pth``, an HF export directory,
a ``.gguf`` file, or the literal ``mock``. Everything runs on CPU; llama.cpp is
always invoked with ``-ngl 0`` (the GPU belongs to the training job).

Exit code 0 on success, 1 if a backend fails (with the error shown in its panel).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends as B  # noqa: E402

WIDTH = 78
BAR = "=" * WIDTH
THIN = "-" * WIDTH


def _box_title(text: str) -> str:
    text = " %s " % text
    pad = max(0, WIDTH - len(text) - 2)
    left = pad // 2
    return "+" + "=" * left + text + "=" * (pad - left) + "+"


def _wrap(text: str, width: int = WIDTH - 4) -> list:
    """Wrap long lines for the terminal box (counts wide chars as 1)."""
    out = []
    for raw in text.split("\n"):
        line = raw.rstrip()
        if not line:
            out.append("")
            continue
        while len(line) > width:
            out.append(line[:width])
            line = line[width:]
        out.append(line)
    return out


def _panel(label: str, result: dict | None, error: str | None, prompt: str) -> str:
    lines = []
    if error:
        lines.append(_box_title(label + " -- ERROR"))
        lines.append("| backend : %s" % (result or {}).get("backend", "?"))
        for chunk in _wrap(error, WIDTH - 4):
            lines.append("| " + chunk)
        lines.append("+" + "=" * (WIDTH - 2) + "+")
        return "\n".join(lines)

    lines.append(_box_title(label))
    lines.append("| backend : %-20s device : %-4s mode : %s"
                 % (result["backend"], result["device"], result.get("prompt_mode", "?")))
    meta = "| tokens  : %-6d tok/s : %-8.2f elapsed : %.3f s" % (
        result["tokens"], result["tok_per_sec"], result["elapsed_s"])
    lines.append(meta)
    extra_bits = []
    if result.get("dtype"):
        extra_bits.append("dtype=%s" % result["dtype"])
    if result.get("params"):
        extra_bits.append("%.1fM params" % result["params"])
    if result.get("gguf_size_mb"):
        extra_bits.append("gguf=%.1f MiB" % result["gguf_size_mb"])
    if result.get("ngl") is not None and result["backend"] == "gguf":
        extra_bits.append("ngl=%d" % result["ngl"])
    if result.get("prompt_tokens"):
        extra_bits.append("prompt_tokens=%d" % result["prompt_tokens"])
    if extra_bits:
        lines.append("| " + "  ".join(extra_bits))
    lines.append(THIN)
    body = result["text"]
    if not body.strip():
        body = "<empty completion>"
    for chunk in _wrap(body, WIDTH - 4):
        lines.append("| " + chunk)
    lines.append("+" + "=" * (WIDTH - 2) + "+")
    return "\n".join(lines)


def run_side(backend, prompt: str, max_new_tokens: int, temperature: float):
    t0 = time.time()
    try:
        res = backend.generate(prompt, max_new_tokens, temperature)
        res["total_s"] = round(time.time() - t0, 3)
        return res, None
    except Exception as exc:  # noqa: BLE001 - show the error in the panel
        import traceback
        return None, "%s: %s\n%s" % (type(exc).__name__, exc, traceback.format_exc(limit=3))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Side-by-side SFT vs GRPO demo (CPU only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--prompt", required=True, help="user prompt (Chinese or English)")
    ap.add_argument("--sft", required=True, help="SFT model: .pth | HF dir | .gguf | mock")
    ap.add_argument("--grpo", required=True, help="GRPO model: .pth | HF dir | .gguf | mock")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--temp", type=float, default=0.7,
                    help="sampling temperature; <=0 means greedy")
    ap.add_argument("--prompt-mode", default=B.DEFAULT_PROMPT_MODE, choices=list(B.PROMPT_MODES),
                    help="chat = closed empty <think> block (eval-harness default)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"],
                    help="torch dtype for .pth / HF backends (CPU fp32 is fastest)")
    ap.add_argument("--threads", type=int, default=B.DEFAULT_THREADS,
                    help="torch / llama.cpp CPU threads")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", action="store_true", help="machine-readable output (no box)")
    ap.add_argument("--out", default=None, help="also write the JSON result to this path")
    ap.add_argument("--quiet", action="store_true", help="suppress the human-readable box")
    return ap.parse_args(argv)


def main(argv=None):
    B._utf8_console()
    args = parse_args(argv)
    B.set_seed(args.seed)

    t_start = time.time()
    sft, grpo = None, None
    load_errors = {}
    t0 = time.time()
    try:
        sft = B.make_backend(args.sft, args.prompt_mode, dtype=args.dtype, threads=args.threads, seed=args.seed)
    except Exception as exc:  # noqa: BLE001
        load_errors["sft"] = "%s: %s" % (type(exc).__name__, exc)
    sft_load_s = time.time() - t0
    t0 = time.time()
    try:
        grpo = B.make_backend(args.grpo, args.prompt_mode, dtype=args.dtype, threads=args.threads, seed=args.seed)
    except Exception as exc:  # noqa: BLE001
        load_errors["grpo"] = "%s: %s" % (type(exc).__name__, exc)
    grpo_load_s = time.time() - t0

    results = {}
    errors = dict(load_errors)
    for side, backend in (("sft", sft), ("grpo", grpo)):
        if backend is None:
            results[side] = None
            continue
        res, err = run_side(backend, args.prompt, args.max_new_tokens, args.temp)
        results[side] = res
        if err:
            errors[side] = err
        try:
            backend.close()
        except Exception:  # noqa: BLE001
            pass

    payload = {
        "prompt": args.prompt,
        "prompt_formatted": B.format_prompt(args.prompt, args.prompt_mode),
        "prompt_mode": args.prompt_mode,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temp,
        "dtype": args.dtype,
        "threads": args.threads,
        "seed": args.seed,
        "device": "cpu",
        "gpu_free_mib": B.gpu_free_mib(),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "load_time_s": {"sft": round(sft_load_s, 3), "grpo": round(grpo_load_s, 3)},
        "sft": results["sft"],
        "grpo": results["grpo"],
        "errors": errors or None,
        "wall_time_s": round(time.time() - t_start, 3),
    }
    for side in ("sft", "grpo"):
        if results[side]:
            results[side].pop("prompt", None)  # the prompt is already in the payload

    if args.out:
        out = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        if not args.quiet and not args.json:
            print("[json] %s" % out)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif not args.quiet:
        print(BAR)
        print("MiniMind Phase-4 demo  |  SFT vs GRPO  |  device=cpu  |  mode=%s  |  temp=%.2f  |  max_new=%d"
              % (args.prompt_mode, args.temp, args.max_new_tokens))
        print(BAR)
        print("PROMPT")
        for chunk in _wrap(args.prompt, WIDTH - 4):
            print("  " + chunk)
        print()
        print(_panel("SFT  (%s)" % args.sft, results["sft"], errors.get("sft"), args.prompt))
        print(_panel("GRPO (%s)" % args.grpo, results["grpo"], errors.get("grpo"), args.prompt))
        print("device=cpu (GPU untouched) | wall %.2fs | gpu free %s MiB"
              % (payload["wall_time_s"],
                 payload["gpu_free_mib"] if payload["gpu_free_mib"] is not None else "n/a"))

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

