#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fixed demo battery: ~12 prompts x 2 checkpoints -> reports/demo_battery.json

Prompts: 9 taken verbatim from the frozen held-out benchmark
(``eval/heldout_tasks.jsonl`` -- read only, never modified) covering all four
families, plus 3 chit-chat prompts. The held-out answers are *not* leaked to the
model; they are only used to grade the completions with ``eval/grader.py`` so
the report shows whether the plumbing produces gradeable text.

Works unchanged with the smoke weights (garbage text is expected) and with the
real ``full_sft_768.pth`` / ``grpo_768.pth`` -- nothing is hard-coded.

Example::

    python run_compare_battery.py ^
        --sft  D:\\llm-rl-lab\\src\\minimind-master\\out\\sft_smoke_768.pth ^
        --grpo D:\\llm-rl-lab\\src\\minimind-master\\out\\grpo_smoke_768.pth ^
        --out  D:\\llm-rl-lab\\reports\\demo_battery.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import backends as B  # noqa: E402

PROJECT_ROOT = os.path.dirname(_HERE)
TASKS_PATH = os.path.join(PROJECT_ROOT, "eval", "heldout_tasks.jsonl")
GRADER_DIR = os.path.join(PROJECT_ROOT, "eval")

#: Held-out task ids reused for the demo battery (all four families, 9 tasks).
HELDOUT_IDS = ["arith_01", "arith_05", "format_01", "format_04",
               "logic_02", "logic_03", "multi_01", "multi_03", "format_06"]

#: 3 free-form chit-chat prompts -> 9 held-out + 3 chit-chat = 12 total.
EXTRA_PROMPTS = [
    {"id": "chat_01", "family": "chit-chat",
     "prompt": "你好，请用一句话介绍一下你自己。"},
    {"id": "chat_02", "family": "chit-chat",
     "prompt": "用一句话解释什么是人工智能。"},
    {"id": "chat_03", "family": "chit-chat",
     "prompt": "推荐一部适合周末看的电影，并说明理由。"},
]


def load_heldout_prompts(path: str) -> dict:
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["id"]] = rec
    return out


def build_prompt_list() -> list:
    heldout = load_heldout_prompts(TASKS_PATH)
    prompts = []
    for tid in HELDOUT_IDS:
        if tid in heldout:
            rec = heldout[tid]
            prompts.append({"id": tid, "family": rec["family"], "prompt": rec["prompt"],
                            "source": "eval/heldout_tasks.jsonl", "task": rec})
        else:  # benchmark file missing -> keep the battery runnable
            prompts.append({"id": tid, "family": "unknown", "prompt": "[missing held-out task %s]" % tid,
                            "source": "missing", "task": None})
    for extra in EXTRA_PROMPTS:
        prompts.append(dict(extra, source="inline", task=None))
    return prompts


def grade(prompt_rec: dict, text: str) -> dict | None:
    """Grade with the frozen grader if we have the task record."""
    task = prompt_rec.get("task")
    if not task:
        return None
    if GRADER_DIR not in sys.path:
        sys.path.insert(0, GRADER_DIR)
    try:
        import grader as G  # noqa: WPS433 - local import keeps torch optional
        g = G.grade(task, text)
        return {k: g.get(k) for k in ("correct", "reason", "extracted") if k in g}
    except Exception as exc:  # noqa: BLE001
        return {"error": "%s: %s" % (type(exc).__name__, exc)}


def run_one(backend, prompt_rec, args):
    t0 = time.time()
    try:
        res = backend.generate(prompt_rec["prompt"], args.max_new_tokens, args.temp)
    except Exception as exc:  # noqa: BLE001
        return {"error": "%s: %s" % (type(exc).__name__, exc),
                "elapsed_s": round(time.time() - t0, 3)}
    res.pop("prompt", None)
    res["total_s"] = round(time.time() - t0, 3)
    graded = grade(prompt_rec, res["text"])
    if graded is not None:
        res["graded"] = graded
    return res


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="SFT-vs-GRPO demo battery (CPU only)")
    ap.add_argument("--sft", required=True, help="SFT model (.pth | HF dir | .gguf | mock)")
    ap.add_argument("--grpo", required=True, help="GRPO model (.pth | HF dir | .gguf | mock)")
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "reports", "demo_battery.json"))
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--prompt-mode", default=B.DEFAULT_PROMPT_MODE, choices=list(B.PROMPT_MODES))
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--threads", type=int, default=B.DEFAULT_THREADS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N prompts")
    return ap.parse_args(argv)


def main(argv=None):
    B._utf8_console()
    args = parse_args(argv)
    B.set_seed(args.seed)
    prompts = build_prompt_list()
    if args.limit:
        prompts = prompts[:args.limit]

    t_start = time.time()
    models = {}
    for side, spec in (("sft", args.sft), ("grpo", args.grpo)):
        t0 = time.time()
        backend = B.make_backend(spec, args.prompt_mode, dtype=args.dtype, threads=args.threads, seed=args.seed)
        info = backend.info()
        info["load_time_s"] = round(time.time() - t0, 3)
        models[side] = {"spec": spec, "info": info, "backend": backend}
        print("[load] %-4s %-70s %.2fs" % (side, spec, info["load_time_s"]))

    rows = []
    for i, rec in enumerate(prompts, 1):
        row = {"id": rec["id"], "family": rec["family"], "source": rec["source"],
               "prompt": rec["prompt"]}
        for side in ("sft", "grpo"):
            row[side] = run_one(models[side]["backend"], rec, args)
            r = row[side]
            if "error" in r:
                print("[%2d/%2d] %-24s %-4s ERROR %s" % (i, len(prompts), rec["id"], side, r["error"][:60]))
            else:
                print("[%2d/%2d] %-24s %-4s tok=%3d %7.1f t/s  %r"
                      % (i, len(prompts), rec["id"], side, r["tokens"], r["tok_per_sec"],
                         r["text"][:42]))
        rows.append(row)

    for side in ("sft", "grpo"):
        models[side]["backend"].close()

    def summarize(side):
        ok = [r[side] for r in rows if "error" not in r[side]]
        graded = [r[side]["graded"] for r in rows if "graded" in r[side]]
        return {
            "prompts": len(rows),
            "ok": len(ok),
            "errors": len(rows) - len(ok),
            "mean_tok_per_sec": round(sum(x["tok_per_sec"] for x in ok) / len(ok), 2) if ok else 0.0,
            "total_tokens": sum(x["tokens"] for x in ok),
            "total_elapsed_s": round(sum(x["elapsed_s"] for x in ok), 2),
            "graded_prompts": len(graded),
            "graded_correct": sum(1 for g in graded if g.get("correct")),
            "empty_completions": sum(1 for x in ok if not x["text"].strip()),
        }

    payload = {
        "report_version": 1,
        "kind": "phase4_demo_battery",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "device": "cpu",
        "gpu_free_mib": B.gpu_free_mib(),
        "ngl": 0,
        "prompt_mode": args.prompt_mode,
        "prompt_template": B.PROMPT_TEMPLATES[args.prompt_mode],
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temp,
        "greedy": args.temp <= 0,
        "dtype": args.dtype,
        "threads": args.threads,
        "seed": args.seed,
        "heldout_task_file": TASKS_PATH,
        "heldout_ids_used": HELDOUT_IDS,
        "models": {k: {"spec": v["spec"], "info": v["info"]} for k, v in models.items()},
        "summary": {"sft": summarize("sft"), "grpo": summarize("grpo")},
        "wall_time_s": round(time.time() - t_start, 2),
        "results": rows,
    }
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print("\n[summary] sft  : %s" % json.dumps(payload["summary"]["sft"], ensure_ascii=False))
    print("[summary] grpo : %s" % json.dumps(payload["summary"]["grpo"], ensure_ascii=False))
    print("[report] %s  (%.1fs)" % (out, payload["wall_time_s"]))
    n_err = sum(1 for r in rows for side in ("sft", "grpo") if "error" in r[side])
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())

