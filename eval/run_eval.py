#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Held-out evaluation runner for MiniMind checkpoints.

Loads a native MiniMind ``.pth`` state_dict (fp16), generates one answer per
task with greedy decoding (or ``--sample-k N`` majority voting), grades the
answers with the pure-python ``grader.py``, and writes a JSON report.

Typical use::

    python eval/run_eval.py \
        --weight src/minimind-master/out/sft_smoke_768.pth \
        --hidden 768 --layers 8 --max-new-tokens 96 \
        --device cuda \
        --out reports/sft_baseline_smoke.json

The report records the exact prompt template, generation params, seed,
``heldout_tasks.jsonl`` sha256 and (if present) whether that hash matches
``lock_hashes.json``, so SFT-vs-GRPO comparisons can be checked for benchmark
drift.

The GGUF backend (llama.cpp quantized checkpoints) is intentionally left as a
clearly-marked stub: ``--backend gguf`` raises ``NotImplementedError``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from collections import Counter
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import grader as G  # noqa: E402  (same-directory pure-python grader)

PROMPT_MODES = ("chat", "chat_think", "raw")

# Human-readable templates. ``chat`` is the repo's canonical non-thinking eval
# format: the tokenizer chat template closes an empty <think> block so the
# model is expected to answer directly. ``chat_think`` leaves <think> open
# (repo's open_thinking=True). ``raw`` is the bare prompt from the task brief.
PROMPT_TEMPLATES = {
    "chat": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "chat_think": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "raw": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def setup_seed(seed):
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:  # noqa: BLE001 - numpy is optional for the runner
        pass
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_tasks(path, limit=None):
    tasks = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit("bad JSON on %s:%d: %s" % (path, line_no, exc))
    errors = G.validate_tasks(tasks)
    if errors:
        raise SystemExit("heldout task file invalid:\n  - " + "\n  - ".join(errors))
    if limit:
        tasks = tasks[:limit]
    return tasks


def resolve_device(requested, min_free_mib=4000):
    """Resolve cuda/cpu, falling back to CPU when VRAM is tight (per brief)."""
    import torch
    if requested == "cpu":
        return "cpu", "requested_cpu"
    if not torch.cuda.is_available():
        return "cpu", "cuda_unavailable"
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        free_mib = free_b / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        free_mib = None
    if free_mib is not None and free_mib < min_free_mib:
        return "cpu", "cuda_free_vram_%dMiB_below_%dMiB" % (int(free_mib), min_free_mib)
    return "cuda", "cuda_free_vram_%dMiB" % int(free_mib) if free_mib is not None else "cuda"


def build_prompt(tokenizer, question, mode):
    if mode == "raw":
        return PROMPT_TEMPLATES["raw"].format(prompt=question)
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=(mode == "chat_think"),
        )
    except TypeError:
        # Older tokenizer configs without open_thinking support.
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )


# --------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------
def load_model_and_tokenizer(args):
    import torch
    sys.path.insert(0, args.minimind_root)
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    config = MiniMindConfig(hidden_size=args.hidden, num_hidden_layers=args.layers)
    model = MiniMindForCausalLM(config)
    state = torch.load(args.weight, map_location="cpu")
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[load] missing keys: %d (first: %s)" % (len(missing), missing[:5]))
    if unexpected:
        print("[load] unexpected keys: %d (first: %s)" % (len(unexpected), unexpected[:5]))
    if len(missing) > 0.25 * max(1, len(list(model.state_dict()))):
        print("[load] WARNING: more than 25%% of model weights are missing from %s" % args.weight)

    if args.device == "cuda":
        model = model.half()
        dtype = "float16"
    else:
        model = model.float()
        dtype = "float32"
    model = model.to(args.device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    return model, tokenizer, {"dtype": dtype, "params_total": n_params,
                              "missing_keys": len(missing), "unexpected_keys": len(unexpected)}


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def generate_completions(model, tokenizer, prompt, args, n_return=1):
    """Generate ``n_return`` completions for one prompt (greedy or sampled)."""
    import torch

    enc = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_ids = enc["input_ids"].to(args.device)
    attention_mask = enc["attention_mask"].to(args.device)
    prompt_len = input_ids.shape[1]
    if n_return > 1:
        input_ids = input_ids.repeat(n_return, 1)
        attention_mask = attention_mask.repeat(n_return, 1)

    if args.sample_k > 1:
        do_sample = True
        temperature = args.sample_temp
        top_p = args.sample_top_p
        top_k = args.sample_top_k
    else:
        # Greedy: MiniMind's custom generate divides by temperature, so it must
        # be > 0; 1.0 with top_p=1.0/top_k=0 is exact argmax.
        do_sample = False
        temperature = 1.0
        top_p = 1.0
        top_k = 0

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,  # already repeated above
        )
    completions = []
    gen_token_counts = []
    for row in out:
        new_ids = row[prompt_len:]
        gen_token_counts.append(int(new_ids.shape[0]))
        # Keep special tokens visible: grader.normalize strips them, and the
        # report then shows whether the model closed with <|im_end|>.
        completions.append(tokenizer.decode(new_ids, skip_special_tokens=False))
    return completions, gen_token_counts


def grade_one(task, completions, args):
    """Grade one task; majority-vote when --sample-k > 1."""
    if args.sample_k <= 1:
        completion = completions[0]
        correct, reason = G.check(task, completion)
        return {
            "id": task["id"],
            "family": task["family"],
            "prompt": task["prompt"],
            "completion": completion,
            "correct": bool(correct),
            "reason": reason,
        }

    samples = []
    votes = Counter()
    for completion in completions:
        value, extract_reason = G.extract(task, completion)
        if value is not None:
            votes[value] += 1
        samples.append({"completion": completion, "extracted": value, "extract_reason": extract_reason})
    if not votes:
        return {
            "id": task["id"], "family": task["family"], "prompt": task["prompt"],
            "completion": completions[0], "correct": False,
            "reason": "majority:no_valid_extraction (%s)" % samples[0]["extract_reason"],
            "votes": {}, "samples": samples,
        }
    best_value, best_count = votes.most_common(1)[0]
    correct, reason = G.check(task, best_value)
    return {
        "id": task["id"], "family": task["family"], "prompt": task["prompt"],
        "completion": best_value, "correct": bool(correct),
        "reason": "majority %d/%d -> %r | %s" % (best_count, len(completions), best_value, reason),
        "votes": dict(votes), "samples": samples,
    }


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------
class GGUFGenerationBackend:
    """llama.cpp-backed generation for a quantized .gguf checkpoint.

    Phase-4 needs the *quantized* model's held-out accuracy so the report can state the
    quantization drop instead of asserting it is small. Grading stays in ``grader.py``;
    only generation differs.

    This deliberately **reuses** ``demo/backends.py::GGUFBackend`` rather than keeping a
    second llama.cpp invocation: that class already encodes the b7795 traps (the
    ``-no-cnv`` flag, the UTF-8 prompt temp file, stdout normalisation, timing parsing)
    and was validated at ~493 tok/s on CPU. Two copies would drift.
    """

    def __init__(self, gguf_path, llamacpp_root=None, threads=8, ctx=2048, timeout=600):
        demo_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo")
        if demo_dir not in sys.path:
            sys.path.insert(0, demo_dir)
        import backends as demo_backends  # noqa: E402
        self._impl = demo_backends.GGUFBackend(gguf_path, prompt_mode="raw",
                                               threads=threads, ngl=0, ctx=ctx)
        self.binary = self._impl.binary
        self.gguf_path = gguf_path
        self.timeout = timeout

    def generate(self, prompt, max_new_tokens=64, temperature=0.0):
        """Return [text]; the prompt is passed through verbatim (prompt_mode='raw')."""
        res = self._impl.generate(prompt, max_new_tokens=max_new_tokens,
                                  temperature=temperature)
        return [res["text"]]


def _load_gguf_backend(args):
    """Build the llama.cpp generation backend for ``--backend gguf``."""
    return GGUFGenerationBackend(args.weight, args.llamacpp_root,
                                 threads=args.gguf_threads)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Held-out evaluation harness for MiniMind checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weight", required=True, help="path to MiniMind .pth state_dict")
    parser.add_argument("--hidden", type=int, default=768, help="hidden_size")
    parser.add_argument("--layers", type=int, default=8, help="num_hidden_layers")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="generation cap per task")
    parser.add_argument("--out", default=None, help="report JSON path (default: reports/<stem>_eval.json)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="device; 'auto' falls back to CPU when free VRAM < 4000 MiB")
    parser.add_argument("--backend", choices=["hf", "gguf"], default="hf",
                        help="hf = native MiniMind .pth; gguf = llama.cpp quantized file")
    parser.add_argument("--llamacpp-root", default=None,
                        help="llama.cpp build dir (default: <project>/llamacpp)")
    parser.add_argument("--gguf-threads", type=int, default=8,
                        help="CPU threads for the gguf backend")
    parser.add_argument("--sample-k", type=int, default=1,
                        help=">1 enables sampled majority-vote mode")
    parser.add_argument("--sample-temp", type=float, default=0.7, help="temperature for --sample-k")
    parser.add_argument("--sample-top-p", type=float, default=0.9, help="top_p for --sample-k")
    parser.add_argument("--sample-top-k", type=int, default=50, help="top_k for --sample-k")
    parser.add_argument("--prompt-mode", choices=list(PROMPT_MODES), default="chat",
                        help="chat = repo non-thinking template (empty <think> block)")
    parser.add_argument("--tasks", default=os.path.join(_THIS_DIR, "heldout_tasks.jsonl"),
                        help="held-out tasks JSONL")
    parser.add_argument("--lock", default=os.path.join(_THIS_DIR, "lock_hashes.json"),
                        help="optional frozen-hash lock file")
    parser.add_argument("--minimind-root", default=os.path.join(PROJECT_ROOT, 'src', 'minimind-master'),
                        help="directory containing the 'model' package")
    parser.add_argument("--tokenizer-dir", default=None,
                        help="tokenizer directory (default: <minimind-root>/model)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="debug: only first N tasks")
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    gguf_backend = None
    if args.backend == "gguf":
        args.tokenizer_dir = args.tokenizer_dir or os.path.join(args.minimind_root, "model")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
        gguf_backend = _load_gguf_backend(args)
        model = None
        print("[backend] gguf via llama.cpp: %s (ngl=0)" % os.path.basename(gguf_backend.binary))
    else:
        args.tokenizer_dir = args.tokenizer_dir or os.path.join(args.minimind_root, "model")
        setup_seed(args.seed)
        import torch

    wall_start = time.time()
    if gguf_backend is not None:
        device, device_reason = "cpu", "gguf backend runs llama.cpp with -ngl 0"
        args.device = device
    else:
        device, device_reason = resolve_device(args.device)
        args.device = device
    if args.device == "cpu":
        print("[device] CPU (%s)" % device_reason)

    tasks_path = os.path.abspath(args.tasks)
    tasks = load_tasks(tasks_path, args.limit)
    tasks_sha = sha256_file(tasks_path)

    lock_check = None
    lock_path = os.path.abspath(args.lock)
    if os.path.exists(lock_path):
        try:
            with open(lock_path, "r", encoding="utf-8") as fh:
                lock = json.load(fh)
            expected = (lock.get("files") or {}).get("heldout_tasks.jsonl", {}).get("sha256")
            lock_check = {"expected": expected, "actual": tasks_sha,
                          "match": expected == tasks_sha}
            if expected != tasks_sha:
                print("[lock] WARNING: heldout_tasks.jsonl sha256 does not match %s" % lock_path)
        except Exception as exc:  # noqa: BLE001
            print("[lock] WARNING: could not read %s: %s" % (lock_path, exc))

    t_load = time.time()
    if gguf_backend is not None:
        model_info = {"dtype": "gguf", "params_total": 0,
                      "missing_keys": 0, "unexpected_keys": 0}
    else:
        model, tokenizer, model_info = load_model_and_tokenizer(args)
    load_time = time.time() - t_load
    if not args.quiet:
        print("[load] %s | %.2fM params | dtype=%s | device=%s | %.2fs"
              % (args.weight, model_info["params_total"] / 1e6, model_info["dtype"],
                 args.device, load_time))

    # Prompt-template sanity check / example (recorded in the report).
    prompt_example = build_prompt(tokenizer, tasks[0]["prompt"], args.prompt_mode)
    if not args.quiet:
        print("[prompt] mode=%s example=%r" % (args.prompt_mode, prompt_example))

    if args.device == "cuda" and gguf_backend is None:
        torch.cuda.reset_peak_memory_stats()
        model_vram_mb = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
    else:
        model_vram_mb = None

    per_task = []
    t_gen = time.time()
    for i, task in enumerate(tasks, 1):
        prompt = build_prompt(tokenizer, task["prompt"], args.prompt_mode)
        try:
            if gguf_backend is not None:
                completions = gguf_backend.generate(prompt, args.max_new_tokens,
                                                    temperature=0.0)
                gen_tokens = [len(c) for c in completions]
            else:
                completions, gen_tokens = generate_completions(model, tokenizer, prompt, args)
            row = grade_one(task, completions, args)
            row["gen_tokens"] = gen_tokens[0] if gen_tokens else 0
        except Exception as exc:  # noqa: BLE001 - keep the eval run alive
            row = {
                "id": task["id"], "family": task["family"], "prompt": task["prompt"],
                "completion": "", "correct": False,
                "reason": "generation_error:%s:%s" % (type(exc).__name__, exc),
            }
            if not args.quiet:
                print("[error] %s: %s" % (task["id"], traceback.format_exc(limit=2)))
        per_task.append(row)
        if not args.quiet:
            print("[%2d/%2d] %-10s %-12s correct=%s  %s"
                  % (i, len(tasks), row["id"], row["family"], row["correct"], row["reason"][:110]))
    generation_time = time.time() - t_gen
    wall_time = time.time() - wall_start

    if args.device == "cuda" and gguf_backend is None:
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    else:
        peak_vram_mb = None

    families = {}
    for fam in G.TASK_FAMILIES:
        rows = [r for r in per_task if r["family"] == fam]
        if not rows:
            continue
        correct = sum(1 for r in rows if r["correct"])
        families[fam] = {"correct": correct, "total": len(rows),
                         "accuracy": round(correct / len(rows), 4)}
    total_correct = sum(1 for r in per_task if r["correct"])
    overall = {"correct": total_correct, "total": len(per_task),
               "accuracy": round(total_correct / len(per_task), 4) if per_task else 0.0}

    report = {
        "report_version": 1,
        "benchmark": {
            "name": "minimind_heldout_v1",
            "held_out": True,
            "task_file": tasks_path,
            "tasks_sha256": tasks_sha,
            "num_tasks": len(per_task),
            "families": {fam: sum(1 for r in per_task if r["family"] == fam) for fam in G.TASK_FAMILIES},
            "lock_check": lock_check,
        },
        "weight": {
            "path": os.path.abspath(args.weight),
            "sha256": sha256_file(args.weight),
            "bytes": os.path.getsize(args.weight),
        },
        "model": {
            "hidden_size": args.hidden,
            "num_hidden_layers": args.layers,
            "params_total": model_info["params_total"],
            "dtype": model_info["dtype"],
            "missing_keys": model_info["missing_keys"],
            "unexpected_keys": model_info["unexpected_keys"],
        },
        "device": args.device,
        "device_reason": device_reason,
        "backend": args.backend,
        "prompt_mode": args.prompt_mode,
        "prompt_template": PROMPT_TEMPLATES[args.prompt_mode],
        "prompt_example": prompt_example,
        "generation": {
            "mode": "majority_vote" if args.sample_k > 1 else "greedy",
            "do_sample": args.sample_k > 1,
            "temperature": args.sample_temp if args.sample_k > 1 else 1.0,
            "top_p": args.sample_top_p if args.sample_k > 1 else 1.0,
            "top_k": args.sample_top_k if args.sample_k > 1 else 0,
            "repetition_penalty": 1.0,
            "max_new_tokens": args.max_new_tokens,
            "eos_token_id": int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None,
            "sample_k": args.sample_k,
            "prompt_mode": args.prompt_mode,
        },
        "model_path": os.path.abspath(args.weight),
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "wall_time_s": round(wall_time, 3),
        "load_time_s": round(load_time, 3),
        "generation_time_s": round(generation_time, 3),
        "peak_vram_mb": round(peak_vram_mb, 1) if peak_vram_mb is not None else None,
        "model_vram_mb": round(model_vram_mb, 1) if model_vram_mb is not None else None,
        "overall": overall,
        "per_family": families,
        "per_task": per_task,
    }

    out_path = args.out
    if not out_path:
        reports_dir = os.path.join(os.path.dirname(_THIS_DIR), "reports")
        out_path = os.path.join(reports_dir, os.path.splitext(os.path.basename(args.weight))[0] + "_eval.json")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print("[result] overall %d/%d (%.2f%%) | %s"
          % (total_correct, len(per_task), 100.0 * overall["accuracy"],
             "  ".join("%s %.1f%%" % (fam, 100.0 * v["accuracy"]) for fam, v in families.items())))
    if peak_vram_mb is not None:
        print("[result] peak VRAM %.1f MiB | wall %.2fs | gen %.2fs"
              % (peak_vram_mb, wall_time, generation_time))
    print("[report] %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
