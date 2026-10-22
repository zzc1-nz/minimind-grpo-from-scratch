#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Measured CPU throughput of every demo backend (feeds reports/demo_app.md).

Runs each available model through the real backend layer, 3 repetitions of 64
new tokens, and prints a table plus the peak working set of the process. Also
asserts that CUDA is invisible (the training job owns the GPU).

    python bench_backends.py --reps 3 --max-new-tokens 64
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
OUT_DIR = os.path.join(PROJECT_ROOT, "src", "minimind-master", "out")
GGUF_DIR = os.path.join(PROJECT_ROOT, "models", "gguf")
HF_DIR = os.path.join(PROJECT_ROOT, "models", "hf")

CANDIDATES = [
    ("hf .pth fp32 (sft_smoke)", os.path.join(OUT_DIR, "sft_smoke_768.pth"), "float32"),
    ("hf .pth fp16 (sft_smoke)", os.path.join(OUT_DIR, "sft_smoke_768.pth"), "float16"),
    ("hf dir fp32 (sft_smoke HF export)", os.path.join(HF_DIR, "sft_smoke_768"), "float32"),
    ("gguf f16", os.path.join(GGUF_DIR, "sft_smoke_768-f16.gguf"), None),
    ("gguf Q8_0", os.path.join(GGUF_DIR, "sft_smoke_768-Q8_0.gguf"), None),
    ("gguf Q4_K_M", os.path.join(GGUF_DIR, "sft_smoke_768-Q4_K_M.gguf"), None),
    ("mock", "mock", None),
]

PROMPTS = [
    "3 + 5 等于多少？请只回复一个数字，不要写任何解释或计算过程。",
    "用一句话解释什么是人工智能。",
]


def rss_mib() -> float:
    try:
        import psutil  # not installed in the main venv; fall back below
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:  # noqa: BLE001
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(counters)
            ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
            return counters.PeakWorkingSetSize / (1024 * 1024)
        except Exception:  # noqa: BLE001
            return -1.0


def main(argv=None):
    B._utf8_console()
    ap = argparse.ArgumentParser(description="benchmark demo backends on CPU")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--threads", type=int, default=B.DEFAULT_THREADS)
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "reports", "demo_bench.json"))
    args = ap.parse_args(argv)

    import torch
    print("[env] torch %s | transformers %s | CUDA_VISIBLE_DEVICES=%r | torch.cuda.is_available(): %s | torch threads: %d"
          % (torch.__version__, __import__("transformers").__version__,
             os.environ.get("CUDA_VISIBLE_DEVICES"), torch.cuda.is_available(),
             torch.get_num_threads()))
    if torch.cuda.is_available():
        raise SystemExit(
            "CUDA is visible in this process -- refusing to run: the GPU is reserved for the\n"
            "~26 h training job. backends.py clears CUDA_VISIBLE_DEVICES on import; do not set\n"
            "MINIMIND_DEMO_ALLOW_GPU=1 here.")
    gpu0 = B.gpu_free_mib()
    print("[env] nvidia-smi free VRAM before: %s MiB | threads=%d | prompts=%d | reps=%d | n=%d"
          % (gpu0, args.threads, len(PROMPTS), args.reps, args.max_new_tokens))

    rows = []
    for label, spec, dtype in CANDIDATES:
        if spec != "mock" and not os.path.exists(spec):
            print("[skip] %-34s (missing: %s)" % (label, spec))
            continue
        t0 = time.time()
        try:
            backend = B.make_backend(spec, B.DEFAULT_PROMPT_MODE,
                                     dtype=dtype or "float32", threads=args.threads, seed=42)
        except Exception as exc:  # noqa: BLE001
            print("[fail] %-34s %s: %s" % (label, type(exc).__name__, exc))
            continue
        load_s = time.time() - t0
        per_rep = []
        texts = []
        for rep in range(args.reps):
            for prompt in PROMPTS:
                r = backend.generate(prompt, args.max_new_tokens, args.temp)
                per_rep.append(r)
                if rep == 0:
                    texts.append(r["text"][:80])
        tps = [r["tok_per_sec"] for r in per_rep]
        tokens = [r["tokens"] for r in per_rep]
        row = {
            "label": label, "spec": spec, "dtype": dtype,
            "backend": per_rep[0]["backend"],
            "load_s": round(load_s, 2),
            "mean_tok_per_sec": round(sum(tps) / len(tps), 2),
            "min_tok_per_sec": round(min(tps), 2),
            "max_tok_per_sec": round(max(tps), 2),
            "mean_tokens": round(sum(tokens) / len(tokens), 1),
            "mean_elapsed_s": round(sum(r["elapsed_s"] for r in per_rep) / len(per_rep), 3),
            "ngl": per_rep[0].get("ngl"),
            "prompt_tokens": per_rep[0].get("prompt_tokens"),
            "sample_text": texts[0] if texts else "",
            "runs": len(per_rep),
        }
        rows.append(row)
        print("[ok]   %-34s %-5s %7.2f t/s (min %7.2f max %7.2f) load %.2fs"
              % (label, row["backend"], row["mean_tok_per_sec"],
                 row["min_tok_per_sec"], row["max_tok_per_sec"], load_s))
        backend.close()

    gpu1 = B.gpu_free_mib()
    payload = {
        "report_version": 1, "kind": "demo_backend_bench",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "device": "cpu", "ngl": 0, "threads": args.threads,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "cuda_visible": False,
        "gpu_free_mib_before": gpu0, "gpu_free_mib_after": gpu1,
        "max_new_tokens": args.max_new_tokens, "temperature": args.temp,
        "prompts": PROMPTS, "reps": args.reps,
        "peak_working_set_mib": round(rss_mib(), 1),
        "rows": rows,
    }
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print("[env] nvidia-smi free VRAM after: %s MiB | peak working set %.0f MiB"
          % (gpu1, payload["peak_working_set_mib"]))
    print("[report] %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

