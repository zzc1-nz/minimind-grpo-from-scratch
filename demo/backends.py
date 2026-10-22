#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Uniform inference backends for the Phase-4 local demo (CPU only).

Three backends implement one interface::

    generate(prompt, max_new_tokens, temperature) -> dict(
        text, tokens, tok_per_sec, backend, device, ...)

* ``HFBackend``   -- native MiniMind ``.pth`` state_dict **or** an HF
                     (``Qwen3ForCausalLM``) directory, on CPU.
* ``GGUFBackend`` -- shells out to llama.cpp ``llama-completion.exe -ngl 0``.
* ``MockBackend`` -- canned text, for UI work with no model at all.

CRITICAL: this module never touches CUDA. ``torch.cuda`` is not imported and
every backend pins ``device="cpu"``. llama.cpp is always invoked with
``-ngl 0``. The GPU is reserved for the ~26 h pretraining job.

The chat prompt is built by a *single* shared function (:func:`format_prompt`)
so the HF and GGUF backends see byte-identical prompt text. This is the
canonical MiniMind format used by ``eval/run_eval.py --prompt-mode chat``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

# --------------------------------------------------------------------------
# project constants
# --------------------------------------------------------------------------
PROJECT_ROOT = r"D:\llm-rl-lab"
MINIMIND_ROOT = os.path.join(PROJECT_ROOT, "src", "minimind-master")
MINIMIND_MODEL_DIR = os.path.join(MINIMIND_ROOT, "model")  # tokenizer lives here
LLAMACPP_CPU_DIR = os.path.join(PROJECT_ROOT, "llamacpp", "cpu")
LLAMACPP_CUDA_DIR = os.path.join(PROJECT_ROOT, "llamacpp", "cuda124")
LLAMA_CLI = os.path.join(LLAMACPP_CPU_DIR, "llama-cli.exe")
LLAMA_COMPLETION = os.path.join(LLAMACPP_CPU_DIR, "llama-completion.exe")
CHAT_TEMPLATE_FILE = os.path.join(PROJECT_ROOT, "llamacpp", "chat_template_minimind.jinja")

#: Hard safety net for the "do not allocate CUDA memory" constraint: hide every
#: GPU from CUDA *before* torch is imported, so a stray ``.to("cuda")`` cannot
#: even succeed. ``-1`` is the value that actually hides the GPU from torch --
#: an empty string is treated as "unset" and still exposes it (measured).
#: Set ``MINIMIND_DEMO_ALLOW_GPU=1`` to opt out (never do that while the ~26 h
#: pretraining job owns the GPU).
if os.environ.get("MINIMIND_DEMO_ALLOW_GPU", "0").strip() not in ("1", "true", "yes", "on"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#: Canonical prompt templates. ``chat`` is what ``eval/run_eval.py`` uses as
#: its default (``--prompt-mode chat``) and what the SFT baseline was trained
#: with: a *closed, empty* <think> block. ``chat_think`` leaves the block open
#: (the GRPO reward expects a </think>), ``raw`` has no think block at all.
PROMPT_TEMPLATES = {
    "chat": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "chat_think": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "raw": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
}
PROMPT_MODES = tuple(PROMPT_TEMPLATES)
DEFAULT_PROMPT_MODE = "chat"

#: llama.cpp CPU thread cap (16 GB box, other work running: never exceed 8).
DEFAULT_THREADS = 8


def format_prompt(prompt: str, mode: str = DEFAULT_PROMPT_MODE) -> str:
    """The one and only prompt formatter used by every backend.

    Keeping this a single function is deliberate: SFT-vs-GRPO-vs-GGUF
    comparisons are only meaningful if every backend receives exactly the same
    bytes. llama.cpp's own ``--chat-template-file`` machinery is therefore
    *bypassed* -- we hand it the already-formatted string via ``-f``.
    """
    if mode not in PROMPT_TEMPLATES:
        raise ValueError("unknown prompt mode %r (choose from %s)" % (mode, ", ".join(PROMPT_MODES)))
    return PROMPT_TEMPLATES[mode].format(prompt=prompt)


def _utf8_console() -> None:
    """Make Windows consoles print Chinese instead of dying on GBK."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - non-tty streams may not support it
            pass


def set_seed(seed: int = 42) -> None:
    """Seed python/numpy/torch so a sampled run is reproducible.

    Without this, temperature > 0 gives different text on every run and the
    numbers quoted in a report cannot be reproduced. CPU-only: no CUDA seeding
    is attempted (and CUDA is hidden anyway).
    """
    import random

    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:  # noqa: BLE001 - numpy is optional
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)  # MiniMind's generate is fine either way
    except Exception:  # noqa: BLE001
        pass


def detect_kind(spec: str) -> str:
    """Classify a model path: ``pth`` | ``hf`` | ``gguf`` | ``mock``."""
    low = spec.strip().lower()
    if low in ("mock", "none", ""):
        return "mock"
    if low.endswith(".pth") or low.endswith(".pt") or low.endswith(".bin"):
        return "pth"
    if low.endswith(".gguf"):
        return "gguf"
    if os.path.isdir(spec):
        if os.path.exists(os.path.join(spec, "config.json")):
            return "hf"
        raise ValueError("%s is a directory but has no config.json; not an HF export" % spec)
    raise ValueError("cannot tell what kind of model %r is (expected .pth, .gguf, HF dir or 'mock')" % spec)


# --------------------------------------------------------------------------
# base
# --------------------------------------------------------------------------
class Backend:
    """Common interface: ``generate()`` returns a dict, never raises on OOM."""

    name = "base"

    def __init__(self, spec: str, prompt_mode: str = DEFAULT_PROMPT_MODE, **kwargs):
        self.spec = spec
        self.prompt_mode = prompt_mode
        self.load_time_s = 0.0

    # -- to be implemented --------------------------------------------------
    def generate(self, prompt: str, max_new_tokens: int = 48, temperature: float = 0.7) -> dict:
        raise NotImplementedError

    def info(self) -> dict:
        return {"backend": self.name, "spec": self.spec, "device": "cpu",
                "prompt_mode": self.prompt_mode, "load_time_s": round(self.load_time_s, 3)}

    def close(self) -> None:
        """Drop model references (best effort)."""

    # -- shared helper ------------------------------------------------------
    def _result(self, text: str, tokens: int, elapsed_s: float, **extra) -> dict:
        out = {
            "text": text,
            "tokens": int(tokens),
            "elapsed_s": round(elapsed_s, 4),
            "tok_per_sec": round(tokens / elapsed_s, 2) if elapsed_s > 0 else 0.0,
            "backend": self.name,
            "device": "cpu",
            "prompt_mode": self.prompt_mode,
            "spec": self.spec,
        }
        out.update(extra)
        return out


# --------------------------------------------------------------------------
# HF backend (native MiniMind .pth, or an HF Qwen3 directory)
# --------------------------------------------------------------------------
class HFBackend(Backend):
    """MiniMind native ``.pth`` or HF export, always on CPU."""

    name = "hf"

    def __init__(self, spec: str, prompt_mode: str = DEFAULT_PROMPT_MODE,
                 dtype: str = "float32", threads: int = 0, hidden: int = 768,
                 layers: int = 8, minimind_root: str = MINIMIND_ROOT, seed: int = 42, **kwargs):
        super().__init__(spec, prompt_mode, **kwargs)
        self.kind = detect_kind(spec)
        if self.kind not in ("pth", "hf"):
            raise ValueError("HFBackend cannot load %r (kind=%s)" % (spec, self.kind))
        self.dtype_name = dtype
        self.threads = int(threads or 0)
        self.hidden = hidden
        self.layers = layers
        self.minimind_root = minimind_root
        self.seed = int(seed) if seed is not None else None
        self.model = None
        self.tokenizer = None
        self.params = None
        self.torch_dtype = None
        self._lock = threading.Lock()
        self._load()

    # -- loading ------------------------------------------------------------
    def _load(self) -> None:
        import torch
        from transformers import AutoTokenizer

        if self.threads > 0:
            torch.set_num_threads(self.threads)
        dtype = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}[self.dtype_name]
        self.torch_dtype = dtype
        t0 = time.time()

        if self.kind == "pth":
            if self.minimind_root not in sys.path:
                sys.path.insert(0, self.minimind_root)
            from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

            tokenizer = AutoTokenizer.from_pretrained(MINIMIND_MODEL_DIR)
            config = MiniMindConfig(hidden_size=self.hidden, num_hidden_layers=self.layers)
            model = MiniMindForCausalLM(config)
            state = torch.load(self.spec, map_location="cpu")
            if isinstance(state, dict):
                for key in ("model_state_dict", "state_dict", "model"):
                    if key in state and isinstance(state[key], dict):
                        state = state[key]
                        break
            missing, unexpected = model.load_state_dict(state, strict=False)
            self.load_report = {"missing_keys": len(missing), "unexpected_keys": len(unexpected)}
            model = model.to(dtype).eval()
            model_name = "MiniMindForCausalLM(native)"
        else:  # hf directory
            from transformers import AutoModelForCausalLM

            tokenizer = AutoTokenizer.from_pretrained(self.spec)
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.spec, trust_remote_code=True, dtype=dtype)
            except TypeError:  # transformers < 4.56 spells it torch_dtype
                model = AutoModelForCausalLM.from_pretrained(
                    self.spec, trust_remote_code=True, torch_dtype=dtype)
            model = model.eval()
            self.load_report = {"missing_keys": 0, "unexpected_keys": 0}
            model_name = type(model).__name__

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0
        self.tokenizer = tokenizer
        self.model = model
        self.model_name = model_name
        self.params = sum(p.numel() for p in model.parameters())
        self.load_time_s = time.time() - t0

    # -- generation ---------------------------------------------------------
    def generate(self, prompt: str, max_new_tokens: int = 48, temperature: float = 0.7) -> dict:
        import torch

        text_prompt = format_prompt(prompt, self.prompt_mode)
        enc = self.tokenizer(text_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        prompt_tokens = int(input_ids.shape[1])

        # temperature <= 0 means greedy. MiniMind's custom generate() divides
        # logits by temperature, so greedy must pass 1.0 with do_sample=False.
        greedy = temperature is None or float(temperature) <= 0.0
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=not greedy,
            temperature=1.0 if greedy else float(temperature),
            top_p=1.0 if greedy else 0.9,
            top_k=0,
            repetition_penalty=1.0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        with self._lock, torch.inference_mode():
            if self.seed is not None:
                torch.manual_seed(self.seed)  # reproducible sampling per call
            t0 = time.time()
            out = self.model.generate(**gen_kwargs)
            elapsed = time.time() - t0
        new_ids = out[0][prompt_tokens:]
        tokens = int(new_ids.shape[0])
        text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
        return self._result(
            text, tokens, elapsed,
            prompt_tokens=prompt_tokens,
            params=round(self.params / 1e6, 3) if self.params else None,
            model_name=self.model_name,
            dtype=self.dtype_name,
            prompt=text_prompt,
            greedy=greedy,
        )

    def info(self) -> dict:
        d = super().info()
        d.update({"model_name": getattr(self, "model_name", None),
                  "dtype": self.dtype_name,
                  "params_m": round(self.params / 1e6, 3) if self.params else None,
                  "kind": self.kind})
        return d

    def close(self) -> None:
        self.model = None
        self.tokenizer = None


# --------------------------------------------------------------------------
# GGUF backend (llama.cpp, CPU)
# --------------------------------------------------------------------------
class GGUFBackend(Backend):
    """Runs a ``.gguf`` through llama.cpp on CPU (``-ngl 0``).

    b7795 trap: ``llama-cli.exe`` **rejects** ``-no-cnv`` ("please use
    llama-completion instead"); the non-interactive completion tool in this
    build is ``llama-completion.exe``. We use that.

    Windows traps handled here:
    * non-ASCII ``-p`` arguments get mangled by the PowerShell/ANSI path, so the
      prompt is always written to a UTF-8 (no BOM) temp file and passed with
      ``-f``;
    * stdout carries ``\\r\\n`` and an `` [end of text]`` marker -> normalised;
    * timings are parsed from stderr (``common_perf_print``).
    """

    name = "gguf"

    _EVAL_RE = re.compile(r"eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*runs.*?([0-9.]+)\s*tokens per second")
    _PROMPT_RE = re.compile(r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([0-9.]+)\s*tokens per second")
    _TOTAL_RE = re.compile(r"total time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens")

    def __init__(self, spec: str, prompt_mode: str = DEFAULT_PROMPT_MODE,
                 threads: int = DEFAULT_THREADS, ngl: int = 0, ctx: int = 2048,
                 seed: int = 42, binary: str | None = None, **kwargs):
        super().__init__(spec, prompt_mode, **kwargs)
        if detect_kind(spec) != "gguf":
            raise ValueError("GGUFBackend needs a .gguf file, got %r" % spec)
        if not os.path.exists(spec):
            raise FileNotFoundError(spec)
        if int(ngl) != 0:
            raise ValueError("refusing to run llama.cpp with -ngl %r: the GPU is reserved for "
                             "the training job. This demo is CPU-only (-ngl 0)." % ngl)
        self.threads = int(threads)
        self.ctx = int(ctx)
        self.seed = int(seed)
        self.binary = binary or self._find_binary()
        self.size_mb = round(os.path.getsize(spec) / (1024.0 * 1024.0), 1)
        self._lock = threading.Lock()
        self.load_time_s = 0.0  # llama.cpp loads per invocation

    @staticmethod
    def _find_binary() -> str:
        for cand in (LLAMA_COMPLETION, os.path.join(LLAMACPP_CUDA_DIR, "llama-completion.exe"),
                     LLAMA_CLI):
            if os.path.exists(cand):
                return cand
        raise FileNotFoundError(
            "no llama.cpp completion binary found under %s" % LLAMACPP_CPU_DIR)

    def build_command(self, prompt_file: str, max_new_tokens: int, temperature: float) -> list:
        greedy = temperature is None or float(temperature) <= 0.0
        cmd = [
            self.binary,
            "-m", self.spec,
            "-ngl", "0",                      # CPU only -- never touch the GPU
            "-t", str(self.threads),
            "-c", str(self.ctx),
            "-f", prompt_file,
            "-n", str(int(max_new_tokens)),
            "--temp", "0.0" if greedy else ("%.4f" % float(temperature)),
            "--top-p", "1.0" if greedy else "0.9",
            "--top-k", "0" if greedy else "40",
            "--min-p", "0.0",
            "-s", str(self.seed),
            "-no-cnv",                        # raw completion, not a chat loop
            "--no-display-prompt",
            "--no-warmup",
        ]
        return cmd

    def generate(self, prompt: str, max_new_tokens: int = 48, temperature: float = 0.7) -> dict:
        text_prompt = format_prompt(prompt, self.prompt_mode)
        tmpdir = tempfile.mkdtemp(prefix="minimind_demo_")
        prompt_file = os.path.join(tmpdir, "prompt.txt")
        try:
            with open(prompt_file, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text_prompt)
            cmd = self.build_command(prompt_file, max_new_tokens, temperature)
            with self._lock:
                t0 = time.time()
                proc = subprocess.run(cmd, capture_output=True, timeout=600)
                elapsed = time.time() - t0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        stderr = proc.stderr.decode("utf-8", "replace")
        stdout = proc.stdout.decode("utf-8", "replace")
        if proc.returncode != 0:
            raise RuntimeError("llama.cpp failed (rc=%d): %s"
                               % (proc.returncode, stderr.strip()[-800:]))

        text = stdout.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace(" [end of text]", "").strip()

        # llama.cpp off-by-one convention: the first prompt token is the initial
        # input (not "evaluated") and the last sampled token is never decoded,
        # so both counters are one less than the real token counts.
        m = self._EVAL_RE.search(stderr)
        if m:
            eval_runs = int(m.group(2))
            tokens = eval_runs + 1
            gen_tps = float(m.group(3))
        else:
            tokens = 0
            gen_tps = 0.0
        mp = self._PROMPT_RE.search(stderr)
        prompt_tokens = (int(mp.group(2)) + 1) if mp else 0
        prompt_tps = float(mp.group(3)) if mp else 0.0
        mt = self._TOTAL_RE.search(stderr)
        total_ms = float(mt.group(1)) if mt else None

        return self._result(
            text, tokens, elapsed,
            prompt_tokens=prompt_tokens,
            prompt_tok_per_sec=round(prompt_tps, 2),
            tok_per_sec=round(gen_tps, 2),        # llama.cpp's own decode rate
            wall_tok_per_sec=round(tokens / elapsed, 2) if elapsed > 0 else 0.0,
            llama_total_ms=total_ms,
            ngl=0,
            threads=self.threads,
            gguf_size_mb=self.size_mb,
            binary=os.path.basename(self.binary),
            prompt=text_prompt,
            greedy=(temperature is None or float(temperature) <= 0.0),
        )

    def info(self) -> dict:
        d = super().info()
        d.update({"ngl": 0, "threads": self.threads, "ctx": self.ctx,
                  "gguf_size_mb": self.size_mb, "binary": os.path.basename(self.binary)})
        return d


# --------------------------------------------------------------------------
# Mock backend
# --------------------------------------------------------------------------
class MockBackend(Backend):
    """No model: deterministic canned answers so the UI/CLI can be built."""

    name = "mock"

    def __init__(self, spec: str = "mock", prompt_mode: str = DEFAULT_PROMPT_MODE, label: str = "MOCK", **kwargs):
        super().__init__(spec, prompt_mode, **kwargs)
        self.label = label
        self.load_time_s = 0.0

    def generate(self, prompt: str, max_new_tokens: int = 48, temperature: float = 0.7) -> dict:
        time.sleep(0.05)
        tokens = max(1, min(int(max_new_tokens), 12))
        text = ("[%s mock] 这是占位回答：%s" % (self.label, prompt.strip()[:40]))
        return self._result(text, tokens, 0.05, prompt_tokens=len(prompt), prompt=format_prompt(prompt, self.prompt_mode),
                            greedy=(temperature is None or float(temperature) <= 0.0))


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------
def make_backend(spec: str, prompt_mode: str = DEFAULT_PROMPT_MODE, kind: str = "auto",
                 dtype: str = "float32", threads: int = 0, **kwargs) -> Backend:
    """Build the right backend for ``spec`` (``.pth`` / HF dir / ``.gguf`` / mock)."""
    if kind == "auto":
        kind = detect_kind(spec)
    if kind == "mock":
        return MockBackend(spec, prompt_mode, **kwargs)
    if kind in ("pth", "hf"):
        return HFBackend(spec, prompt_mode, dtype=dtype, threads=threads, **kwargs)
    if kind == "gguf":
        return GGUFBackend(spec, prompt_mode,
                           threads=threads or DEFAULT_THREADS, **kwargs)
    raise ValueError("unknown backend kind %r" % kind)


def gpu_free_mib() -> int | None:
    """Free VRAM in MiB, from nvidia-smi (read-only; allocates nothing)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, timeout=20)
        return int(r.stdout.decode("ascii", "replace").strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    _utf8_console()
    import argparse
    import json

    ap = argparse.ArgumentParser(description="probe one backend on CPU")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="3 + 5 等于多少？请只回复一个数字。")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--prompt-mode", default=DEFAULT_PROMPT_MODE, choices=list(PROMPT_MODES))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--dtype", default="float32")
    a = ap.parse_args()
    b = make_backend(a.model, a.prompt_mode, dtype=a.dtype, threads=a.threads)
    print(json.dumps(b.info(), ensure_ascii=False, indent=2))
    r = b.generate(a.prompt, a.max_new_tokens, a.temp)
    r.pop("prompt", None)
    print(json.dumps(r, ensure_ascii=False, indent=2))
