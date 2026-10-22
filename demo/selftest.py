#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Self-check for the Phase-4 demo app (no model needed, fast, CPU only).

    python selftest.py          # asserts only
    python selftest.py -v       # + a tiny live generation on the smoke weights

Checks: prompt-template invariants, backend auto-detection, the ``-ngl 0`` guard,
CUDA invisibility, the stdlib UI handler (in-process HTTP), and the CLI contract
(``--json`` shape / exit codes). Designed to be runnable *while training owns the
GPU*: it never touches CUDA.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import backends as B  # noqa: E402
import app as A  # noqa: E402

FAILS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("  ok   " if cond else "  FAIL ") + name + (("  [%s]" % detail) if detail else ""))
    if not cond:
        FAILS.append(name)


def test_prompt_templates():
    print("[1] prompt templates")
    q = "3 + 5 等于多少？"
    chat = B.format_prompt(q, "chat")
    check("chat ends with closed empty think block",
          chat.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), repr(chat[-40:]))
    check("chat starts with the user turn",
          chat.startswith("<|im_start|>user\n" + q + "<|im_end|>\n"), repr(chat[:30]))
    check("chat_think leaves <think> open", B.format_prompt(q, "chat_think").endswith("<think>\n"))
    raw = B.format_prompt(q, "raw")
    check("raw == think-free chat template (rendered jinja)",
          raw == "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % q, repr(raw))
    try:
        B.format_prompt(q, "nope")
        check("unknown prompt mode rejected", False)
    except ValueError:
        check("unknown prompt mode rejected", True)


def test_detect_kind():
    print("[2] backend auto-detection")
    out = os.path.join(B.PROJECT_ROOT, "src", "minimind-master", "out")
    check("pth", B.detect_kind(os.path.join(out, "sft_smoke_768.pth")) == "pth")
    check("hf dir", B.detect_kind(os.path.join(B.PROJECT_ROOT, "models", "hf", "sft_smoke_768")) == "hf")
    check("gguf", B.detect_kind(os.path.join(B.PROJECT_ROOT, "models", "gguf",
                                             "sft_smoke_768-Q4_K_M.gguf")) == "gguf")
    check("mock", B.detect_kind("mock") == "mock")
    try:
        B.detect_kind(os.path.join(B.PROJECT_ROOT, "demo"))
        check("non-HF dir rejected", False)
    except ValueError:
        check("non-HF dir rejected", True)


def test_guards():
    print("[3] CPU-only guards")
    check("CUDA_VISIBLE_DEVICES hidden from torch", os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
          repr(os.environ.get("CUDA_VISIBLE_DEVICES")))
    try:
        import torch
        check("torch.cuda.is_available() is False", torch.cuda.is_available() is False)
    except Exception as exc:  # noqa: BLE001
        check("torch.cuda.is_available() is False", False, str(exc))
    try:
        B.GGUFBackend(os.path.join(B.PROJECT_ROOT, "models", "gguf", "sft_smoke_768-Q4_K_M.gguf"),
                      ngl=99)
        check("GGUFBackend refuses -ngl != 0", False)
    except ValueError:
        check("GGUFBackend refuses -ngl != 0", True)
    cmd = B.GGUFBackend(os.path.join(B.PROJECT_ROOT, "models", "gguf", "sft_smoke_768-Q4_K_M.gguf")
                        ).build_command("p.txt", 8, 0.7)
    check("llama.cpp command contains -ngl 0", "-ngl" in cmd and cmd[cmd.index("-ngl") + 1] == "0")
    check("llama.cpp command contains -no-cnv", "-no-cnv" in cmd)
    check("llama.cpp threads <= 8", int(cmd[cmd.index("-t") + 1]) <= 8, cmd[cmd.index("-t") + 1])


def test_cli_json():
    print("[4] CLI contract (mock backend)")
    import contextlib
    import io
    import compare_cli
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = compare_cli.main(["--prompt", "你好", "--sft", "mock", "--grpo", "mock",
                               "--max-new-tokens", "8", "--json", "--quiet"])
    check("mock/mock CLI exits 0", rc == 0, "rc=%s" % rc)
    payload = json.loads(buf.getvalue())
    keys = {"text", "tokens", "tok_per_sec", "backend", "device"}
    check("--json has both sides with required keys",
          keys <= set(payload["sft"]) and keys <= set(payload["grpo"]))
    check("--json records device=cpu", payload["device"] == "cpu")
    with contextlib.redirect_stdout(io.StringIO()):
        rc = compare_cli.main(["--prompt", "x", "--sft", "D:\\nope\\missing.pth",
                               "--grpo", "mock", "--max-new-tokens", "8", "--json", "--quiet"])
    check("missing weight CLI exits 1", rc == 1, "rc=%s" % rc)


def test_http_api():
    print("[5] stdlib UI (in-process)")
    pool = A.BackendPool(B.DEFAULT_PROMPT_MODE, "float32", B.DEFAULT_THREADS)
    stats = {"t0": time.time(), "verbose": False}
    from http.server import ThreadingHTTPServer
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), A.make_handler(pool, stats))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % port
    try:
        with urllib.request.urlopen(base + "/healthz", timeout=20) as r:
            health = json.loads(r.read().decode("utf-8"))
        check("/healthz 200 + device cpu", health.get("status") == "ok" and health.get("device") == "cpu")
        with urllib.request.urlopen(base + "/", timeout=20) as r:
            html = r.read().decode("utf-8")
        check("GET / serves the page", "Run both" in html and "/api/generate" in html)
        check("page has both panels", "id=\"sftout\"" in html and "id=\"grpoout\"" in html)
        body = json.dumps({"prompt": "你好", "sft": "mock", "grpo": "mock",
                           "max_new_tokens": 8, "temperature": 0.7}).encode("utf-8")
        req = urllib.request.Request(base + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = json.loads(r.read().decode("utf-8"))
        check("POST /api/generate 200 + both sides",
              payload["sft"]["backend"] == "mock" and payload["grpo"]["backend"] == "mock")
        for side in ("sft", "grpo"):
            keys = {"text", "tokens", "tok_per_sec", "backend", "device", "elapsed_s"}
            check("result[%s] has the required keys" % side, keys <= set(payload[side]), str(sorted(payload[side])))
        try:
            urllib.request.urlopen(urllib.request.Request(
                base + "/api/generate", data=b'{"prompt":""}',
                headers={"Content-Type": "application/json"}), timeout=20)
            check("empty prompt -> HTTP 400", False)
        except urllib.error.HTTPError as exc:
            check("empty prompt -> HTTP 400", exc.code == 400, str(exc.code))
        try:
            urllib.request.urlopen(base + "/nope", timeout=20)
            check("unknown path -> 404", False)
        except urllib.error.HTTPError as exc:
            check("unknown path -> 404", exc.code == 404, str(exc.code))
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_live_generation():
    print("[6] live generation on the smoke weights (optional)")
    spec = os.path.join(B.PROJECT_ROOT, "src", "minimind-master", "out", "sft_smoke_768.pth")
    if not os.path.exists(spec):
        print("  skip (no smoke weights)")
        return
    be = B.make_backend(spec)
    r = be.generate("3+5等于几？只回复数字", 16, 0.7)
    check("HF .pth generates tokens", r["tokens"] > 0 and r["tok_per_sec"] > 0,
          "%d tok, %.1f t/s" % (r["tokens"], r["tok_per_sec"]))
    be.close()
    gg = os.path.join(B.PROJECT_ROOT, "models", "gguf", "sft_smoke_768-Q4_K_M.gguf")
    if os.path.exists(gg):
        be = B.make_backend(gg)
        r = be.generate("3+5等于几？只回复数字", 16, 0.7)
        check("GGUF generates tokens with ngl=0", r["tokens"] > 0 and r.get("ngl") == 0,
              "%d tok, %.1f t/s" % (r["tokens"], r["tok_per_sec"]))
        be.close()


def test_determinism():
    print("[7] reproducibility (seed=42)")
    spec = os.path.join(B.PROJECT_ROOT, "src", "minimind-master", "out", "sft_smoke_768.pth")
    if not os.path.exists(spec):
        print("  skip (no smoke weights)")
        return
    a = B.make_backend(spec, seed=42)
    r1 = a.generate("3+5等于几？只回复数字", 16, 0.7)["text"]
    r2 = a.generate("3+5等于几？只回复数字", 16, 0.7)["text"]
    a.close()
    b = B.make_backend(spec, seed=42)
    r3 = b.generate("3+5等于几？只回复数字", 16, 0.7)["text"]
    b.close()
    check("same backend twice -> identical text", r1 == r2, repr(r1[:24]))
    check("fresh backend, same seed -> identical text", r1 == r3, repr(r1[:24]))


def main():
    B._utf8_console()
    print("demo selftest -- device=cpu, ngl=0, no CUDA")
    test_prompt_templates()
    test_detect_kind()
    test_guards()
    test_cli_json()
    test_http_api()
    if "-v" in sys.argv or "--live" in sys.argv:
        test_live_generation()
        test_determinism()
    print()
    if FAILS:
        print("FAILED: %d check(s): %s" % (len(FAILS), ", ".join(FAILS)))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
