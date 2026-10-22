#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase-4 local web UI: same prompt, SFT vs GRPO side by side, CPU only.

Two front-ends, one backend layer:

* ``--ui gradio``   -> Gradio app on http://127.0.0.1:7860 (needs the demo venv,
  see ``reports/demo_app.md``; the main ``.venv`` is deliberately untouched).
* ``--ui stdlib``   -> zero-dependency ``http.server`` app on
  http://127.0.0.1:7861 with a single HTML page and ``POST /api/generate``.

Both are CPU-only: no CUDA is initialised (``CUDA_VISIBLE_DEVICES`` is cleared
in ``backends.py``) and llama.cpp always runs with ``-ngl 0``.

Run it as a short-lived managed job only -- never leave it running::

    python app.py --ui stdlib --port 7861
    # then: curl http://127.0.0.1:7861/healthz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import backends as B  # noqa: E402

PROJECT_ROOT = os.path.dirname(_HERE)
OUT_DIR = os.path.join(PROJECT_ROOT, "src", "minimind-master", "out")
GGUF_DIR = os.path.join(PROJECT_ROOT, "models", "gguf")
HF_DIR = os.path.join(PROJECT_ROOT, "models", "hf")

#: Choices offered in the per-side dropdowns. ``None`` = not present on disk.
#: Real checkpoints are listed under the names the pipeline actually writes, so the
#: demo picks them up automatically once Phase 2/3/4 finish -- no code edit needed.
BACKEND_CHOICES = [
    ("Real SFT (full_sft_768.pth)", os.path.join(OUT_DIR, "full_sft_768.pth")),
    ("Real GRPO (grpo_stage3_full_768.pth)", os.path.join(OUT_DIR, "grpo_stage3_full_768.pth")),
    ("Real GRPO (grpo_768.pth)", os.path.join(OUT_DIR, "grpo_768.pth")),
    ("Pretrain (pretrain_768.pth)", os.path.join(OUT_DIR, "pretrain_768.pth")),
    # NOTE: pretrain_768.pth exists from the FIRST checkpoint onwards (every 2000 steps),
    # so during Phase 1 it holds a partially-trained model. It is deliberately NOT the
    # default for either side; the demo defaults to full_sft / grpo when they exist.
    ("Real SFT GGUF Q4_K_M", os.path.join(GGUF_DIR, "full_sft_768-Q4_K_M.gguf")),
    ("Real SFT GGUF f16", os.path.join(GGUF_DIR, "full_sft_768-f16.gguf")),
    ("Real GRPO GGUF Q4_K_M", os.path.join(GGUF_DIR, "grpo_stage3_full_768-Q4_K_M.gguf")),
    ("Real GRPO GGUF f16", os.path.join(GGUF_DIR, "grpo_stage3_full_768-f16.gguf")),
    ("Real SFT HF export (Qwen3 dir)", os.path.join(HF_DIR, "full_sft_768")),
    ("Real GRPO HF export (Qwen3 dir)", os.path.join(HF_DIR, "grpo_stage3_full_768")),
    ("SFT smoke (.pth, native MiniMind)", os.path.join(OUT_DIR, "sft_smoke_768.pth")),
    ("GRPO smoke (.pth, native MiniMind)", os.path.join(OUT_DIR, "grpo_smoke_768.pth")),
    ("Pretrain smoke (.pth, native MiniMind)", os.path.join(OUT_DIR, "pretrain_smoke_768.pth")),
    ("SFT HF export (Qwen3 dir)", os.path.join(HF_DIR, "sft_smoke_768")),
    ("GRPO HF export (Qwen3 dir)", os.path.join(HF_DIR, "grpo_smoke_768")),
    ("GGUF f16 (sft smoke)", os.path.join(GGUF_DIR, "sft_smoke_768-f16.gguf")),
    ("GGUF Q4_K_M (sft smoke)", os.path.join(GGUF_DIR, "sft_smoke_768-Q4_K_M.gguf")),
    ("GGUF Q8_0 (sft smoke)", os.path.join(GGUF_DIR, "sft_smoke_768-Q8_0.gguf")),
    ("GGUF f16 (grpo smoke)", os.path.join(GGUF_DIR, "grpo_smoke_768-f16.gguf")),
    ("GGUF Q4_K_M (grpo smoke)", os.path.join(GGUF_DIR, "grpo_smoke_768-Q4_K_M.gguf")),
    ("Mock (no model)", "mock"),
]


def available_choices() -> list:
    out = []
    for label, spec in BACKEND_CHOICES:
        if spec == "mock" or os.path.exists(spec):
            out.append({"label": label, "spec": spec})
    return out


def default_spec(prefer: str) -> str:
    """First existing spec matching ``prefer`` (sft/grpo) else mock."""
    for label, spec in BACKEND_CHOICES:
        if prefer in label.lower() and (spec == "mock" or os.path.exists(spec)):
            return spec
    return "mock"


class BackendPool:
    """Lazily load one backend per spec and reuse it across requests."""

    def __init__(self, prompt_mode: str, dtype: str, threads: int, seed: int = 42):
        self.prompt_mode = prompt_mode
        self.dtype = dtype
        self.threads = threads
        self.seed = seed
        self._cache = {}
        self._lock = threading.Lock()

    def get(self, spec: str):
        with self._lock:
            if spec not in self._cache:
                self._cache[spec] = B.make_backend(
                    spec, self.prompt_mode, dtype=self.dtype, threads=self.threads,
                    seed=self.seed)
            return self._cache[spec]

    def describe(self) -> dict:
        return {spec: be.info() for spec, be in self._cache.items()}


def run_pair(pool: BackendPool, prompt: str, sft_spec: str, grpo_spec: str,
             max_new_tokens: int, temperature: float) -> dict:
    payload = {
        "prompt": prompt,
        "prompt_formatted": B.format_prompt(prompt, pool.prompt_mode),
        "prompt_mode": pool.prompt_mode,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "device": "cpu",
        "gpu_free_mib": B.gpu_free_mib(),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    for side, spec in (("sft", sft_spec), ("grpo", grpo_spec)):
        t0 = time.time()
        try:
            backend = pool.get(spec)
            res = backend.generate(prompt, max_new_tokens, temperature)
            res.pop("prompt", None)
            res["total_s"] = round(time.time() - t0, 3)
            payload[side] = res
        except Exception as exc:  # noqa: BLE001 - surface it in the panel
            payload[side] = {"error": "%s: %s" % (type(exc).__name__, exc),
                             "traceback": traceback.format_exc(limit=4),
                             "backend": B.detect_kind(spec) if spec else "?",
                             "spec": spec}
    payload["wall_time_s"] = round(sum((payload[s].get("total_s") or 0) for s in ("sft", "grpo")), 3)
    return payload


# --------------------------------------------------------------------------
# Gradio UI
# --------------------------------------------------------------------------
def build_gradio(pool: BackendPool, host: str, port: int):
    import gradio as gr

    choices = available_choices()
    labels = [c["label"] for c in choices]
    specs = {c["label"]: c["spec"] for c in choices}
    sft_default = next((c["label"] for c in choices if "sft" in c["label"].lower() and "grpo" not in c["label"].lower()), labels[0])
    grpo_default = next((c["label"] for c in choices if "grpo" in c["label"].lower()), labels[-1])

    def _fmt(res: dict) -> str:
        if "error" in res:
            return "**ERROR** (%s)\n\n```\n%s\n```" % (res.get("backend", "?"), res["error"])
        head = ("**backend** `%s` | **device** %s | **tokens** %d | **tok/s** %.2f | **elapsed** %.3f s"
                % (res["backend"], res["device"], res["tokens"], res["tok_per_sec"], res["elapsed_s"]))
        extra = []
        for key, label in (("dtype", "dtype"), ("params", "params(M)"), ("gguf_size_mb", "gguf(MiB)"),
                           ("ngl", "ngl"), ("prompt_tokens", "prompt_tokens"), ("model_name", "model")):
            if res.get(key) is not None:
                extra.append("%s=%s" % (label, res[key]))
        if extra:
            head += "\n\n`" + "  ".join(extra) + "`"
        body = res["text"].strip() or "<empty completion>"
        return head + "\n\n---\n\n```\n" + body + "\n```"

    def go(prompt, sft_label, grpo_label, max_new_tokens, temperature):
        if not prompt or not prompt.strip():
            msg = "**Please enter a prompt.**"
            return msg, msg, ""
        payload = run_pair(pool, prompt, specs[sft_label], specs[grpo_label],
                           int(max_new_tokens), float(temperature))
        meta = ("wall %.3fs | device=cpu | prompt_mode=%s | GPU free %s MiB | %s"
                % (payload["wall_time_s"], payload["prompt_mode"],
                   payload["gpu_free_mib"], payload["timestamp"]))
        return _fmt(payload["sft"]), _fmt(payload["grpo"]), meta

    with gr.Blocks(title="MiniMind SFT vs GRPO demo") as demo:
        gr.Markdown("# MiniMind 64M — SFT vs GRPO (CPU only)\n"
                    "Same prompt, two checkpoints. **No CUDA**: the GPU is reserved for training; "
                    "llama.cpp runs with `-ngl 0`.")
        with gr.Row():
            prompt_box = gr.Textbox(label="Prompt", lines=3,
                                    value="3+5等于几？只回复数字",
                                    placeholder="输入提示词 / type a prompt")
        with gr.Row():
            sft_dd = gr.Dropdown(labels, value=sft_default, label="SFT backend")
            grpo_dd = gr.Dropdown(labels, value=grpo_default, label="GRPO backend")
            max_tok = gr.Slider(8, 256, value=48, step=8, label="max_new_tokens")
            temp = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="temperature (0 = greedy)")
        run_btn = gr.Button("Run both ▶", variant="primary")
        with gr.Row():
            sft_out = gr.Markdown(label="SFT output")
            grpo_out = gr.Markdown(label="GRPO output")
        meta_out = gr.Markdown()
        run_btn.click(go, [prompt_box, sft_dd, grpo_dd, max_tok, temp], [sft_out, grpo_out, meta_out])
        prompt_box.submit(go, [prompt_box, sft_dd, grpo_dd, max_tok, temp], [sft_out, grpo_out, meta_out])
    return demo


# --------------------------------------------------------------------------
# stdlib http.server UI
# --------------------------------------------------------------------------
PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>MiniMind 64M — SFT vs GRPO (CPU)</title>
<style>
 :root{color-scheme:dark light}
 body{font:15px/1.5 -apple-system,Segoe UI,Roboto,"Microsoft YaHei",sans-serif;margin:0;padding:24px;
      background:#0f1117;color:#e6e6e6}
 h1{font-size:20px;margin:0 0 4px} .sub{color:#9aa4b2;margin:0 0 18px;font-size:13px}
 .row{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px}
 textarea{width:100%;min-height:70px;background:#171a22;color:#e6e6e6;border:1px solid #2c3242;
          border-radius:8px;padding:10px;font:inherit;box-sizing:border-box}
 label{font-size:12px;color:#9aa4b2;display:block;margin-bottom:4px}
 select,input{background:#171a22;color:#e6e6e6;border:1px solid #2c3242;border-radius:8px;padding:7px}
 button{background:#3b82f6;color:#fff;border:0;border-radius:8px;padding:11px 20px;font-weight:600;cursor:pointer}
 button:disabled{opacity:.55;cursor:progress}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:16px}
 .card{background:#151922;border:1px solid #242a36;border-radius:10px;padding:12px;min-height:120px}
 .card h2{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:#7dd3fc;margin:0 0 8px}
 .meta{font:12px/1.6 ui-monospace,Consolas,monospace;color:#93c5fd;margin-bottom:8px;white-space:pre-wrap}
 pre{white-space:pre-wrap;word-break:break-word;font:13px/1.55 ui-monospace,Consolas,monospace;margin:0}
 .foot{margin-top:14px;font-size:12px;color:#6b7280}
 .err{color:#fca5a5}
</style></head><body>
<h1>MiniMind 64M — SFT vs GRPO</h1>
<p class="sub">Same prompt, two checkpoints, side by side. <b>CPU only</b> — the GPU is reserved for training
(llama.cpp runs with <code>-ngl 0</code>).</p>
<label for="prompt">Prompt</label>
<textarea id="prompt">3+5等于几？只回复数字</textarea>
<div class="row">
  <div><label for="sft">SFT backend</label><select id="sft"></select></div>
  <div><label for="grpo">GRPO backend</label><select id="grpo"></select></div>
  <div><label for="mt">max_new_tokens</label><input id="mt" type="number" value="48" min="8" max="256" step="8" style="width:90px"></div>
  <div><label for="tp">temperature (0 = greedy)</label><input id="tp" type="number" value="0.7" min="0" max="1.5" step="0.1" style="width:90px"></div>
</div>
<button id="run">Run both ▶</button>
<div class="grid">
  <div class="card"><h2>SFT</h2><div class="meta" id="sftmeta">—</div><pre id="sftout">—</pre></div>
  <div class="card"><h2>GRPO</h2><div class="meta" id="grpometa">—</div><pre id="grpoout">—</pre></div>
</div>
<div class="foot" id="foot">ready</div>
<script>
const CHOICES = __CHOICES__;
const DEFAULT_SFT = __DEFAULT_SFT__, DEFAULT_GRPO = __DEFAULT_GRPO__;
function fill(id, def){
  const sel = document.getElementById(id);
  CHOICES.forEach(c => { const o=document.createElement('option'); o.value=c.spec; o.textContent=c.label; sel.appendChild(o); });
  sel.value = def;
}
fill('sft', DEFAULT_SFT); fill('grpo', DEFAULT_GRPO);
function render(side, r){
  const meta = document.getElementById(side+'meta'), out = document.getElementById(side+'out');
  if (r.error){ meta.innerHTML = '<span class="err">ERROR '+r.backend+'</span>'; out.textContent = r.error; return; }
  const bits = [r.backend, r.device, 'tokens='+r.tokens, 'tok/s='+r.tok_per_sec.toFixed(2),
                'elapsed='+r.elapsed_s.toFixed(3)+'s'];
  if (r.dtype) bits.push('dtype='+r.dtype);
  if (r.params) bits.push(r.params+'M params');
  if (r.gguf_size_mb) bits.push('gguf='+r.gguf_size_mb+'MiB');
  if (r.ngl !== undefined) bits.push('ngl='+r.ngl);
  meta.textContent = bits.join(' | ');
  out.textContent = r.text.trim() || '<empty completion>';
}
document.getElementById('run').onclick = async () => {
  const btn = document.getElementById('run'); btn.disabled = true;
  document.getElementById('foot').textContent = 'running…';
  try{
    const body = { prompt: document.getElementById('prompt').value,
                   sft: document.getElementById('sft').value,
                   grpo: document.getElementById('grpo').value,
                   max_new_tokens: parseInt(document.getElementById('mt').value),
                   temperature: parseFloat(document.getElementById('tp').value) };
    const t0 = performance.now();
    const resp = await fetch('/api/generate', {method:'POST', headers:{'Content-Type':'application/json'},
                                               body: JSON.stringify(body)});
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || ('HTTP '+resp.status));
    render('sft', data.sft); render('grpo', data.grpo);
    document.getElementById('foot').textContent =
      'wall '+((performance.now()-t0)/1000).toFixed(2)+'s | device='+data.device+
      ' | prompt_mode='+data.prompt_mode+' | GPU free '+data.gpu_free_mib+' MiB';
  }catch(e){
    document.getElementById('foot').innerHTML = '<span class="err">'+e+'</span>';
  }finally{ btn.disabled = false; }
};
</script></body></html>
"""


def make_handler(pool: BackendPool, stats: dict):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MiniMindDemo/1.0"

        def log_message(self, fmt, *args):  # quieter console
            if stats.get("verbose"):
                sys.stderr.write("[http] %s\n" % (fmt % args))

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self):  # noqa: N802
            if self.path in ("/", "/index.html"):
                html = (PAGE.replace("__CHOICES__", json.dumps(available_choices(), ensure_ascii=False))
                            .replace("__DEFAULT_SFT__", json.dumps(default_spec("sft")))
                            .replace("__DEFAULT_GRPO__", json.dumps(default_spec("grpo"))))
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/healthz":
                self._json(200, {"status": "ok", "device": "cpu", "prompt_mode": pool.prompt_mode,
                                 "loaded": list(pool.describe().keys()),
                                 "gpu_free_mib": B.gpu_free_mib(), "uptime_s": round(time.time() - stats["t0"], 1)})
            elif self.path == "/api/backends":
                self._json(200, {"choices": available_choices()})
            else:
                self._json(404, {"error": "not found", "path": self.path})

        def do_POST(self):  # noqa: N802
            if self.path != "/api/generate":
                self._json(404, {"error": "not found", "path": self.path})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception as exc:  # noqa: BLE001
                self._json(400, {"error": "bad json: %s" % exc})
                return
            prompt = (req.get("prompt") or "").strip()
            if not prompt:
                self._json(400, {"error": "empty prompt"})
                return
            try:
                payload = run_pair(pool, prompt,
                                   req.get("sft") or default_spec("sft"),
                                   req.get("grpo") or default_spec("grpo"),
                                   int(req.get("max_new_tokens") or 48),
                                   float(req.get("temperature") if req.get("temperature") is not None else 0.7))
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": "%s: %s" % (type(exc).__name__, exc),
                                 "traceback": traceback.format_exc(limit=4)})
                return
            self._json(200, payload)

    return Handler


def serve_stdlib(pool: BackendPool, host: str, port: int, verbose: bool = False) -> None:
    stats = {"t0": time.time(), "verbose": verbose}
    httpd = ThreadingHTTPServer((host, port), make_handler(pool, stats))
    print("[ui] stdlib http.server on http://%s:%d  (CPU only, no CUDA)" % (host, port), flush=True)
    print("[ui] health: http://%s:%d/healthz   api: POST /api/generate" % (host, port), flush=True)
    print("[ui] Ctrl+C to stop. Do not leave this running.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        print("[ui] stopped", flush=True)


def main(argv=None):
    B._utf8_console()
    ap = argparse.ArgumentParser(description="MiniMind SFT-vs-GRPO demo web UI (CPU only)")
    ap.add_argument("--ui", choices=["gradio", "stdlib"], default="gradio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None,
                    help="default: 7860 for gradio, 7861 for stdlib")
    ap.add_argument("--prompt-mode", default=B.DEFAULT_PROMPT_MODE, choices=list(B.PROMPT_MODES))
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--threads", type=int, default=B.DEFAULT_THREADS)
    ap.add_argument("--seed", type=int, default=42, help="seed for reproducible sampling")
    ap.add_argument("--preload", action="store_true", help="load the default backends at startup")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    B.set_seed(args.seed)
    pool = BackendPool(args.prompt_mode, args.dtype, args.threads, seed=args.seed)
    if args.preload:
        for spec in (default_spec("sft"), default_spec("grpo")):
            t0 = time.time()
            pool.get(spec)
            print("[preload] %s (%.2fs)" % (spec, time.time() - t0), flush=True)

    if args.ui == "gradio":
        port = args.port or 7860
        demo = build_gradio(pool, args.host, port)
        print("[ui] gradio on http://%s:%d  (CPU only, no CUDA)" % (args.host, port), flush=True)
        try:
            demo.launch(server_name=args.host, server_port=port, share=False,
                        inbrowser=False, quiet=not args.verbose)
        except TypeError:  # older/newer gradio kwarg sets
            demo.launch(server_name=args.host, server_port=port, share=False, inbrowser=False)
    else:
        port = args.port or 7861
        serve_stdlib(pool, args.host, port, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
