# Phase-4 local demo app — SFT vs GRPO side by side (CPU only)

**Status: ✅ complete and smoke-tested on the 200-step smoke checkpoints.**
Host: Windows 11, RTX 4060 Laptop 8 GB, Python 3.11.9, uv.
Every number below was measured on this machine on 2026-10-21 while the ~26 h
pretraining job was running; **no CUDA memory was allocated by anything in this
directory** (verified: free VRAM stayed at ~2.0–2.1 GiB, `torch.cuda.is_available()`
is forced `False`, every llama.cpp call uses `-ngl 0`).

> ⚠️ **The smoke weights are garbage.** `sft_smoke_768.pth` / `grpo_smoke_768.pth`
> are 200-step checkpoints: they emit `"\n"`, `好的，`, `我需要…` and stray
> `<think>`/`<|im_end|>` tokens. That is **expected and is not a bug in the demo**.
> What is validated here is the *plumbing*: loading, prompt templating, token
> accounting, timing, GGUF/llama.cpp subprocess handling, the CLI, the web UI,
> the battery and the report generator. Replace the weights with
> `full_sft_768.pth` / `grpo_768.pth` (see §7) and the same commands produce the
> real comparison.

---

## 1. Files

| Path | What |
|---|---|
| `demo/backends.py` | uniform `generate(prompt, max_new_tokens, temperature) -> dict(text, tokens, tok_per_sec, backend, device, …)`; `HFBackend` (`.pth` native **or** HF dir), `GGUFBackend` (llama.cpp subprocess), `MockBackend`; single shared `format_prompt()` |
| `demo/compare_cli.py` | the minimal CLI: prompt → SFT panel → GRPO panel, `--json` for machines |
| `demo/app.py` | web UI: Gradio (`:7860`) **and** a stdlib `http.server` fallback (`:7861`) with `POST /api/generate` |
| `demo/run_compare_battery.py` | 12 fixed prompts × 2 checkpoints → `reports/demo_battery.json` |
| `demo/bench_backends.py` | measured CPU tok/s per backend → `reports/demo_bench.json` |
| `demo/selftest.py` | 32 assertions (templates, guards, CLI contract, in-process HTTP API, live generation, reproducibility) |
| `reports/demo_app.md` | this file |
| `reports/demo_battery.json` | canonical battery output — **currently smoke weights**; §7.3 overwrites it with the real checkpoints |
| `reports/demo_battery_smoke_gguf.json` | battery with GGUF Q4_K_M SFT vs GGUF Q4_K_M GRPO |
| `reports/demo_bench.json` | backend matrix numbers |

`demo/.venv/` is a **separate** 390 MB venv created only for Gradio (see §4.1); the
main `D:\llm-rl-lab\.venv` is untouched.

---

## 2. Backend matrix — measured CPU throughput

From `reports/demo_bench.json`: 2 prompts × 3 reps × 64 new tokens,
`--threads 8`, temperature 0.7, `CUDA_VISIBLE_DEVICES=-1`, while training ran.

| backend | model | mean tok/s | min–max | load | notes |
|---|---|---:|---:|---:|---|
| `hf` native `.pth` fp32 | `sft_smoke_768.pth` | **38.3** | 37.3–39.4 | 1.5 s | `MiniMindForCausalLM`, 63.9 M params, 255 MB RAM |
| `hf` native `.pth` fp16 | `sft_smoke_768.pth` | 33.9 | 32.7–34.9 | 0.7 s | **fp16 is *slower* on CPU** — don't use it |
| `hf` HF dir fp32 | `models/hf/sft_smoke_768` | **38.2** | 36.5–39.4 | 0.2 s | `Qwen3ForCausalLM`, numerically identical |
| `gguf` f16 | `sft_smoke_768-f16.gguf` (122 MiB) | **246.7** | 232.9–261.5 | 0.35 s/invocation | llama.cpp `-ngl 0 -t 8` |
| `gguf` Q8_0 | `sft_smoke_768-Q8_0.gguf` (65 MiB) | **393.4** | 345.8–427.7 | 0.3 s | |
| `gguf` Q4_K_M | `sft_smoke_768-Q4_K_M.gguf` (40.5 MiB) | **518.4** | 492.8–552.3 | 0.2 s | fastest; matches the 398 t/s llama-bench number (this run is warmer/less contended) |
| `mock` | — | 240.0 | — | 0 | canned text, for UI work |

Peak working set of the benchmark process: **870 MiB** (budget was 3 GB).
Free VRAM before/after: **2118 → 2117 MiB** (no GPU use; the small delta is the
training job's own fluctuation).

**How tok/s is measured**
* HF backends: `tokens / wall_time` of the `model.generate()` call, token count =
  length of the decoded new ids (special tokens included, matching `run_eval.py`).
* GGUF backend: llama.cpp's own `eval time` rate from stderr
  (`common_perf_print: eval time = … tokens per second`), which excludes prompt
  eval and sampling. `wall_tok_per_sec` (including process start + prompt eval)
  is also recorded in the JSON — it is much lower (e.g. 150–200 t/s at n=48)
  because each call is a fresh process.

**Why GGUF is 13× faster than torch on CPU:** llama.cpp uses hand-written
quantized GEMV kernels and 8 threads, while this 63.9 M torch model is
memory-bound at ~38 t/s on fp32 regardless of thread count (14 threads measured
in `logs/demo_probe_dtype.py` gives the same number). Q4_K_M is also *more*
faithful than f16 for this model per `reports/deploy_probe.md` §6 (PPL 373.28 vs
376.49), so **Q4_K_M is the recommended demo backend**.

---

## 3. CLI — invocation and real output

```powershell
$env:PYTHONIOENCODING='utf-8'; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8
cd D:\llm-rl-lab\demo

# native .pth (slow path, ~38 t/s, no conversion needed)
D:\llm-rl-lab\.venv\Scripts\python.exe compare_cli.py `
  --prompt "3+5等于几？只回复数字" `
  --sft  D:\llm-rl-lab\src\minimind-master\out\sft_smoke_768.pth `
  --grpo D:\llm-rl-lab\src\minimind-master\out\grpo_smoke_768.pth `
  --max-new-tokens 24 --temp 0.7

# GGUF (fast path, ~500 t/s)
D:\llm-rl-lab\.venv\Scripts\python.exe compare_cli.py `
  --prompt "用一句话解释什么是人工智能。" `
  --sft  D:\llm-rl-lab\models\gguf\sft_smoke_768-Q4_K_M.gguf `
  --grpo D:\llm-rl-lab\models\gguf\grpo_smoke_768-Q4_K_M.gguf `
  --max-new-tokens 48 --temp 0.7

# machine-readable (used by the battery / report tooling)
... compare_cli.py --prompt "..." --sft <a> --grpo <b> --json --out logs/run.json
```

`--sft` / `--grpo` accept **any** of: a native `.pth`, an HF export directory,
a `.gguf`, or the literal `mock`. `--temp 0` = greedy. Exit code 0 on success,
1 if a side fails (the error is shown inside that side's panel).

To save machine-readable output use **`--out <path>`** (written as UTF-8), not a
PowerShell redirect: `compare_cli.py --json > file.json` produces **UTF-16** under
PowerShell 5/7 default redirection and `json.load(open(...))` then fails with
`UnicodeDecodeError: 0xff in position 0`.

Measured text output, native `.pth` (full box, `logs/demo_cli_final_text.log`):

```
==============================================================================
MiniMind Phase-4 demo  |  SFT vs GRPO  |  device=cpu  |  mode=chat  |  temp=0.70  |  max_new=24
==============================================================================
PROMPT
  3+5等于几？只回复数字

+====== SFT  (D:\llm-rl-lab\src\minimind-master\out\sft_smoke_768.pth) ======+
| backend : hf                   device : cpu  mode : chat
| tokens  : 24     tok/s : 39.48    elapsed : 0.608 s
| dtype=float32  63.9M params  prompt_tokens=27
------------------------------------------------------------------------------
| 
| ...
| ```
| 好的，
+============================================================================+
+===== GRPO (D:\llm-rl-lab\src\minimind-master\out\grpo_smoke_768.pth) ======+
| backend : hf                   device : cpu  mode : chat
| tokens  : 24     tok/s : 39.53    elapsed : 0.607 s
| dtype=float32  63.9M params  prompt_tokens=27
------------------------------------------------------------------------------
| 
| ...
| ```
| 好的，
+============================================================================+
device=cpu (GPU untouched) | wall 9.03s | gpu free 2104 MiB
```

Measured text output, GGUF Q4_K_M (full box, `logs/demo_cli_gguf_text.log`) —
this is the shape the Lead will see with the real weights:

```
+======== SFT  (D:\llm-rl-lab\models\gguf\sft_smoke_768-Q4_K_M.gguf) ========+
| backend : gguf                 device : cpu  mode : chat
| tokens  : 48     tok/s : 523.95   elapsed : 0.333 s
| gguf=40.5 MiB  ngl=0  prompt_tokens=25
------------------------------------------------------------------------------
| <think>
| 好的，
| 好的，我需要想问是人类的开露，我需要回答有没有不透露。
| 最后，用户知道需要用户可能能要可能想可能需要不欢迎情。
+============================================================================+
+======= GRPO (D:\llm-rl-lab\models\gguf\grpo_smoke_768-Q4_K_M.gguf) ========+
| backend : gguf                 device : cpu  mode : chat
| tokens  : 48     tok/s : 484.26   elapsed : 0.237 s
| gguf=40.5 MiB  ngl=0  prompt_tokens=25
------------------------------------------------------------------------------
| <think>
| 好的，
| 好的，我需要想问到需要提供具体人情。
| 好的，用户问你想到高效的复杂，
| </think>
| 我需要
+============================================================================+
device=cpu (GPU untouched) | wall 0.62s | gpu free 2109 MiB
```

(The `wall 9.03s` for the `.pth` run is dominated by the ~7–9 s *first* torch
import + model build in a cold process; generation itself is 0.6 s. The GGUF run
pays ~0.2–0.35 s of llama.cpp model load per invocation instead.)

---

## 4. Web UI

Two front-ends, identical backend layer. Both bind `127.0.0.1` and are meant to
be run as a **short-lived** job only.

### 4.1 Gradio (preferred) — `http://127.0.0.1:7860`

Gradio is **not** installed in the main venv and installing it there would have
downgraded `pandas 3.0.6 → 2.3.3`, `pillow 12.3.0 → 11.3.0`, `rich 13.7.1 → 15.0.0`
and `huggingface-hub 0.36.2 → 2.2.0` (the last one **breaks transformers 4.57.6**:
`ImportError: huggingface-hub>=0.34.0,<1.0 is required`). So Gradio lives in an
isolated venv that *reuses* the main site-packages through a `.pth` file:

```powershell
$env:UV_CACHE_DIR='D:\llm-rl-lab\.cache\uv'
# one-time (already done): isolated venv + a .pth bridge so it reuses torch/transformers
D:\llm-rl-lab\.venv\Scripts\python.exe -m venv D:\llm-rl-lab\demo\.venv
$sp = 'D:\llm-rl-lab\demo\.venv\Lib\site-packages'
[System.IO.File]::WriteAllText("$sp\llm_rl_lab_site.pth",
    "D:\llm-rl-lab\.venv\Lib\site-packages`r`n", (New-Object System.Text.UTF8Encoding($false)))
C:\Users\24657\.local\bin\uv.exe pip install --python D:\llm-rl-lab\demo\.venv\Scripts\python.exe `
    "gradio<6" "huggingface-hub>=0.34,<1.0" `
    --index-url https://mirrors.aliyun.com/pypi/simple/ --extra-index-url https://pypi.org/simple

# run (short-lived!)
$env:GRADIO_ANALYTICS_ENABLED='False'; $env:HF_HUB_OFFLINE='1'
D:\llm-rl-lab\demo\.venv\Scripts\python.exe D:\llm-rl-lab\demo\app.py --ui gradio --port 7860
```

Verified in that venv: `gradio 5.50.0`, `torch 2.6.0+cu124`, `transformers 4.57.6`,
`numpy 2.4.6` (venv-local), `pandas 2.3.3` (venv-local) — torch/transformers
import and generate fine alongside Gradio, and the main venv still reports
`torch 2.6.0+cu124 / transformers 4.57.6 / numpy 2.2.6` afterwards.

UI: prompt textbox, **Run both ▶** button, SFT and GRPO panels, a backend dropdown
per side (all present models: `.pth` smoke/real, HF dirs, GGUF f16/Q8_0/Q4_K_M for
both SFT and GRPO, and Mock), plus `max_new_tokens` and `temperature` sliders.
Each panel shows backend, device, tokens, tok/s, elapsed and dtype/params/gguf/ngl.

Smoke test (server started as a background job, hit, then killed) — mixed
backends, i.e. the HF `.pth` on the left and GGUF Q4_K_M on the right:

```
PAGE 200
EVENT 200
SFT : **backend** `hf`   | **device** cpu | **tokens** 24 | **tok/s** 35.59 | **elapsed** 0.674 s
      `dtype=float32  params(M)=63.912  prompt_tokens=31  model=MiniMindForCausalLM(native)`
GRPO: **backend** `gguf` | **device** cpu | **tokens** 24 | **tok/s** 400.02 | **elapsed** 0.241 s
      `gguf(MiB)=40.5  ngl=0  prompt_tokens=31`
wall 8.138s | device=cpu | prompt_mode=chat | GPU free 2050 MiB
port 7860 open after kill: False
```

(A pure-GGUF pair measured earlier in the same way: SFT 473.33 t/s / GRPO
450.48 t/s, wall 0.457 s. Exercised through `/gradio_api/call/go` + the event
stream, i.e. exactly the function the button calls.)

### 4.2 stdlib `http.server` fallback — `http://127.0.0.1:7861`

Zero dependencies, works in the main venv:

```powershell
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\demo\app.py --ui stdlib --port 7861
# health:  curl http://127.0.0.1:7861/healthz
# api:     POST http://127.0.0.1:7861/api/generate  {"prompt","sft","grpo","max_new_tokens","temperature"}
```

Smoke test (started, curled, killed):

```
HEALTH 200: {"status":"ok","device":"cpu","prompt_mode":"chat","loaded":[],"gpu_free_mib":2104,...}
PAGE   200 len=5823 hasRunBoth=True
mixed  API 200: sft=hf/16tok/37.88tps   grpo=gguf/ngl0/16tok/435.63tps
earlier runs: sft hf 35.58 t/s | grpo hf 36.47 t/s;  gguf Q4 vs gguf f16 432.91 / 204.72 t/s
error paths: missing .pth -> per-side error in JSON (HTTP 200, other side still runs);
             empty prompt -> HTTP 400; unknown path -> HTTP 404
port 7861 open after kill: False
```

**Both servers were killed after the smoke test; nothing is left listening.**

---

## 5. Prompt formatting — one shared function, and why llama.cpp's chat template is bypassed

`backends.format_prompt(prompt, mode)` is the **only** place a prompt string is
built, and every backend (HF `.pth`, HF dir, GGUF) receives exactly those bytes.
The GGUF backend writes the string to a UTF-8 (no BOM) temp file and passes it to
llama.cpp with `-f`, so llama.cpp's own template machinery is never involved.
This guarantees a byte-identical prompt across backends — the precondition for a
fair SFT-vs-GRPO comparison.

| mode | prompt tail | when |
|---|---|---|
| `chat` (**default**) | `…<|im_start|>assistant\n<think>\n\n</think>\n\n` | canonical: what `eval/run_eval.py --prompt-mode chat` uses and what the SFT baseline is evaluated with |
| `chat_think` | `…<|im_start|>assistant\n<think>\n` | GRPO/thinking checkpoints (reward expects a `</think>`) |
| `raw` | `…<|im_start|>assistant\n` | no think block at all |

Verified (assertion in `selftest.py`): `raw` is **byte-identical** to what the
previous teammate's think-free `llamacpp/chat_template_minimind.jinja` renders for
a single user turn (`<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n`).
So if the Lead prefers the think-free deployment style, `--prompt-mode raw` is the
exact equivalent — and because the template file is no longer needed for the CLI
path, there is no risk of llama.cpp silently injecting an empty `<think>` block.

⚠️ This is a **real decision the Lead must make once the real weights exist**: SFT
and GRPO may want *different* native modes (`chat` vs `chat_think`, see
`reports/eval_harness.md` §"Caveats" #1). The demo supports both, but the report
of a comparison must record which mode was used — the CLI, the UI and the battery
all stamp `prompt_mode` into every result and JSON report.

---

## 6. Smoke battery — 12 prompts × 2 checkpoints

`demo/run_compare_battery.py`: 9 prompts taken **verbatim** from the frozen
`eval/heldout_tasks.jsonl` (2 arithmetic, 2 format, 2 logic, 2 multistep, 1
regex-format — all four families) + 3 chit-chat prompts. The held-out answers are
only used to grade with `eval/grader.py`; nothing is hard-coded, so the script
works unchanged with the real checkpoints.

```powershell
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\demo\run_compare_battery.py `
  --sft  D:\llm-rl-lab\src\minimind-master\out\sft_smoke_768.pth `
  --grpo D:\llm-rl-lab\src\minimind-master\out\grpo_smoke_768.pth `
  --out  D:\llm-rl-lab\reports\demo_battery.json
```

| run | SFT | GRPO | report |
|---|---|---|---|
| native `.pth` (canonical) | 38.25 t/s, 576 tok, 15.08 s, **0/9** graded | 37.97 t/s, 576 tok, 15.19 s, **0/9** graded | `reports/demo_battery.json` |
| GGUF Q4_K_M (fast) | **493.6 t/s**, 3.20 s total | **494.3 t/s**, 3.12 s total | `reports/demo_battery_smoke_gguf.json` |

Both runs: **12/12 prompts answered, 0 errors, 0 empty completions** — every code
path (prompt build → tokenize → generate → decode → token count → timing → grading
→ JSON) executed end-to-end. Wall time for the whole 24-generation battery: 38.1 s
(native `.pth`) / 6.4 s (GGUF).

**Reproducibility.** `seed=42` is now threaded into every backend
(`backends.set_seed()` + `torch.manual_seed(seed)` before each HF `generate()`), so
two independent battery processes produce **byte-identical** completions (verified
by diffing two runs: `REPRODUCIBLE: True`). llama.cpp is seeded with `-s 42`
per invocation. Before this fix the sampled output changed every run, which would
have made the numbers in a report unquotable.

**Real example completions (plumbing proof, not model quality).** With the seeded
run, 200-step checkpoints collapse to the same few tokens on every arithmetic
prompt — that *is* the expected behaviour of an undertrained model:

| prompt | SFT (`.pth`) | GRPO (`.pth`) |
|---|---|---|
| `3 + 5 等于多少？…` (arith_01) | `\n\n…好的，` → `no_integer_in_completion` | `\n\n…好的，` → `no_integer_in_completion` |
| `0 - 9 …` (arith_05) | `\n…` → `empty_completion` | `\n…0/\n… = without` → `mismatch:got=0 expected=-9` |
| `2+6=…` (format_06) | `\n…0，\n 3\n```\n   =def-` → `regex_fullmatch_failed` | same → `regex_fullmatch_failed` |
| `你好，请用一句话介绍一下你自己。` (chat_01) | `好的，我需要或者成为问与微提自人。我需要许多情感。…` | `好的，我需要或者成为问与微提自人。我需要许多情感。…` |
| `用一句话解释什么是人工智能。` (GGUF Q4) | `好的，我需要想问是人类的开露，我需要回答有没有不透露。最后，用户知道…` | `好的，我需要想问到需要提供具体人情。好的，用户问你想到高效的复杂，</think>我需要` |

With the seeded run **both** sides score 0/9 — the earlier 1/9 was sampling luck
(a stray `c` matched `logic_02`'s `choice` checker). Graded accuracy must be
ignored until the real weights land; what matters here is that all 9 held-out
prompts produced gradeable completions with a concrete `reason` from
`eval/grader.py`, i.e. the grading path is wired correctly.

Both other prompt modes were exercised end-to-end too (mixed HF+GGUF, `--json`):
`--prompt-mode chat_think` (prompt ends `…assistant\n<think>\n`) and
`--prompt-mode raw` (prompt ends `…assistant\n`, byte-identical to the think-free
chat template) both return 16 tokens from each side, exit 0.

---

## 7. Commands to run once the real weights exist

Everything below was executed end-to-end on the smoke weights (that is how the
demo artifacts were produced), so only the file names change.

### 7.1 Step 0 — verify the training job is still fine, and that VRAM is untouched

```powershell
nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader   # expect ~2 GiB free
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*train_pretrain*' } | Select-Object ProcessId
```

### 7.2 Step 1 — CLI, both checkpoints native (no conversion, ~38 t/s)

```powershell
$env:PYTHONIOENCODING='utf-8'; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8
cd D:\llm-rl-lab\demo
D:\llm-rl-lab\.venv\Scripts\python.exe compare_cli.py `
  --prompt "3+5等于几？只回复数字" `
  --sft  D:\llm-rl-lab\src\minimind-master\out\full_sft_768.pth `
  --grpo D:\llm-rl-lab\src\minimind-master\out\grpo_768.pth `
  --max-new-tokens 64 --temp 0.7
```

If the GRPO checkpoint is a *thinking* model, use `--prompt-mode chat_think` for
both sides (and record it). Compare with `--prompt-mode chat` as a sensitivity
check — never compare two runs with different `prompt_mode`.

### 7.3 Step 2 — battery + report

```powershell
D:\llm-rl-lab\.venv\Scripts\python.exe run_compare_battery.py `
  --sft  D:\llm-rl-lab\src\minimind-master\out\full_sft_768.pth `
  --grpo D:\llm-rl-lab\src\minimind-master\out\grpo_768.pth `
  --max-new-tokens 64 --temp 0.7 `
  --out  D:\llm-rl-lab\reports\demo_battery.json
D:\llm-rl-lab\.venv\Scripts\python.exe bench_backends.py --reps 3 --max-new-tokens 64
```

### 7.4 Step 3 — GGUF conversion of the real weights (proven path, `reports/deploy_probe.md` §8)

⚠️ **PowerShell trap in the obvious loop form:** `@(@('a','b'), @('c','d'))` gets
flattened to `'a','b','c','d'` by the `foreach` assignment, so `$pair[0]` is the
first *character* (`s`, not `full_sft_768`). The loop below therefore iterates over
plain weight names. This exact block was executed end-to-end during this task
(with the smoke checkpoint into a scratch dir) — all four steps `rc=0`, and the
resulting GGUF ran through `compare_cli.py` successfully.

```powershell
$env:UV_CACHE_DIR='D:\llm-rl-lab\.cache\uv'; $env:PYTHONIOENCODING='utf-8'
$OUT = 'D:\llm-rl-lab\src\minimind-master\out'
$HF  = 'D:\llm-rl-lab\models\hf'
$GG  = 'D:\llm-rl-lab\models\gguf'

foreach ($w in @('full_sft_768', 'grpo_768')) {
  # 1. .pth -> HF (Qwen3 rebuild, parity-asserted)
  & D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\llamacpp\scripts\convert_pth_to_hf.py `
      "$OUT\$w.pth" "$HF\${w}_real"
  # 1b. Qwen3Config leaves bos/eos/pad = None; llama.cpp needs them as stop tokens
  & D:\llm-rl-lab\.venv\Scripts\python.exe -c "import json;p=r'$HF\${w}_real\config.json';d=json.load(open(p,encoding='utf-8'));d.update(bos_token_id=1,eos_token_id=2,pad_token_id=0);json.dump(d,open(p,'w',encoding='utf-8'),indent=2,ensure_ascii=False)"
  # 2. HF -> GGUF f16   (MUST use the PATCHED converter copy in D:\llm-rl-lab\llamacpp\)
  Set-Location D:\llm-rl-lab\llamacpp
  & D:\llm-rl-lab\.venv\Scripts\python.exe .\convert_hf_to_gguf.py "$HF\${w}_real" `
      --outfile "$GG\${w}_real-f16.gguf" --outtype f16
  # 3. quantize
  Set-Location D:\llm-rl-lab\llamacpp\cpu
  .\llama-quantize.exe "$GG\${w}_real-f16.gguf" "$GG\${w}_real-Q4_K_M.gguf" Q4_K_M
}
```

⚠️ **Use `D:\llm-rl-lab\llamacpp\convert_hf_to_gguf.py`, never a fresh upstream
copy** — that file carries the 8-line `chkhsh a39372c9… → "gpt-2"` pre-tokenizer
patch (`deploy_probe.md` §4.2). Without it the conversion dies with
`NotImplementedError: BPE pre-tokenizer was not recognized`. Expect the benign
`WARNING: 8 of 57 tensor(s) required fallback quantization` (ffn_down rows are
2432, not a multiple of 256) — Q8_0 fallback, slightly *better* than Q6_K.

This exact block was executed for `grpo_smoke_768.pth` during this task
(`logs/demo_hf_convert_grpo.log`, `logs/demo_gguf_convert_grpo_f16.log`,
`logs/demo_gguf_quantize_grpo_q4km.log`), producing
`models/hf/grpo_smoke_768/`, `models/gguf/grpo_smoke_768-f16.gguf` (122.2 MiB)
and `models/gguf/grpo_smoke_768-Q4_K_M.gguf` (40.5 MiB, same 8-fallback warning),
so the path is validated on a *GRPO* checkpoint too, not just SFT.

### 7.5 Step 4 — CLI/UI on the real GGUF pair

```powershell
cd D:\llm-rl-lab\demo
D:\llm-rl-lab\.venv\Scripts\python.exe compare_cli.py `
  --prompt "3+5等于几？只回复数字" `
  --sft  D:\llm-rl-lab\models\gguf\full_sft_768_real-Q4_K_M.gguf `
  --grpo D:\llm-rl-lab\models\gguf\grpo_768_real-Q4_K_M.gguf `
  --max-new-tokens 64 --temp 0.7

D:\llm-rl-lab\.venv\Scripts\python.exe run_compare_battery.py `
  --sft  D:\llm-rl-lab\models\gguf\full_sft_768_real-Q4_K_M.gguf `
  --grpo D:\llm-rl-lab\models\gguf\grpo_768_real-Q4_K_M.gguf `
  --out  D:\llm-rl-lab\reports\demo_battery_gguf_real.json
```

### 7.6 Step 5 — web UI (short-lived only)

```powershell
# Gradio (demo venv, has torch+transformers via the .pth bridge)
$env:GRADIO_ANALYTICS_ENABLED='False'; $env:HF_HUB_OFFLINE='1'
D:\llm-rl-lab\demo\.venv\Scripts\python.exe D:\llm-rl-lab\demo\app.py --ui gradio --port 7860
#   -> http://127.0.0.1:7860

# or stdlib fallback (main venv)
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\demo\app.py --ui stdlib --port 7861
#   -> http://127.0.0.1:7861
```

The UI dropdowns already list `full_sft_768.pth` / `grpo_768.pth` by their final
names (`OUT_DIR/full_sft_768.pth`, `OUT_DIR/grpo_768.pth`) — they appear
automatically once the files exist. For the real-GGUF names above, add them to
`BACKEND_CHOICES` in `app.py` or point the API at the spec directly (the
`/api/generate` endpoint takes raw paths).

### 7.7 Step 6 — selftest

```powershell
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\demo\selftest.py -v   # 32/32 ok
```

---

## 8. Traps found and handled (worth knowing before Phase 4 is signed off)

1. **`llama-cli.exe` no longer supports `-no-cnv` in b7795.** It prints
   `--no-conversation is not supported by llama-cli / please use llama-completion
   instead` and **exits 1**. The non-interactive binary is
   `llamacpp/cpu/llama-completion.exe` (same build, same flags otherwise). The
   GGUF backend prefers `llama-completion.exe` and only falls back to `llama-cli`.
2. **llama.cpp's `-f` file and trailing newlines.** Prompt bytes are preserved
   exactly (verified: llama.cpp tokenized the prompt to ids `[1,832,311,…,234]`,
   byte-identical to the HF tokenizer's 31 ids), so no template/newline drift.
3. **llama.cpp token-count off-by-one.** `common_perf_print` reports
   `prompt eval time = … / N tokens` and `eval time = … / M runs` where N and M are
   each one less than the true counts (the first prompt token is the initial input,
   the last sampled token is never decoded). The backend adds 1 to both; a
   `n=48` request reports 48 generated tokens and prompt_tokens=27, matching HF.
4. **Windows console.** `print()` of Chinese dies on the GBK codec — every entry
   point calls `sys.stdout.reconfigure(encoding="utf-8")` and the report commands
   set `$env:PYTHONIOENCODING='utf-8'`. Non-ASCII `-p` arguments to llama.cpp get
   mangled by the PowerShell/ANSI path, which is why the prompt always goes
   through a UTF-8 temp file.
5. **`CUDA_VISIBLE_DEVICES=""` does NOT hide the GPU from torch** (measured:
   `torch.cuda.is_available()` still `True`). `-1` does. `backends.py` sets `-1` at
   import time and `selftest.py` asserts it, so a stray `.to("cuda")` in future code
   cannot allocate.
6. **Installing Gradio into the main venv breaks transformers.** `uv pip install
   gradio` resolves `huggingface-hub==2.2.0`, and transformers 4.57.6 then refuses
   to import. Hence the isolated `demo/.venv` with a `.pth` bridge to the main
   site-packages and an explicit `huggingface-hub>=0.34,<1.0` pin.
7. **`torch.cuda.mem_get_info`/`torch.cuda.*` is never called** by any demo code
   path; free VRAM is read with `nvidia-smi --query-gpu=memory.free`, which
   allocates nothing.
8. **PowerShell redirection writes UTF-16.** `compare_cli.py --json > out.json`
   produces a UTF-16 file (BOM `0xff 0xfe`) and `json.load(open(..., encoding='utf-8'))`
   then fails with `UnicodeDecodeError`. Use `--out <path>` instead — it writes UTF-8.
9. **PowerShell flattens nested arrays in `foreach`.** `foreach ($pair in
   @(@('a','b'), @('c','d')))` iterates over the four *strings*, so `$pair[0]` is
   `'a'`, and `$pair[0]` of `'full_sft_768'` is the single character `f`. The GGUF
   conversion block in §7.4 therefore loops over plain weight names.
10. **llama.cpp `-t 8` cap is honoured, and `--threads 0` means "don't touch
    torch's default"** (torch already defaults to 14 here); the demo only lowers it
    when `--threads` is explicitly positive, keeping total CPU use under the agreed
    budget.

---

## 9. What is smoke-tested (all measured, all reproducible)

| # | item | result |
|---|---|---|
| 1 | CLI, `.pth` SFT vs `.pth` GRPO | ✅ exit 0, both panels, 39.5/39.5 t/s (`logs/demo_cli_final_text.log`; 38.2/37.6 t/s in `logs/demo_cli_pth_smoke.log`) |
| 2 | CLI, GGUF Q4_K_M SFT vs GGUF Q4_K_M GRPO | ✅ exit 0, 524.0/484.3 t/s (`logs/demo_cli_gguf_text.log`; 500.3/506.1 in `logs/demo_cli_gguf_q4_smoke.log`) |
| 3 | CLI, GGUF Q4 vs GGUF f16 | ✅ exit 0 (`logs/demo_cli_gguf_smoke.log`, 523.4 vs 244.7 t/s) |
| 4 | CLI, mixed HF dir SFT vs GGUF Q4 GRPO | ✅ exit 0 (37.7 / 541.7 t/s, `logs/demo_cli_mixed_smoke.log`) |
| 5 | CLI `--json` + `--out` | ✅ schema complete, `device=cpu`, `gpu_free_mib` recorded; `--out` writes UTF-8 |
| 6 | CLI error path (missing weight) | ✅ exit 1, error inside the panel, other side still runs |
| 7 | battery, 12 prompts × 2 `.pth` | ✅ 12/12, 0 errors → `reports/demo_battery.json` |
| 8 | battery, 12 prompts × 2 GGUF Q4 | ✅ 12/12, 0 errors → `reports/demo_battery_smoke_gguf.json` |
| 9 | Gradio UI `:7860` | ✅ page 200, `go()` event 200, killed, port closed |
| 10 | stdlib UI `:7861` | ✅ page/health/API 200, mixed backends, error paths, killed, port closed |
| 11 | backend matrix bench | ✅ 7 configurations → `reports/demo_bench.json` |
| 12 | selftest | ✅ **32/32 checks passed** (`logs/demo_selftest.log`) |
| 13 | GRPO GGUF pipeline (pth→HF→GGUF f16→Q4_K_M) | ✅ parity 3.58e-06, 122.2 → 40.5 MiB |
| 14 | CUDA untouched | ✅ free VRAM 2015→2104 MiB across the session (training job's own fluctuation; bench measured 2118→2117), `torch.cuda.is_available()==False` |
| 15 | RAM | ✅ peak working set 870 MiB (limit 3 GB) |

---

## 9b. Log / artifact index

| Path | What |
|---|---|
| `logs/demo_selftest.log` | 32/32 selftest checks |
| `logs/demo_bench.log` + `reports/demo_bench.json` | backend matrix |
| `logs/demo_cli_pth_smoke.{log,json}` | CLI, .pth SFT vs .pth GRPO |
| `logs/demo_cli_gguf_text.log` | CLI, GGUF Q4 SFT vs GGUF Q4 GRPO (report quotes this) |
| `logs/demo_cli_gguf_smoke.{log,json}` | CLI, GGUF Q4 vs GGUF f16 |
| `logs/demo_cli_gguf_q4_smoke.{log,json}` | CLI, GGUF Q4 vs GGUF Q4 (grpo converted here) |
| `logs/demo_cli_mixed_smoke.{log,json}` | CLI, HF-dir SFT vs GGUF-Q4 GRPO |
| `logs/demo_cli_final_text.log` | CLI box quoted in §3 |
| `logs/demo_cli_max64.json` | 64-token CLI run (token accounting check) |
| `logs/demo_battery_smoke.log` | battery run log (native .pth) |
| `logs/demo_battery_smoke_gguf.log` | battery run log (GGUF Q4) |
| `logs/demo_hf_convert_grpo.log` | pth→HF for grpo_smoke (parity 3.58e-06) |
| `logs/demo_gguf_convert_grpo_f16.log` | HF→GGUF f16 for grpo_smoke |
| `logs/demo_gguf_quantize_grpo_q4km.log` | Q4_K_M for grpo_smoke (8-fallback warning) |
| `logs/llama_cli_help_b7795.txt` | full `llama-cli --help` dump used to pick flags |
| `models/hf/grpo_smoke_768/` | new GRPO HF export (Qwen3ForCausalLM, 122 MiB) |
| `models/gguf/grpo_smoke_768-f16.gguf` | 122.2 MiB |
| `models/gguf/grpo_smoke_768-Q4_K_M.gguf` | 40.5 MiB |

---
## 10. Anything that could block Phase 4 with the real weights

1. **Prompt mode must be chosen per checkpoint.** SFT (non-thinking) wants `chat`;
   a thinking GRPO checkpoint wants `chat_think` (and then `--max-new-tokens`
   should be ≥128–192 or the answer may be truncated before `</think>`). The demo
   defaults to `chat` and records the mode everywhere, but the Lead must decide and
   state which mode the headline comparison used. Not a code blocker.
2. **Native `.pth` is slow (~38 t/s) and pays a ~7–9 s cold torch import.** For a
   live demo prefer GGUF Q4_K_M (~500 t/s, 0.2–0.35 s per call). Both are wired;
   just pick per audience.
3. **GGUF is generated text-inexact vs HF** for a near-flat distribution (already
   documented in `deploy_probe.md` §6: greedy diverges after ~8 tokens, PPL
   indistinguishable). For a *qualitative* demo this is fine; for graded numbers
   use the native `.pth` path (or accept Q4 as "deployment view").
4. **A real GRPO checkpoint may be a different architecture flag** (e.g. MoE or a
   different `use_moe`) — `HFBackend` builds `MiniMindConfig(hidden_size=768,
   num_hidden_layers=8)` with `use_moe=False` defaults, matching every checkpoint
   in `out/` today. If a future checkpoint is MoE, `--hidden/--layers` are
   configurable but `use_moe` is not exposed; the HF-directory backend has no such
   assumption. Not currently blocking (no MoE checkpoint exists).
5. **Disk:** `demo/.venv` is 390 MB. Delete it if space matters; the stdlib UI and
   the CLI need nothing from it.
6. **The UI dropdowns use final file names.** `full_sft_768.pth` / `grpo_768.pth`
   appear automatically once they exist; the real-GGUF names from §7.4
   (`full_sft_768_real-Q4_K_M.gguf` etc.) need to be added to `BACKEND_CHOICES` in
   `demo/app.py` — or just pass the path to `/api/generate`, which accepts raw
   specs.
7. **No server is running now.** Both smoke tests started a background job, curled
   it, and killed it; `Test-NetConnection` on 7860/7861 returns `False` (the ports
   must be free before launching the UI again).

