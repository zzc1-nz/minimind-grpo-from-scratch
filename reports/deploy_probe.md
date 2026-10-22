# Phase 4 Deployment Probe — MiniMind .pth → HF → GGUF Q4_K_M → llama.cpp

**Status: ✅ UNBLOCKED — full deployment path works end-to-end on this machine.**
Probe date: 2026-10-21. Host: Windows 11, RTX 4060 Laptop 8 GB, Python 3.11.9, uv.
Test weight: `D:\llm-rl-lab\src\minimind-master\out\sft_smoke_768.pth` (63.91 M params, fp16 state_dict).

---

## 1. Headline numbers

| Item | Result |
|---|---|
| HF conversion | ✅ `Qwen3ForCausalLM`, logit-parity vs native MiniMind **4.29e-06** |
| HF safetensors size | **121.9 MiB** (`model.safetensors`, fp16) |
| llama.cpp binary | ✅ **b7795** (build `bd544c94a`), **CUDA 12.4 x64** + CPU x64 fallback |
| GGUF f16 | ✅ **122.2 MiB** (`sft_smoke_768-f16.gguf`, 128,093,952 B) |
| GGUF Q4_K_M | ✅ **40.5 MiB** (`sft_smoke_768-Q4_K_M.gguf`, 42,481,920 B) — **3.0× compression** |
| GGUF Q8_0 (extra) | ✅ 65.0 MiB (68,189,952 B), for quality comparison |
| Tokenizer parity | ✅ **12/12** texts byte-identical vs HF tokenizer |
| Quantization quality | Q4_K_M PPL **373.28** vs f16 **376.49** (Δ **−0.9 %**) |
| GPU speed | **939 t/s** generation, 14,562 t/s prompt (llama-bench, `-ngl 99`) |
| CPU speed | **398 t/s** generation, 1,966 t/s prompt (14 threads, `-ngl 0`) |
| Peak VRAM | **3,092 MiB** delta (3,879 MiB absolute) at `-ngl 99`, ctx 512 |
| Blockers remaining | **None** that stop Phase 4 |

---

## 2. HF conversion

### 2.1 The key insight (avoids the "custom architecture" trap entirely)

MiniMind's `model_minimind.py` decoder is **structurally identical to Qwen3**: RMSNorm,
RoPE θ=1e6, GQA 8/4 heads, head_dim 96, SwiGLU, tied embeddings, **and QK-norm**
(`q_norm`/`k_norm` over `head_dim`) — that last detail is what makes it Qwen3 rather
than Qwen2/Llama. So instead of teaching llama.cpp a new architecture, we rebuild a
stock `Qwen3ForCausalLM` and load the state dict with `strict=True`.

Verified numerically — max absolute logit difference between native MiniMind and the
Qwen3 rebuild on the same token ids:

```
[load] missing=[] unexpected=[]
[parity] max |MiniMind - Qwen3| logit diff = 4.291534e-06
[params] 63.91M
```

`missing=[] unexpected=[]` on a **strict=False** call that we then assert on: there is
no key remapping at all. `model.embed_tokens.weight` and `lm_head.weight` are bit-equal
(tied) in the source `.pth`, so Qwen3's tied-embedding path is exact.

This means **`convert_hf_to_gguf.py` needs no architecture support for MiniMind** — it
only ever sees `Qwen3ForCausalLM`, which it supports natively (`MODEL_ARCH.QWEN3`).
The repo's own `scripts/convert_model.py` already does the same thing
(`convert_torch2transformers()` builds a `Qwen3ForCausalLM`); we re-implemented it as a
standalone script so it can be pointed at any smoke checkpoint and so it emits a parity
assertion.

### 2.2 Commands

```powershell
$env:UV_CACHE_DIR='D:\llm-rl-lab\.cache\uv'
& D:\llm-rl-lab\.venv\Scripts\python.exe `
    D:\llm-rl-lab\llamacpp\scripts\convert_pth_to_hf.py `
    D:\llm-rl-lab\src\minimind-master\out\sft_smoke_768.pth `
    D:\llm-rl-lab\models\hf\sft_smoke_768
```

Script: `D:\llm-rl-lab\llamacpp\scripts\convert_pth_to_hf.py`
Log: `D:\llm-rl-lab\logs\hf_convert.log`

One post-step was needed: `Qwen3Config` defaults `bos_token_id`/`eos_token_id` to
`None`, but llama.cpp needs them to emit stop tokens. Fixed in-place in `config.json`:

```python
d['bos_token_id'] = 1      # <|im_start|>
d['eos_token_id'] = 2      # <|im_end|>
d['pad_token_id'] = 0      # <|endoftext|>
```

### 2.3 Output files — `D:\llm-rl-lab\models\hf\sft_smoke_768\`

| File | Bytes | Note |
|---|---|---|
| `model.safetensors` | 127,834,168 | fp16, single shard |
| `config.json` | 952 | `architectures: ["Qwen3ForCausalLM"]`, head_dim 96, GQA 8/4 |
| `tokenizer.json` | 451,182 | 6,400-token BPE, copied verbatim from repo |
| `tokenizer_config.json` | 13,308 | chat template + added_tokens_decoder |
| `chat_template.jinja` | 3,979 | |
| `special_tokens_map.json` | 1,111 | |
| `generation_config.json` | 73 | |

### 2.4 Load + generate verification

```powershell
$env:PYTHONIOENCODING='utf-8'
& D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\llamacpp\scripts\verify_hf.py
```

```
loaded: Qwen3ForCausalLM params: 63.912192 M
vocab: 6400 eos: 2 bos: 1
prompt ids: [[1, 832, 311, 234, 1968, 294, 960, 2919, 2360, 441, 1153, 302, 2, 234, 1, 1388, 570, 811, 234]]
=== CLEAN ===
好的，我无法可能到情感的事情的电影。我需要提供一些人和关于电影，用户回答的语言和准确。
我需要想保持好，我是没有看。
我需要推荐一篇电影，我是我想对没有不错。
```

`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` loads cleanly
(no remote code needed — it resolves to native `Qwen3ForCausalLM`). Output is
chat-formatted, on-topic-ish Chinese, grammatically rough — **this is a 63.9 M smoke
model, not a quality signal**. Text is *coherent-ish*, not garbage: correct
`<|im_start|>`/`<|im_end|>` handling, plausible Chinese tokens, no repetition loops
beyond a mild "我需要" tic.

⚠️ Windows console gotcha: `print()` of Chinese crashes with
`UnicodeEncodeError: 'gbk' codec`. **Always set `$env:PYTHONIOENCODING='utf-8'`.**

---

## 3. llama.cpp prebuilt binary

### 3.1 Tag discovery (api.github.com via ghproxy is 403 — as the lead found)

Working methods, in order of reliability:

1. **`releases/latest` → follow the 302 → parse the tag from `Location`.** This is the
   only way to get the *current* tag, and it works:
   ```powershell
   curl.exe -sIL --max-time 60 "https://ghproxy.net/https://github.com/ggml-org/llama.cpp/releases/latest" |
       Select-String -Pattern "^(HTTP|location)"
   # HTTP/1.1 302 FOUND
   # Location: /https://github.com/ggml-org/llama.cpp/releases/tag/v0.7.0
   ```
2. **`releases/expanded_assets/<tag>`** — returns a small HTML fragment listing every
   asset. This is the money endpoint. `releases/tag/<tag>` returns a 401 KB JS-heavy SPA
   page with **no** asset names in it (grep for `bin-` finds nothing); `expanded_assets`
   returns 77 KB of plain `href="/ggml-org/llama.cpp/releases/download/..."` links.
3. **Probe tags by HTTP status** — `expanded_assets/<tag>` → 200 vs 404 is a clean
   existence test. `releases.atom` is useless through ghproxy (returns 14 bytes).

⚠️ **Do not trust `v0.7.0`.** It is the newest *semver-tagged* release, but its
`expanded_assets` contains exactly **one** file: `nightly-tag.txt`. llama.cpp now ships
binaries only under `bNNNN` build-number tags, and `v0.7.0`'s `nightly-tag.txt` is a
pointer, not a build. Probing `bNNNN` found the live range: `b7500`→404, `b7600`→200,
`b7700`→200, `b7780`→404, **`b7795`→200**, `b7800`→404.

### 3.2 Downloaded assets (tag `b7795`)

| Asset | Size | Extracted to |
|---|---|---|
| `llama-b7795-bin-win-cuda-12.4-x64.zip` | 207.4 MB | `D:\llm-rl-lab\llamacpp\cuda124\` |
| `cudart-llama-bin-win-cuda-12.4-x64.zip` | 373.3 MB | `D:\llm-rl-lab\llamacpp\cudart124\` |
| `llama-b7795-bin-win-cpu-x64.zip` | 28.5 MB | `D:\llm-rl-lab\llamacpp\cpu\` |

```powershell
curl.exe -sL --max-time 900 -o D:\llm-rl-lab\llamacpp\downloads\cuda124.zip `
  "https://ghproxy.net/https://github.com/ggml-org/llama.cpp/releases/download/b7795/llama-b7795-bin-win-cuda-12.4-x64.zip"
# ... same for cudart and cpu, then:
Expand-Archive -Path <zip> -DestinationPath D:\llm-rl-lab\llamacpp\<dir> -Force
```

### 3.3 ⚠️ The CUDA build ships **without** its CUDA runtime DLLs

Extracting `llama-b7795-bin-win-cuda-12.4-x64.zip` gives `ggml-cuda.dll` (445 MB) but
`llama-cli.exe --list-devices` reports **`Available devices:` (empty)** — it silently
falls back to CPU. The runtime DLLs are in the *separate* `cudart-*.zip`. Fix:

```powershell
Copy-Item D:\llm-rl-lab\llamacpp\cudart124\*.dll D:\llm-rl-lab\llamacpp\cuda124\ -Force
```

`cudart124` contains `cublas64_12.dll` (100 MB), `cublasLt64_12.dll` (473 MB),
`cudart64_12.dll` (554 KB). After the copy:

```
Available devices:
  CUDA0: NVIDIA GeForce RTX 4060 Laptop GPU (8187 MiB, 7106 MiB free)
load_backend: loaded CUDA backend from D:\llm-rl-lab\llamacpp\cuda124\ggml-cuda.dll
```

**This is a real trap** — without it you would conclude "CUDA build works but GPU isn't
used" and ship a 2.4× slower path. Total footprint of the working CUDA dir ≈ 1.0 GB.

Verified binary:

```
$ D:\llm-rl-lab\llamacpp\cuda124\llama-cli.exe --version
version: 7795 (bd544c94a)
built with Clang 19.1.5 for Windows x86_64
```

No build tools needed — prebuilt binaries work on this machine as-is.

---

## 4. HF → GGUF conversion

### 4.1 Dependencies

`uv` is **not** importable as a module (`python -m uv` fails); use the standalone exe:

```powershell
$env:UV_CACHE_DIR='D:\llm-rl-lab\.cache\uv'
& C:\Users\24657\.local\bin\uv.exe pip install `
    --python D:\llm-rl-lab\.venv\Scripts\python.exe `
    gguf sentencepiece `
    --index-url https://mirrors.aliyun.com/pypi/simple/ `
    --extra-index-url https://pypi.org/simple
# + gguf==0.19.0   (sentencepiece 0.2.0 already present)
```

Converter script fetched at the matching tag:

```powershell
curl.exe -sL --max-time 120 -o D:\llm-rl-lab\llamacpp\convert_hf_to_gguf.py `
  "https://ghproxy.net/https://raw.githubusercontent.com/ggml-org/llama.cpp/b7795/convert_hf_to_gguf.py"
# 537,123 bytes
```

### 4.2 ⚠️ BLOCKER: unrecognized BPE pre-tokenizer — and the minimal fix

First attempt failed exactly where predicted, but **not** on architecture — the model
loaded fine as `Qwen3ForCausalLM`. It failed on the tokenizer:

```
INFO:hf-to-gguf:Model architecture: Qwen3ForCausalLM
...
WARNING:hf-to-gguf:** WARNING: The BPE pre-tokenizer was not recognized!
WARNING:hf-to-gguf:** chkhsh:  a39372c92460bbbac950b7bc25ddc61768b9aa10ae4bd07d3336ef66f6b9fd41
NotImplementedError: BPE pre-tokenizer was not recognized - update get_vocab_base_pre()
```

**Root cause.** `convert_hf_to_gguf.py` fingerprints the tokenizer's pre-tokenizer by
hashing the token ids of a canary string, then looks the hash up in a generated table
mapping hash → llama.cpp pre-tokenizer name. MiniMind trained its BPE with
`tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)` (see
`trainer/train_tokenizer.py:47`), whose `tokenizer.json` stores only
`{"type":"ByteLevel", ..., "use_regex": true}` — the regex is *implicit* and equals the
stock GPT-2 pattern. That exact configuration simply isn't in the table, so conversion
aborts before writing anything.

**Why the fix is exact, not a fudge.** `use_regex: true` + `ByteLevel` is by definition
the GPT-2 regex `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)`.
llama.cpp's `LLAMA_VOCAB_PRE_TYPE_GPT2` branch (`src/llama-vocab.cpp:351`) uses that
literal regex. So `gpt-2` is the *correct* mapping. Confirmed empirically in §4.5:
**12/12 texts tokenize byte-identically** between HF and llama.cpp.

**The patch** — 8 lines added to our *copy* of the converter
(`D:\llm-rl-lab\llamacpp\convert_hf_to_gguf.py`, original repo untouched), inserted
just before the `if res is None:` failure branch at line 1262:

```python
        # --- LOCAL PATCH (llm-rl-lab) -------------------------------------
        # MiniMind (jingyaogong/minimind) trains its 6400-token BPE with
        # tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False), i.e. the
        # stock GPT-2 regex. Its hash is not in the generated table above.
        # Mapping it to "gpt-2" is exact, not an approximation.
        if chkhsh == "a39372c92460bbbac950b7bc25ddc61768b9aa10ae4bd07d3336ef66f6b9fd41":
            # ref: MiniMind tokenizer.json (ByteLevel, use_regex=True)
            res = "gpt-2"
        # --- END LOCAL PATCH ----------------------------------------------
```

That is the **entire** converter change. No architecture patching was required — the
Qwen3 rebuild in §2 already took care of that. No SentencePiece model needed, no
`--vocab-only` detour.

### 4.3 f16 conversion

```powershell
$env:UV_CACHE_DIR='D:\llm-rl-lab\.cache\uv'
$env:PYTHONIOENCODING='utf-8'
Set-Location D:\llm-rl-lab\llamacpp
& D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\llamacpp\convert_hf_to_gguf.py `
    D:\llm-rl-lab\models\hf\sft_smoke_768 `
    --outfile D:\llm-rl-lab\models\gguf\sft_smoke_768-f16.gguf `
    --outtype f16
```

```
INFO:hf-to-gguf:Model architecture: Qwen3ForCausalLM
INFO:hf-to-gguf:gguf: head count = 8 / key-value head count = 4 / rope theta = 1000000.0
INFO:gguf.gguf_writer:...: n_tensors = 90, total_size = 127.9M
INFO:hf-to-gguf:Model successfully exported to ...\sft_smoke_768-f16.gguf
```

Log: `D:\llm-rl-lab\logs\gguf_convert_f16.log`

### 4.4 Quantization

```powershell
Set-Location D:\llm-rl-lab\llamacpp\cuda124
.\llama-quantize.exe D:\llm-rl-lab\models\gguf\sft_smoke_768-f16.gguf `
    D:\llm-rl-lab\models\gguf\sft_smoke_768-Q4_K_M.gguf Q4_K_M
```

```
llama_model_quantize_impl: model size  =   121.93 MiB
llama_model_quantize_impl: quant size  =    40.28 MiB
llama_model_quantize_impl: WARNING: 8 of 57 tensor(s) required fallback quantization
```

**Known cosmetic warning (harmless).** 8 `ffn_down` tensors (`[2432, 768]`) fall back to
**Q8_0** instead of Q6_K because Q6_K requires the row length to be divisible by 256 and
2432 is not. This *improves* quality (Q8_0 > Q6_K) and costs ~1.5 MiB; no action needed.
Root cause: `intermediate_size = ceil(768·π/64)·64 = 2432`, a non-256-multiple. A real
MiniMind-768 has the same shape, so **expect this warning on the production model too**.

Q8_0 also produced for comparison:

```powershell
.\llama-quantize.exe ...\sft_smoke_768-f16.gguf ...\sft_smoke_768-Q8_0.gguf Q8_0
# quant size = 64.80 MiB
```

### 4.5 Tokenizer parity — the proof the patch is right

```powershell
$env:PYTHONIOENCODING='utf-8'
& D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\llamacpp\scripts\tok_parity.py
```

Compares `llama-tokenize.exe --ids --no-bos` against
`AutoTokenizer(..., add_special_tokens=False).input_ids` over Chinese, English, code,
emoji, digits/dates, mixed-case/whitespace, chat markup, and indentation:

```
TOKENIZER PARITY: 12/12 texts identical
```

GGUF metadata confirms the intended configuration:

```
general.architecture      = qwen3
general.size_label        = 64M
qwen3.block_count         = 8
qwen3.embedding_length    = 768
qwen3.feed_forward_length = 2432
qwen3.attention.head_count    = 8
qwen3.attention.head_count_kv = 4
qwen3.attention.key_length    = 96
qwen3.rope.freq_base      = 1000000.0
tokenizer.ggml.model      = gpt2
tokenizer.ggml.pre        = gpt-2          <-- the patch landed
tokenizer.ggml.bos_token_id = 1
tokenizer.ggml.eos_token_id = 2
tokenizer.ggml.add_bos_token = False
tokenizer.ggml.add_eos_token = False
```

---

## 5. Local run

### 5.1 GPU (`-ngl 99`)

```powershell
Set-Location D:\llm-rl-lab\llamacpp\cuda124
$env:PYTHONIOENCODING='utf-8'; chcp 65001 | Out-Null
[Console]::OutputEncoding=[System.Text.Encoding]::UTF8

# UTF-8 prompt file - see gotcha below
[System.IO.File]::WriteAllText("D:\llm-rl-lab\logs\prompt_zh.txt",
  "<|im_start|>user`n你好，请介绍一下你自己。<|im_end|>`n<|im_start|>assistant`n",
  (New-Object System.Text.UTF8Encoding($false)))

.\llama-cli.exe -m D:\llm-rl-lab\models\gguf\sft_smoke_768-Q4_K_M.gguf `
    -ngl 99 -f D:\llm-rl-lab\logs\prompt_zh.txt `
    -n 64 --temp 0.7 --top-p 0.9 --seed 42 --no-warmup -st --no-display-prompt
```

```
[ Prompt: 859.1 t/s | Generation: 923.8 t/s ]

llama_memory_breakdown_print: | memory breakdown [MiB]          | total   free    self   model   context   compute
llama_memory_breakdown_print: |   - CUDA0 (RTX 4060 Laptop GPU) |  8187 = 6564 + ( 526 =    40 +     384 +     102 )
llama_memory_breakdown_print: |   - Host                        |                   69 =     3 +       0 +      65
```

Output (`--seed 42`):

> 我需要在我要一些好的电影，我想希望我能要对我需要用户。
> 首先，我需要没有提供我的信息，用户需要回答不有由我需要提供关于人类。
> 我需要提供电影，要想可能不想可能到有数据或问题，用户

Log: `D:\llm-rl-lab\logs\run_gpu_ngl99_b.log`

### 5.2 CPU (`-ngl 0`)

```powershell
Set-Location D:\llm-rl-lab\llamacpp\cpu
.\llama-cli.exe -m D:\llm-rl-lab\models\gguf\sft_smoke_768-Q4_K_M.gguf `
    -ngl 0 -f D:\llm-rl-lab\logs\prompt_zh.txt -n 64 --temp 0.7 --top-p 0.9 --seed 42
```

```
[ Prompt: 1846.7 t/s | Generation: 263.1 t/s ]
```

Log: `D:\llm-rl-lab\logs\run_cpu_ngl0.log`

### 5.3 Benchmark (3 runs, ±σ)

```powershell
Set-Location D:\llm-rl-lab\llamacpp\cuda124   # and ...\cpu for the CPU row
.\llama-bench.exe -m D:\llm-rl-lab\models\gguf\sft_smoke_768-Q4_K_M.gguf -ngl 99 -p 64 -n 64 -r 3
```

| backend | ngl | pp64 (t/s) | tg64 (t/s) |
|---|---:|---:|---:|
| CUDA (RTX 4060 Laptop) | 99 | 14,562.15 ± 2255.83 | **939.44 ± 22.93** |
| CPU (14 threads) | 0 | 1,966.01 ± 14.14 | **398.24 ± 8.42** |

→ GPU is **2.4× faster at generation**; but CPU-only is perfectly usable for a 64 M
model. Given the GPU is shared with the lead, **`-ngl 0` is a completely acceptable
production choice here** and leaves the GPU free.

### 5.4 Peak VRAM

Sampled `nvidia-smi` every 250 ms during a 128-token GPU run:

```
baseline VRAM used: 787 MiB
PEAK VRAM used: 3879 MiB  (delta 3092 MiB)
```

Log: `D:\llm-rl-lab\logs\vram_peak.log`. llama.cpp's own accounting at ctx 512:
**526 MiB self** (40 model + 384 KV cache + 102 compute) plus 1,097 MiB unaccounted
CUDA context/driver overhead. Budget **~3.1 GiB** for `-ngl 99` at 512 ctx; KV cache
scales with `-c`, so at 8192 ctx expect roughly +5 GiB more. **This is the real
constraint for Phase 4 on 8 GB** — see §7.

### 5.5 OpenAI-compatible server — ✅ works

```powershell
Set-Location D:\llm-rl-lab\llamacpp\cuda124
Start-Process -FilePath .\llama-server.exe -ArgumentList @(
  '-m','D:\llm-rl-lab\models\gguf\sft_smoke_768-Q4_K_M.gguf',
  '-ngl','99','--host','127.0.0.1','--port','8089','-c','2048',
  '--jinja','--chat-template-file','D:\llm-rl-lab\llamacpp\chat_template_minimind.jinja'
)
```

```powershell
$body = @{ messages=@(@{role='user'; content='用一句话解释什么是人工智能。'});
           temperature=0.7; max_tokens=48; stream=$false } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri "http://127.0.0.1:8089/v1/chat/completions" -Method Post `
    -Body $body -ContentType "application/json; charset=utf-8"
```

```
health: {"status":"ok"}
usage:  {"completion_tokens":48,"prompt_tokens":25,"total_tokens":73}
timings: prompt 1053 t/s, predicted 418 t/s
```

Both `/v1/chat/completions` and the raw `/completion` endpoint work. This is the
Phase 4 serving surface and it needs no Python, no torch, no transformers.

### 5.6 ⚠️ The repo's chat template injects an empty `<think>` block

`model/tokenizer_config.json` ends its generation prompt with:

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if open_thinking is defined and open_thinking is true %}
        {{- '<think>\n' }}
    {%- else %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
```

So a naive chat request prompts the model with `<|im_start|>assistant\n<think>\n\n</think>\n\n`.
On the undertrained smoke model this makes it emit **extra literal `<think>` tokens**:

> 好的， `<think>` `</think>` `<think>` 好的，用户问的是…

This is faithful to the repo (`dataset/lm_dataset.py:31-38` shows MiniMind deliberately
trains *both* with and without the empty-think block, stripping it ~80 % of the time),
so the template is correct for a real model — but for **non-reasoning deployment** the
cleanest path is to override it. I wrote a minimal template with no think block:

`D:\llm-rl-lab\llamacpp\chat_template_minimind.jinja`

```jinja
{%- if messages[0].role == 'system' %}{{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}{%- endif %}{%- for message in messages %}{%- if message.role != 'system' %}{{- '<|im_start|>' + message.role + '\n' + (message.content if message.content is string else '') + '<|im_end|>' + '\n' }}{%- endif %}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}
```

With `--chat-template-file` the `<think>` artifacts disappear. **Recommend this for
Phase 4** unless the production MiniMind checkpoint is a thinking model.

### 5.7 ⚠️ Prompt encoding on Windows

`llama-cli -p "<chinese>"` through PowerShell mangles non-ASCII into `��ã������` (the
prompt still "works" but is garbage-in). Two robust workarounds, both used above:

- `-f <utf8-file>` with the file written via `[System.IO.File]::WriteAllText(..., UTF8Encoding($false))` — **preferred**; or
- `--binary-file` / `--escape` with `\u` escapes.

Also needed for readable output: `chcp 65001` + `[Console]::OutputEncoding=UTF8`.

---

## 6. Quantization fidelity — is Q4_K_M actually safe?

Perplexity on 60 real SFT conversations from `D:\llm-rl-lab\data\sft_t2t_mini.jsonl`
(20,868 chars, 4 chunks @ 512 ctx):

```powershell
.\llama-perplexity.exe -m <gguf> -ngl 99 -f D:\llm-rl-lab\logs\ppl_corpus.txt -c 512 --chunks 4
```

| Model | PPL | vs f16 |
|---|---:|---:|
| f16 | 376.49 ± 27.38 | — |
| Q8_0 | 376.86 ± 27.42 | +0.1 % |
| **Q4_K_M** | **373.28 ± 27.21** | **−0.9 %** |

Q4_K_M is **statistically indistinguishable from f16** (well inside the ±27 error bar).
The absolute PPL of ~376 is high, but that reflects the **undertrained smoke model**, not
the quantization — and it's the same number for f16.

Caveat on greedy determinism: HF and GGUF agree exactly for the first ~8 tokens then
diverge (`好的，我需要提供回答。` vs `好的，我需要提供回答的电影，我需要提供回答。`).
That is expected for a model whose next-token distribution is nearly flat (PPL 376 →
very low confidence margins), where f16-vs-Q4 and llama.cpp-vs-torch numerics flip
near-ties. The PPL table is the meaningful measure and it says Q4_K_M is faithful.
Script: `D:\llm-rl-lab\llamacpp\scripts\parity_hf_gguf.py`.

---

## 7. Blockers for Phase 4 — assessment

| # | Risk | Status | Mitigation |
|---|---|---|---|
| 1 | Custom arch `MiniMindForCausalLM` unsupported by llama.cpp | ✅ **Gone** | Rebuild as stock `Qwen3ForCausalLM`; logit parity 4.3e-06 |
| 2 | Custom 6400-token BPE tokenizer unsupported | ✅ **Gone** | One-line hash→`gpt-2` mapping in converter copy; 12/12 token parity |
| 3 | No MSVC/cmake → cannot build llama.cpp | ✅ **Gone** | Prebuilt `b7795` win-cuda-12.4 / win-cpu-x64 via ghproxy |
| 4 | CUDA build silently CPU-only (missing cudart DLLs) | ✅ **Fixed** | Merge `cudart-*.zip` DLLs into the CUDA dir |
| 5 | 8 `ffn_down` tensors fall back Q6_K→Q8_0 | ✅ Benign | Row len 2432 not /256; Q8_0 is *better*; +1.5 MiB |
| 6 | Chat template injects empty `<think>` | ✅ **Fixed** | `--chat-template-file chat_template_minimind.jinja` |
| 7 | **VRAM: 3.1 GiB at 512 ctx on a shared 8 GB GPU** | ⚠️ **Watch** | Use `-ngl 0` (398 t/s, plenty) or cap `-c`. At 8192 ctx, KV cache alone ≈ +5 GiB → **will OOM**. Recommend `-c 2048` max on GPU, or CPU. |
| 8 | 64 M model output quality is poor | ℹ️ Expected | Smoke-weight artifact, **not** a deployment defect. Path is proven; re-run on real `full_sft_768.pth`. |

**Bottom line: nothing blocks Phase 4.** The pipeline is validated end-to-end. Items 7
and 8 are operational/quality notes, not blockers.

---

## 8. Reproduce in one pass

```powershell
$env:UV_CACHE_DIR='D:\llm-rl-lab\.cache\uv'; $env:PYTHONIOENCODING='utf-8'

# 1. pth -> HF (Qwen3 rebuild, parity-asserted)
& D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\llamacpp\scripts\convert_pth_to_hf.py `
    D:\llm-rl-lab\src\minimind-master\out\sft_smoke_768.pth D:\llm-rl-lab\models\hf\sft_smoke_768

# 1b. bos/eos/pad ids (Qwen3Config leaves them None)
& D:\llm-rl-lab\.venv\Scripts\python.exe -c "import json;p=r'D:\llm-rl-lab\models\hf\sft_smoke_768\config.json';d=json.load(open(p,encoding='utf-8'));d.update(bos_token_id=1,eos_token_id=2,pad_token_id=0);json.dump(d,open(p,'w',encoding='utf-8'),indent=2,ensure_ascii=False)"

# 2. HF -> GGUF f16  (uses patched converter copy)
Set-Location D:\llm-rl-lab\llamacpp
& D:\llm-rl-lab\.venv\Scripts\python.exe .\convert_hf_to_gguf.py D:\llm-rl-lab\models\hf\sft_smoke_768 `
    --outfile D:\llm-rl-lab\models\gguf\sft_smoke_768-f16.gguf --outtype f16

# 3. quantize
Set-Location D:\llm-rl-lab\llamacpp\cuda124
.\llama-quantize.exe D:\llm-rl-lab\models\gguf\sft_smoke_768-f16.gguf D:\llm-rl-lab\models\gguf\sft_smoke_768-Q4_K_M.gguf Q4_K_M

# 4. run (GPU)
.\llama-cli.exe -m D:\llm-rl-lab\models\gguf\sft_smoke_768-Q4_K_M.gguf -ngl 99 `
    -f D:\llm-rl-lab\logs\prompt_zh.txt -n 64 --temp 0.7 --top-p 0.9 -st --no-display-prompt

# 5. serve
.\llama-server.exe -m D:\llm-rl-lab\models\gguf\sft_smoke_768-Q4_K_M.gguf -ngl 99 `
    --host 127.0.0.1 --port 8089 -c 2048 --jinja `
    --chat-template-file D:\llm-rl-lab\llamacpp\chat_template_minimind.jinja
```

**Scaling to a real MiniMind-768**: identical, just point step 1 at
`full_sft_768.pth` / `grpo_768.pth` and change output dirs. Expect ~same sizes
(63.9 M params → Q4_K_M ≈ 40 MiB) and the same benign Q8_0 fallback warning.

---

## 9. Artifact index

| Path | What |
|---|---|
| `D:\llm-rl-lab\models\hf\sft_smoke_768\` | HF Qwen3 checkpoint (121.9 MiB) |
| `D:\llm-rl-lab\models\gguf\sft_smoke_768-f16.gguf` | f16 GGUF, 128,093,952 B |
| `D:\llm-rl-lab\models\gguf\sft_smoke_768-Q4_K_M.gguf` | **Q4_K_M GGUF, 42,481,920 B** |
| `D:\llm-rl-lab\models\gguf\sft_smoke_768-Q8_0.gguf` | Q8_0 GGUF, 67,960,832 B |
| `D:\llm-rl-lab\llamacpp\cuda124\` | llama.cpp b7795 CUDA 12.4 + cudart DLLs |
| `D:\llm-rl-lab\llamacpp\cpu\` | llama.cpp b7795 CPU x64 |
| `D:\llm-rl-lab\llamacpp\convert_hf_to_gguf.py` | converter copy **with the 1-hash patch** |
| `D:\llm-rl-lab\llamacpp\chat_template_minimind.jinja` | think-free chat template |
| `D:\llm-rl-lab\llamacpp\scripts\convert_pth_to_hf.py` | pth→HF + parity assert |
| `D:\llm-rl-lab\llamacpp\scripts\verify_hf.py` | HF load+generate check |
| `D:\llm-rl-lab\llamacpp\scripts\tok_parity.py` | 12-text tokenizer parity |
| `D:\llm-rl-lab\llamacpp\scripts\parity_hf_gguf.py` | HF vs GGUF greedy compare |
| `D:\llm-rl-lab\llamacpp\scripts\make_ppl_corpus.py` | builds PPL corpus from SFT data |
| `D:\llm-rl-lab\logs\hf_convert.log` | HF conversion log |
| `D:\llm-rl-lab\logs\hf_verify.log` | HF load/generate log |
| `D:\llm-rl-lab\logs\gguf_convert_f16.log` | GGUF f16 conversion log (incl. the blocker) |
| `D:\llm-rl-lab\logs\gguf_quantize_q4km.log` | Q4_K_M quantize log |
| `D:\llm-rl-lab\logs\tok_parity.log` | 12/12 tokenizer parity |
| `D:\llm-rl-lab\logs\bench_gpu_q4km.log` | GPU benchmark |
| `D:\llm-rl-lab\logs\bench_cpu_q4km.log` | CPU benchmark |
| `D:\llm-rl-lab\logs\run_gpu_ngl99_b.log` | GPU inference log |
| `D:\llm-rl-lab\logs\run_cpu_ngl0.log` | CPU inference log |
| `D:\llm-rl-lab\logs\vram_peak.log` | peak VRAM measurement |
| `D:\llm-rl-lab\logs\server2_stdout.log` | llama-server log |
| `D:\llm-rl-lab\logs\llamacpp_assets_b7795.html` | release asset listing |
| `D:\llm-rl-lab\logs\llama-vocab.cpp` | upstream pre-tokenizer regex table (evidence) |
