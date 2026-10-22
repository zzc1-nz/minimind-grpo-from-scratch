# Phase 0 — Environment & Minimum End-to-End Run

**Status: PASS.** All three training paths (pretrain / SFT / GRPO) start and step on
this machine, on Windows, natively, with **no WSL2 fallback and no flash-attn /
bitsandbytes / triton-by-default dependency problems.** Peak VRAM stayed under 7 GB in
every configuration that is actually used downstream.

Every number in this report was measured on this machine; nothing is estimated.

---

## 1. Host & GPU

| Item | Measured value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Total VRAM | 8,188 MiB |
| Compute capability | 8.9 (Ada, sm_89), 24 SMs |
| Driver | 610.47 |
| bf16 support | True |
| torch CUDA build | cu124 (CUDA 12.4), cuDNN 9.1.0 |
| CPU | i7-13700H (reported by user) |
| RAM | 16 GB DDR5 (hard constraint; confirmed — see §6) |
| Disk used | Project root `D:\llm-rl-lab` (D: had 140.6 GB free, 113 GB free after data + model downloads) |

**Baseline VRAM was NOT the assumed 1.2 GB.** A resident Ollama `llama-server`
(pid 29596, `--model .../sha256-65b8fcd9...`) held **4,655 MiB** of dedicated VRAM,
leaving only **2,058 MiB** free. It was stopped (`Stop-Process`), after which free VRAM
became **6,759–7,404 MiB**. All experiments below were run with it stopped.

> Lesson recorded for the README: on this box, "6.7 GB usable" is only true after
> killing the resident Ollama server. Leaving it running makes even a 64M model OOM.

## 2. Toolchain

| Component | Version | Notes |
|---|---|---|
| Python | 3.11.9 | `uv venv` at `D:\llm-rl-lab\.venv` |
| uv | 0.13.0 | `UV_CACHE_DIR=D:\llm-rl-lab\.cache\uv` (default cache on C: was **not writable**) |
| torch | 2.6.0+cu124 | from `download.pytorch.org/whl/cu124`, 325 s |
| torchvision | 0.21.0+cu124 | |
| transformers | 4.57.6 | MiniMind pinned version |
| datasets | 3.6.0 | |
| numpy | **2.2.6** | pinned 1.26.4 from `requirements.txt` **breaks pyarrow** — see §5 |
| triton | triton-windows **3.2.0.post21** | required for `torch.compile`; 3.8.0 does **not** work |
| modelscope | 1.37.0 | used only for API metadata; downloads used raw HTTP Range |
| Git | 2.54.0 | |
| cmake / MSVC (`cl`) | **absent** | no source builds possible → GGUF path uses prebuilt llama.cpp |

## 3. Network (measured, this machine)

| Endpoint | Result |
|---|---|
| `pypi.org` | 200, 377 ms |
| `mirrors.aliyun.com/pypi` | 200, **82 ms** (used as primary index) |
| `modelscope.cn` | 200, 213 ms |
| `hf-mirror.com` | 200, 197 ms |
| `github.com` | **timeout (8 s)** |
| `ghproxy.net` | 200, 788 ms (GitHub proxy — used for the MiniMind repo + llama.cpp binaries) |
| `gitee.com` / `gitcode.com` | 200 but the MiniMind mirror paths 404/405 |

**ModelScope throughput, measured:** a single connection is slow, but
**16-way parallel HTTP Range (`curl.exe -r`) reached 33.7 MB/s**, which is how all
datasets and the reward model were fetched:

| Artifact | Size | Method | Time |
|---|---|---|---|
| MiniMind repo (zip via ghproxy) | 10.3 MB | single GET | 10.7 s |
| `pretrain_t2t_mini.jsonl` | 1,183.5 MB | 16-way Range | — |
| `sft_t2t_mini.jsonl` | 1,658.5 MB | 16-way Range | — |
| `rlaif.jsonl` | 22.7 MB | 4-way Range | — |
| `pretrain_t2t.jsonl` (full) | **7.71 GB** | 24-way Range | **110 s** |
| `internlm2-1_8b-reward` | 3,167 GiB | 16-way Range | 49 s |

`curl.exe -s -r start-end` against
`https://www.modelscope.cn/api/v1/datasets/gongjy/minimind_dataset/repo?Revision=master&FilePath=<file>`
returns **HTTP 206** and is byte-exact (verified by SHA-256 against the ModelScope
listing for the reward model shards).

## 4. MiniMind repo

Cloned from `https://ghproxy.net/https://github.com/jingyaogong/minimind/archive/refs/heads/master.zip`
→ `D:\llm-rl-lab\src\minimind-master`. GitHub direct is unreachable, so the archive was
fetched through ghproxy; **there is no `.git` directory** (zip download), so the repo
revision is pinned by the zip checksum rather than a commit hash.

Structure used: `model/model_minimind.py` (Transformer), `dataset/lm_dataset.py`
(Pretrain/SFT/RLAIF datasets), `trainer/train_pretrain.py`, `trainer/train_full_sft.py`,
`trainer/train_grpo.py`, `trainer/rollout_engine.py`, `trainer/trainer_utils.py`.

**Nothing in the repo was modified.** Where a change was needed it was made in a
*documented thin derivative* under `D:\llm-rl-lab\custom\` (see §7).

### Model size

`MiniMindConfig(hidden_size, num_hidden_layers)` parameter counts, measured by
instantiating the model:

| hidden | layers | params |
|---|---|---|
| 512 | 8 | **30.03 M** (this is MiniMind's "26M" family) |
| 640 | 8 | 45.40 M |
| **768** | **8** | **63.91 M** ← chosen ("64M" family) |

`vocab_size=6400`, `num_attention_heads=8`, `num_key_value_heads=4`, `head_dim=96`.
Architecturally this is a Qwen3-style decoder (RMSNorm + RoPE + GQA + SwiGLU + QK-norm),
which later turned out to matter for GGUF export (§8).

## 5. Dependency problems actually hit (and how they were resolved)

These are the real blockers; each was resolved on Windows, so **WSL2 was not needed**.

| # | Problem | Resolution |
|---|---|---|
| 1 | `uv` default cache on `C:\Users\...\AppData\Local\uv\cache` is **not writable** → every uv command failed | set `UV_CACHE_DIR=D:\llm-rl-lab\.cache\uv` |
| 2 | `ujson==5.1.0` and `jieba==0.42.1` have **no wheel for this platform** and need MSVC → `Microsoft Visual C++ 14.0 is required` | both are only used by `scripts/chat_api.py`, `scripts/web_demo.py` (Flask/Streamlit demo), **not** by any training path → skipped. All training imports verified. |
| 3 | `pyarrow requires NumPy 2.0 or newer, found 1.26.4` (repo pins `numpy==1.26.4`) | upgraded to `numpy==2.2.6`; training code only uses `np.ndarray` basics, verified working |
| 4 | `torch.compile` fails: `Cannot find a working triton installation` | `triton-windows==3.8.0.post29` installs but then fails with `ImportError: cannot import name 'AttrsDescriptor' from 'triton.compiler.compiler'` (too new for torch 2.6). **`triton-windows==3.2.0.post21` works.** |
| 5 | `train_pretrain.py --save_interval 0` → `ZeroDivisionError: integer modulo by zero` | repo bug; use `save_interval >= 1` (or `100000` to effectively disable) |
| 6 | DataLoader workers crash: `_pickle.UnpicklingError: pickle data was truncated` / `OSError: [Errno 22]` | the 4.2 GB `np.memmap` dataset object was being pickled to workers on Windows spawn. Fixed by `__getstate__`/`__setstate__` (send path + shape, re-open the memmap inside each worker). |
| 7 | A JSONL file written by PowerShell `Set-Content` gets a **UTF-8 BOM** → `datasets` `DatasetGenerationError` | always write jsonl from Python with `encoding='utf-8', newline='\n'` |
| 8 | `eval_llm.py`/trainers resolve `../out` and `../../internlm2-1_8b-reward` relative to `trainer/`, not to the project root | always `cd D:\llm-rl-lab\src\minimind-master\trainer` and pass `--save_dir` / `--reward_model_path` explicitly |

**Not needed / not a problem:** `flash-attn` (the model falls back to
`torch.nn.functional.scaled_dot_product_attention`, which is enabled and used),
`bitsandbytes` (no quantization needed during training), `sglang` (optional rollout
backend; the repo's native `TorchRolloutEngine` is used instead), `deepspeed`,
`wandb`/`swanlab` (metrics are written to JSONL instead — no account, no network).

## 6. RAM pressure (16 GB hard constraint)

Observed and worked around:

* A first attempt to tokenize the 7.71 GB corpus with a 20,000-document batch grew the
  Python process to **5.6 GB working set** and the machine to **2.2 GB available RAM**,
  and the process **stalled** (no forward progress for >2 min). Killed.
* Rewritten as a bounded-buffer streaming tokenizer (`batch-docs=2000`): the same job
  then ran at a **flat 3.6 M tokens/s** and finished the full 8,468,827-document corpus
  in **1,033 s** with no memory growth.
* Consequence: **dataloader worker count is kept at 4** and `num_workers=0` is used for
  short smoke runs. Do not raise it while other desktop apps are open.

## 7. Throughput and VRAM measurements

All numbers: `hidden=768, layers=8` (63.91M), bf16 autocast, AdamW, seq_len as listed,
batch = 16,384-token steps where stated. Measured with
`torch.cuda.max_memory_reserved()`.

### 7a. Vanilla MiniMind path (per-document tokenization + padding to `max_seq_len`)

| seq | batch | compile | s/step | tokens/s | peak reserved |
|---|---|---|---|---|---|
| 340 | 16 | off | 0.353 | 15,395 | 3,700 MiB |
| 340 | 32 | off | 0.744 | 14,620 | 6,560 MiB |
| 512 | 16 | off | 0.556 | 14,739 | 5,124 MiB |
| 768 | 16 | off | 0.870 | 14,129 | 7,184 MiB |
| 340 | 16 | **on** | — | 22,122 | 2,400 MiB |
| 340 | 32 | on | 0.526 | 20,698 | 4,100 MiB |
| 340 | 48 | on | 0.670 | 24,346 | 5,468 MiB |
| 512 | 32 | on | 0.707 | 23,187 | 5,468 MiB |
| 256 | 64 | on | 0.654 | 25,040 | 5,472 MiB |

**`torch.compile` is worth 1.47x and uses *less* VRAM** (inductor fuses and does not
keep the extra autograd buffers). Compile cost is ~30–42 s once per process.

### 7b. Packed-corpus path (the one actually used for Phase 1)

MiniMind's `PretrainDataset` tokenizes inside `__getitem__` and **pads every document to
`max_seq_len`**; documents average 261 tokens, so ~40 % of every step was pad tokens, and
dataloader time was ~30 % of step time. Replaced with a pre-tokenized, packed block
dataset (see §8). Measured on the real 2.209 B-token corpus:

| config | s/step | tokens/s | peak reserved |
|---|---|---|---|
| bs 32, seq 512, compile, 120 steps | — | 17,342 | 5,288 MiB |
| bs 32, seq 512, compile, 400 steps (steady) | 0.74 | **22,184** | 5,288 MiB |
| bs 40, seq 512, compile, 60 steps | 1.61 | — | **6,634 MiB** |
| bs 48, seq 512, compile | — | OOM | > 6.7 GB |

→ **chosen production config: bs 32, seq 512, compile → 16,384 tokens/step,
~22,000 tokens/s, 5,288 MiB peak reserved (64 % of the 8,188 MiB card).**

### 7c. Generation (rollout) throughput — matters for GRPO cost

64M model, `max_new_tokens=128`, do_sample:

| batch/group | s | gen tokens/s | peak reserved |
|---|---|---|---|
| 1 | 2.45 | 52 | 428 MiB |
| 4 | 2.37 | 216 | 444 MiB |
| 8 | 2.38 | 430 | 458 MiB |
| 16 | 2.16 | 950 | 522 MiB |
| 32 | 2.38 | 1,721 | 620 MiB |

Decode is **latency-bound, not throughput-bound**: 128 tokens take ~2.4 s regardless of
batch size, so GRPO should use a **large group size** (8–32) — the samples are almost free.

## 8. Tokenization / data facts measured

The full `pretrain_t2t.jsonl` was tokenized with the MiniMind 6,400-vocab tokenizer
(`tokenizer._tokenizer.encode_batch`, 3.6 M tokens/s, 1,033 s):

| Metric | Value |
|---|---|
| documents | 8,468,827 |
| tokens (incl. one EOS per document) | **2,208,852,597** |
| blocks of 512 (dropped 117 tail tokens) | 4,314,165 |
| **tokens actually consumed by training** | **2,208,841,728** |
| disk (`uint16`) | 4,217.7 MB |

Digit tokenization (critical for arithmetic tasks, verified):

* single digits `0–9` are single tokens, ids **51–60**
* **every integer 0–25 is a single token**; `-9` is two tokens (`48, 60`)
* `123` → `[6318]` (one token), but `26` → `[53, 57]` — 2-digit handling is inconsistent
  above 25
* full-width `７` is **not** normalised by the tokenizer (the grader NFKC-folds it instead)

Chat template quirk: `tokenizer.apply_chat_template(..., open_thinking=False)` emits
`<|im_start|>assistant\n<think>\n\n</think>\n\n` — an **empty, already-closed** think
block. The repo's SFT pipeline removes it 80 % of the time (`post_processing_chat`).
Consequence for evaluation: **SFT and GRPO must be evaluated with the same prompt mode**;
the held-out harness records `prompt_mode` in every report so two runs can never be
silently compared across formats.

## 9. The three training paths — smoke results

All three ran on the **63.91 M** model, `seed=42`, from the repo's own scripts unless
noted. Loss curves below are real logged values.

### 9a. Pretrain (`trainer/train_pretrain.py`, repo code unchanged)

200 steps, bs 16, seq 340, lr 5e-4, 3,200 documents:

```
step  20  loss 7.3078
step  60  loss 7.1141
step 120  loss 6.3928
step 200  loss 6.1006     (monotone, no divergence)
```
elapsed 81.9 s, **peak reserved 3,622 MiB**.

### 9b. SFT (`trainer/train_full_sft.py`, repo code unchanged)

200 steps, bs 8, seq 768, lr 1e-5, 1,600 conversations, warm-started from the pretrain
smoke weight:

```
step  20  loss 6.1669
step  80  loss 5.4639
step 200  loss 5.5102
```
elapsed 92.6 s, **peak reserved 4,008 MiB**.

### 9c. GRPO — official script (`trainer/train_grpo.py`) WITH the 1.8B reward model

MiniMind's GRPO scores rollouts with `internlm2-1_8b-reward` (a neural, LLM-judge-style
reward). It was downloaded and it works:

* `Shanghai_AI_Laboratory/internlm2-1_8b-reward`, 3.167 GiB, SHA-256 verified
* `trust_remote_code=True` works on transformers 4.57.6
* `get_score` discriminates: correct answer `1+1等于2` → **0.8916**, wrong `1+1等于3` → **0.5288**
* peak VRAM to load it: **3,254 MiB**

Run: group=4, 100 steps, gen_len=96, rlaif data:

```
step   1  Reward -1.73  KL 0.1148  AdvStd 0.0000
step  20  Reward -1.60  KL 0.0477  AdvStd 1.1544
step  50  Reward -1.90  KL 0.0580  AdvStd 1.1544
step 100  Reward -2.31  KL 0.0262  AdvStd 1.1535
```
elapsed 240.5 s, **peak reserved 5,322 MiB**.

At group=8 / gen_len=256 (the config the spec asks for) it still runs but costs
**peak reserved 6,890 MiB** — i.e. **6.73 GB, 99 % of the 8,188 MiB card.** That is
inside the "< 7 GB" acceptance bar but leaves no margin at all.

**This is why Phase 3 does not use this path.** The spec requires
*"奖励函数只选可程序化验证的任务：输出必须合法且计算结果正确，奖励 1/0，不用 LLM 裁判"*.
The neural reward model above is exactly the LLM-judge the spec forbids. So Phase 3 uses a
**thin derivative** of the same GRPO trainer whose only change is the reward function:

`custom/train_grpo_rule.py` — keeps MiniMind's GRPO math verbatim (group-normalised
advantages, PPO clip, k3 KL estimator, `TorchRolloutEngine` rollout, token masks) and
replaces `calculate_rewards()` with a pure `1.0 / 0.0` programmatic checker
(`custom/verifiable_reward.py`, self-test **14/14 PASS**).

Smoke of that path: 3 steps, group=8, gen_len=32 → runs, and because there is no 1.8 B
reward model, **peak reserved is only 1,940 MiB**; a 30-step probe on an easy task reached
**2,270 MiB**. It also logs per-step `sampled_tokens`, `reward`, `acc`, `KL`, `clip_frac`,
`entropy`, `degenerate_group_frac` and a `collapsed` flag to JSONL.

> Phase-0 GRPO minimum acceptance was run **both** ways, as required: the official
> script (dependency + VRAM check, with the neural reward) and the rule-reward variant
> that Phase 3 will actually use.

### 9d. Reward signal sanity (important, recorded honestly)

A 30-step rule-reward GRPO probe on an easy task (200 prompts, `a+b` with `a,b∈[0,4]`)
using the **200-step smoke weights** (i.e. a model that has barely learned anything):

```
step  5  reward 0.078  acc 0.031  advstd 1.01
step 15  reward 0.169  acc 0.125  advstd 1.02
step 30  reward 0.156  acc 0.062  advstd 0.51
```

`adv_std ≈ 1.0` means group-normalisation is producing real gradient signal, and
`acc` is non-zero on a mostly-untrained model — so the reward is **not** uniformly 0 and
the pipeline can learn. On the full 600-task set the same probe got `reward=0.000,
adv_std=0.000` — i.e. **all-wrong groups give zero advantage** (the documented GRPO
sparsity risk). Phase 3 therefore starts from an easy-task curriculum, exactly as the
spec's risk section prescribes.

## 10. Phase-4 de-risking (done early, while Phase 1 trains)

Because Phase 1 is a ~28 h GPU run, the deployment path was proven out in parallel:

| Stage | Result |
|---|---|
| `.pth` → HF (`Qwen3ForCausalLM`) | ✅ logit parity vs native MiniMind **4.29e-06** |
| llama.cpp | ✅ prebuilt **b7795** (CUDA 12.4 + CPU), via ghproxy (no MSVC needed) |
| GGUF f16 / Q4_K_M / Q8_0 | ✅ **122.2 / 40.5 / 65.0 MB** |
| tokenizer parity HF vs llama.cpp | ✅ **12/12** byte-identical |
| quant fidelity (smoke weights) | Q4_K_M PPL **373.28** vs f16 **376.49** → **−0.9 %** |
| local run | GPU **939 tok/s** gen @ `-ngl 99`, CPU **398 tok/s** @ `-ngl 0`, peak 3,092 MiB |
| serving | ✅ `llama-server` `/v1/chat/completions` verified |

Two traps found and documented (both would have silently broken Phase 4):
the CUDA zip ships **without** its CUDA runtime DLLs (silently falls back to CPU until
the separate `cudart-*.zip` is merged in), and GitHub's `releases/latest` points at a
`v0.7.0` decoy whose only asset is `nightly-tag.txt` — real binaries live under
`bNNNN` tags. Full detail: `reports/deploy_probe.md`.

## 11. Phase-0 acceptance checklist

| Requirement | Result |
|---|---|
| pretrain path starts and loss decreases | ✅ 7.31 → 6.10 over 200 steps |
| SFT path starts and loss decreases | ✅ 6.17 → 5.51 over 200 steps |
| GRPO path starts (official script) | ✅ 100 steps, group=4 |
| GRPO path starts (rule-reward variant used in Phase 3) | ✅ 3 steps + 30-step probe |
| peak VRAM < 7 GB | ✅ 5,288 MiB (pretrain prod) / 4,008 MiB (SFT) / 1,940–2,270 MiB (rule GRPO); 6,890 MiB worst case for the *neural-reward* GRPO that is not used |
| environment report | ✅ this file |
| Windows dependency blockers | ✅ triton/compile solved (`triton-windows 3.2.0.post21`), no flash-attn/bnb needed |
| WSL2 needed? | ❌ not needed — everything runs natively on Windows |

## 12. Files produced by Phase 0

```
D:\llm-rl-lab\
  .venv\                              uv virtualenv (torch 2.6.0+cu124)
  src\minimind-master\                MiniMind repo, unmodified
  data\
    pretrain_t2t.jsonl                7.71 GB  full pretrain corpus
    pretrain_t2t_mini.jsonl           1.18 GB
    sft_t2t_mini.jsonl                1.66 GB
    rlaif.jsonl                        22.7 MB
    tokens\pretrain_blocks.bin        4.22 GB  4,314,165 x 512 uint16
    tokens\pretrain_meta.json         exact token counts
    phase0\                           3200/1600/100-sample smoke subsets
    grpo\train_grpo.jsonl             600 rule-verifiable tasks (seed 42)
    sft\sft_aux.jsonl                 120 rule-verifiable tasks (seed 1337)
  models\internlm2-1_8b-reward\       3.17 GB (Phase-0 dependency probe only)
  models\hf\sft_smoke_768\            HF export probe
  models\gguf\sft_smoke_768-*.gguf    f16 / Q4_K_M / Q8_0
  llamacpp\                           llama.cpp b7795 binaries
  custom\                             thin derivatives + data/tooling scripts
  eval\                               held-out benchmark (24 tasks) + grader + runner
  scripts\                            benchmarks, VRAM wrapper
  logs\                               every raw measurement used above
  reports\env_report.md               this file
  reports\reward_model_probe.md       internlm2 reward model probe
  reports\deploy_probe.md             Phase-4 de-risking report
  reports\eval_harness.md             held-out benchmark design + smoke run
```

## 13. Cost model for the remaining phases (from measured numbers)

| Phase | Work | Measured rate | Projected |
|---|---|---|---|
| 1 | 1 epoch × 2,208,841,728 tokens, 134,817 steps @ bs32/seq512 | ~22,000 tok/s | **≈ 28 h** |
| 2 | SFT (general + aux), ~250 M tokens | to be measured | ~3–4 h |
| 3 | rule-reward GRPO, group 8–16, gen_len 32–64 | ~1,000 sampled tok/s, 2,270 MiB | ~2–4 h for 1–2 k steps |
| 4 | GGUF Q4 export + demo | pipeline proven | < 1 h |
| 5 | README + bullets | — | < 1 h |

### 13a. Revised Phase-1 budget (decision recorded)

The full-corpus run (134,817 steps ≈ 28 h) was **started and then deliberately stopped at
step 4,400** (72.1 M tokens, loss 2.3236, 5,292 MiB peak, 23,070 tok/s — all healthy).
Reason: 28 h of GPU time blocks Phases 2–4 entirely, and the run had only reached **3.3 %**.

It was restarted with an explicit **1.2 B-token budget: `--max_steps 73242`**, so the
cosine LR schedule (`get_lr`, MiniMind's own) **anneals to its floor inside the budget**
instead of being truncated mid-schedule. That is the important difference: stopping a
cosine run at 3 % leaves the model at full learning rate and a non-annealed loss, which
would make the SFT/GRPO numbers downstream incomparable to a properly finished schedule.

| | full corpus | **chosen budget** |
|---|---|---|
| steps | 134,817 | **73,242** |
| tokens | 2,208,841,728 | **1,199,996,928** |
| wall time @ 22,000 tok/s | ≈ 28 h | **≈ 15 h** |
| LR at end | 0.05 × peak (truncated) | **0.1 × peak (schedule complete)** |
| spec target (10–50 亿 tokens) | 22.1 亿 | **12.0 亿 — in range** |

The un-run portion of the corpus stays on disk (`data/tokens/pretrain_blocks.bin`) and the
launcher supports `--max_steps`, so extending to the full 2.2 B later is a one-flag change
if the schedule is rebuilt accordingly. The aborted run's metrics are preserved verbatim in
`logs/pretrain_metrics_aborted_2.2Bplan.jsonl` (44 rows) and are **not** mixed into the
Phase-1 report.

