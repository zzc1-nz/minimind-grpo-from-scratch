# Phase-2 SFT pipeline — data mix, exact token accounting, CPU validation

Status: **complete and CPU-validated.** The GPU was never touched: the Phase-1 pretraining
job kept its 5.9 GB of VRAM the whole time (`nvidia-smi` free memory stayed at 2,018–2,054
MiB before and after every run below). No `.cuda()`, no `device='cuda'`, no CUDA context —
`train_full_sft_counted.py` skips all `torch.cuda.*` calls when `--device cpu` (the repo's
`setup_seed` would otherwise lazily create a CUDA context).

Every number in this report was measured on this machine; nothing is estimated.

---

## 1. Files

| Path | What it is |
|---|---|
| `custom/build_sft_seed.py` | builds the task-shaped "format seeding" set, `data/sft/sft_seed.jsonl` (seed 1337) |
| `custom/build_sft_data.py` | builds the SFT training mix, `data/sft/sft_train.jsonl` + `data/sft/sft_meta.json` |
| `custom/train_full_sft_counted.py` | thin derivative of `trainer/train_full_sft.py` with exact token accounting |
| `custom/verify_sft_token_accounting.py` | independent re-tokenisation check of the trainer's accounting |
| `data/sft/sft_seed.jsonl` / `sft_seed_meta.json` | 2,000 easy format-seeding tasks |
| `data/sft/sft_train.jsonl` / `sft_meta.json` | 200,000-record SFT mix + measured composition/tokens |
| `logs/sft_val/` | every raw artifact of the CPU validation (metrics JSONL, summary, batch dump, reconciliation report) |

The MiniMind repo was **not** modified, and `eval/` was not touched.

---

## 2. SFT dataset composition (`data/sft/sft_train.jsonl`)

Built with the defaults, seed 42, `--max_general 160000 --aux_repeat 20`:

| | records | share of records | labeled tokens (measured) | share of supervised tokens |
|---|---|---|---|---|
| general chat (`data/sft_t2t_mini.jsonl`) | **160,000** | 80.0 % | 66,675,244 | 99.39 % |
| task-shaped (`data/sft/sft_seed.jsonl` × 20) | **40,000** | **20.0 %** | 408,040 | **0.61 %** |
| **total** | **200,000** | 100 % | **67,083,284** | 100 % |

```
out_bytes        307,881,880  (293.6 MB)
out_records      200,000
out_sha256       f22f685712944e6361629c5304ada02eafbceaaeb5fc9f907cfc501c1a95b00f
BOM              none  (written by Python, encoding='utf-8', newline='\n')
heldout leak     0     (exact-substring scan over every user turn of the written file)
```

### Exact token totals (measured, every record rendered + tokenized)

| | full rendered prompt | at `max_seq_len 768` |
|---|---|---|
| prompt tokens (input_ids) | **80,977,303** | **80,094,979** |
| labeled tokens (`labels != -100`) | **67,083,284** | **66,236,849** |
| records truncated at 768 | — | **8,774** (4.4 %) |
| labeled fraction of prompt | 82.8 % | 82.7 % |
| padded tokens/epoch @ bs 16 | — | **153,600,000** (= 200,000 × 768) |

Per source, at 768: general 78,398,679 prompt / 65,828,809 labeled; task-shaped
1,696,300 prompt / 408,040 labeled (mean 42.4 prompt and 10.2 supervised tokens per
task-shaped record — that is the point: the *answer* is 1–3 tokens).

**Cross-check against the real trainer dataset.** The numbers above come from a
deterministic render (no `pre_processing_chat` system-prompt injection). Running the exact
dataset class the trainer uses (`SFTDatasetKeepThink`, max_length 768, preprocessing RNG
seeded 42, whole file in written order, 255 s) gives:

```
tokens_total        153,600,000   (= 200,000 records × 768, i.e. padding included)
tokens_supervised    66,202,998   (vs 66,236,849 deterministic -> 0.05 % lower, from
                                   system-prompt-induced truncation of 33,851 tokens)
zero-label records            1
task-shaped records      40,000   -> task_supervised   408,040
general records         160,000   -> general_supervised 65,794,958
task_fraction_by_record      0.20
task_fraction_by_supervised  0.0062
```

So the measured per-epoch numbers for the real run are: **153.6 M total (padded) tokens,
66.20 M supervised tokens, 40,000 task-shaped records (20.0 % of records, 0.62 % of
supervised tokens)**.

### Truncation cost at other seq lens (measured, so the choice is data-driven)

| max_seq_len | prompt tokens | labeled tokens | truncated records | padded/epoch @ bs16 |
|---|---|---|---|---|
| 512 | 71,579,782 | 58,086,663 | 72,270 (36.1 %) | 102,400,000 |
| 640 | 77,838,867 | 64,083,875 | 29,362 (14.7 %) | 128,000,000 |
| **768** | **80,094,979** | **66,236,849** | **8,774 (4.4 %)** | **153,600,000** |
| 896 | 80,680,335 | 66,790,813 | 2,062 (1.0 %) | 179,200,000 |

**768 is the right choice:** going 768 → 512 saves 33 % of padded compute but throws away
12.3 % of the supervised tokens (8.15 M) and truncates 36 % of conversations mid-answer;
896 recovers only 0.55 M more supervised tokens for 17 % more compute. This matches the
repo default (`--max_seq_len 768`) and is now measured rather than assumed.

### Seed set (`data/sft/sft_seed.jsonl`, seed 1337, 2,000 records)

Built by `custom/build_sft_seed.py`; **only** the deliberately easy shapes the Lead asked
for, each with many phrasings so the format is learned rather than the instances:

| shape | records | family | checker | instance space |
|---|---|---|---|---|
| `add` — `a + b`, a,b∈[0,4] | 400 | arithmetic | `exact_int` | 23 distinct (a,b) after leak exclusion, answers 0..8 |
| `cmp` — `a > b ?`, a,b∈[0,9], a≠b | 400 | logic | `choice` YES/NO | 45 instances |
| `json_sum` — `{"sum": a+b}` | 300 | format | `exact_json` | 23 instances, 9 distinct answers |
| `choice_sum` — A/B/C of `a+b` | 300 | format | `choice` | 23 instances |
| `parity` — 奇数/偶数, n∈[1,20] | 300 | logic | `label` | 14 numbers after leak exclusion |
| `max3` — max of three, n∈[0,20] | 300 | logic | `exact_int` | 25 triples |
| **total** | **2,000** | | | 6 distinct prompts shapes, 1,510 distinct prompt strings |

Excluded on purpose (this is GRPO's headroom): **all subtraction** (the held-out arithmetic
family is `3+5, 7-2, 9+0, 2+9, 0-9, 8-8, 6+4, 4-9`; two answers are negative, which the
Lead forbade), multi-step `(a+b)*c` / sum-of-first-n / count-to-n, sequences, A>B>C
ordering, and any `a,b > 9` arithmetic. Verified: 0 records contain a subtraction pattern
or mention multiplication.

**Held-out leak checks (all 0, the builder aborts otherwise):**

| check | result |
|---|---|
| exact prompt equality vs `eval/heldout_tasks.jsonl` | 0 |
| prompt substring containment (either direction) | 0 |
| arithmetic instance identity `(a,b)` for `+` — extracted from held-out prompts by regex | 0 |
| parity instance identity `n` for held-out 奇数/偶数 prompts | 0 |
| max-of-three triple identity for the held-out 最大 prompt | 0 |

Excluded held-out instances (so they cannot be memorised): add `(1,1),(2,3),(2,6),(2,9),
(3,5),(3,6),(6,4),(9,0)`, sub `(0,9),(4,9),(7,2),(7,4),(8,8)`, parity `3,5,6,8,9,13`,
max3 `(4,6,9)`. Note the excluded set is larger than strictly needed because
`2 + 6` appears in the held-out `format_06` regex task and `2 + 3` in `multi_01`; both are
kept out of the seed set.

Every one of the 2,000 seed answers is validated with `custom/verifiable_reward.check()`
before writing — the same contract GRPO's reward and `eval/grader.py` use (0 failures).

### Idempotency

Two full builds (`data/sft/sft_train.jsonl` and a rebuild into `logs/sft_val/`) produced the
**same sha256** `f22f6857…5b00f`; wall 92.1 s and 91.7 s, peak RSS 1,371 MB. The 200k-record
mix is therefore byte-reproducible from `seed=42`.

---

## 3. `custom/train_full_sft_counted.py`

Thin derivative of `trainer/train_full_sft.py`: MiniMind's model, `SFTDataset`,
AdamW, `get_lr`, GradScaler and `lm_checkpoint` resume code are kept. Added:

* **Exact token accounting from the tensors actually fed to the model**:
  `cum_tokens_total += input_ids.numel()`, `cum_tokens_supervised += (labels != -100).sum()`,
  per log step written to `--metrics_path` as
  `{step, loss, lr, cum_tokens_total, cum_tokens_supervised, tokens_per_sec, elapsed_sec,
  eta_hours, gnorm, peak_reserved_MiB}` plus `step_tokens_total`, `step_tokens_supervised`,
  `epoch`, `aux_loss`, `logits_loss` so the accounting is auditable without recomputing diffs.
* `--use_compile`, `--num_workers`, `--max_steps`, `--metrics_path`, `--seed`,
  `--checkpoint_dir`, `--dump_batches`.
* Checkpoints: `{save_weight}_{hidden}.pth` **plus** an immutable
  `{save_weight}_{hidden}_step{N}.pth`, plus MiniMind's own `_resume.pth`.
* Final `*_summary.json`: `total_tokens`, `supervised_tokens`, `wall_sec`, `peak_*_MiB`.
* `torch.amp.autocast('cuda', …)` / `torch.amp.GradScaler('cuda', …)` instead of the repo's
  deprecated `torch.cuda.amp.*` (no FutureWarnings on torch 2.6).
* **Prompt-format consistency (Lead's rule)**: `SFTDatasetKeepThink` is a byte-for-byte copy
  of the repo's `__getitem__` with one argument changed —
  `post_processing_chat(prompt, remove_empty_think=False)` — so the empty
  `<think>\n\n</think>\n\n` block is kept 100 % of the time instead of 20 %, matching the
  canonical `eval/run_eval.py --prompt-mode chat` prompt (`eval_llm.py --open_thinking 0`).
  The `<think>` tokens are assistant-turn tokens and are supervised (measured example:
  supervision for `10 是奇数还是偶数？…` = `'<think>\n\n</think>\n\n偶数<|im_end|>\n'`).
  The aux/seed records already carry the closed think block in the data file (the chat
  template always emits it for assistant turns), so no data change was needed for them.

  Verified against the frozen harness report: `reports/sft_baseline_smoke.json` records
  `prompt_mode: chat` and `prompt_template:
  '<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'`.
  The training sequence for a seed record is exactly that string plus
  `{answer}<|im_end|>\n`, with the labels covering `<think>\n\n</think>\n\n{answer}<|im_end|>\n`
  — i.e. **the eval prompt is a strict prefix of the training sequence**, and the
  conditional the eval asks for (`P(answer | … </think>\n\n)`) is inside the supervised
  span. Measured example, seed record `10 是奇数还是偶数？只回答"奇数"或"偶数"。`:
  supervised token text = `'<think>\n\n</think>\n\n偶数<|im_end|>\n'` (10 tokens).
  Without the Lead's change, 80 % of samples would have dropped the `<think>` block and the
  model would never have been trained at that conditioning context.

---

## 4. CPU validation (the GPU stayed free)

Command actually run (from `D:\llm-rl-lab\src\minimind-master\trainer`, 200-record slice of
the real `sft_train.jsonl`, `--device cpu --from_weight none`):

```powershell
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\custom\train_full_sft_counted.py `
  --device cpu --hidden_size 128 --num_hidden_layers 2 --batch_size 2 --max_seq_len 128 `
  --max_steps 8 --num_workers 0 --from_weight none --epochs 1 --log_interval 1 `
  --save_interval 4 --data_path D:\llm-rl-lab\logs\sft_val\sft_slice200.jsonl `
  --save_dir D:\llm-rl-lab\logs\sft_val\out --save_weight sft_cpuval `
  --checkpoint_dir D:\llm-rl-lab\logs\sft_val\checkpoints `
  --metrics_path D:\llm-rl-lab\logs\sft_val\cpu_val_metrics.jsonl `
  --dump_batches D:\llm-rl-lab\logs\sft_val\cpu_val_batches.json
```

Result: **8/8 steps, loss finite throughout (8.5937 → 8.7958, range 8.5937–8.7999), wall
0.6 s compute / 12.1 s process, 1.26 M params, peak VRAM 0 MiB (CPU), `gnorm` 2.4–13.5.**
Checkpoint code exercised (`sft_cpuval_128.pth` + `_step4.pth` + `_step8.pth` + `_resume.pth`),
summary JSON produced, metrics JSONL produced.

### Token reconciliation (independent, PASS)

`custom/verify_sft_token_accounting.py` re-reads the trainer's metrics, re-creates the same
`SFTDatasetKeepThink`, re-seeds `random/np/torch` in the trainer's order, replays the exact
index order the trainer dumped, and recomputes both counters from the returned tensors:

| step | `cum_tokens_total` trainer / manual | `cum_tokens_supervised` trainer / manual |
|---|---|---|
| 1 | 256 / 256 | 9 / 9 |
| 2 | 512 / 512 | 192 / 192 |
| 3 | 768 / 768 | 368 / 368 |
| 4 | 1,024 / 1,024 | 492 / 492 |
| 5 | 1,280 / 1,280 | 698 / 698 |
| 6 | 1,536 / 1,536 | 862 / 862 |
| 7 | 1,792 / 1,792 | 1,070 / 1,070 |
| 8 | 2,048 / 2,048 | 1,172 / 1,172 |

```
total_equals_steps_batch_seq     true   (2048 = 8 steps × 2 batch × 128 seq_len)
cum_total_strictly_increasing    true
cum_supervised_strictly_increasing true
supervised_less_than_total       true
all_losses_finite                true
per_step_match                   true   (0 mismatches)
summary_matches_metrics          true   (total 2,048 / supervised 1,172)
RECONCILIATION PASS
```

The dataset pads every sample to `max_seq_len`, so total tokens are exactly
`steps × batch × max_seq_len`; supervised tokens are strictly less (57.2 % here) and match a
manual per-record `(labels != -100)` sum exactly. A second run with
`--accumulation_steps 2 --max_seq_len 192 --log_interval 2 --use_compile 1` also reconciled
(3,072 / 1,979, `RECONCILIATION PASS`), which covers the log-interval-skipping and
accumulation paths.

---

## 5. Real run — recommended launch command

Measured throughput basis: the repo's own SFT smoke run (env_report §9b) did 200 steps at
bs 8 / seq 768 in 92.6 s ≈ **13,300 tok/s**; the Phase-1 pretrain job is running at
**22,994–23,050 tok/s** at bs 32 / seq 512 (read from `logs/pretrain_metrics.jsonl`, steps
2000–4100). SFT is slower per token because 48 % of every step is padding
(`200,000 × 768 = 153.6 M` padded vs 80.1 M real prompt tokens) and because the dataloader
re-tokenizes on the fly. **Assumption used for sizing: 15,000 tok/s** (between the two,
deliberately conservative; compile should push it toward 18–20 k, but that is unmeasured at
seq 768).

At 15,000 tok/s: `153.6 M / 15,000 ≈ 10,240 s ≈ 2.84 h` for one epoch — inside the 3–4 h
budget, and one epoch is the right call (the mix is 200 k records; a second epoch adds
~3 h for a second pass over the same 2,000 seed instances).
```powershell
cd D:\llm-rl-lab\src\minimind-master\trainer
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\custom\train_full_sft_counted.py `
  --data_path D:\llm-rl-lab\data\sft\sft_train.jsonl `
  --from_weight pretrain --save_dir D:\llm-rl-lab\src\minimind-master\out `
  --save_weight full_sft `
  --hidden_size 768 --num_hidden_layers 8 `
  --batch_size 16 --max_seq_len 768 --epochs 1 `
  --learning_rate 1e-5 --accumulation_steps 1 --grad_clip 1.0 `
  --use_compile 1 --num_workers 4 --seed 42 `
  --log_interval 50 --save_interval 1000 `
  --checkpoint_dir D:\llm-rl-lab\src\minimind-master\checkpoints `
  --metrics_path D:\llm-rl-lab\logs\sft_metrics.jsonl
```

* **Batch 16 / seq 768**: 12,560 steps, 12,288 padded tokens/step. VRAM at this shape is the
  one open risk: env_report §7a measured **7,184 MiB peak reserved** for bs 16 / seq 768
  **without** compile — too close to 8,188 MiB. With compile the measured peak at
  seq 512 / bs 32 (16,384 tok/step, i.e. *more* tokens per step) was only **5,468 MiB**, so
  bs 16 / seq 768 + compile should land ≈ 5–5.5 GB, but that exact shape is unmeasured.
  **Check the first 100 steps' `peak_reserved_MiB` in the metrics JSONL; if it is > 6,500 MiB,
  restart with `--batch_size 8`** (measured 4,008 MiB without compile at bs 8 / seq 768;
  same step count doubles to 25,120, same total tokens, ~3.2 h at 13.3 k tok/s).
* **lr 1e-5** = the repo default for full SFT (env_report §9b used exactly this).
* **`--epochs 1`** with `get_lr`'s cosine horizon over the whole 12,560 steps.
* `--save_interval 1000` → 12 checkpoints (`full_sft_768_step1000.pth` …) plus the
  rolling `full_sft_768.pth`, plus `checkpoints/full_sft_768_resume.pth` for
  `--from_resume 1`.
* **`--from_weight pretrain`** reads `out/pretrain_768.pth`, which the Phase-1 job is
  writing (it already exists, step 2,000 checkpoint at 05:31).

**Before launching:** wait until the pretrain job has finished. The SFT run needs ~5–6 GB
and the pretrain job holds 5.9 GB; they cannot coexist on this 8 GB card.

Afterwards, evaluate with the canonical prompt mode (matches training exactly):

```powershell
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\eval\run_eval.py `
  --weight D:\llm-rl-lab\src\minimind-master\out\full_sft_768.pth `
  --hidden 768 --layers 8 --max-new-tokens 96 --device cuda --prompt-mode chat `
  --out D:\llm-rl-lab\reports\sft_baseline_full.json
```

---

## 6. Things that could break Phase 2 (read before launching)

1. **The 20 % task-shaped share is 0.61 % of supervised tokens.** Records are 20 % but the
   seed answers are 1–3 tokens, so the format signal is carried by *steps that contain a
   task-shaped record* (≈ 20 % of batch slots), not by token volume. That is enough for the
   model to learn "answer directly" but the arithmetic mapping is only weakly constrained —
   which is exactly the intended "knows the format, not the reasoning" regime. If the
   held-out accuracy after SFT is still 0, check the *format* (is the completion a bare
   digit?) before concluding the model cannot do the task; the eval report records the raw
   completions.
2. **VRAM at bs 16 / seq 768 + compile is unmeasured** (7,184 MiB measured *without* compile
   is too close to the 8,188 MiB card). Fallback: `--batch_size 8` (4,008 MiB measured).
3. **Do not run SFT concurrently with the pretrain job.** 5.9 GB + ~5 GB does not fit.
4. **The 2,000 seed records repeat 20× each**, so the model will see the same 2,000 prompt
   strings 20 times per epoch. Over-fitting to those instances is possible; that is
   acceptable for a format primer but means SFT-only held-out accuracy is not evidence about
   generalisation — the GRPO phase is where the reasoning range is supposed to come from.
   Held-out instances are excluded, so nothing can be memorised directly.
5. **`pre_processing_chat` still injects a random system prompt 20 % of the time** (repo
   behaviour, kept unchanged). It only prepends masked tokens, so it does not change
   supervised counts except through truncation at 768 (measured: 66.24 M supervised at 768
   with truncation vs 67.08 M untruncated).
6. **Padding waste is 48 %** (153.6 M padded vs 80.1 M real prompt tokens). MiniMind's SFT
   dataset pads every sample; fixing it needs sequence packing, which would be a larger
   deviation from the repo and is not attempted here.
7. `data/sft/sft_t2t_mini.jsonl` has **one malformed line** (record #905,640 is truncated in
   the downloaded file). The builder skips it and records `general_malformed_lines_skipped: 1`
   in the meta; it is not in the mix.
8. The **older 120-task `data/sft/sft_aux.jsonl`** mix is still reproducible
   (`--aux_path …sft_aux.jsonl --aux_repeat 8 --max_general 200000`) but it was 0.48 % of
   records and is superseded by the seed set.

---

## 7. Reproduction

```powershell
# 1. seed set (2,000 easy format tasks, seed 1337, leak-checked)
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\custom\build_sft_seed.py

# 2. training mix (160,000 general + 2,000x20 seed = 200,000 records, seed 42)
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\custom\build_sft_data.py

# 3. CPU validation (no CUDA; tiny model, 200-record slice, 8 steps)
#    ... see §4 for the full command ...
# 4. independent token reconciliation
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\custom\verify_sft_token_accounting.py `
  --metrics D:\llm-rl-lab\logs\sft_val\cpu_val_metrics.jsonl `
  --batches D:\llm-rl-lab\logs\sft_val\cpu_val_batches.json `
  --data_path D:\llm-rl-lab\logs\sft_val\sft_slice200.jsonl `
  --max_seq_len 128 --hidden_size 128 --num_hidden_layers 2
```
