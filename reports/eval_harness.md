# Held-out evaluation harness (Phase-2 infrastructure)

Status: **complete and smoke-tested** on `sft_smoke_768.pth` (200-step weights).
Frozen benchmark v1: `minimind_heldout_v1`, seed 42.

## Files

| Path | Purpose | Frozen? |
|---|---|---|
| `eval/heldout_tasks.jsonl` | 24 hand-written held-out tasks (4 families) | **yes** (sha256 in lock) |
| `eval/grader.py` | pure-python grader, stdlib only, self-test | **yes** (sha256 in lock) |
| `eval/run_eval.py` | model loading + generation + grading + JSON report | no (harness code) |
| `eval/lock_hashes.json` | sha256 of the two frozen files + created_at | — |
| `reports/sft_baseline_smoke.json` | canonical smoke-run report | — |

Frozen hashes (`eval/lock_hashes.json`, created 2026-10-21T04:17:43+08:00):

```
heldout_tasks.jsonl  6d873f2ab30ac0b31017e1cdaf62f78fc54e8358fdaedf31b1e103e6235b88df  (5283 B, 24 records)
grader.py            c053ec4fbdc522ecc9296365f8276603745bbfdaae1a73839708c6aaf72fe521  (26170 B)
```

`run_eval.py` recomputes the tasks sha256 and compares it to the lock on every run
(`benchmark.lock_check.match` in the report; warning on stdout if it drifts).

## Task families (24 total, no dataset downloads, no LLM judge)

| family | count | checkers | content |
|---|---|---|---|
| `arithmetic` | 8 | `exact_int` | `a+b`, `a-b`, a,b∈0..9; answers 8, 5, 9, 11, -9, 0, 10, -5 |
| `format` | 6 | `exact_json` ×2, `choice` ×3, `regex` ×1 | `{"sum": 9}`, `{"diff": 3}`, YES/NO, A/B/C, `2+6=8` fixed-format equation |
| `logic` | 6 | `choice` ×3, `exact_int` ×3 | A>B>C ordering, A<B<C ordering, two sequences (+2, +2), parity (13 odd), max of three |
| `multistep` | 4 | `exact_int` | `(a+b)*c`, `(a-b)*c`, count 1..6, sum 1..5 |

Every record has `id`, `family`, `prompt` (Chinese, explicit about output format),
`answer` (canonical string), `checker`, checker params (`json_keys` / `choices` /
`regex` where applicable), and `"held_out": true`. All answers are unique,
deterministic, and programmatically verifiable.

Task ids: `arith_01..08`, `format_01..06`, `logic_01..06`, `multi_01..04`.

## Grader design (`eval/grader.py`)

Stdlib only (`json`, `re`, `unicodedata`, `ast`); importable without torch/numpy,
so the same grader can later be reused inside a GRPO reward function.

API: `normalize(text)`, `extract(task, completion) -> (canonical|None, reason)`,
`check(task, completion) -> (bool, reason)`, `grade(task, completion) -> dict`,
`validate_tasks(tasks) -> [errors]`.

`normalize()` pipeline:
1. remove leaked special tokens (`<|im_end|>`, `<|im_start|>`, `<|endoftext|>`, …);
2. strip thinking: keep only text after the **last** `</think>`; if a `<think>` is
   still unclosed, discard everything from it onward;
3. Unicode **NFKC** (full-width digits/punctuation → ASCII: `８`→`8`, `＋`→`+`, `？`→`?`);
4. strip whitespace and repeated trailing punctuation (`. 。 ！ ? , ：` etc.).

Checker semantics:
* `exact_int` — first try a whole-string integer after stripping wrappers
  (`答案是`, `结果：`, leading `=`…); else a Chinese numeral (0..99, optional 负);
  else take the **last integer** in the text. So `8`, ` 8 `, `8。`, `8<|im_end|>`,
  `答案是 8`, `3+5=8`, `８`, `八` all grade as 8; `7` or `答案是七` fail when 8 is expected.
  `- 9`, `−9`, `负九` grade as -9.
* `exact_json` — string-aware balanced-brace scan for the first `{...}`, `json.loads`
  with `ast.literal_eval` fallback, then compare **only** the required
  `json_keys` values (numeric strings coerced, extra keys ignored).
* `choice` — first occurrence of any allowed choice string wins (casefolded for
  ASCII, exact for Chinese). Per spec; compliant single-token answers are
  unambiguous.
* `regex` — `re.fullmatch` on the normalized text: fixed-format tasks require the
  whole answer to match (`2+6=8`, `2 + 6 = 8`, `2+6=8。` pass; `答案是 2+6=8` fails).

Self-test: `python eval/grader.py` → **95 checks, all pass** (v1.0.0). It covers
hand-written pass/fail examples for every checker, special-token/think/full-width
robustness, `grade()` shape, and — when `heldout_tasks.jsonl` is present — full
file validation plus a check that each task's canonical answer passes its own
checker. Exit code is non-zero on any failure.

## How to invoke

Greedy (canonical held-out run):

```powershell
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\eval\run_eval.py `
  --weight D:\llm-rl-lab\src\minimind-master\out\sft_smoke_768.pth `
  --hidden 768 --layers 8 --max-new-tokens 96 --device cuda `
  --out D:\llm-rl-lab\reports\sft_baseline_smoke.json
```

Sampled majority vote (`k` samples per task, temperature 0.7 / top_p 0.9 by default):

```powershell
... run_eval.py --weight <ckpt.pth> --sample-k 5 --max-new-tokens 128 --out <report.json>
```

Options: `--prompt-mode {chat,chat_think,raw}`, `--sample-temp`, `--sample-top-p`,
`--sample-top-k`, `--seed 42`, `--limit N` (debug), `--device {auto,cuda,cpu}`,
`--backend {hf,gguf}`. `--device auto` falls back to CPU when free VRAM < 4000 MiB.

`--backend gguf` is a clearly-marked stub and raises `NotImplementedError`
(planned Phase-3 llama.cpp path; grading/reporting is backend-agnostic).

Report JSON includes: per-task `{id, family, prompt, completion, correct, reason,
gen_tokens}` (+ `samples`/`votes` in sample-k mode), `per_family` accuracy,
`overall`, `weight` path + sha256, `model` config/dtype/param count,
`prompt_mode`, `prompt_template`, `prompt_example`, `generation` params,
`max_new_tokens`, `seed`, `timestamp`, `benchmark.tasks_sha256`,
`benchmark.lock_check`, `device`, `device_reason`, `wall_time_s`,
`load_time_s`, `generation_time_s`, `peak_vram_mb`.

## Smoke-run result — `sft_smoke_768.pth`

Command: greedy, `chat` prompt mode, `max_new_tokens=96`, seed 42, cuda fp16.
Report: `reports/sft_baseline_smoke.json`.

| metric | value |
|---|---|
| overall | **0 / 24 (0.00 %)** |
| arithmetic | 0 / 8 (0.0 %) |
| format | 0 / 6 (0.0 %) |
| logic | 0 / 6 (0.0 %) |
| multistep | 0 / 4 (0.0 %) |
| peak VRAM | **146.5 MiB** (model resident 134.3 MiB) |
| wall time | **67.4 s** total (load 7.0 s, generation 60.2 s); 47–67 s across runs due to shared-GPU contention |
| tasks sha256 match | true |

Expected: the smoke checkpoint is 200 steps, so greedy decoding degenerates to
repeating token 234 (`"\n"`) for all 96 new tokens on every task, and
`grader.normalize` correctly returns `""` → `empty_completion`. This is a
successful harness test, not a model result.

Non-empty paths were validated too: with `--sample-k 3 --sample-temp 1.2`
(`chat_think`), sampled completions contained text and the grader extracted
integer answers (e.g. `'2'`, graded `mismatch:got=2 expected=5`); `raw` and
`chat_think` prompt modes were exercised end-to-end. CPU fp16→fp32 path verified
on 1 task. GGUF stub verified to raise.

## Tokenizer / format findings (important for arithmetic)

MiniMind tokenizer is a custom 6400-vocab BPE (`model/tokenizer.json`).
`bos=<|im_start|>=1`, `eos=<|im_end|>=2`, `pad=<|endoftext|>=0`.

Digits and numbers:
* single ASCII digits `0`..`9` are single tokens, ids **51..60** (`'7'`→`[58]`,
  `'8'`→`[59]`, `'9'`→`[60]`, `'0'`→`[51]`);
* **every integer 0..25 is a single token**; so all benchmark answers (0..20) are
  one token each — the model must learn a direct prompt→answer-token mapping,
  not digit composition. Negative answers are two tokens: `'-9'` → `[48, 60]`
  (`'-'` id 48);
* 2-digit tokenization is **inconsistent** above 25: `123`→`[6318]` (one token),
  `45` is one token (part of `456`→`[4374, 57]`), but `26`→`[53,57]`,
  `46`→`[55,57]`, `99`→`[60,59]`. Single-token ints ≤100:
  `0-25, 28, 29, 30, 33, 40, 45, 48, 50, 60, 64, 67, 70, 75, 80, 90, 99, 100`;
  everything else splits into two digit tokens;
* `' 7'` (space + digit) is itself one token (1538), and `'答案是 7'` tokenizes as
  `[4881, 357, 1538]` — a leading space changes tokenization;
* full-width digits are **not** normalized by the tokenizer (`'７'`→`[292, 281]`),
  so a model is unlikely to emit them; `grader.normalize` still NFKC-folds them;
* `'7。'` → `[58, 302]` (`。` is its own token), `'7.'` → `[58, 49]`.

Consequence: the 24-task benchmark is a **single-token answer-retrieval** test.
That is fine for SFT-vs-GRPO comparison and for GRPO's programmatic reward, but
do not read it as multi-digit arithmetic capability. If larger answers are ever
added, they must account for the inconsistent 2-digit tokenization (a future
benchmark version, not an edit to the frozen v1).

Chat template:
* `apply_chat_template([...], add_generation_prompt=True, open_thinking=False)`
  ends with `...<|im_start|>assistant\n<think>\n\n</think>\n\n` (closed empty think
  block). This is the repo's canonical **non-thinking** eval format (same as
  `eval_llm.py --open_thinking 0`) and is the harness default (`--prompt-mode chat`).
* `open_thinking=True` ends with `...assistant\n<think>\n` (open block) →
  `--prompt-mode chat_think`. The repo's GRPO reward expects a `</think>` block
  (`train_grpo.py` splits on `</think>`), so **GRPO checkpoints likely need
  `chat_think`**; SFT non-thinking checkpoints need `chat`.
* `--prompt-mode raw` uses the brief's bare prompt
  `<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n` (no think block).

The grader strips thinking in all modes: it keeps text after the last
`</think>` (or drops an unclosed `<think>`), so the same tasks grade answers
regardless of prompt mode.

## Caveats before using this for the SFT-vs-GRPO comparison

1. **Prompt mode must match training format.** Default `chat` (empty think block)
   is right for non-thinking SFT; GRPO trained with thinking should be run with
   `--prompt-mode chat_think`. The report records the mode — never compare two
   reports with different `prompt_mode` / `prompt_template` / `max_new_tokens`
   as if they were the same benchmark. Recommended: run both checkpoints in their
   native mode, and (optional) cross-run both in both modes as a sensitivity check.
2. **`max_new_tokens=96` may truncate thinking models.** With `chat_think` a model
   may spend >96 tokens before `</think>` and never reach the answer, producing
   false negatives. Use 128–192 for GRPO/thinking checkpoints (recorded in report).
3. **Greedy is the canonical baseline** (`do_sample=False`, temperature 1.0,
   top_p 1.0, top_k 0 — MiniMind's custom `generate` divides by temperature, so
   it must stay > 0). `--sample-k N` enables seeded majority vote; sample-k and
   greedy numbers are not interchangeable.
4. **Small benchmark:** 24 tasks → per-family granularity is coarse (12.5 % for
   arithmetic, 16.7 % for format/logic, 25 % for multistep). Report raw counts
   alongside accuracy; don't over-interpret 1–2 task deltas. The file is frozen;
   changing it invalidates comparisons and requires a new `lock_hashes.json`.
5. **Grader leniency is intentional but bounded.** `exact_int` takes the last
   integer (so leaked expressions grade), `choice` takes the first allowed string
   (verbose answers can be penalised), `regex` requires the whole answer to match.
   These rules are documented in `grader.py` and covered by the self-test.
6. **Determinism:** seed 42, greedy decoding, one prompt per batch (no padding),
   so runs are reproducible on the same GPU/driver. Sampling uses global RNG in
   task order and is seeded, but GPU kernels can introduce minor nondeterminism.
7. **VRAM/time:** peak 146.5 MiB, ~47 s for 24 tasks — cheap enough to run
   alongside training and safe on the shared 8 GB GPU. `--device auto` falls back
   to CPU when free VRAM < 4000 MiB.
8. **Smoke weights are garbage** (200 steps each): 0 % is expected. Treat the
   harness as validated; only real trained checkpoints give meaningful numbers.
   If a real SFT checkpoint still emits only `"\n"`/`<|im_end|>`, suspect the
   prompt format or the fine-tune, not the harness.

## Integrity / reproduction

```powershell
# grader + task-file validation (95 checks)
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\eval\grader.py

# canonical baseline (must reproduce 0/24, 146.5 MiB peak; wall 47–67 s depending on GPU contention)
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\eval\run_eval.py `
  --weight D:\llm-rl-lab\src\minimind-master\out\sft_smoke_768.pth `
  --out D:\llm-rl-lab\reports\sft_baseline_smoke.json
```
