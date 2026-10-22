# Phase-3 experiment harness — GRPO curriculum, SFT-vs-GRPO driver, token accounting

Status: **complete and CPU-validated. The GPU was never touched.**
Written while the ~26-hour Phase-1 pretraining job owned the GPU. Every command
below ran with `--device cpu` on the 63.91 M model (or with no model at all);
free VRAM was only ever read, never allocated.

Every number in this report was measured on this machine from an artifact on
disk. Where a number cannot exist yet it is written `n/a (requires real
checkpoint)` and the reason is stated. Nothing is estimated silently.

---

## 0. What was built

| Path | What it is |
|---|---|
| `custom/build_grpo_curriculum.py` | derives the difficulty-ordered curriculum from `data/grpo/train_grpo.jsonl` (no hand-labelling) |
| `data/grpo/stage1_easy.jsonl` | 150 easiest tasks — where GRPO must first get non-zero advantage |
| `data/grpo/stage2_medium.jsonl` | easy + medium, 350 tasks |
| `data/grpo/stage3_full.jsonl` | all 600 tasks, same difficulty order |
| `data/grpo/curriculum_stats.json` | the full difficulty model, histogram, per-stage composition, sha256s, integrity checks |
| `tools/grpo_healthcheck.py` | diagnostic: is reward rising / how sparse is the signal / is the policy collapsing → verdict |
| `tools/_selftest_healthcheck.py` | 13-case self-test covering every verdict branch + 4 malformed-file cases |
| `tools/compare_sft_grpo.py` | Phase-3 headline driver: runs both evals, builds the auditable comparison + McNemar + token accounting + verdict |
| `tools/run_phase3.py` | staged launcher; **dry-run by default**, `--execute` to actually run |
| `reports/grpo_vs_sft_SMOKE.json` | smoke-weight comparison artifact (harness test, **not** a result) |
| `reports/grpo_vs_sft_SMOKE_ruleReward.json` | same, against the rule-reward smoke checkpoint, to prove the token-accounting path populates |
| `logs/grpo_easy_probe_healthcheck.json`, `logs/grpo_rule_smoke_healthcheck.json` | machine-readable healthcheck analyses of the two existing metric files |

`data/grpo/train_grpo.jsonl` is **untouched** (sha256 `57576f34…c37c1c`, the same
hash recorded in `data/grpo/train_grpo_stats.json`). `eval/` and
`src/minimind-master/` were not modified.

---

## 1. The curriculum

### 1.1 Why a curriculum, with the measured evidence

GRPO normalises advantages *inside* a group of `num_generations` samples of the
same prompt:

```
A_i = (r_i - mean(r)) / (std(r) + 1e-4)
```

If every sample in a group gets the same reward — all wrong, or all right —
`std(r) = 0`, every advantage is 0, and the group contributes **no gradient**.
This project measured both regimes on the same model:

| artifact | task set | steps | reward | adv_std | degenerate_group_frac |
|---|---|---|---|---|---|
| `logs/grpo_rule_smoke.jsonl` | full 600 tasks | 3 | **0.000** (all steps) | **0.000** | **1.0** (every step) |
| `logs/grpo_easy_probe.jsonl` | 200 easy `a+b`, a,b∈[0,4] | 30 | 0.053 → 0.156 | **0.890 mean** | 0.208 mean |

Same model, same reward, same trainer. On the 600-task set the groups were
uniformly wrong and nothing could be learned; on an easy set the groups were
mixed and the reward rose. That is the entire justification for starting easy
and widening: **the curriculum is not a preference, it is a prerequisite for
the gradient to exist.**

### 1.2 The difficulty score (computed, not hand-labelled)

Each task gets an integer `difficulty` = sum of the applicable components. Every
component is derived from the task content itself (prompt text + answer), so a
re-rolled dataset classifies the same way:

| component | weight | why |
|---|---|---|
| answer magnitude | `floor(log2(|answer|+1)) * 2` | ints 0–25 are one token; 2-digit values above 25 split inconsistently (env_report §8), so a larger answer is a real step up |
| negative answer | +3 | `'-'` is its own token (id 48) and the prompt must condition on "负号" |
| extra arithmetic ops | +2 each beyond the first | counted from the prompt, not from a family label |
| negation / polarity | +2 | prompt contains 负 / 小于 / 最小 / 不是 … |
| JSON output | +4 | must emit a well-formed object, not a bare integer |
| multi-step | +3 | two or more separate computation clauses |
| letter choice | +1 | answer is A/B/C among distractors |
| long prompt | +1 | prompt > 80 chars |
| family prior | arithmetic 0, format +1, logic +1, multistep +2 | mirrors the generator weights in `build_rl_data.py` (arithmetic 1.0 > format/logic 0.75 > multistep 0.5) |

Ties are broken deterministically: `(difficulty, family_rank, |answer|, id)`.
**No RNG is used**, so the build is byte-reproducible — verified: two consecutive
runs produced identical sha256 for all three stage files.

### 1.3 The resulting distribution (all 600 tasks)

`difficulty` 0–19, mean 7.64.

```
 0-1    82  ############################################################
 2-3    65  ############################################################
 4-5    34  ##################################
 6-7    76  ############################################################
 8-9   181  ############################################################
10-11   65  ############################################################
12-13   24  ########################
14-15   39  #######################################
16-17   32  ################################
18-19    2  ##
```

The mass at 8–9 is the `max_of_three` + `add_sub_neg` band; the thin tail at
14–19 is the multi-step arithmetic (`two_step_mul`, `mul_then_sub`,
`sum_first_n`), which is the genuinely hardest material and is deliberately kept
out of stages 1–2.

Mean difficulty per prompt shape (this is what makes the ordering sane — the
ranking matches intuition, but it was computed first):

| shape | n | difficulty min | max | mean |
|---|---|---|---|---|
| `cmp_yesno` (a > b ?) | 46 | 1 | 1 | 1.00 |
| `parity` (odd/even) | 30 | 1 | 1 | 1.00 |
| `choice_sum` (A/B/C of a+b) | 51 | 2 | 2 | 2.00 |
| `order_three` (A>B>C) | 6 | 4 | 4 | 4.00 |
| `add_sub_single` | 121 | 0 | 10 | 5.42 |
| `mul_single` | 37 | 4 | 12 | 8.32 |
| `max_of_three` | 114 | 7 | 9 | 8.79 |
| `add_sub_neg` | 42 | 7 | 13 | 8.86 |
| `json_sum` | 53 | 7 | 13 | 10.47 |
| `sum_first_n` | 11 | 9 | 17 | 13.36 |
| `mul_then_sub` | 33 | 9 | 17 | 14.27 |
| `two_step_mul` ((a+b)·c) | 56 | 11 | 19 | 15.61 |

### 1.4 The three stages

| stage | tasks | difficulty range | mean | negation | JSON | multi-step | families |
|---|---|---|---|---|---|---|---|
| **stage1_easy** | **150** | 0–4 | 1.45 | **0** | **0** | **0** | arithmetic 23, format 97, logic 30 |
| **stage2_medium** | **350** | 0–9 | 4.63 | 32 | 16 | 0 | arithmetic 172, format 113, logic 65 |
| **stage3_full** | **600** | 0–19 | 7.64 | 42 | 53 | 100 | arithmetic 200, format 150, logic 150, multistep 100 |

Stage 1 shape mix: `add_sub_single` 23, `choice_sum` 51, `cmp_yesno` 46,
`parity` 30 — four shapes, three families, difficulty 0/1/2/4 only (6 + 76 + 65
+ 3 tasks). It contains **no negation, no JSON, no multi-step**, which is exactly
the "single-step, small answers, no negation" brief.

`stage1_easy ⊂ stage2_medium ⊂ stage3_full` is **asserted** at build time, so a
staged run never sees a harder task before an easier one and can resume from the
previous stage's weights without revisiting anything out of order.

**Integrity (all enforced by the builder, which aborts on failure):**

* every one of the 600 answers passes its own `verifiable_reward.check()` — the
  same contract the GRPO reward and `eval/grader.py` use → **0 failures**;
* no curriculum prompt appears in the frozen held-out set → **0 leaks**
  (held-out sha256 `6d873f2a…b88df`, matching `eval/lock_hashes.json`);
* stage1 contains no hard task (neg/json/multi) → asserted, not assumed.

---

## 2. Healthcheck verdicts on the two existing metric files

`tools/grpo_healthcheck.py` reads a per-step GRPO metrics JSONL and answers the
five spec questions with real numbers, then prints one verdict line. It always
exits 0 (it is a diagnostic, not a gate).

Verdict rules, in order (all thresholds are named constants in the source):

1. `n_steps < --min-steps` (default 20) → **INSUFFICIENT_DATA**
2. entropy collapsing (tail/head < 0.1, or any step below the trainer's own
   0.05 floor) **or** mean `kl_mean` > 0.5 → **DEGENERATE**
3. ≥ 50 % of steps have `adv_std < 1e-3` → **SPARSE**
4. `reward_mean_20` falls by ≥ 0.02 with a negative slope → **DEGENERATE**
5. `reward_mean_20` gain ≥ 0.02 **and** OLS slope > 0 → **LEARNING**
6. otherwise (flat reward) → **SPARSE**

### 2.1 `logs/grpo_easy_probe.jsonl` → **LEARNING**

30 steps, 200 easy `a+b` tasks, group 8.

| question | measured answer |
|---|---|
| reward rising? | `reward_mean_20` **0.0918 → 0.1329**, gain **+0.0411**; OLS slope **+0.00344/step**, r = 0.535. First→last step 0.0531 → 0.1562 |
| sparsity | **0/30 steps** had `adv_std < 1e-3` (**0.0 %**); mean `adv_std` **0.890**; mean `degenerate_group_frac` **0.208** (max 0.75) |
| KL / entropy / clip | mean `kl_mean` **−0.0371** (max |kl| 0.2439); mean entropy **2.897**, head 2.525 → tail 3.223 (tail/head **1.28**, *rising*, not collapsing); mean `clip_frac` **0.000** |
| `collapsed` flags | **12/30** flagged — **0** from the entropy arm, **12** from the **length arm**. `train_grpo_rule.py` defines `collapsed = (entropy < 0.05) OR (resp_len_mean ≥ 0.99·max_gen_len)`; the length arm means **"hit max_gen_len"**, NOT "the policy collapsed". Max `resp_len_mean` was 24.0, so these steps hit the generation cap. |
| acc rising? tracks reward? | acc **0.0703 → 0.0891** (gain +0.0188, slope +0.00170, rising); Pearson(acc, reward) = **0.938** → the two counters agree, as they must under a binary reward |

Verdict: `LEARNING — reward_mean_20 gain20 = 0.0411 >= 0.02 with OLS slope
0.003439/step > 0, and only 0.0% of steps are signal-free (< 50.0%)`.

### 2.2 `logs/grpo_rule_smoke.jsonl` → **INSUFFICIENT_DATA**

3 steps, full 600 tasks, group 8.

| question | measured answer |
|---|---|
| reward rising? | **0.000 → 0.000** at every step; gain 0.000; slope 0.000 (r undefined — zero variance) |
| sparsity | **3/3 steps (100 %)** had `adv_std < 1e-3`; mean `adv_std` **0.000**; mean `degenerate_group_frac` **1.000** |
| KL / entropy / clip | mean `kl_mean` **0.0328**; entropy **2.829** flat (tail/head 1.000); `clip_frac` **0.000** |
| `collapsed` flags | **1/3** — **0** entropy arm, **1** length arm (`resp_len_mean` 32.0 = the `max_gen_len` cap) |
| acc rising? | **0.000 → 0.000**; acc-vs-reward r undefined |

Verdict: `INSUFFICIENT_DATA — n_steps=3 < --min-steps=20: not enough steps to
distinguish a trend from noise`.

This is the correct verdict and it is also the **most important number in this
report**: 100 % of steps were signal-free, mean `adv_std` exactly 0.0. With only
3 steps the tool refuses to call it, but the sparsity statistics are already
unambiguous and are exactly why stage 1 exists. If a real 600-task run looks
like this at step 300, the verdict becomes **SPARSE** and the curriculum has
failed to reach the easy enough regime.

### 2.3 Self-test

`tools/_selftest_healthcheck.py` — **13/13 PASS**: all four verdicts
(LEARNING / SPARSE / DEGENERATE / INSUFFICIENT_DATA) on synthetic fixtures,
both real files, and four malformed inputs (missing file, empty file, garbage
file, partially-flushed last line) — every one exits 0 without crashing.

---

## 3. The smoke comparison (harness test, not a result)

Artifact: **`reports/grpo_vs_sft_SMOKE.json`**.
Command actually run (CPU, no CUDA context):

```powershell
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\tools\compare_sft_grpo.py `
  --sft  D:\llm-rl-lab\src\minimind-master\out\sft_smoke_768.pth `
  --grpo D:\llm-rl-lab\src\minimind-master\out\grpo_smoke_768.pth `
  --out D:\llm-rl-lab\reports\grpo_vs_sft_SMOKE.json `
  --hidden 768 --layers 8 --max-new-tokens 64 --device cpu
```

Both checkpoints were evaluated by `eval/run_eval.py` with `--prompt-mode chat`,
`--max-new-tokens 64`, `--seed 42`, greedy — identical conditions, and the
artifact records both prompt templates and refuses to compare if they differ
(both are `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`).
Measured CPU cost: SFT eval **46.0 s** wall (7.9 s load, 38.1 s generation) and
GRPO eval **45.8 s** (7.9 s load, 37.9 s generation); `peak_vram_mb` is `null`
on CPU, as expected.

| | SFT | GRPO | delta |
|---|---|---|---|
| **overall** | **0/24 (0.0 %)** | **0/24 (0.0 %)** | **0 tasks, 0.0 pp** |
| arithmetic | 0/8 | 0/8 | 0 |
| format | 0/6 | 0/6 | 0 |
| logic | 0/6 | 0/6 | 0 |
| multistep | 0/4 | 0/4 | 0 |

McNemar 2×2: both-right **0**, SFT-only-right **0**, GRPO-only-right **0**,
both-wrong **24**; discordant pairs **0** → exact p = **1.0000** (not
significant). With 0 discordant pairs no test can show an effect.

**Verdict (generated from the numbers):** *"No measurable difference: both
checkpoints score 0/24 and 0/24. A 0-vs-0 comparison cannot show a training
effect. Reason candidate: these are SMOKE checkpoints (a few hundred steps on a
tiny data slice), not trained models — their completions are expected to be
degenerate, and they are. Treat this artifact as a harness test, not a result."*
Rules fired: `both_zero`, `smoke_weights`.

Why 0/24 is the expected, correct outcome: the smoke weights emit `"\n"` for
all 64 new tokens on every task (`empty_completion`), exactly as documented for
the 96-token canonical run in `eval_harness.md` §"Smoke-run result". The harness
is proven; the checkpoints are not trained.

### 3.1 Token accounting, proven on a real artifact

`reports/grpo_vs_sft_SMOKE.json` deliberately reports the GRPO token count as
`n/a (requires real checkpoint: missing grpo)` — because `grpo_smoke_768.pth` was
produced by the *official* `train_grpo.py` (neural reward), which writes no
per-step metrics at all. The tool refuses to borrow another run's numbers.

To prove the accounting path fully populates, the same comparison was run
against the checkpoint whose metrics genuinely belong to it
(`reports/grpo_vs_sft_SMOKE_ruleReward.json`), with an **attribution check**:
the summary's `final_weight` must equal the evaluated checkpoint, otherwise the
metrics are rejected. Result:

```
grand total  2,362,454,016 tokens  (= pretrain 2,208,852,480 + SFT 153,600,000 + GRPO 1,536)
components   {"pretrain": 2208852480, "sft": 153600000, "grpo": 1536}
grpo sampled 1,536 tokens     grpo prompt 230 tokens (reported separately, NOT added)
attribution  checked=true ok=true summary_final_weight=...grpo_rule_smoke_768.pth
consumed so far  n/a (requires real checkpoint: missing sft)   <- correct: no SFT run exists yet
```

The `accounting_notes` list in the artifact explains each basis:

* **PRETRAIN** — `tokens_consumed_by_training = steps × tokens_per_step`, read
  from `logs/pretrain_metrics_summary.json` if it exists (it does not), else the
  last `cum_tokens` in `logs/pretrain_metrics.jsonl` (it is **live**, so this
  number grows while the pretraining job runs — the self-test derives its
  expectation from the file rather than hard-coding it). The corpus budget in
  `data/tokens/pretrain_meta.json` is `n_blocks × seq_len = 4,314,165 × 512 =
  2,208,852,480`, which matches `total_tokens_consumed` exactly; packed blocks
  contain no padding.
* **SFT** — `padded_total = out_records × max_seq_len = 200,000 × 768 =
  153,600,000` (MiniMind pads every sample; measured **56.88 % padding waste**),
  and `supervised = 66,236,849` (labels ≠ −100, assistant turns only). Both are
  the measured one-epoch budget from `data/sft/sft_meta.json`. A real SFT run's
  own trainer counters are used for "consumed so far" once it exists; today they
  do not, so that field is explicitly `n/a`.
* **GRPO** — `sampled_tokens` is the measured completion tokens
  (`completion_mask.sum()` accumulated); `prompt_tokens` is separate and **not**
  added, because the same prompts are re-encoded every step (compute, not data).
* **Grand total** counts padded SFT tokens (the model consumed them) and excludes
  GRPO prompt tokens and SFT label tokens (the latter are a subset of the padded
  total, so adding them would double count). When a component is missing the
  total is reported as **n/a with the missing component named**, never as a
  silently smaller number.

### 3.2 Self-test of the driver

`python tools/compare_sft_grpo.py --self-test` — **23/23 PASS**: 11 exact
McNemar reference values computed with `math.comb` (b=0→1.0, b=2→0.5, b=3→0.25,
b=5→0.0625, b=6→0.03125, b=10→2/1024, b=2/c=2→1.0, b=8/c=1→0.0390625, plus
symmetry), and 12 token-accounting identities including
`padded = records × max_seq_len`, `cum_tokens = step × tokens_per_step`, and the
grand-total sum.

---

## 4. How to run Phase 3 for real (once the GPU is free)

`tools/run_phase3.py` prints the exact commands; it **does not execute them
unless `--execute` is passed**. `--execute` additionally refuses to start if free
VRAM < 5000 MiB (the pretraining job reserves 5,288 MiB, so a concurrent launch
would fail or OOM).

### 4.1 Preview (safe at any time, creates no CUDA context)

```powershell
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\tools\run_phase3.py
```

### 4.2 The real run

```powershell
# 1. build the curriculum (already done; idempotent, byte-reproducible)
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\custom\build_grpo_curriculum.py

# 2. run the three stages + a healthcheck after each (stops on a trainer failure)
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\tools\run_phase3.py --execute `
  --from-weight full_sft --device cuda

# 3. compare the real SFT and GRPO checkpoints on the held-out benchmark
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\tools\compare_sft_grpo.py `
  --sft  D:\llm-rl-lab\src\minimind-master\out\full_sft_768.pth `
  --grpo D:\llm-rl-lab\src\minimind-master\out\grpo_stage3_full_768.pth `
  --out D:\llm-rl-lab\reports\grpo_vs_sft.json `
  --hidden 768 --layers 8 --max-new-tokens 64 --device cuda
```

The commands the launcher produces (verified by the dry run above; `--max_steps`
is **per epoch** in `train_grpo_rule.py`, so the launcher converts the target step
count into `--epochs` and passes one full epoch as `--max_steps` — this is the
one non-obvious detail):

```
stage1_easy    150 tasks, 37 iters/epoch, --epochs 9 -> 333 steps, from full_sft
stage2_medium  350 tasks, 87 iters/epoch, --epochs 7 -> 609 steps, from grpo_stage1_easy
stage3_full    600 tasks, 150 iters/epoch, --epochs 6 -> 900 steps, from grpo_stage2_medium
```

Full stage-1 command as printed:

```powershell
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\custom\train_grpo_rule.py `
  --data_path D:\llm-rl-lab\data\grpo\stage1_easy.jsonl --from_weight full_sft `
  --save_weight grpo_stage1_easy --save_dir D:\llm-rl-lab\src\minimind-master\out `
  --hidden_size 768 --num_hidden_layers 8 --batch_size 4 --num_generations 16 `
  --max_prompt_len 256 --max_gen_len 48 --epochs 9 --learning_rate 5e-07 --beta 0.01 `
  --epsilon 0.2 --temperature 0.9 --accumulation_steps 1 --grad_clip 1.0 `
  --lr_schedule cosine --reward_mode binary --max_steps 37 --save_interval 111 `
  --log_interval 10 --metrics_path D:\llm-rl-lab\logs\grpo_stage1_easy.jsonl `
  --seed 42 --device cuda
```

### 4.3 Recommended hyperparameters and their justification

| arg | value | justification from a measurement in this repo |
|---|---|---|
| `--num_generations` | **16** | Decode is latency-bound: 128 new tokens take ~2.4 s at *any* batch size (430 / 950 / 1,721 tok/s at group 8 / 16 / 32, env_report §7c). Group 16 nearly doubles the chance of a mixed group vs group 8 for ~0.3 s more per step. |
| `--max_gen_len` | **48** | Answers here are 1–3 tokens (all ints 0–25 are single tokens). 32 already suffices; 48 leaves room for `<|im_end|>` and avoids the every-sample-hits-the-cap mode observed in the smoke run (`resp_len_mean` 32.0 = cap). 64+ adds latency with no reward benefit. |
| `--beta` | **0.01** | Repo default and the value in every measured GRPO run here; mean `kl_mean` stayed small (−0.037 … 0.033) at 0.01, so there is no evidence to change it. |
| `--learning_rate` | **5e-7** | Repo GRPO default on an already-SFT policy; measured runs at 5e-7…1e-6 did not diverge. |
| `--batch_size` | **4** | Group 16 × batch 4 = 64 rollouts/step. VRAM measured for exactly this regime: group 8–16 at gen_len 32–64 ≈ **2,270 MiB** (env_report §9c/9d). |
| `--temperature` | **0.9** | Required for group diversity: a greedy rollout makes every sample of a group identical → `adv_std` always 0. |
| `--max_prompt_len` | **256** | Longest task prompt is far below this; repo default. |
| `--lr_schedule` | **cosine** | Decay over the stage's steps so later steps do not overshoot a small policy change. |
| `--seed` | **42** | Project-wide. |

### 4.4 VRAM / time budget

* **VRAM ~2,270 MiB peak reserved** (measured, `logs/grpo_easy_probe_summary.json`;
  the shorter smoke run measured 1,940 MiB in `logs/grpo_rule_smoke_summary.json`).
  The pretraining job reserves 5,288 MiB, so **GRPO must not run concurrently with
  pretraining** — 5,288 + 2,270 > 8,188 MiB. This is a scheduling constraint, not
  a tuning one.
* **Time**: measured 0.77 s/step at group 8 / gen_len 24 (30 steps in 23.2 s).
  Projected ~1.5 s/step at group 16 / gen_len 48: stage1 ≈ 8 min, stage2 ≈ 15 min,
  stage3 ≈ 23 min → **≈ 45 min total for 1,842 steps**, under 1 h. Phase-3 cost
  is dominated by SFT, not GRPO.

---

## 5. Risk register

Each risk is stated with the evidence already on disk and the concrete check
that would confirm or refute it.

### R1 — Sparse reward: GRPO cannot learn from all-wrong groups *(highest)*

**Evidence.** On the full 600-task set the smoke model produced
`reward_mean 0.000`, `adv_std 0.000`, `degenerate_group_frac 1.000` for **3/3
steps** (`logs/grpo_rule_smoke.jsonl`); the healthcheck reports
**100 % of steps signal-free**. On 200 easy tasks the same model produced
`adv_std ≈ 0.890` and rising reward. The advantage formula has `std(r)` in the
denominator, so a uniform group is *exactly* zero gradient — this is not a
tuning issue.

**Mitigation.** The staged curriculum (stage 1 is 4 easy shapes, difficulty
≤ 4, no negation/JSON/multi-step). Run the healthcheck after every stage: if
stage 1 still returns `SPARSE` (≥ 50 % of steps with `adv_std < 1e-3`), the
easy set is still too hard for this checkpoint — the next move is a smaller
stage 1 (or the `easy_add.jsonl` 200-task probe set that already worked), **not**
more steps on stage 3.

**Watch.** `frac_steps_no_signal` in `logs/grpo_stage*_healthcheck.json`.

### R2 — Capacity ceiling: 63.9 M parameters may be unable to do the task at all

**Evidence.** The held-out benchmark is a single-token answer-retrieval test on
24 tasks (eval_harness.md §"Tokenizer findings"), and the smoke checkpoints score
0/24 because they are untrained, not because the task is hard. But the hardest
curriculum material is 2-step arithmetic (`(a+b)·c`, difficulty 11–19) and a
63.91 M-parameter model trained on 66 M supervised SFT tokens may simply not have
the capacity. The `verdict` block in `compare_sft_grpo.py` selects
"capacity or task difficulty" **only when** the training advantage signal was
non-degenerate (`adv_std` not ≈ 0) yet the held-out score did not move — i.e. the
artifact distinguishes "cannot learn" from "not given a signal".

**Check.** If the real run shows rising training reward but a flat held-out
score, that is distribution over-fitting, not capacity; if training reward is
also flat while `adv_std > 0`, it is capacity.

### R3 — Reward hacking via format-only answers

**Evidence.** The reward is a pure checker on the *answer string*
(`custom/verifiable_reward.py`). For `format` tasks a model can satisfy the
checker — emit `{"sum": N}`, `YES`/`NO`, or a letter — without doing the
arithmetic, and the `shaped` mode even pays +0.1 for a parseable but wrong
answer. The `format` family is 150/600 of the training set. The driver has an
explicit rule for this: if **every** gain is in `format` and arithmetic/multistep
did not improve, the verdict fires `format_only_gain` and says the result is
format compliance, not reasoning.

**Watch.** `verdict.rules_fired` and the per-family deltas in
`reports/grpo_vs_sft.json`. `--reward_mode binary` (the default) avoids the
shaped-reward variant of this risk entirely.

### R4 — Group-size / VRAM tradeoff

**Evidence.** Samples are nearly free (latency-bound decode) but they are not
free in *memory*: group 8 at gen_len 256 cost 6,890 MiB in the official GRPO
path (env_report §9c), i.e. 84 % of the card. The rule-reward path is far
cheaper (1,940–2,270 MiB measured) because it has no 1.8 B reward model, but
raising group size and gen_len together would still walk toward the ceiling, and
the card is shared with a 5,288 MiB pretraining job until that job finishes.

**Mitigation.** group 16 / gen_len 48 / batch 4 ≈ 2,270 MiB measured regime.
`--execute` refuses to launch below 5,000 MiB free VRAM. If VRAM is tight,
reduce `--batch_size` before `--num_generations`: group size is what creates the
learning signal (R1), batch size is not.

### R5 — Benchmark too small to read a delta (secondary, but real)

**Evidence.** 24 tasks → 1 task = 4.2 pp overall; per family 1 task = 12.5 pp
(arithmetic) or 25 pp (multistep). The driver therefore reports an exact McNemar
p-value per task and the artifact's verdict fires `small_delta_not_significant`
when |delta| ≤ 2 and p ≥ 0.05. A "+1 task improvement" will be labelled as
within chance, not as a result.

**Note.** The file is frozen (`eval/lock_hashes.json`); changing it invalidates
every comparison. A bigger benchmark is a new version, not an edit.

---

## 6. Reproduction

```powershell
# curriculum (deterministic, no RNG; two runs give identical sha256)
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\custom\build_grpo_curriculum.py

# healthcheck self-test (13 cases) and the two real verdicts
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\tools\_selftest_healthcheck.py
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\tools\grpo_healthcheck.py `
  D:\llm-rl-lab\logs\grpo_easy_probe.jsonl
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\tools\grpo_healthcheck.py `
  D:\llm-rl-lab\logs\grpo_rule_smoke.jsonl

# comparison driver self-test (McNemar + token accounting)
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\tools\compare_sft_grpo.py --self-test

# the smoke artifact (CPU; measured 46.0 s + 45.8 s wall for the two 24-task evals)
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\tools\compare_sft_grpo.py `
  --sft  D:\llm-rl-lab\src\minimind-master\out\sft_smoke_768.pth `
  --grpo D:\llm-rl-lab\src\minimind-master\out\grpo_smoke_768.pth `
  --out D:\llm-rl-lab\reports\grpo_vs_sft_SMOKE.json `
  --hidden 768 --layers 8 --max-new-tokens 64 --device cpu

# launcher preview (executes nothing)
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\tools\run_phase3.py
```

**Not measured, and honestly labelled as such in every artifact:**

* the real SFT checkpoint (`full_sft_768.pth`) — the SFT run has not happened;
  `sft.measured_run` is `n/a (no SFT metrics at logs/sft_metrics.jsonl)`;
* the real GRPO checkpoints — `grpo_vs_sft.json` does not exist yet;
* GRPO token counts for `grpo_smoke_768.pth` — that checkpoint came from the
  official trainer, which writes no per-step metrics; the driver refuses to
  attribute `grpo_rule_smoke_summary.json` to it (attribution check by
  `final_weight`);
* "consumed so far" tokens — needs the SFT run to exist; reported `n/a` with the
  missing component named rather than as a smaller total.
