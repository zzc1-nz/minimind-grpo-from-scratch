# Reporting & plotting tooling (Phase 1 / Phase 3)

**Status: complete and validated.** Three deliverable tools (`plot_curves.py`,
`make_tables.py`, `gen_readme_sections.py`), one shared stdlib-only library
(`metrics_lib.py`), one regression suite (`selfcheck_tools.py` — **56/56 checks pass
from a clean checkout**), two figures under `D:\llm-rl-lab\figures\`. Pure CPU: **no
CUDA call, no torch import, no model load** anywhere in this tooling (verified:
`matplotlib` does not pull `torch` into `sys.modules`). Everything runs while the 28 h
pretrain job owns the GPU.

Every number printed by these tools is read from a file at run time. Missing inputs
produce explicit `n/a (file not found: ...)` / `TODO: ...` text — never a fabricated
value, never a silent zero.

---

## 1. Files

| Path | What it is |
|---|---|
| `tools/metrics_lib.py` | shared stdlib-only helpers: tolerant JSONL reader, number/markdown formatting, series extraction |
| `tools/plot_curves.py` | metrics JSONL → PNG (pretrain 2-panel, GRPO 3-panel) |
| `tools/make_tables.py` | summary JSONs + eval reports → markdown tables |
| `tools/gen_readme_sections.py` | the data-driven README fragments (metrics / eval / tokens / figures) |
| `tools/selfcheck_tools.py` | regression suite covering every documented edge case |
| `figures/pretrain_curve_partial.png` | 1600x900, ~115 KB — Phase-1 partial run (regenerated as the run grows) |
| `figures/grpo_probe_curves.png` | 1600x900, 188,213 B — GRPO rule-reward probe |
| `reports/tables_current.md` | snapshot of `make_tables.py` output |
| `reports/readme_sections_current.md` | snapshot of `gen_readme_sections.py` output |

All tools import `metrics_lib` from their own directory, so they work from any CWD:

```powershell
$py = 'D:\llm-rl-lab\.venv\Scripts\python.exe'
```

---

## 2. `plot_curves.py`

```powershell
& $py D:\llm-rl-lab\tools\plot_curves.py `
    --metrics <jsonl> --out <png> [--kind pretrain|grpo|auto] [--title T]
```

Options: `--note` (repeatable subtitle line), `--total-steps N` (enables the
`PARTIAL: N of M planned` banner), `--dpi` (150), `--width/--height` (1600/900).

**`pretrain`** — two panels:

* (a) loss vs **cumulative tokens in B** (`cum_tokens / 1e9`). Light raw trace +
  a trailing moving average whose window is `n/200` (so a 134,817-point run still
  gets ~200 smoothed points). The x tick step is a "nice" 1/2/5×10ᵏ value and the
  label decimals are derived from that step, so the labels are exact at both ends
  of the run (`0.0005` for an early partial run, `0.5` for a full run).
* (b) tokens/sec vs step, with the **peak `peak_reserved_MiB` annotated** as a
  dotted reference line. `last N tok/s` is labelled in the corner.

**`grpo`** — three panels: (a) `reward_mean` + `reward_mean_20`, (b) `acc` + `adv_std`,
(c) `kl_mean` + `entropy`. Every step with `collapsed == true` is shaded across all
three panels, with merged spans (adjacent steps become one band) and a legend entry
`collapsed (N step(s))`. Panel (c) switches to a twin y axis when the two series
differ by >4×, because `kl_mean ≈ -0.01` against `entropy ≈ 2.9` would otherwise
draw KL as a flat line on zero.

**`auto`** — GRPO if the record has `reward_mean`+`kl_mean`+`acc`; pretrain if it has
`loss`+`tokens_per_sec`+`cum_tokens`. The chosen kind is always printed.

**Style** — white background, major+minor grid, top/right spines off, 1600x900 at
dpi 150, fixed colours, no seaborn. `matplotlib.use("Agg")` is set before `pyplot`,
so no window is ever opened. PNG metadata is fixed (`Creation Time: None`), so
re-running on a frozen input is **byte-identical** (verified by SHA-256).

### Partial / broken input handling

`read_jsonl()` returns both the records and a `stats` dict, and `plot_curves` prints
warnings to **stderr** (not only into the figure subtitle), because a caller piping
into a build log must see that data was dropped:

| Input | Behaviour |
|---|---|
| trailing line with no `\n` that does not parse | counted `skipped_partial`, dropped, **stderr warning**, figure still drawn |
| trailing line with no `\n` that *does* parse | kept (a finished writer may omit the final newline) |
| unparsable line in the middle | counted `skipped_invalid` with line numbers, dropped, warning, figure drawn |
| UTF-8 BOM / CRLF (PowerShell `Set-Content`) | parsed correctly |
| empty file (0 bytes) | `no usable records ...`, exit 0, **no PNG written** |
| fewer than 2 usable points | `only N usable '<key>' point(s) ... (need >= 2)`, exit 0, **no PNG written** |
| missing file | exit **2** (a real error), no traceback |
| a metric missing at some steps | plotted as a **gap** (`None`), never as 0.0 |
| existing stale PNG at `--out` | left untouched, and stderr says it is stale |

Refusing to write a PNG for empty/1-point input is deliberate: a blank or
single-dot figure in a README is worse than an explicit message.

**Downsampling / RAM.** A 134,817-step file (32.8 MB) renders in **3.0 s** at
**357 MB peak working set** and produces a **122 KB** PNG — not a 200 MB one. Points
are decimated to ≤4,000 for the raw trace and ≤1,500 for the moving average, always
keeping the final point so the endpoint stays visible.

---

## 3. `make_tables.py`

```powershell
& $py D:\llm-rl-lab\tools\make_tables.py `
    [--out tables.md] `
    [--eval REPORT1.json REPORT2.json ...] `
    [--summary EXTRA_summary.json ...] `
    [--sft-summary logs/sft_summary.json] [--grpo-summary ...] `
    [--section runs|eval|tokens ...]
```

### 3a. Training runs

Auto-discovers `logs/*_summary.json`, sorts by phase, and reads each row from three
sources, all named in the `source` column:

* steps / tokens / wall time / peak VRAM ← the summary JSON;
* the run config ← the matching `logs/vram_train_*.json` (matched by `save_weight`
  in the recorded argv; latest mtime wins);
* final loss ← the last record of the sibling `logs/*.jsonl`.

The canonical Phase-1 production run gets a **dedicated row even though its summary
does not exist yet**: while the run is in progress the row is filled from the live
JSONL and labelled `(live JSONL, N logged points)`, `N of 134,817 planned (x %
complete)`. Once the run ends and the summary appears, the same row switches to the
summary values automatically. When neither exists, the row shows
`n/a (file not found: logs/pretrain_metrics_summary.json)`.

`config` is `n/a (no run argv recorded)` when no `vram_train_*.json` matches — this
is the honest answer, not a guess. GRPO rows are footnoted: their `tokens` column is
`cum_sampled_tokens` and their `final loss` is the last PPO policy loss, not a
cross-entropy loss.

### 3b. Eval comparison

```powershell
& $py tools\make_tables.py --section eval --eval reports\a.json reports\b.json
```

Prints overall + per-family accuracy side by side as `0.0% (0/24)`, plus a delta
column when exactly two reports are given, plus a provenance table (benchmark,
tasks sha256, prompt_mode, max_new_tokens, decoding, device, wall time, peak VRAM,
report path).

**Comparability guard.** If `prompt_mode`, `max_new_tokens`, `benchmark.tasks_sha256`
or `generation.mode` differ between the reports, the table is preceded by a loud
`> [!WARNING]` block naming both values for each mismatched field — and **the table
is still printed**, because a mismatch is itself a finding. Example from the
self-test:

```
> [!WARNING]
> **These results are not directly comparable.**
> - `prompt_mode` differs between reports: `sft_smoke_768 [chat, 96 tok]` = chat,
>   `grpo_easy_probe_768 [chat_think, 128 tok]` = chat_think. Accuracy is NOT
>   comparable across prompt formats.
> - `max_new_tokens` differs between reports: ... = 96, ... = 128. A shorter budget
>   can truncate thinking models and produce false negatives.
```

Identical labels are disambiguated (`... #2`) so no two columns look the same.
Unreadable reports are listed as `- <path>: <reason>`, and with fewer than 2 readable
reports the section says `n/a (need >= 2 readable eval reports, got N)`.

### 3c. Token accounting

Sums pretrain / SFT / GRPO tokens from the summary files, with an explicit
`TOTAL (known components)` row that names what it excludes. **GRPO uses
`cum_sampled_tokens`**; `cum_prompt_tokens` is shown in the detail column but not
added to the total (it is input, not sampled output). A missing SFT summary prints
`n/a (no SFT summary file supplied: pass --sft-summary <path>)`.

---

## 4. `gen_readme_sections.py`

```powershell
& $py D:\llm-rl-lab\tools\gen_readme_sections.py `
    [--out readme_sections.md] [--eval a.json b.json] [--section metrics|eval|tokens|figures]
```

Emits **only** the data-driven fragments, each wrapped in paste-safe markers:

```
<!-- BEGIN GENERATED: metrics_table (tools/gen_readme_sections.py) -->
...
<!-- END GENERATED: metrics_table -->
```

* `metrics_table` — status (`partial -- N of 134,817 steps (x%)`), logged points,
  cumulative tokens, tokens/step, last and recent-mean throughput, peak VRAM,
  first/last loss with step numbers, elapsed, ETA; plus a GRPO probe table
  (steps, reward first→last, final acc/adv_std, collapsed count, peak VRAM).
* `eval_table` — delegates to `make_tables.table_eval_comparison` so there is exactly
  one implementation of the comparison + warning logic. With <2 reports it prints a
  `TODO` and lists the reports that *are* on disk (path, prompt_mode,
  max_new_tokens, overall accuracy).
* `token_accounting` — delegates to `make_tables.table_token_accounting`.
* `figures` — markdown image references for each expected PNG with a real
  `WxH px / N bytes` caption (read from the PNG IHDR header) and the exact
  regeneration command. A figure that does not exist yet gets a `TODO` marker and is
  listed under `_Missing figures (TODO)_`.

Nothing is invented: while `logs/pretrain_metrics_summary.json` is absent, the run
wall time row reads `TODO: Phase-1 summary is only written when the run ends --
missing logs/pretrain_metrics_summary.json`, and `pretrain_curve_full.png` is a TODO
until the run finishes. `gen_readme_sections.py` is idempotent (verified).

---

## 5. Validation — every edge case actually tested

`tools\selfcheck_tools.py` builds all fixtures itself under `tools\_selftest\`
(delete it any time; re-running recreates it). **56/56 checks pass from a clean
checkout.** Run it with:

```powershell
& $py D:\llm-rl-lab\tools\selfcheck_tools.py        # add -v for the full list
```

| # | Edge case | Asserted outcome |
|---|---|---|
| 1 | **truncated final line** (file still being appended to) | exit 0, PNG written, `partial` warning on stderr |
| 2 | **empty file** (0 bytes) | exit 0, `no usable records`, no traceback, **no PNG** |
| 3 | **1 point only** | exit 0, `need >= 2` message, no traceback, **no PNG** |
| 4 | missing metrics file | exit 2, no traceback |
| 5 | **UTF-8 BOM + CRLF** (PowerShell-written) | parsed, figure written |
| 6 | unparsable **middle** line | skipped with line number, figure still written |
| 7 | a metric missing at one step | figure written, series has a gap (no 0.0) |
| 8 | 2,697-point run (the suite) and a full 134,817-point / 32.8 MB run (measured separately) | downsampled, PNG < 1 MB (122 KB actual), 3.0 s, 357 MB peak working set |
| 9 | `auto` kind inference | grpo and pretrain both inferred correctly |
| 10 | GRPO `collapsed` marking | all 12 collapsed steps found and reported |
| 11 | **mismatched `prompt_mode` / `max_new_tokens`** | `[!WARNING]` block names both values, table still printed, delta column present |
| 12 | missing eval reports | exit 0, explicit `n/a ... file not found` cells |
| 13 | only 1 eval report | exit 0, `need >= 2` |
| 14 | same report passed twice | labels disambiguated `#2`, no ambiguous columns |
| 15 | eval report lacking `overall`/`per_family` | `n/a` cells, `n/a (absent)` in the warning, no crash |
| 16 | missing training summary | explicit `n/a` row, no crash |
| 17 | malformed summary (`[1,2,3]`) | exit 0, no traceback, reported as `unexpected JSON type` |
| 18 | GRPO token accounting | uses `cum_sampled_tokens`, prompt tokens listed separately |
| 19 | idempotency | `make_tables` and `gen_readme_sections` byte-identical across runs |
| 20 | PNG idempotency | byte-identical SHA-256 on a frozen input |
| 21 | figures | both exist, >10 KB, valid PNG, 1600x900, open in PIL |

---

## 6. Generated figures

| Figure | Size | Source | Regenerate |
|---|---|---|---|
| `figures/pretrain_curve_partial.png` | 1600x900, ~115 KB | `logs/pretrain_metrics.jsonl` (live, still growing) | see below |
| `figures/grpo_probe_curves.png` | 1600x900, 188,213 B | `logs/grpo_easy_probe.jsonl` (30 steps) | see below |

The partial figure's byte size changes on every regeneration because the underlying
run is still appending records; that is expected. On a **frozen** input the output is
byte-identical (verified by SHA-256 in the self-check).

```powershell
& $py tools\plot_curves.py --metrics logs\pretrain_metrics.jsonl `
    --out figures\pretrain_curve_partial.png --kind pretrain --total-steps 134817 `
    --title "Phase-1 pretrain -- partial run (loss vs tokens, throughput)"

& $py tools\plot_curves.py --metrics logs\grpo_easy_probe.jsonl `
    --out figures\grpo_probe_curves.png --kind grpo --total-steps 30 `
    --title "GRPO rule-reward probe (30 steps, easy add tasks)"
```

The partial figure carries `PARTIAL: N steps of 134,817 planned` in its subtitle, so
it cannot be mistaken for a finished run. Re-run the same command at the end of the
28 h job to produce the full-run figure (`figures/pretrain_curve_full.png`).

---

## 7. Schema findings (documented fields vs. what the files actually contain)

Everything below was observed in the real files, not assumed. None of it broke the
tools, but the lead should know about it for the reports.

1. **`pretrain_metrics.jsonl` rows are 100 steps apart, not 1.** The live Phase-1
   process was launched with `--log_interval 100` (confirmed from its command line,
   and the observed gaps in the file are exactly 100 steps), so the file will hold
   ~1,349 rows for a 134,817-step run, not 134k. The `PARTIAL: N of M` banner reports
   the **last logged step**, which is always a multiple of 100 and lags the true
   current step by up to 99. Note this differs from the `--log_interval 50` default
   in `custom/run_phase1_pretrain.py` — the job that is actually running did not use
   that launcher, so trust the file, not the launcher.
2. **`tokens_per_sec` is a run-average, not an instantaneous rate.** It is
   `run_step * tokens_per_step / elapsed`, so it converges upward and is not
   comparable to the instantaneous tok/s figures in `reports/env_report.md`
   (e.g. 22,184). Do not quote both in one sentence as if they were the same metric.
3. **`peak_reserved_MiB` drifts upward over a run** (5,288 → 5,292 observed) because
   it is a high-water mark read fresh at each log step; it can also rise if another
   process takes memory. Quote the summary value once the run ends.
4. **Token accounting is internally inconsistent by 10,752 tokens.** `data/tokens/
   pretrain_meta.json` says `total_tokens_consumed = 2,208,852,480` (= `n_blocks ×
   512`), but training consumes `steps × 16,384 = 2,208,841,728` because `drop_last=True`
   discards the last 21 blocks. `env_report.md` quotes the latter (2,208,841,728);
   the meta file quotes the former. **Use `steps × tokens_per_step` for "tokens
   actually trained on"**; the 10,752 difference is the dropped tail. The tools take
   the summary's `total_tokens`, which is the `cum_tokens` accumulator and therefore
   already the drop_last-corrected value.
5. **`grpo_*_summary.json` has no `total_tokens` field** — only `cum_sampled_tokens`
   and `cum_prompt_tokens`. Any tool that assumes `total_tokens` everywhere will
   silently print `n/a` for GRPO. `make_tables.py` falls back explicitly.
6. **`grpo_easy_probe` has no `logs/vram_train_*.json` metadata** — the probe was run
   directly, not through `scripts/run_with_mem.py`, so its run argv is unrecorded.
   Its config row therefore reads `n/a (no run argv recorded)`; only its summary
   numbers (steps, tokens, wall, VRAM) are available. `grpo_rule_smoke` does have one.
7. **`collapsed` in the GRPO probe means "truncated", not "entropy collapsed".**
   `train_grpo_rule.py` sets `collapsed = (ent < 0.05 or resp_len >= max_gen_len*0.99)`.
   In `grpo_easy_probe` the entropy never drops below **2.09** (the entropy arm never
   fires), and the 12 `collapsed` steps are exactly the ones whose `resp_len_mean` is
   ≥ 23.76 = 0.99 × 24 (observed range 23.78–24.00), while every non-collapsed step is
   ≤ 23.75. So the flag is triggered purely by the **response-length** arm: the
   rollouts hit `max_gen_len=24`. Reading it as "policy collapse" would be wrong.
   The figure legend says `collapsed (12 step(s))` without asserting a cause, and the
   README text should do the same.
8. **Eval report fields are duplicated at two levels.** `max_new_tokens` exists both
   top-level and at `generation.max_new_tokens`; `prompt_mode` both top-level and at
   `generation.prompt_mode`. `tasks_sha256` is **only** at `benchmark.tasks_sha256`
   (there is no top-level copy). The tools read the top-level value and fall back to
   the nested one; they read the sha from `benchmark`.
9. **`peak_vram_mb` (eval, 146.5 MB) and `peak_reserved_MiB` (training, 5,288 MiB) are
   different units and different measurement methods** — one is an eval-process
   allocation peak in MB, the other a trainer high-water reserved peak in MiB. They
   appear in separate tables and are never compared.
10. **`pretrain_metrics_summary.json` is written only at the very end of the run**, and
    `train_pretrain_packed.py` opens the metrics JSONL with mode `'w'` — so a resumed
    run **truncates** the existing metrics file. If the 28 h job is ever restarted
    with `--from_resume`, the previously logged points are lost unless the file is
    copied first. Worth flagging before any resume.

---

## 8. What the lead should do at the end of Phase 1

```powershell
$py = 'D:\llm-rl-lab\.venv\Scripts\python.exe'
# 1. full-run figures (both now exist)
& $py tools\plot_curves.py --metrics logs\pretrain_metrics.jsonl `
    --out figures\pretrain_curve_full.png --kind pretrain --total-steps 134817
# 2. README fragments, with the real eval reports once SFT/GRPO checkpoints exist
& $py tools\gen_readme_sections.py --out reports\readme_sections.md `
    --eval reports\eval_sft.json reports\eval_grpo.json
# 3. paste each BEGIN/END block into README.md, then re-run to refresh the numbers
```

Until then the fragments correctly contain `TODO` markers for the eval comparison and
the full-run figure, and no placeholder numbers.
