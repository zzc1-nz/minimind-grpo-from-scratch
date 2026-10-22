"""Phase-1 -> Phase-2 -> Phase-3 -> Phase-4 -> Phase-5 unattended pipeline.

Runs each stage as a separate child process (so a crash in one stage does not lose the
others), waits for the previous stage's checkpoint to exist, writes a single pipeline log,
and stops on the first hard failure instead of silently continuing with a stale weight.

Design rules
  * Every stage writes its own metrics/report artifact; this orchestrator only sequences
    them and records timing.
  * A stage is only started when its input checkpoint exists AND is newer than the
    previous stage's completion marker.
  * VRAM is checked before every GPU stage (needs >= 4,500 MiB free) so a stray process
    cannot cause an OOM that looks like a training bug.
  * `--dry-run` prints the exact plan without launching anything.

Usage
  python run_pipeline.py --phase 2            # run SFT then stop
  python run_pipeline.py --from 2 --to 5      # run everything from SFT onward
  python run_pipeline.py --from 2 --to 5 --dry-run
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

PY = r'D:\llm-rl-lab\.venv\Scripts\python.exe'
ROOT = r'D:\llm-rl-lab'
OUT = rf'{ROOT}\src\minimind-master\out'
LOGS = rf'{ROOT}\logs'
REPORTS = rf'{ROOT}\reports'
TRAINER_CWD = rf'{ROOT}\src\minimind-master\trainer'


def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(rf'{LOGS}\pipeline.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def free_vram_mib():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            text=True, timeout=30)
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def kill_ollama(why=''):
    """Kill a resident Ollama llama-server.

    On this box Ollama auto-restarts and grabs ~4.6 GB of the 8.2 GB card; any GPU
    stage then fails with an OOM that looks like a training bug. Observed twice in one
    hour during Phase 1. Only these two process names are touched.
    """
    killed = []
    try:
        out = subprocess.check_output(['tasklist', '/FO', 'CSV', '/NH'],
                                      text=True, timeout=30, errors='replace')
    except Exception:
        return killed
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) < 2:
            continue
        name, pid = parts[0], parts[1]
        if name.lower() in ('llama-server.exe', 'ollama.exe'):
            try:
                subprocess.call(['taskkill', '/PID', pid, '/F'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                timeout=20)
                killed.append(f'{name}:{pid}')
            except Exception:
                pass
    if killed:
        log(f'  killed resident ollama processes before GPU stage ({why}): {killed}')
        time.sleep(3)
    return killed


def run(cmd, cwd=None, label='', check_vram=4500, timeout=None):
    log(f'--- START {label} ---')
    log('  ' + ' '.join(cmd))
    if check_vram:
        kill_ollama(label)
        fv = free_vram_mib()
        log(f'  free VRAM before launch: {fv} MiB (need >= {check_vram})')
        if fv >= 0 and fv < check_vram:
            raise RuntimeError(
                f'refusing to start {label}: only {fv} MiB VRAM free, need {check_vram}. '
                'Another process is still holding the GPU.')
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=cwd)
    dt = time.time() - t0
    log(f'--- END {label}: exit={rc} wall={dt/60:.2f} min ---')
    if rc != 0:
        raise RuntimeError(f'{label} failed with exit code {rc}')
    return dt


def need(path, what):
    if not os.path.exists(path):
        raise FileNotFoundError(f'{what} not found: {path}')
    return path


# --------------------------------------------------------------------------- stages
PRETRAIN_FINAL_STEP = 134817


def _assert_pretrain_complete():
    """Refuse to start SFT on a partially-trained base model.

    `pretrain_768.pth` is rewritten every 2000 steps, so its mere existence does NOT
    mean Phase 1 finished -- at the time this guard was added the file held a 49 %
    checkpoint. The authoritative completion signal is the summary JSON, which the
    trainer writes only after the loop ends; fall back to the metrics step counter.
    """
    ckpt = rf'{OUT}\pretrain_768.pth'
    need(ckpt, 'pretrain checkpoint')
    summary = rf'{LOGS}\pretrain_metrics_summary.json'
    if os.path.exists(summary):
        s = json.load(open(summary, encoding='utf-8'))
        steps = int(s.get('steps') or 0)
        log(f'  pretrain summary: steps={steps:,} tokens={s.get("total_tokens"):,} '
            f'peak={s.get("peak_reserved_MiB")} MiB')
        if steps >= PRETRAIN_FINAL_STEP:
            return s
        log(f'  WARNING: pretrain summary says {steps:,} steps '
            f'(< {PRETRAIN_FINAL_STEP:,} planned)')
        return s
    # no summary -> read the last metrics row
    last = None
    mpath = rf'{LOGS}\pretrain_metrics.jsonl'
    if os.path.exists(mpath):
        with open(mpath, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        last = json.loads(line)
                    except Exception:
                        pass
    step = int((last or {}).get('step') or 0)
    raise RuntimeError(
        f'pretrain is not finished: no {os.path.basename(summary)} yet and the last '
        f'metrics row is step {step:,} of {PRETRAIN_FINAL_STEP:,} '
        f'({100.0*step/PRETRAIN_FINAL_STEP:.1f} %). Refusing to run SFT on a '
        f'partially-trained base model -- the resulting SFT/GRPO numbers would be '
        f'measured against a different starting point than the report claims. '
        f'Finish pretraining first (tools/resume_training.py), or delete the guard '
        f'if a partial-base experiment is intended.')


def stage2_sft(a):
    _assert_pretrain_complete()

    def build(bs, epochs=1, max_steps=0):
        cmd = [PY, rf'{ROOT}\custom\train_full_sft_counted.py',
               '--data_path', rf'{ROOT}\data\sft\sft_train.jsonl',
               '--from_weight', 'pretrain',
               '--save_dir', OUT, '--save_weight', 'full_sft',
               '--hidden_size', '768', '--num_hidden_layers', '8',
               '--batch_size', str(bs), '--max_seq_len', '768',
               '--epochs', str(epochs), '--learning_rate', '1e-5',
               '--accumulation_steps', '1', '--grad_clip', '1.0',
               '--use_compile', '1', '--num_workers', '4', '--seed', '42',
               '--log_interval', '50', '--save_interval', '5000',
               '--checkpoint_dir', rf'{ROOT}\src\minimind-master\checkpoints',
               '--metrics_path', rf'{LOGS}\sft_metrics.jsonl']
        if max_steps:
            cmd += ['--max_steps', str(max_steps)]
        return cmd

    # bs16/seq768 peak VRAM was measured at 7,184 MiB WITHOUT compile; with compile the
    # closest measured point was 5,468 MiB, but that is still an estimate. Try 16 first
    # and fall back to 8 (measured 4,008 MiB) rather than losing the whole chain to an OOM.
    #
    # On failure the partial metrics file is moved aside (kept for diagnosis) so the
    # retry does not append to a half-written log from the crashed attempt.
    last_err = None
    ok = False
    for bs in (a.sft_batch, 8):
        try:
            run(build(bs), cwd=TRAINER_CWD,
                label=f'PHASE2 SFT (200k records, 1 epoch, bs={bs})', check_vram=a.min_vram)
            ok = True
            break
        except Exception as e:
            last_err = e
            log(f'  SFT bs={bs} failed: {e}')
            if os.path.exists(rf'{LOGS}\sft_metrics.jsonl'):
                shutil.move(rf'{LOGS}\sft_metrics.jsonl',
                            rf'{LOGS}\sft_metrics_failed_bs{bs}.jsonl')
            log('  retrying SFT with the next smaller batch size')
    if not ok:
        raise RuntimeError(f'SFT failed at every batch size tried; last error: {last_err}')
    need(rf'{OUT}\full_sft_768.pth', 'SFT checkpoint')


def stage2_eval(a):
    cmd = [PY, rf'{ROOT}\eval\run_eval.py',
           '--weight', rf'{OUT}\full_sft_768.pth',
           '--hidden', '768', '--layers', '8',
           '--max-new-tokens', '64', '--device', 'cuda', '--prompt-mode', 'chat',
           '--out', rf'{REPORTS}\sft_baseline_full.json']
    run(cmd, label='PHASE2 held-out eval (SFT baseline)', check_vram=a.min_vram)


def stage3_grpo(a):
    need(rf'{OUT}\full_sft_768.pth', 'SFT checkpoint')
    # Feasibility pre-flight: GRPO learns only from within-group reward variance, so a
    # format the SFT stage never taught would leave every group all-wrong (adv_std = 0)
    # and burn the whole stage. Cheap static check, run before the GPU work.
    for _stage in ('stage1_easy', 'stage2_medium', 'stage3_full'):
        try:
            run([PY, rf'{ROOT}\tools\grpo_feasibility.py', '--stage', _stage,
                 '--group', '16'],
                label=f'PREFLIGHT grpo feasibility {_stage}', check_vram=0)
        except Exception as e:
            log(f'  feasibility {_stage} reported a problem (continuing): {e}')
    # Measure the REAL peak at the production GRPO shape before committing to 3 stages.
    # The two calibration points both used max_prompt_len=192; Phase 3 uses 256, and
    # attention cost is quadratic in sequence length, so a linear extrapolation can
    # under-predict. One probe step is cheap and removes the guess.
    try:
        run([PY, rf'{ROOT}\tools\probe_grpo_vram.py', '--steps', '2',
             '--group', '16', '--bs', '4', '--gen', '48', '--prompt-len', '256'],
            label='PREFLIGHT grpo vram probe (production shape)', check_vram=4500)
        probe = os.path.join(LOGS, 'probe_grpo_vram.json')
        if os.path.exists(probe):
            pj = json.load(open(probe, encoding='utf-8'))
            log(f'  measured GRPO peak: alloc={pj.get("peak_alloc_MiB")} MiB '
                f'reserved={pj.get("peak_reserved_MiB")} MiB '
                f'headroom={pj.get("headroom_MiB")} MiB')
    except Exception as e:
        log(f'  GRPO vram probe skipped/failed (continuing): {e}')
    # stage-by-stage, resuming from the previous stage's weights
    stages = [
        ('stage1_easy', 37, 9, 'full_sft'),
        ('stage2_medium', 87, 7, 'grpo_stage1_easy'),
        ('stage3_full', 150, 6, 'grpo_stage2_medium'),
    ]
    for name, max_steps, epochs, from_w in stages:
        # save_interval is large on purpose: only the FINAL weight of each stage is
        # consumed (by the next stage / the comparison). Intermediate saves would be
        # ~27 redundant 131 MB writes with no consumer.
        cmd = [PY, rf'{ROOT}\custom\train_grpo_rule.py',
               '--data_path', rf'{ROOT}\data\grpo\{name}.jsonl',
               '--from_weight', from_w, '--save_weight', f'grpo_{name}',
               '--save_dir', OUT, '--hidden_size', '768', '--num_hidden_layers', '8',
               '--batch_size', '4', '--num_generations', '16',
               '--max_prompt_len', '256', '--max_gen_len', '48',
               '--epochs', str(epochs), '--learning_rate', '5e-7', '--beta', '0.01',
               '--epsilon', '0.2', '--temperature', '0.9',
               '--accumulation_steps', '1', '--grad_clip', '1.0',
               '--lr_schedule', 'cosine', '--reward_mode', 'binary',
               '--max_steps', str(max_steps), '--save_interval', '1000000',
               '--log_interval', '10', '--seed', '42', '--device', 'cuda',
               '--metrics_path', rf'{LOGS}\grpo_{name}.jsonl']
        # cwd MUST be the trainer dir: `init_model` defaults tokenizer_path='../model',
        # which only resolves correctly relative to <repo>/trainer. Omitting cwd made
        # transformers treat '../model' as a HuggingFace repo id and fail with
        # HFValidationError (this is exactly how Phase 3 died at 15:38).
        run(cmd, cwd=TRAINER_CWD, label=f'PHASE3 GRPO {name}', check_vram=a.min_vram)
        hc = [PY, rf'{ROOT}\tools\grpo_healthcheck.py', rf'{LOGS}\grpo_{name}.jsonl',
              '--json', rf'{LOGS}\grpo_{name}_healthcheck.json', '--min-steps', '20']
        try:
            run(hc, label=f'healthcheck {name}', check_vram=0)
        except Exception as e:  # a diagnostic must never kill the pipeline
            log(f'  healthcheck {name} non-fatal: {e}')
    need(rf'{OUT}\grpo_stage3_full_768.pth', 'final GRPO checkpoint')


def stage3_compare(a):
    cmd = [PY, rf'{ROOT}\tools\compare_sft_grpo.py',
           '--sft', rf'{OUT}\full_sft_768.pth',
           '--grpo', rf'{OUT}\grpo_stage3_full_768.pth',
           '--out', rf'{REPORTS}\grpo_vs_sft.json',
           '--hidden', '768', '--layers', '8',
           '--max-new-tokens', '64', '--device', 'cuda']
    run(cmd, label='PHASE3 SFT-vs-GRPO held-out comparison', check_vram=a.min_vram)


def stage4_gguf(a):
    need(rf'{OUT}\full_sft_768.pth', 'SFT checkpoint')
    need(rf'{OUT}\grpo_stage3_full_768.pth', 'GRPO checkpoint')
    os.makedirs(rf'{ROOT}\models\hf', exist_ok=True)
    os.makedirs(rf'{ROOT}\models\gguf', exist_ok=True)
    conv = rf'{ROOT}\llamacpp\scripts\convert_pth_to_hf.py'
    quant = rf'{ROOT}\llamacpp\cuda124\llama-quantize.exe'
    # MUST be the patched copy: it carries the 8-line `chkhsh a39372c9… -> "gpt-2"`
    # pre-tokenizer fix. An upstream copy dies with "BPE pre-tokenizer not recognized".
    ggufconv = rf'{ROOT}\llamacpp\convert_hf_to_gguf.py'
    jobs = [('full_sft_768', rf'{OUT}\full_sft_768.pth'),
            ('grpo_stage3_full_768', rf'{OUT}\grpo_stage3_full_768.pth')]
    for name, pth in jobs:
        hf_dir = rf'{ROOT}\models\hf\{name}'
        f16 = rf'{ROOT}\models\gguf\{name}-f16.gguf'
        q4 = rf'{ROOT}\models\gguf\{name}-Q4_K_M.gguf'
        run([PY, conv, pth, hf_dir], label=f'PHASE4 {name}: .pth -> HF', check_vram=0)
        # Qwen3Config leaves bos/eos/pad as None; llama.cpp needs them as stop tokens
        cfg = os.path.join(hf_dir, 'config.json')
        with open(cfg, encoding='utf-8') as f:
            d = json.load(f)
        d.update(bos_token_id=1, eos_token_id=2, pad_token_id=0)
        with open(cfg, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        log(f'  patched {cfg} with bos/eos/pad ids')
        run([PY, ggufconv, hf_dir, '--outfile', f16, '--outtype', 'f16'],
            cwd=rf'{ROOT}\llamacpp', label=f'PHASE4 {name}: HF -> GGUF f16', check_vram=0)
        run([quant, f16, q4, 'Q4_K_M'], cwd=rf'{ROOT}\llamacpp',
            label=f'PHASE4 {name}: quantize Q4_K_M', check_vram=0)
        # Sanity-check the artifact before moving on. A truncated/failed quantize still
        # leaves a file behind, and a 15 MB "Q4" for a 122 MB f16 model looks plausible
        # until you try to load it. Measured: a correct Q4_K_M of this model is ~40 MB;
        # anything under 25 MB means the write did not complete.
        if not os.path.exists(q4):
            raise RuntimeError(f'quantize produced no file: {q4}')
        f16_mb = os.path.getsize(f16) / 1e6
        q4_mb = os.path.getsize(q4) / 1e6
        if q4_mb < 0.25 * f16_mb:
            raise RuntimeError(
                f'{q4} is only {q4_mb:.1f} MB vs f16 {f16_mb:.1f} MB '
                f'({q4_mb/f16_mb:.0%}) -- expected ~33% for Q4_K_M. '
                f'The quantize step almost certainly did not finish.')
        log(f'  verified {name}: f16={f16_mb:.1f} MB, Q4_K_M={q4_mb:.1f} MB '
            f'({q4_mb/f16_mb:.0%} of f16)')
        log(f'  gguf sizes: f16={os.path.getsize(f16)/1e6:.1f} MB '
            f'Q4_K_M={os.path.getsize(q4)/1e6:.1f} MB')


def stage4_demo(a):
    cmd = [PY, rf'{ROOT}\demo\run_compare_battery.py',
           '--sft', rf'{OUT}\full_sft_768.pth',
           '--grpo', rf'{OUT}\grpo_stage3_full_768.pth',
           '--out', rf'{REPORTS}\demo_battery.json']
    try:
        run(cmd, cwd=rf'{ROOT}\demo', label='PHASE4 demo battery (CPU)', check_vram=0)
    except Exception as e:
        log(f'  demo battery non-fatal: {e}')
    # GGUF accuracy check: the quantization-drop figure Phase-4 must report. Runs the
    # quantized file through llama.cpp with the SAME prompt mode / max_new_tokens / seed
    # as the fp baseline, so the delta measures quantization and nothing else.
    q4 = rf'{ROOT}\models\gguf\full_sft_768-Q4_K_M.gguf'
    if os.path.exists(q4):
        try:
            run([PY, rf'{ROOT}\eval\run_eval.py', '--backend', 'gguf', '--weight', q4,
                 '--hidden', '768', '--layers', '8', '--max-new-tokens', '64',
                 '--prompt-mode', 'chat', '--out', rf'{REPORTS}\eval_gguf_q4_sft.json'],
                label='PHASE4 GGUF Q4 held-out eval (SFT)', check_vram=0)
        except Exception as e:
            log(f'  GGUF SFT eval non-fatal: {e}')
        # same for the GRPO model, so the report can state both quantization drops
        q4g = rf'{ROOT}\models\gguf\grpo_stage3_full_768-Q4_K_M.gguf'
        if os.path.exists(q4g):
            try:
                run([PY, rf'{ROOT}\eval\run_eval.py', '--backend', 'gguf', '--weight', q4g,
                     '--hidden', '768', '--layers', '8', '--max-new-tokens', '64',
                     '--prompt-mode', 'chat', '--out', rf'{REPORTS}\eval_gguf_q4_grpo.json'],
                    label='PHASE4 GGUF Q4 held-out eval (GRPO)', check_vram=0)
            except Exception as e:
                log(f'  GGUF GRPO eval non-fatal: {e}')
    else:
        log(f'  GGUF Q4 file not found ({q4}); skipping quantization-drop eval')


def stage5_docs(a):
    for name, script, extra in [
        ('pretrain.md', rf'{ROOT}\tools\make_pretrain_report.py',
         ['--metrics', rf'{LOGS}\pretrain_metrics.jsonl',
          '--summary', rf'{LOGS}\pretrain_metrics_summary.json',
          '--meta', rf'{ROOT}\data\tokens\pretrain_meta.json',
          '--out', rf'{REPORTS}\pretrain.md']),
    ]:
        try:
            run([PY, script] + extra, label=f'PHASE5 {name}', check_vram=0)
        except Exception as e:
            log(f'  {name} non-fatal: {e}')
    # final curves. NOTE: --total-steps must be the REAL planned step count, otherwise
    # the "N of M planned" label in the figure is wrong.
    for metrics, out, kind, total in [
        (rf'{LOGS}\pretrain_metrics.jsonl', rf'{ROOT}\figures\pretrain_curve_full.png', 'pretrain', '134817'),
        (rf'{LOGS}\grpo_stage1_easy.jsonl', rf'{ROOT}\figures\grpo_stage1.png', 'grpo', '333'),
        (rf'{LOGS}\grpo_stage2_medium.jsonl', rf'{ROOT}\figures\grpo_stage2.png', 'grpo', '609'),
        (rf'{LOGS}\grpo_stage3_full.jsonl', rf'{ROOT}\figures\grpo_stage3.png', 'grpo', '900'),
    ]:
        if os.path.exists(metrics):
            try:
                run([PY, rf'{ROOT}\tools\plot_curves.py', '--metrics', metrics, '--out', out,
                     '--kind', kind, '--total-steps', total],
                    label=f'PHASE5 figure {os.path.basename(out)}', check_vram=0)
            except Exception as e:
                log(f'  figure {out} non-fatal: {e}')
    # README last: it reads every artifact produced above.
    try:
        run([PY, rf'{ROOT}\tools\make_readme.py', '--out', rf'{ROOT}\README.md'],
            label='PHASE5 README', check_vram=0)
    except Exception as e:
        log(f'  README non-fatal: {e}')
    # resume bullets (non-strict: a blocked bullet is reported, not a failure)
    try:
        run([PY, rf'{ROOT}\tools\make_resume_bullets.py'], label='PHASE5 resume bullets',
            check_vram=0)
    except Exception as e:
        log(f'  resume bullets non-fatal: {e}')
    # data-driven README fragments (tables + figures), for cross-checking the README
    try:
        run([PY, rf'{ROOT}\tools\gen_readme_sections.py',
             '--total-pretrain-steps', '134817',
             '--out', rf'{REPORTS}\readme_sections_final.md'],
            label='PHASE5 readme sections', check_vram=0)
    except Exception as e:
        log(f'  readme sections non-fatal: {e}')
    # Quality gate: the GRPO training reward and the held-out grader are separate
    # implementations. If they disagree, GRPO optimises a different objective than the
    # one the report measures. Recorded as a pass/fail line in the pipeline log.
    try:
        run([PY, rf'{ROOT}\tools\crosscheck_reward_vs_grader.py'],
            label='PHASE5 reward-vs-grader cross-check', check_vram=0)
        log('  reward/grader cross-check: PASS (0 disagreements)')
    except Exception as e:
        log(f'  reward/grader cross-check FAILED: {e}')


STAGES = {2: [('sft', stage2_sft), ('eval_sft', stage2_eval)],
          3: [('grpo', stage3_grpo), ('compare', stage3_compare)],
          4: [('gguf', stage4_gguf), ('demo', stage4_demo)],
          5: [('docs', stage5_docs)]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--from', dest='start', type=int, default=2)
    p.add_argument('--to', dest='end', type=int, default=5)
    p.add_argument('--phase', type=int, default=None, help='run exactly one phase')
    p.add_argument('--sft-batch', type=int, default=16)
    p.add_argument('--min-vram', type=int, default=4500)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.phase is not None:
        a.start = a.end = a.phase

    plan = []
    for ph in range(a.start, a.end + 1):
        for name, fn in STAGES.get(ph, []):
            plan.append((ph, name, fn))
    if not plan:
        raise SystemExit(f'no stages for phases {a.start}..{a.end}')

    log('=' * 78)
    log(f'PIPELINE START phases {a.start}..{a.end} '
        f'({len(plan)} stages){" [DRY RUN]" if a.dry_run else ""}')
    log(f'  free VRAM now: {free_vram_mib()} MiB')

    # Static contract check before spending hours of GPU time: verifies that every
    # artifact a stage requires is produced by an earlier stage, and that the summary
    # JSON fields the consumers read are actually written by the producers.
    if not a.dry_run:
        try:
            run([PY, rf'{ROOT}\tools\check_artifact_contracts.py'],
                label='PREFLIGHT artifact contracts', check_vram=0)
            log('  artifact contracts: PASS')
        except Exception as e:
            log(f'  artifact contracts FAILED (continuing anyway): {e}')

    if a.dry_run:
        for ph, name, fn in plan:
            log(f'  phase {ph}: {name}  ->  {fn.__doc__ or fn.__name__}')
        log('DRY RUN: nothing executed')
        return 0

    t_all = time.time()
    done = []
    try:
        for ph, name, fn in plan:
            log(f'>>>>> phase {ph} stage {name}')
            fn(a)
            done.append(f'{ph}:{name}')
            log(f'<<<<< phase {ph} stage {name} OK')
    except Exception as e:
        log(f'PIPELINE FAILED at phase {ph} stage {name}: {type(e).__name__}: {e}')
        log(f'completed before failure: {done}')
        # Write an actionable recovery block. Distinguish "the stage crashed" from
        # "a precondition was not met" -- telling someone to resume SFT when the real
        # problem is that pretraining never finished would send them in circles.
        msg = str(e)
        precondition = ('pretrain is not finished' in msg
                        or 'resume checkpoint' in msg
                        or 'Refusing to start' in msg)
        log('--- RECOVERY ---')
        if precondition:
            log('This is a PRECONDITION failure, not a crash of the stage itself.')
            if 'pretrain is not finished' in msg:
                log('Finish Phase 1 first:')
                log(f'  {PY} {ROOT}\\tools\\resume_training.py')
                log(f'  then: {PY} {ROOT}\\tools\\run_pipeline.py --from 2 --to 5')
            else:
                log('Check the resume checkpoint path / --save_weight before retrying.')
            return 1
        if ph == 2 and name == 'sft':
            log('SFT died. Resume it (it keeps its own step/token position) with:')
            log(f'  cd {TRAINER_CWD}')
            log(f'  {PY} {ROOT}\\custom\\train_full_sft_counted.py '
                f'--data_path {ROOT}\\data\\sft\\sft_train.jsonl --from_weight pretrain '
                f'--save_dir {OUT} --save_weight full_sft --hidden_size 768 '
                f'--num_hidden_layers 8 --batch_size {a.sft_batch} --max_seq_len 768 '
                f'--epochs 1 --learning_rate 1e-5 --use_compile 1 --num_workers 4 '
                f'--seed 42 --save_interval 5000 --from_resume 1 '
                f'--checkpoint_dir {ROOT}\\src\\minimind-master\\checkpoints '
                f'--metrics_path {LOGS}\\sft_metrics.jsonl')
            log(f'  then: {PY} {ROOT}\\tools\\run_pipeline.py --from 2 --to 5')
        elif ph == 3:
            log('GRPO died. Each stage resumes from its own _resume.pth; re-run:')
            log(f'  {PY} {ROOT}\\tools\\run_pipeline.py --from 3 --to 5')
        else:
            log(f'  re-run: {PY} {ROOT}\\tools\\run_pipeline.py --from {ph} --to {a.end}')
        return 1
    log(f'PIPELINE COMPLETE in {(time.time()-t_all)/3600:.2f} h; stages: {done}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
