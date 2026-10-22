"""End-to-end integration test of the whole pipeline on CPU with a tiny model.

The individual stages have been tested separately; this exercises the *chain*:
  pretrain -> SFT -> held-out eval -> GRPO (3 curriculum stages) -> compare -> GGUF eval stub

It uses a 2-layer/128-dim model and a few hundred records so it finishes in a couple of
minutes on CPU and never touches the GPU.

What it proves (and what it does NOT):
  * every stage's CLI contract matches what the previous stage produces;
  * the artifacts each stage writes are the ones the next stage reads;
  * the token accounting adds up across the chain.
  * It does NOT prove model quality -- a 128-dim model trained for seconds is noise.

Usage: python test_e2e_cpu.py [--keep]
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
WORK = os.path.join(ROOT, 'logs', '_e2e')
TRAINER_CWD = os.path.join(ROOT, 'src', 'minimind-master', 'trainer')
CKPT_DIR = os.path.join(ROOT, 'src', 'minimind-master', 'checkpoints')

HID, LAYERS = 128, 2
BS, SEQ = 4, 64


def run(cmd, cwd=None, label='', timeout=900):
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       encoding='utf-8', errors='replace', timeout=timeout)
    dt = time.time() - t0
    ok = r.returncode == 0
    print(f'  [{"OK " if ok else "FAIL"}] {label}  ({dt:.1f}s)')
    if not ok:
        tail = (r.stderr or r.stdout or '').strip().splitlines()[-6:]
        for line in tail:
            print('        ' + line)
    return ok, r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--keep', action='store_true', help='keep the work dir')
    a = p.parse_args()

    if os.path.exists(WORK):
        shutil.rmtree(WORK, ignore_errors=True)
    for d in (WORK, os.path.join(WORK, 'out')):
        os.makedirs(d, exist_ok=True)

    # mini SFT corpus: the first 200 records of the real SFT mix, so the test exercises
    # the same schema (multi-turn `conversations`) that Phase 2 will use.
    sft_data = os.path.join(WORK, 'sft200.jsonl')
    with open(os.path.join(ROOT, 'data', 'sft', 'sft_train.jsonl'), encoding='utf-8') as f, \
            open(sft_data, 'w', encoding='utf-8', newline='\n') as g:
        for i, line in enumerate(f):
            if i >= 200:
                break
            g.write(line)

    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = '-1'
    env['PYTHONIOENCODING'] = 'utf-8'
    os.environ.update(env)

    results = []

    def step(name, cmd, cwd=None):
        ok, r = run(cmd, cwd=cwd, label=name)
        results.append((name, ok))
        return ok, r

    print('=' * 70)
    print('E2E integration test (CPU, tiny model)')
    print('=' * 70)

    # ---- 1. mini corpus -> packed blocks -----------------------------------
    import numpy as np
    n_blocks = 512
    rng = np.random.default_rng(42)
    binp = os.path.join(WORK, 'blocks.bin')
    rng.integers(0, 6400, size=n_blocks * SEQ, dtype=np.uint16).tofile(binp)
    print(f'  corpus: {n_blocks} blocks x {SEQ} tokens')

    # ---- 2. pretrain --------------------------------------------------------
    ok, _ = step('pretrain', [
        PY, os.path.join(ROOT, 'custom', 'train_pretrain_packed.py'),
        '--data_path', binp, '--max_seq_len', str(SEQ), '--batch_size', str(BS),
        '--epochs', '1', '--max_steps', '40', '--learning_rate', '5e-4',
        '--log_interval', '10', '--save_interval', '20', '--num_workers', '0',
        '--use_compile', '0', '--hidden_size', str(HID), '--num_hidden_layers', str(LAYERS),
        '--device', 'cpu', '--dtype', 'float32', '--seed', '42',
        '--save_dir', os.path.join(WORK, 'out'), '--save_weight', 'pretrain',
        '--metrics_path', os.path.join(WORK, 'pretrain.jsonl')],
        cwd=TRAINER_CWD)
    pretrain_ckpt = os.path.join(WORK, 'out', f'pretrain_{HID}.pth')

    # ---- 3. SFT -------------------------------------------------------------
    ok, _ = step('sft', [
        PY, os.path.join(ROOT, 'custom', 'train_full_sft_counted.py'),
        '--data_path', sft_data, '--from_weight', 'pretrain',
        '--save_dir', os.path.join(WORK, 'out'), '--save_weight', 'full_sft',
        '--hidden_size', str(HID), '--num_hidden_layers', str(LAYERS),
        '--batch_size', '2', '--max_seq_len', '128', '--epochs', '1', '--max_steps', '20',
        '--learning_rate', '1e-5', '--num_workers', '0', '--device', 'cpu',
        '--use_compile', '0', '--log_interval', '10', '--save_interval', '10',
        '--checkpoint_dir', os.path.join(WORK, 'ckpt'),
        '--metrics_path', os.path.join(WORK, 'sft.jsonl')],
        cwd=TRAINER_CWD)

    # ---- 4. held-out eval on the SFT checkpoint ----------------------------
    sft_report = os.path.join(WORK, 'eval_sft.json')
    ok, _ = step('eval (SFT, hf backend)', [
        PY, os.path.join(ROOT, 'eval', 'run_eval.py'),
        '--weight', os.path.join(WORK, 'out', f'full_sft_{HID}.pth'),
        '--hidden', str(HID), '--layers', str(LAYERS),
        '--max-new-tokens', '12', '--device', 'cpu', '--prompt-mode', 'chat',
        '--out', sft_report], cwd=ROOT)

    # ---- 5. GRPO, 3 curriculum stages --------------------------------------
    prev = 'full_sft'
    for stage, steps in (('stage1_easy', 6), ('stage2_medium', 6), ('stage3_full', 6)):
        ok, _ = step(f'grpo {stage}', [
            PY, os.path.join(ROOT, 'custom', 'train_grpo_rule.py'),
            '--data_path', os.path.join(ROOT, 'data', 'grpo', f'{stage}.jsonl'),
            '--from_weight', prev, '--save_weight', f'grpo_{stage}',
            '--save_dir', os.path.join(WORK, 'out'),
            '--hidden_size', str(HID), '--num_hidden_layers', str(LAYERS),
            '--batch_size', '2', '--num_generations', '4',
            '--max_prompt_len', '128', '--max_gen_len', '8',
            '--epochs', '1', '--max_steps', str(steps), '--save_interval', '1000',
            '--log_interval', '2', '--device', 'cpu', '--seed', '42',
            '--metrics_path', os.path.join(WORK, f'grpo_{stage}.jsonl')],
            cwd=TRAINER_CWD)
        prev = f'grpo_{stage}'

    # ---- 6. healthcheck ----------------------------------------------------
    ok, _ = step('healthcheck stage3', [
        PY, os.path.join(ROOT, 'tools', 'grpo_healthcheck.py'),
        os.path.join(WORK, 'grpo_stage3_full.jsonl'),
        '--json', os.path.join(WORK, 'hc3.json'), '--min-steps', '3'], cwd=ROOT)

    # ---- 7. compare SFT vs GRPO -------------------------------------------
    ok, _ = step('compare SFT vs GRPO', [
        PY, os.path.join(ROOT, 'tools', 'compare_sft_grpo.py'),
        '--sft', os.path.join(WORK, 'out', f'full_sft_{HID}.pth'),
        '--grpo', os.path.join(WORK, 'out', f'grpo_stage3_full_{HID}.pth'),
        '--out', os.path.join(WORK, 'grpo_vs_sft.json'),
        '--hidden', str(HID), '--layers', str(LAYERS),
        '--max-new-tokens', '12', '--device', 'cpu'], cwd=ROOT)

    # ---- 8. chance baseline ------------------------------------------------
    ok, _ = step('chance baseline', [
        PY, os.path.join(ROOT, 'tools', 'chance_baseline.py'),
        '--json'], cwd=ROOT)

    # ---- 9. reward/grader crosscheck --------------------------------------
    ok, _ = step('reward vs grader crosscheck', [
        PY, os.path.join(ROOT, 'tools', 'crosscheck_reward_vs_grader.py')], cwd=ROOT)

    # ---- 10. README generation --------------------------------------------
    ok, _ = step('make_readme', [
        PY, os.path.join(ROOT, 'tools', 'make_readme.py'),
        '--out', os.path.join(WORK, 'README.md')], cwd=ROOT)

    # ---- 11. GGUF export chain (.pth -> HF -> GGUF f16 -> Q4_K_M) ----------
    # This is Phase-4's core and it has two known traps (missing CUDA runtime DLLs,
    # the decoy release tag), so it must be exercised in the chain, not just manually.
    gguf_dir = os.path.join(WORK, 'gguf')
    os.makedirs(gguf_dir, exist_ok=True)
    hf_dir = os.path.join(WORK, 'hf')
    conv = os.path.join(ROOT, 'llamacpp', 'scripts', 'convert_pth_to_hf.py')
    ggufconv = os.path.join(ROOT, 'llamacpp', 'convert_hf_to_gguf.py')
    quant = os.path.join(ROOT, 'llamacpp', 'cuda124', 'llama-quantize.exe')
    sft_ckpt = os.path.join(WORK, 'out', f'full_sft_{HID}.pth')

    ok_hf, _ = step('gguf: .pth -> HF', [PY, conv, sft_ckpt, hf_dir, str(HID), str(LAYERS)],
                    cwd=ROOT)
    if ok_hf:
        # Qwen3Config leaves bos/eos/pad None; llama.cpp needs them
        cfg = os.path.join(hf_dir, 'config.json')
        with open(cfg, encoding='utf-8') as f:
            d = json.load(f)
        d.update(bos_token_id=1, eos_token_id=2, pad_token_id=0)
        with open(cfg, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        f16 = os.path.join(gguf_dir, 'm-f16.gguf')
        ok_f16, _ = step('gguf: HF -> GGUF f16',
                         [PY, ggufconv, hf_dir, '--outfile', f16, '--outtype', 'f16'],
                         cwd=os.path.join(ROOT, 'llamacpp'))
        if ok_f16:
            q4 = os.path.join(gguf_dir, 'm-Q4_K_M.gguf')
            ok_q4, _ = step('gguf: quantize Q4_K_M', [quant, f16, q4, 'Q4_K_M'],
                            cwd=os.path.join(ROOT, 'llamacpp'))
        else:
            ok_q4 = False
    else:
        ok_f16 = ok_q4 = False

    # ---- assertions --------------------------------------------------------
    print()
    print('-' * 70)
    print('ARTIFACT CHECKS')
    checks = []

    def check(name, cond, detail=''):
        checks.append((name, bool(cond), detail))
        print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'  {detail}' if detail else ''))

    def jload(p):
        try:
            with open(p, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def jlines(p):
        out = []
        try:
            with open(p, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        out.append(json.loads(line))
        except Exception:
            pass
        return out

    check('pretrain checkpoint written', os.path.exists(pretrain_ckpt))
    check('SFT checkpoint written',
          os.path.exists(os.path.join(WORK, 'out', f'full_sft_{HID}.pth')))
    check('GRPO stage3 checkpoint written',
          os.path.exists(os.path.join(WORK, 'out', f'grpo_stage3_full_{HID}.pth')))

    pm = jlines(os.path.join(WORK, 'pretrain.jsonl'))
    check('pretrain metrics non-empty', len(pm) > 0, f'{len(pm)} rows')
    if pm:
        check('pretrain token accounting monotonic',
              all(pm[i]['cum_tokens'] < pm[i + 1]['cum_tokens']
                  for i in range(len(pm) - 1)),
              f'last={pm[-1]["cum_tokens"]}')

    sm = jlines(os.path.join(WORK, 'sft.jsonl'))
    check('SFT metrics non-empty', len(sm) > 0, f'{len(sm)} rows')
    if sm:
        check('SFT supervised < total tokens',
              sm[-1]['cum_tokens_supervised'] < sm[-1]['cum_tokens_total'],
              f'{sm[-1]["cum_tokens_supervised"]} < {sm[-1]["cum_tokens_total"]}')

    for stage in ('stage1_easy', 'stage2_medium', 'stage3_full'):
        gm = jlines(os.path.join(WORK, f'grpo_{stage}.jsonl'))
        check(f'GRPO {stage} metrics non-empty', len(gm) > 0, f'{len(gm)} rows')

    ev = jload(sft_report)
    check('held-out eval report has 24 tasks',
          ev is not None and ev.get('overall', {}).get('total') == 24,
          f'{ev.get("overall") if ev else "no report"}')

    cmp3 = jload(os.path.join(WORK, 'grpo_vs_sft.json'))
    check('comparison artifact written', cmp3 is not None)
    if cmp3:
        check('comparison has chance baseline',
              (cmp3.get('verdict') or {}).get('chance_baseline') is not None)
        check('comparison token accounting present',
              'token_accounting' in cmp3)

    hc = jload(os.path.join(WORK, 'hc3.json'))
    check('healthcheck produced a verdict',
          hc is not None and bool(hc.get('verdict')),
          f'{hc.get("verdict") if hc else "none"}')

    rd = os.path.join(WORK, 'README.md')
    if os.path.exists(rd):
        txt = open(rd, encoding='utf-8').read()
        check('README generated with sections', '## 1. Results' in txt)
        check('README has no broken bold',
              all(l.count('**') % 2 == 0 for l in txt.split('\n')
                  if not l.strip().startswith('```')))
    else:
        check('README generated', False)

    # GGUF chain artifacts + the size-ratio guard that catches a truncated quantize.
    # NOTE: the ratio band is model-size dependent. The 63.91 M production model gives
    # ~0.33 (measured), but a 1.26 M test model is dominated by fixed-size tensors
    # (embedding, norms) that do not compress, so ~0.52 is correct for it. The guard
    # here only needs to catch a *truncated* file, which shows up as < 0.2.
    check('GGUF f16 written', ok_f16 and os.path.exists(os.path.join(gguf_dir, 'm-f16.gguf')))
    if ok_q4 and os.path.exists(os.path.join(gguf_dir, 'm-Q4_K_M.gguf')):
        f16_sz = os.path.getsize(os.path.join(gguf_dir, 'm-f16.gguf'))
        q4_sz = os.path.getsize(os.path.join(gguf_dir, 'm-Q4_K_M.gguf'))
        ratio = q4_sz / f16_sz
        check('GGUF Q4_K_M written', True, f'{q4_sz/1e6:.1f} MB')
        check('Q4/f16 ratio not truncated (>0.2)',
              ratio > 0.2, f'ratio={ratio:.3f} (production 63.91M model measures ~0.33)')
    else:
        check('GGUF Q4_K_M written', False)

    print()
    stages_ok = all(ok for _, ok in results)
    checks_ok = all(ok for _, ok, _ in checks)
    print('-' * 70)
    print(f'stages: {sum(1 for _, o in results if o)}/{len(results)} passed')
    print(f'checks: {sum(1 for _, o, _ in checks if o)}/{len(checks)} passed')
    print('RESULT:', 'ALL PASS' if (stages_ok and checks_ok) else 'FAILURES PRESENT')

    out = {'stages': [{'name': n, 'ok': o} for n, o in results],
           'checks': [{'name': n, 'ok': o, 'detail': d} for n, o, d in checks],
           'all_pass': stages_ok and checks_ok}
    with open(os.path.join(ROOT, 'logs', 'e2e_cpu_test.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    if not a.keep:
        shutil.rmtree(WORK, ignore_errors=True)
    return 0 if out['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
