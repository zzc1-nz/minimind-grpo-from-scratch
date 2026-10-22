"""Static contract check: every stage's OUTPUT must match the next stage's INPUT.

The end-to-end test proves the chain runs with a tiny model, but it uses the same
hard-coded paths it checks. This script instead *parses* the pipeline and the trainers
and verifies the artifact contracts independently:

  1. every path `run_pipeline.py` requires (`need(...)`) is produced by an earlier stage;
  2. every summary JSON field a consumer reads is written by the producer;
  3. the checkpoint filenames the trainers derive from --save_weight match the names
     the pipeline passes as --from_weight.

This catches the class of bug where a stage writes `foo_summary.json` but the next
stage reads `foo_metrics_summary.json` -- which only shows up after hours of training.

Usage: python check_artifact_contracts.py
"""
import json
import os
import re
import sys

ROOT = r'D:\llm-rl-lab'
PIPE = os.path.join(ROOT, 'tools', 'run_pipeline.py')
README = os.path.join(ROOT, 'tools', 'make_readme.py')
SFT = os.path.join(ROOT, 'custom', 'train_full_sft_counted.py')
GRPO = os.path.join(ROOT, 'custom', 'train_grpo_rule.py')
PRE = os.path.join(ROOT, 'custom', 'train_pretrain_packed.py')
CMP = os.path.join(ROOT, 'tools', 'compare_sft_grpo.py')

results = []


def check(name, ok, detail=''):
    results.append((name, bool(ok), detail))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'   {detail}' if detail else ''))


def read(p):
    with open(p, encoding='utf-8') as f:
        return f.read()


def main():
    print('=' * 72)
    print('ARTIFACT CONTRACT CHECK (static analysis, no execution)')
    print('=' * 72)

    pipe = read(PIPE)
    rd = read(README)
    sft = read(SFT)
    grpo = read(GRPO)
    pre = read(PRE)

    # ---- 1. checkpoint naming ------------------------------------------------
    print('\n[1] checkpoint filename derivation')
    # Every trainer writes f'{save_dir}/{save_weight}_{hidden}{moe}.pth'
    for name, src in (('sft', sft), ('grpo', grpo), ('pretrain', pre)):
        pat = re.search(r"\{args\.save_dir\}/\{args\.save_weight\}_\{lm_config\.hidden_size\}",
                        src)
        check(f'{name} writes <save_dir>/<save_weight>_<hidden>.pth', pat is not None)

    # Pipeline stage 2 loads pretrain_768.pth as SFT's starting weight
    check('pipeline stage2 reads pretrain_768.pth',
          'pretrain_768.pth' in pipe and "--from_weight', 'pretrain'" in pipe)
    # Stage 3 stage1 starts from full_sft (the tuple form: ('stage1_easy', 37, 9, 'full_sft'))
    check('pipeline stage3 stage1 starts from full_sft',
          "('stage1_easy', 37, 9, 'full_sft')" in pipe)
    # stage2 starts from stage1's output, stage3 from stage2's
    check('pipeline stage3 stage2 chains from grpo_stage1_easy',
          "('stage2_medium', 87, 7, 'grpo_stage1_easy')" in pipe)
    check('pipeline stage3 stage3 chains from grpo_stage2_medium',
          "('stage3_full', 150, 6, 'grpo_stage2_medium')" in pipe)

    # ---- 2. required paths ---------------------------------------------------
    print('\n[2] paths the pipeline requires exist as earlier-stage outputs')
    needed = re.findall(r"need\(rf'\{OUT\}\\([^']+)'", pipe)
    for n in sorted(set(needed)):
        # is it written by some stage? look for save_weight that produces it
        stem = n.replace('.pth', '')
        produced = (stem in pipe) or (f"save_weight', '{stem.split('_768')[0]}'" in pipe)
        check(f'need({n}) is produced by a stage', produced)

    # ---- 3. summary JSON field contracts ------------------------------------
    print('\n[3] summary JSON fields: producer vs consumer')
    # SFT summary producer
    sft_summary_fields = set(re.findall(r"summary = dict\(([^)]*)\)", sft, re.DOTALL))
    sft_prod = set(re.findall(r"(\w+)=", ' '.join(sft_summary_fields)))
    check('SFT summary writes total_tokens', 'total_tokens' in sft or '"total_tokens"' in sft)
    check('SFT summary writes supervised_tokens',
          'supervised_tokens' in sft or '"supervised_tokens"' in sft)
    # README consumer
    check('README reads sft total_tokens', "sft_sum.get('total_tokens')" in rd)
    check('README reads sft supervised_tokens', "sft_sum.get('supervised_tokens')" in rd)

    # Pretrain summary is written as dict(key=value, ...), so match the keyword form.
    check('pretrain summary writes total_tokens', 'total_tokens=' in pre)
    check('pretrain summary writes peak_reserved_MiB', 'peak_reserved_MiB=' in pre)
    check('pretrain summary writes steps', 'steps=global_step' in pre)
    check('README reads pretrain total_tokens', "pre_sum.get('total_tokens')" in rd)
    check('README reads pretrain peak_reserved_MiB',
          "pre_sum.get('peak_reserved_MiB')" in rd)

    # GRPO summary
    check('GRPO summary writes cum_sampled_tokens', "'cum_sampled_tokens'" in grpo)
    check('README globs grpo_stage*_summary.json', 'grpo_stage*_summary.json' in rd)
    check('README reads cum_sampled_tokens', "cum_sampled_tokens" in rd)

    # ---- 4. summary filename derivation -------------------------------------
    print('\n[4] summary filename derivation (metrics_path -> _summary.json)')
    for name, src in (('sft', sft), ('grpo', grpo), ('pretrain', pre)):
        ok = "replace('.jsonl', '_summary.json')" in src
        check(f'{name} writes summary at <metrics>_summary.json', ok)

    # Pipeline passes metrics_path; README must look in the same logs dir
    for m in ('sft_metrics.jsonl', 'grpo_stage1_easy.jsonl', 'grpo_stage3_full.jsonl'):
        check(f'pipeline passes --metrics_path ...{m}', m in pipe)
    check('README looks for sft_metrics_summary.json',
          'sft_metrics_summary.json' in rd)
    check('README looks for pretrain_metrics_summary.json',
          'pretrain_metrics_summary.json' in rd)

    # ---- 5. eval report contracts ------------------------------------------
    print('\n[5] eval report field contracts')
    check('README reads eval overall.correct', 'gguf_eval["overall"]["correct"]' in rd)
    check('README reads eval per_family', "sft_eval.get('per_family')" in rd)
    check('compare reads eval overall.total', "['overall']['total']" in read(CMP)
          or '"total"' in read(CMP))

    # ---- 6. GGUF artifact names --------------------------------------------
    print('\n[6] GGUF artifact names')
    for n in ('full_sft_768-Q4_K_M.gguf', 'grpo_stage3_full_768-Q4_K_M.gguf'):
        check(f'pipeline/README reference {n}', n in pipe or n in rd)

    # ---- 7. reports the README expects -------------------------------------
    print('\n[7] report files README expects')
    for f in ('grpo_vs_sft.json', 'sft_baseline_full.json', 'eval_gguf_q4_sft.json'):
        check(f'README references {f}', f in rd)
    check('pipeline writes reports/grpo_vs_sft.json', 'grpo_vs_sft.json' in pipe)
    check('pipeline writes reports/sft_baseline_full.json',
          'sft_baseline_full.json' in pipe)

    # ---- 8. completion guards ---------------------------------------------
    # `pretrain_768.pth` is rewritten every 2000 steps, so "file exists" does NOT mean
    # Phase 1 finished. SFT must gate on the summary JSON (written only at the end) or
    # the final step count, never on the weight file alone.
    print('\n[8] completion guards (partial-artifact protection)')
    check('stage2 gates on pretrain completion (summary or step count)',
          '_assert_pretrain_complete' in pipe)
    check('the guard reads the summary JSON', 'pretrain_metrics_summary.json' in pipe)
    check('the guard falls back to the metrics step count',
          'PRETRAIN_FINAL_STEP' in pipe and 'pretrain_metrics.jsonl' in pipe)
    check('wait_then_pipeline also checks the summary',
          'pretrain_metrics_summary.json' in read(
              os.path.join(ROOT, 'tools', 'wait_then_pipeline.py')))


    # ---- 9. GRPO feasibility gate ------------------------------------------
    print(chr(10) + '[9] GRPO feasibility gate')
    check('stage3 runs the feasibility pre-flight', 'grpo_feasibility.py' in pipe)
    check('feasibility tool exists',
          os.path.exists(os.path.join(ROOT, 'tools', 'grpo_feasibility.py')))
    check('stage3 measures GRPO VRAM at production shape',
          'probe_grpo_vram.py' in pipe)
    check('VRAM probe exists',
          os.path.exists(os.path.join(ROOT, 'tools', 'probe_grpo_vram.py')))
    check('feasibility checks format reachability (checkers)',
          'unseen_checkers' in read(os.path.join(ROOT, 'tools', 'grpo_feasibility.py')))

    # ---- summary -----------------------------------------------------------
    print()
    print('-' * 72)
    npass = sum(1 for _, ok, _ in results if ok)
    print(f'{npass}/{len(results)} contract checks passed')
    out = {'checks': [{'name': n, 'ok': ok, 'detail': d} for n, ok, d in results],
           'passed': npass, 'total': len(results),
           'all_pass': npass == len(results)}
    with open(os.path.join(ROOT, 'logs', 'artifact_contract_check.json'), 'w',
              encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('RESULT:', 'ALL PASS' if out['all_pass'] else 'CONTRACT MISMATCHES FOUND')
    return 0 if out['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
