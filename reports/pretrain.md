# Phase 1 — Pretraining (from random initialization)

Status: **complete** — 134,817 optimizer steps, **2,208,841,728 tokens** fed to the model.

Every number below is read from `logs/pretrain_metrics.jsonl` (written by the trainer
at each log interval) and `data/tokens/pretrain_meta.json` (written by the tokenizer).
Nothing is estimated.

## 1. Configuration

| Item | Value |
|---|---|
| Model | MiniMind `MiniMindConfig(hidden_size=768, num_hidden_layers=8)` |
| Parameters | **63,912,192** (63.91 M), trained **from random init** (`--from_weight none`) |
| Vocab | 6,400 (MiniMind tokenizer) |
| Sequence length | 512 tokens, packed blocks (no padding) |
| Batch size | 32 x 512 = 16,384 tokens / step |
| Optimizer | AdamW, lr 5e-4, cosine schedule (`trainer_utils.get_lr`, annealed to 0.1x) |
| Precision | bf16 autocast, `torch.compile` (triton-windows 3.2.0.post21) |
| Seed | 42 |
| Corpus | MiniMind `pretrain_t2t.jsonl`, 8,468,827 documents |

## 2. Token accounting (measured, not estimated)

| Quantity | Value | Basis |
|---|---|---|
| Corpus tokens (tokenizer output, incl. 1 EOS/doc) | **2,208,852,597** | `tokenizer._tokenizer.encode_batch` over every document |
| Packed blocks x seq_len | 2,208,852,480 | 4,314,165 blocks x 512 |
| **Tokens actually consumed by training** | **2,208,841,728** | sum of `input_ids.numel()` over every step |
| Steps | 134,817 | |
| Wall time (all segments) | **24.88 h** | sum of 4 process segments (the run was paused/resumed; the final segment alone was 4.43 h) |

Note: 117 tail tokens are not a whole block and are not used; `drop_last=True` discards the last partial batch, so `steps x tokens/step = 2,208,841,728` is the authoritative figure.

## 3. Loss

| Metric | Value |
|---|---|
| First logged loss (step 100) | 6.4024 |
| Mean of first 10 % of logged points | 2.4760 |
| Mean of last 10 % of logged points | 1.4079 |
| Last logged loss (step 134,817) | 1.4947 |
| Minimum logged loss | 0.7286 (step 129,900) |
| Reduction, first logged -> last logged | **+4.9077** (76.7 %) |
| Reduction, head 10 % -> tail 10 % of logged points | +1.0681 (43.1 %) |

> Two reduction figures are reported on purpose. *First logged -> last logged* is the
> headline number (6.40 -> 1.81). *Head 10 % -> tail 10 %* is the conservative one:
> the first 10 % of logged points are already past the initial cliff, so it measures
> the remaining descent and is the fairer basis for "is it still improving".

![pretrain loss curve](../figures/pretrain_curve_full.png)

## 4. Throughput and memory

| Metric | Value |
|---|---|
| Mean tokens/s (run-average over log points) | 24,646 |
| Peak tokens/s (run-average over log points) | 26,288 |
| **Peak reserved VRAM** | **5,296 MiB** (64.7 % of the 8,188 MiB card) |
| Peak allocated VRAM | 5,265 MiB |

## 5. Checkpoints

```
D:\llm-rl-lab\src\minimind-master\out\pretrain_768.pth            (final)
D:\llm-rl-lab\src\minimind-master\out\pretrain_768_stepN.pth       (every 2000 steps)
D:\llm-rl-lab\src\minimind-master\checkpoints\pretrain_768_resume.pth (optimizer state)
```

## 6. Reproduce

```powershell
# 1) tokenize the corpus into packed blocks (once)
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\custom\pretokenize_packed.py `
  --src D:\llm-rl-lab\data\pretrain_t2t.jsonl --out D:\llm-rl-lab\data\tokens `
  --seq-len 512 --batch-docs 2000 --tag pretrain

# 2) train
cd D:\llm-rl-lab\src\minimind-master\trainer
D:\llm-rl-lab\.venv\Scripts\python.exe D:\llm-rl-lab\custom\train_pretrain_packed.py `
  --data_path D:\llm-rl-lab\data\tokens\pretrain_blocks.bin --max_seq_len 512 `
  --batch_size 32 --epochs 1 --max_steps 134817 --learning_rate 5e-4 `
  --log_interval 100 --save_interval 2000 --num_workers 4 --use_compile 1 `
  --hidden_size 768 --num_hidden_layers 8 --seed 42 `
  --save_dir D:/llm-rl-lab/src/minimind-master/out --save_weight pretrain `
  --metrics_path D:/llm-rl-lab/logs/pretrain_metrics.jsonl
```
