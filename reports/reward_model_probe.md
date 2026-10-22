# Reward Model Probe — internlm2-1_8b-reward

Date: 2026-10-21
Host: Windows, RTX 4060 Laptop (~8 GB total / ~7.3 GB free at probe time)

## 1. Model id

```
Shanghai_AI_Laboratory/internlm2-1_8b-reward
```

The **exact** requested id was available on ModelScope — no fallback to `InternLM/...` was needed.
Revision used: `master`.

Local directory: `D:\llm-rl-lab\models\internlm2-1_8b-reward`

## 2. Downloaded files

15 entries in the repo (14 blobs + 1 empty `reward_bench_results/` tree, not downloaded).

| File | Bytes |
|---|---:|
| `.gitattributes` | 1,519 |
| `README.md` | 16,491 |
| `config.json` | 813 |
| `configuration.json` | 52 |
| `configuration_internlm2.py` | 9,042 |
| `modeling_internlm2.py` | 91,364 |
| `model.safetensors.index.json` | 13,682 |
| `model-00001-of-00002.safetensors` | 1,981,392,544 |
| `model-00002-of-00002.safetensors` | 1,417,790,344 |
| `special_tokens_map.json` | 809 |
| `tokenization_internlm2.py` | 8,806 |
| `tokenization_internlm2_fast.py` | 7,805 |
| `tokenizer.model` | 1,477,754 |
| `tokenizer_config.json` | 2,950 |

**Total: 3,400,813,975 bytes = 3,243.27 MiB = 3.167 GiB**

Integrity: every file's byte size matches the ModelScope listing, and both shard SHA256
hashes match the remote values exactly:

```
b0cbcdad5899d93dd653a4bf7fdf65e1d9668aaf61c0d5d41be1529e96b902ea  model-00001-of-00002.safetensors
6421a9ec59b5e7b127c1b99fb80b88e5e013a9f2769a2f3000b8c99c99ad4630  model-00002-of-00002.safetensors
```

Download method: 16-way parallel HTTP Range download via `curl.exe -r start-end` (server
returned HTTP 206), parts concatenated in order. Sustained ~65 MB/s (68.9 / 63.6 MB/s per
shard), ~49 s wall clock for the 3.4 GB of weights.

## 3. Load verification — SUCCESS

`trust_remote_code=True` works fine under transformers 4.57.6. No traceback.

```
tokenizer : InternLM2TokenizerFast        OK
model     : InternLM2ForRewardModel       OK  (torch.float16)
device    : cuda, eval mode
```

Peak VRAM (after load + both scoring calls):

- **peak allocated: 3,253.9 MiB**
- peak reserved: 3,316.0 MiB
- GPU free before load: 7,106 MiB of 8,187.5 MiB

So the reward model fits comfortably alongside other workloads in the ~6.7 GB usable budget,
leaving roughly 3.4 GB headroom.

### get_score

`model.get_score` **exists and works** (`@torch.no_grad`, signature
`get_score(tokenizer, conversation: List[dict], **kwargs)`; it appends the
`reward_token_id=92527` token and gathers the last-token score).

| Conversation | Score |
|---|---:|
| user `1+1等于几` / assistant `1+1等于2` (correct) | **0.8916015625** |
| user `1+1等于几` / assistant `1+1等于3` (wrong) | 0.52880859375 |

The correct answer scores higher, so the model is discriminating as expected (the absolute
values are both positive, which is normal — MiniMind clamps to [-3, 3] anyway).

### Benign warnings observed (not errors)

```
`torch_dtype` is deprecated! Use `dtype` instead!
past_key_values should not be None in from_legacy_cache()
```

Both are deprecation noise under transformers 4.57.6; neither affects the result. The
`torch_dtype` argument is still accepted.

## 4. Exact working command

Environment: `D:\llm-rl-lab\.venv\Scripts\python.exe`

```powershell
$py = "D:\llm-rl-lab\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
& $py D:\llm-rl-lab\logs\probe_reward.py
```

The probe script (`D:\llm-rl-lab\logs\probe_reward.py`) does exactly this:

```python
import torch
from transformers import AutoTokenizer, AutoModel

PATH = r"D:\llm-rl-lab\models\internlm2-1_8b-reward"

tok = AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(PATH, torch_dtype=torch.float16, trust_remote_code=True)
model = model.to("cuda").eval()

score = model.get_score(tok, [
    {"role": "user", "content": "1+1等于几"},
    {"role": "assistant", "content": "1+1等于2"},
])
print(score)                                     # 0.8916015625
print(torch.cuda.max_memory_allocated() / 1024**2)  # 3253.9 MiB
```

## 5. Integration note for MiniMind GRPO

MiniMind's `trainer/trainer_utils.py::LMForRewardModel` uses the **same** load call
(`AutoTokenizer`/`AutoModel` with `trust_remote_code=True`, `torch_dtype=torch.float16`,
`.to(device).eval()`) and calls `self.model.get_score(self.tokenizer, eval_messages)`.
The probe therefore exercises the real integration path, not just a lookalike.

However, `train_grpo.py` / `train_ppo.py` / `train_agent.py` default
`--reward_model_path` to `../../internlm2-1_8b-reward`, which from
`D:\llm-rl-lab\src\minimind-master\trainer` resolves to
`D:\llm-rl-lab\src\internlm2-1_8b-reward` — **not** the download location. Pass the path
explicitly when launching:

```powershell
& $py train_grpo.py --reward_model_path D:\llm-rl-lab\models\internlm2-1_8b-reward ...
```

(Alternative: create a directory junction at that default path. Not done here, as it would
write outside the permitted directories.)

## 6. Packages installed

**None.** Everything ran with the pre-existing venv (torch 2.6.0+cu124, transformers 4.57.6,
modelscope 1.37.0). The `modelscope` SDK was not even needed — plain `curl.exe` Range
downloads were used.

## 7. Blockers

None. Model downloaded, verified, loads on GPU, and `get_score` returns sensible values.
