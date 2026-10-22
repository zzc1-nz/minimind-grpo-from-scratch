# Resume bullets (measured only)

Format: **action + quantified result**. Every number is read from the run artifacts; nothing is estimated.

1. Built a from-scratch LLM training pipeline (PyTorch, no pretrained weights) and trained a **63.9M-parameter Transformer from random initialization** on a single 8 GB laptop GPU, consuming **2,208,841,728 tokens** (2.21B) in 24.9 h at **5,296 MiB peak VRAM** (64.7% of the card).

2. Implemented **GRPO reinforcement-learning alignment with programmatically verifiable rewards** (binary 1/0 checker, no LLM judge) over a staged easy-to-hard curriculum, sampling **1,639,872 rollout tokens** in 1200 steps; raised held-out accuracy from **7/24 (29.2%) to 14/24 (58.3%)**, delta +7 tasks (McNemar exact p=0.0390625).

3. Deployed the trained model locally as **GGUF Q4_K_M** (**42.5 MB**, 3.0x smaller than f16) and served it with llama.cpp at **518 tok/s on CPU and 939 tok/s on GPU**, with a measured held-out accuracy drop of **2 task(s)** (14 → 12 of 24) after quantization.

---

### Supporting figures

- pretrain: 2,208,841,728 tokens, 5296 MiB peak, 4.4 h
- SFT: 153,600,000 tokens (66,203,231 supervised)
- GRPO: 1,639,872 sampled tokens over 1200 steps
- grand total: **2,364,081,600 tokens**
