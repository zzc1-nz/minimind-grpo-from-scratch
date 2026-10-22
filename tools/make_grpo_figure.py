"""Build the GRPO curriculum figure for the README."""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

LOGS = r'D:\llm-rl-lab\logs'
OUT = r'D:\llm-rl-lab\figures\grpo_curriculum_clean.png'

stages = [
    ('stage1_easy', 'Stage 1: easy (150 tasks, 200 steps)', '#2E86AB'),
    ('stage2_medium', 'Stage 2: medium (350 tasks, 400 steps)', '#A23B72'),
    ('stage3_full', 'Stage 3: full (600 tasks, 600 steps)', '#F18F01'),
]

series = []
for name, label, color in stages:
    rows = []
    with open(rf'{LOGS}\grpo_{name}.jsonl', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    series.append((name, label, color, rows))

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=140)
ax = axes[0]
offset = 0
for name, label, color, rows in series:
    x = np.arange(offset, offset + len(rows))
    reward = np.array([r['reward_mean'] for r in rows])
    smooth = np.array([r['reward_mean_20'] for r in rows])
    ax.scatter(x, reward, s=6, alpha=0.22, color=color)
    ax.plot(x, smooth, lw=2.2, color=color, label=label)
    if offset:
        ax.axvline(offset, ls='--', color='gray', lw=1, alpha=0.7)
    offset += len(rows)
ax.set_xlabel('GRPO step (three stages concatenated)', fontsize=11)
ax.set_ylabel('reward (1.0 = correct answer)', fontsize=11)
ax.set_title('(a) Reward rises across the easy-to-hard curriculum',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(alpha=0.3)
ax.set_ylim(-0.05, 1.05)

ax = axes[1]
offset = 0
for name, label, color, rows in series:
    acc = np.array([r['acc'] for r in rows])
    x = np.arange(offset, offset + len(acc))
    ax.scatter(x, acc, s=6, alpha=0.22, color=color)
    w = 20
    if len(acc) >= w:
        k = np.convolve(acc, np.ones(w) / w, mode='valid')
        ax.plot(np.arange(len(k)) + offset + w - 1, k, lw=2.2, color=color, label=label)
    if offset:
        ax.axvline(offset, ls='--', color='gray', lw=1, alpha=0.7)
    offset += len(acc)
ax.set_xlabel('GRPO step', fontsize=11)
ax.set_ylabel('training accuracy on rollout tasks', fontsize=11)
ax.set_title('(b) Training accuracy on the sampled tasks',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(alpha=0.3)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches='tight')
print('saved', OUT)
