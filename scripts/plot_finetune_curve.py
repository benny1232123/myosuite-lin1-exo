"""Plot the PPO fine-tuning curve from SB3's TensorBoard event files."""

from __future__ import annotations

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    EventAccumulator = None

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TB = r"E:\exo algorithm\myosuite\tb_logs"
files = sorted(glob.glob(os.path.join(TB, "**", "events.out.tfevents.*"), recursive=True))
if not files:
    raise SystemExit(f"no TensorBoard events under {TB} - is training still starting?")
acc = EventAccumulator(os.path.dirname(files[-1]), size_guidance={"scalars": 0})
acc.Reload()

tags = acc.Tags()["scalars"]
print("available tags:", tags)

def series(tag):
    if tag not in tags:
        return None, None
    ev = acc.Scalars(tag)
    return [e.step for e in ev], [e.value for e in ev]

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
want = [
    ("rollout/ep_rew_mean", "回合回报"),
    ("rollout/ep_len_mean", "回合长度"),
    ("train/loss", "训练 loss"),
    ("train/approx_kl", "KL 散度"),
]
for ax, (tag, label) in zip(axes.ravel(), want):
    xs, ys = series(tag)
    if xs:
        ax.plot(xs, ys, color="#1f5fd0", lw=1.2)
    ax.set_xlabel("PPO 更新")
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.grid(alpha=0.3)
    if tag == "rollout/ep_len_mean":
        ax.axhline(1500, color="r", ls="--", lw=1, label="15 s 目标")
        ax.legend()

fig.suptitle("myoLegWalkExo-v0 PPO 微调曲线（TensorBoard）", fontsize=14)
fig.tight_layout()
out = r"E:\exo algorithm\finetune_curve.png"
fig.savefig(out, dpi=130)
print("wrote", out)
