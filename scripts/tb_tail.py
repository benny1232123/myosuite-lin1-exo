"""Print the tail of the SB3 TensorBoard rollout/train scalars for the fine-tune."""

from __future__ import annotations

import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TB = r"E:\exo algorithm\myosuite\tb_logs"
files = sorted(glob.glob(os.path.join(TB, "**", "events.out.tfevents.*"), recursive=True))
if not files:
    raise SystemExit(f"no events under {TB}")

acc = EventAccumulator(os.path.dirname(files[-1]), size_guidance={"scalars": 0})
acc.Reload()
tags = sorted(acc.Tags()["scalars"])
print("tags:", tags)

for tag in ("rollout/ep_len_mean", "rollout/ep_rew_mean", "train/approx_kl",
            "train/explained_variance", "train/value_loss", "time/fps"):
    if tag not in tags:
        print(f"{tag:28s} (missing)")
        continue
    ev = acc.Scalars(tag)
    head = ", ".join(f"{e.value:.1f}" for e in ev[:3])
    tail = ", ".join(f"{e.value:.1f}" for e in ev[-5:])
    print(f"{tag:28s} n={len(ev):3d}  first[{head}]  last[{tail}]")
