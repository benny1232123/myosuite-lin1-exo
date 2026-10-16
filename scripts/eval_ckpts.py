"""Deterministic survival of every fine-tune checkpoint on the exo env.

Training optimises the *stochastic* return, so the best checkpoint is not necessarily the
last one.  This rolls each saved policy greedily and prints how long it walks.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch

import myosuite  # noqa: F401
from myosuite import make_env
from myosuite.utils.checkpoint_utils import find_checkpoint

from stable_baselines3 import PPO

os.environ.setdefault("MUJOCO_GL", "wgl")


def main() -> int:
    sd = torch.load(find_checkpoint(env_id="myoLegWalk-v0"), map_location="cpu")["actor_state_dict"]
    mean = sd["obs_normalizer._mean"].numpy().ravel()
    scale = np.sqrt(sd["obs_normalizer._var"].numpy().ravel()) + 1e-2

    env = make_env("myoLegWalkExo-v0").unwrapped

    def roll(pol, seed=0, n=1500):
        obs, _ = env.reset(seed=seed)
        for i in range(n):
            x = ((np.asarray(obs, dtype=np.float32).ravel() - mean) / scale).astype(np.float32)
            a, _ = pol.predict(x, deterministic=True)
            obs, _r, term, trunc, _info = env.step(a)
            if term or trunc:
                return i + 1
        return n

    paths = sorted(glob.glob("finetune_ckpt/ppo_*_steps.zip"))
    paths.append("finetuned_myoLegWalkExo.zip")
    best = (0, None)
    for p in paths:
        if not os.path.exists(p):
            continue
        pol = PPO.load(p, device="cpu")
        steps = roll(pol)
        # a single seed is noisy; confirm the leaders on two more
        seeds = [roll(pol, seed=s) for s in (1, 2)] if steps > 1000 else []
        print(f"{os.path.basename(p):<32} seed0 {steps:>5}   others {seeds}")
        if steps > best[0]:
            best = (steps, p)
    print(f"BEST: {best[1]}  ({best[0]} steps = {best[0] / 100:.2f} s)")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
