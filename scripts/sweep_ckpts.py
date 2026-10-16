"""Deterministic survival of SB3 checkpoints on the exo env, across several seeds.

Usage:
    python scripts/sweep_ckpts.py <policy.zip> [<policy.zip> ...]

Prints one row per checkpoint: how many control steps it survives for seeds 0..4, using
the float32 rsl_rl observation normalisation that the controller was trained with.
A checkpoint is only worth promoting if it holds all 1500 steps for every seed.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

import myosuite  # noqa: F401
from myosuite import make_env
from myosuite.utils.checkpoint_utils import find_checkpoint

from stable_baselines3 import PPO

STEPS = 1500
SEEDS = range(5)


def main() -> int:
    dt = np.float64 if "--f64" in sys.argv else np.float32
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    sd = torch.load(find_checkpoint(env_id="myoLegWalk-v0"),
                    map_location="cpu")["actor_state_dict"]
    mean = sd["obs_normalizer._mean"].numpy().astype(dt).ravel()
    scale = (np.sqrt(sd["obs_normalizer._var"].numpy().astype(dt).ravel())
             + dt(1e-2))
    env = make_env("myoLegWalkExo-v0").unwrapped

    for path in paths:
        try:
            ppo = PPO.load(path, device="cpu")
        except Exception as exc:  # noqa: BLE001
            print(f"{path}: LOAD FAILED ({exc})")
            continue
        surv = []
        for seed in SEEDS:
            obs, _ = env.reset(seed=seed)
            n = 0
            for i in range(STEPS):
                x = (np.asarray(obs, dtype=dt).ravel() - mean) / scale
                action, _ = ppo.predict(x, deterministic=True)
                obs, _r, terminated, truncated, _info = env.step(action)
                n = i + 1
                if terminated or truncated:
                    break
            surv.append(n)
        ok = sum(1 for v in surv if v >= STEPS)
        print(f"{path}: {surv}  -> passes {ok}/{len(surv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
