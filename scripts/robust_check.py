"""Is the 15 s walk robust, or a knife-edge?

The acceptance script and the checkpoint sweep disagreed on the same policy (749 vs 1500
steps), which means the gait is sensitive.  This rolls the best checkpoint deterministically
over several seeds, with the observation normalisation evaluated in float64 (as the
acceptance script does) and in float32 (as the sweep does).
"""

from __future__ import annotations

import os

import numpy as np
import torch

import myosuite  # noqa: F401
from myosuite import make_env
from myosuite.utils.checkpoint_utils import find_checkpoint

from stable_baselines3 import PPO

os.environ.setdefault("MUJOCO_GL", "wgl")
CKPT = os.environ.get("CKPT", "finetune_ckpt/ppo_1000000_steps.zip")


def main() -> int:
    sd = torch.load(find_checkpoint(env_id="myoLegWalk-v0"), map_location="cpu")["actor_state_dict"]
    mean = sd["obs_normalizer._mean"].numpy().ravel()
    var = sd["obs_normalizer._var"].numpy().ravel()

    env = make_env("myoLegWalkExo-v0").unwrapped
    pol = PPO.load(CKPT, device="cpu")

    def roll(seed, dtype):
        m = np.asarray(mean, dtype=dtype)
        v = np.asarray(var, dtype=dtype)
        scale = np.sqrt(v) + 1e-2
        obs, _ = env.reset(seed=seed)
        for i in range(1500):
            x = ((np.asarray(obs, dtype=dtype).ravel() - m) / scale).astype(np.float32)
            a, _ = pol.predict(x, deterministic=True)
            obs, _r, term, trunc, _info = env.step(a)
            if term or trunc:
                return i + 1
        return 1500

    for dtype in (np.float64, np.float32):
        res = [roll(s, dtype) for s in range(6)]
        print(f"{CKPT}  {dtype.__name__:<8} steps per seed 0..5: {res}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
