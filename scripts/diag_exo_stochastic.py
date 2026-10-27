"""Compare deterministic vs stochastic rollouts of the published baseline on the exo env.

The acceptance test drives the policy deterministically (mean action), while PPO
optimises the stochastic policy (mean + learned std).  For muscle tasks these can differ
a lot, so measure both before blaming the dynamics.
"""

from __future__ import annotations

import numpy as np
import torch

import myosuite  # noqa: F401
from myosuite import make_env
from myosuite.utils.checkpoint_utils import find_checkpoint
from myosuite.utils.rslrl_policy import load_rslrl_policy


def roll(env, policy, stochastic: bool, seed: int, max_steps: int = 1500, std_scale: float = 1.0):
    obs, _ = env.reset(seed=seed)
    total = 0.0
    saved = policy.std.clone() if policy.std is not None else None
    if saved is not None:
        policy.std.copy_(saved * std_scale)
    try:
        for i in range(max_steps):
            a = policy.act(np.atleast_2d(obs), stochastic=stochastic)[0]
            obs, r, terminated, truncated, info = env.step(a)
            total += float(r)
            if terminated or truncated:
                return i + 1, total, info.get("done_type", info.get("termination", "?"))
        return max_steps, total, "full"
    finally:
        if saved is not None:
            policy.std.copy_(saved)


def main() -> int:
    env = make_env("myoLegWalkExo-v0").unwrapped
    ckpt = find_checkpoint(env_id="myoLegWalk-v0")
    a_dim = env.action_space.shape[0]
    policy = load_rslrl_policy(ckpt, a_dim)
    std = policy.std.numpy() if policy.std is not None else None
    print(f"std: mean={std.mean():.4f} min={std.min():.4f} max={std.max():.4f}")

    print("\n-- deterministic (mean action) --")
    for s in range(3):
        n, ret, why = roll(env, policy, False, s)
        print(f"  seed {s}: {n:5d} steps ({n*0.01:5.2f} s)  return={ret:8.1f}  {why}")

    print("\n-- stochastic (sampled, as PPO trains) --")
    for s in range(3):
        n, ret, why = roll(env, policy, True, s)
        print(f"  seed {s}: {n:5d} steps ({n*0.01:5.2f} s)  return={ret:8.1f}  {why}")

    print("\n-- stochastic on the exo env at reduced exploration std --")
    for scale in (1.0, 0.5, 0.25, 0.1, 0.05):
        lens = []
        for s in range(4):
            n, _ret, _why = roll(env, policy, True, s, std_scale=scale)
            lens.append(n)
        print(f"  std x{scale:<4} ({std.mean()*scale:.4f}): "
              f"steps={lens}  mean={np.mean(lens):.0f}  "
              f"survive15={sum(1 for n in lens if n >= 1500)}/4")

    print("\n-- deterministic on the PLAIN env (no exo), same policy --")
    env2 = make_env("myoLegWalk-v0").unwrapped
    for s in range(2):
        n, ret, why = roll(env2, policy, False, s)
        print(f"  seed {s}: {n:5d} steps ({n*0.01:5.2f} s)  return={ret:8.1f}  {why}")
    env.close(); env2.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
