"""One-shot diagnostic for the myoLegWalkExo-v0 fine-tuning configuration.

Answers the questions that decide the PPO settings:

* how large is the raw walk reward per step (=> is reward normalisation needed?),
* how far does the zero-shot baseline actually get before terminating,
* what does the published action std look like (=> is ``log_std = log(std)`` right?),
* what are the raw / normalised observation ranges (=> are the obs_rms stats usable?),
* how big are the normalised advantages (=> why did the first PPO update blow up KL?).

Run from ``E:\\exo algorithm\\myosuite``::

    python scripts/diag_exo_reward.py
"""

from __future__ import annotations

import numpy as np
import torch

import myosuite  # noqa: F401
from myosuite import make_env
from myosuite.utils.checkpoint_utils import find_checkpoint, load_policy


def main() -> int:
    env = make_env("myoLegWalkExo-v0").unwrapped
    ckpt = find_checkpoint(env_id="myoLegWalk-v0")
    sd = torch.load(ckpt, map_location="cpu")["actor_state_dict"]

    # 1. action std from the published policy
    std = sd["distribution.std_param"].detach().cpu().numpy()
    print(f"published std_param: min={std.min():.4f} max={std.max():.4f} "
          f"mean={std.mean():.4f}  (log={np.log(np.clip(std, 1e-3, None)).mean():.4f})")

    # 2. observation normaliser statistics
    mean = sd["obs_normalizer._mean"].numpy().ravel()
    var = sd["obs_normalizer._var"].numpy().ravel()
    print(f"obs_rms: mean in [{mean.min():.3f},{mean.max():.3f}] "
          f"std in [{np.sqrt(var).min():.4f},{np.sqrt(var).max():.4f}]  dim={mean.size}")

    # 3. roll the zero-shot policy
    policy = load_policy(env, ckpt)
    obs, _ = env.reset(seed=0)
    rew, done_at, adv, terms = [], None, [], None
    for i in range(1500):
        a = policy(obs)
        obs, r, terminated, truncated, info = env.step(a)
        rew.append(float(r))
        xn = np.clip((obs - mean) / np.sqrt(var + 1e-8), -10, 10)
        adv.append(float(np.abs(xn).max()))
        if terminated or truncated:
            done_at = i + 1
            terms = info.get("done_type", info.get("termination", "?"))
            break

    rew = np.asarray(rew)
    print(f"\nzero-shot on exo env: survived {'15.0 s (FULL)' if done_at is None else f'{done_at*0.01:.2f} s'}"
          f"  steps={done_at or 1500}  done_type={terms}")
    print(f"reward/step: mean={rew.mean():.4f} min={rew.min():.3f} max={rew.max():.3f} "
          f"std={rew.std():.4f}")
    print(f"episode return (truncated at step {len(rew)}): {rew.sum():.1f}")
    print(f"raw obs |max| over rollout: {max(adv):.2f}  (this is the *normalised* max |obs|)")

    # 4. what a 10-step raw return looks like, i.e. the scale a fresh ret_rms would divide by
    print(f"\nreward magnitude => norm_reward={'N/A' if rew.mean() < 5 else 'was the problem'}")
    print(f"a fresh VecNormalize ret_rms (var=1) would divide reward by "
          f"{np.sqrt(1.0):.2f}; raw mean reward is {rew.mean():.4f}; after 1 discount cycle "
          f"std would be ~{np.std(np.cumsum(rew) / (np.arange(len(rew)) + 1)):.3f}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
