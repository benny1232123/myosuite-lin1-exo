"""Does the SB3 warm start reproduce the published policy's 4 s walk on the exo env?

Training logs show PPO's stochastic episodes dying after ~62 steps, while the same
checkpoint driven directly walks for 408 steps.  That gap means the SB3 setup itself
changes the policy.  Two candidate causes are compared here:

* ``vecnorm``  -- the transform SB3/VecNormalize applies: (x-m)/sqrt(v+1e-8), clip 10
* ``rsl``      -- the transform the checkpoint was trained with, rsl_rl
                 EmpiricalNormalization: (x-m)/sqrt(v+eps), eps = 1e-2, no clip

If the ``rsl`` columns walk and the ``vecnorm`` columns do not, the epsilon mismatch is
the bug and must be fixed in the training environment.
"""

from __future__ import annotations

import numpy as np
import torch
from stable_baselines3 import PPO

import myosuite  # noqa: F401
from myosuite import make_env
from myosuite.utils.checkpoint_utils import find_checkpoint
from myosuite.utils.rslrl_policy import load_rslrl_policy


def roll(env, act, seed=0, max_steps=1500):
    obs, _ = env.reset(seed=seed)
    for i in range(max_steps):
        obs, _r, terminated, truncated, _info = env.step(act(obs))
        if terminated or truncated:
            return i + 1
    return max_steps


def main() -> int:
    ckpt = find_checkpoint(env_id="myoLegWalk-v0")
    sd = torch.load(ckpt, map_location="cpu")["actor_state_dict"]
    mean = sd["obs_normalizer._mean"].numpy().ravel()
    var = sd["obs_normalizer._var"].numpy().ravel()

    env = make_env("myoLegWalkExo-v0").unwrapped

    # reference: the checkpoint driven through its own normaliser
    rsl = load_rslrl_policy(ckpt, env.action_space.shape[0])
    ref = roll(env, lambda o: rsl.act(np.atleast_2d(o))[0])
    print(f"reference (rsl_rl policy)          : {ref} steps")

    # the same tensors inside SB3
    policy = PPO("MlpPolicy", env,
                 policy_kwargs=dict(net_arch=[dict(pi=[128, 128, 128, 128],
                                                   vf=[256, 256, 256, 256])],
                                    activation_fn=torch.nn.ELU),
                 device="cpu", verbose=0)
    asd = sd
    csd = torch.load(ckpt, map_location="cpu")["critic_state_dict"]
    pn, vn = policy.policy.mlp_extractor.policy_net, policy.policy.mlp_extractor.value_net
    for i, k in enumerate(("mlp.0", "mlp.2", "mlp.4", "mlp.6")):
        for m, src in ((pn[2 * i], asd), (vn[2 * i], csd)):
            m.weight.data = torch.as_tensor(src[f"{k}.weight"], dtype=torch.float32)
            m.bias.data = torch.as_tensor(src[f"{k}.bias"], dtype=torch.float32)
    for m, src, k in ((policy.policy.action_net, asd, "mlp.8"),
                      (policy.policy.value_net, csd, "mlp.8")):
        m.weight.data = torch.as_tensor(src[f"{k}.weight"], dtype=torch.float32)
        m.bias.data = torch.as_tensor(src[f"{k}.bias"], dtype=torch.float32)

    def norm_vec(o, eps):
        # exact rsl_rl EmpiricalNormalization: eps is added to std, NOT inside the sqrt
        return ((o - mean) / (np.sqrt(var) + eps)).astype(np.float32)

    def act_factory(eps, seed_std=None):
        def act(o):
            a, _ = policy.predict(norm_vec(o, eps), deterministic=seed_std is None)
            return a

        return act

    for eps in (1e-8, 1e-2):
        n = roll(env, act_factory(eps))
        tag = "rsl_rl formula (std + 1e-2)" if eps == 1e-2 else "VecNormalize formula"
        print(f"SB3 tensors, eps={eps:<8g}             : {n} steps  {tag}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
