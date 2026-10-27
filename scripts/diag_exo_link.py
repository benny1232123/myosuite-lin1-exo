"""Per-side check of the exo thigh link: does it hug the thigh, and does it wobble?

The complaint is asymmetric - one link looks detached from the leg, the other looks like
it is swinging.  Everything here is measured from a real rollout of the acceptance
controller on ``myoLegWalkExo-v0``; nothing is assumed.

Reported per side:
* the perpendicular distance from the link body to the femur axis at reset and over the
  walk (large / varying = the link is standing off the thigh),
* the cuff-weld site gap over the walk (the weld is deliberately soft, so a large or
  oscillating gap is the mechanism behind a wobble),
* the three exo joint coordinates over the walk, as min/max/peak-to-peak.
"""

from __future__ import annotations

import numpy as np

import myosuite  # noqa: F401
from myosuite import make_env
from myosuite.utils.checkpoint_utils import find_checkpoint

POLICY = "walk_exo_policy.zip"
STEPS = 1500


def build_policy(env):
    import torch
    from stable_baselines3 import PPO

    sb3 = PPO.load(POLICY, device="cpu")
    sd = torch.load(find_checkpoint(env_id="myoLegWalk-v0"),
                    map_location="cpu")["actor_state_dict"]
    mean = sd["obs_normalizer._mean"].numpy().astype(np.float32).ravel()
    scale = np.sqrt(sd["obs_normalizer._var"].numpy().astype(np.float32).ravel()) \
        + np.float32(1e-2)
    return lambda obs: sb3.predict(
        ((np.asarray(obs, dtype=np.float32).ravel() - mean) / scale),
        deterministic=True)[0]


def dist_to_femur_axis(model, data, side: str, point: np.ndarray) -> float:
    """Perpendicular distance from a world point to the femur's axis line."""
    import mujoco

    hip = data.xpos[model.body(f"femur_{side}").id].copy()
    knee = data.xpos[model.body(f"tibia_{side}").id].copy()
    axis = knee - hip
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    rel = point - hip
    return float(np.linalg.norm(rel - (rel @ axis) * axis))


def main() -> int:
    import mujoco

    env = make_env("myoLegWalkExo-v0").unwrapped
    model, data = env.model, env.data
    policy = build_policy(env)
    obs, _ = env.reset(seed=0)
    mujoco.mj_forward(model, data)

    dofs = {s: [f"exo_hip_flexion_{s}", f"exo_slot_{s}", f"exo_axial_roll_{s}"]
            for s in ("r", "l")}
    adr = {s: [int(model.joint(n).qposadr[0]) for n in dofs[s]] for s in dofs}
    cuff_site = {s: model.site(f"exo_cuff_site_{s}").id for s in ("r", "l")}
    attach_site = {s: model.site(f"exo_attach_{s}").id for s in ("r", "l")}

    link_off = {s: [] for s in ("r", "l")}
    weld_gap = {s: [] for s in ("r", "l")}
    jq = {s: {n: [] for n in dofs[s]} for s in dofs}

    n_run = STEPS
    for i in range(STEPS):
        for s in ("r", "l"):
            link_off[s].append(dist_to_femur_axis(
                model, data, s, data.xpos[model.body(f"exo_link_{s}").id].copy()))
            weld_gap[s].append(float(np.linalg.norm(
                data.site_xpos[cuff_site[s]] - data.site_xpos[attach_site[s]])))
            for n in dofs[s]:
                jq[s][n].append(float(data.qpos[adr[s][dofs[s].index(n)]]))
        obs, _r, terminated, truncated, _i = env.step(policy(obs))
        mujoco.mj_forward(model, data)
        if terminated or truncated:
            n_run = i + 1
            break

    print(f"steps run: {n_run} ({n_run * 0.01:.2f} s)\n")
    for s in ("r", "l"):
        off = np.array(link_off[s][:n_run])
        gap = np.array(weld_gap[s][:n_run])
        print(f"--- side {s} ---")
        print(f"  link->femur-axis distance [mm]: start {off[0]*1e3:7.1f}  "
              f"mean {off.mean()*1e3:7.1f}  min {off.min()*1e3:7.1f}  "
              f"max {off.max()*1e3:7.1f}")
        print(f"  cuff weld gap [mm]           : start {gap[0]*1e3:7.1f}  "
              f"mean {gap.mean()*1e3:7.1f}  max {gap.max()*1e3:7.1f}")
        for n in dofs[s]:
            v = np.array(jq[s][n][:n_run])
            print(f"  {n:22s}: start {v[0]: .4f}  min {v.min(): .4f}  "
                  f"max {v.max(): .4f}  p2p {v.max()-v.min():.4f}")
        print()

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
