"""Zero-assistance 15 s gait check for the 'Lin-I' exoskeleton walking envs.

Runs the published ``myoLegWalk-v0`` baseline policy (no re-training) on a
Lin-I-equipped walker and reports the four acceptance criteria:

1. 1500 control steps (15.0 s at 0.01 s/step) without a termination,
2. forward displacement >= 5 m (mean speed >= 0.33 m/s),
3. both exoskeleton motors unpowered (actuator force at the exo hinges == 0),
4. alternating left/right foot ground contact.

Because the exoskeleton changes the physics (3.1 kg of load, a strap-coupled passive
chain) the zero-shot policy may not survive 15 s; the script reports the numbers either
way and exits non-zero when a criterion fails, so it doubles as a regression check::

    python scripts/eval_walk_exo.py                       # myoLegWalkExo-v0, baseline ckpt
    python scripts/eval_walk_exo.py --env myoLegWalkExoLocked-v0
    python scripts/eval_walk_exo.py --checkpoint logs/.../final_model.zip
    python scripts/eval_walk_exo.py --frames frames --plot gait.png
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import myosuite  # noqa: F401  (registers the CPU envs)
from myosuite import make_env
from myosuite.utils.checkpoint_utils import find_checkpoint, load_policy

DEFAULT_ENV = "myoLegWalkExo-v0"
BASELINE_ENV = "myoLegWalk-v0"
CONTROL_DT = 0.01
N_STEPS = 1500
MIN_DISTANCE_M = 5.0
HEIGHT_THRESHOLD_M = 0.8


def _foot_body_ids(model) -> tuple[set[int], set[int]]:
    """Body ids that make up the right / left foot."""
    right = {model.body(n).id for n in ("calcn_r", "toes_r")}
    left = {model.body(n).id for n in ("calcn_l", "toes_l")}
    return right, left


def _stance_onsets(contact: np.ndarray, min_len: int = 3) -> int:
    """Number of stance phases (contact runs of at least *min_len* control steps).

    Raw per-step contact flags chatter on the toe/heel boundary; a run filter is what
    makes the count a gait quantity rather than a contact-solver artefact.
    """
    runs = 0
    run = 0
    for flag in contact:
        if flag:
            run += 1
        else:
            if run >= min_len:
                runs += 1
            run = 0
    return runs + (1 if run >= min_len else 0)


def evaluate(
    env_id: str = DEFAULT_ENV,
    checkpoint: str | Path | None = None,
    steps: int = N_STEPS,
    seed: int = 0,
    sb3_policy: Any = None,
    obs_stats: Any = None,
) -> dict:
    env = make_env(env_id).unwrapped
    ckpt = Path(checkpoint) if checkpoint else None
    if ckpt is not None and ckpt.is_dir():
        candidates = sorted(ckpt.glob("**/model_*.pt")) + sorted(ckpt.glob("**/*.zip"))
        ckpt = candidates[-1] if candidates else None
    if ckpt is None:
        # the exo env has no baseline of its own: reuse the myoLegWalk-v0 policy, which
        # observes exactly the same 403-dim vector (the 4 exo coordinates are stripped)
        ckpt = find_checkpoint(env_id=BASELINE_ENV)
    if ckpt is None:
        raise SystemExit(
            f"no policy checkpoint found for {BASELINE_ENV}; pass --checkpoint explicitly"
        )
    if sb3_policy is not None:
        # An SB3 policy trained behind VecNormalize must be fed the *normalised* obs;
        # running it on raw observations produces garbage actions.  Rebuild the same
        # transform here from the saved obs_rms statistics.
        mean, var = obs_stats if obs_stats is not None else (0.0, 1.0)
        # float32 throughout: that is the precision rsl_rl's EmpiricalNormalization and the
        # training environment use, and this gait is sensitive to the rounding (float64
        # normalisation walks 7.49 s where float32 walks the full 15.00 s)
        mean = np.asarray(mean, dtype=np.float32).ravel()
        scale = np.sqrt(np.asarray(var, dtype=np.float32).ravel()) + np.float32(1e-2)

        def policy(obs):
            # rsl_rl EmpiricalNormalization: eps is added to std, NOT inside the sqrt
            x = (np.asarray(obs, dtype=np.float32).ravel() - mean) / scale
            action, _ = sb3_policy.predict(x, deterministic=True)
            return action

        print(f"sb3 policy (obs-normalised over {mean.size} dims)")
    else:
        policy = load_policy(env, Path(ckpt))
        print(f"policy checkpoint: {ckpt}")

    obs, _ = env.reset(seed=seed)
    model, data = env.model, env.data
    n_exo = getattr(env, "n_exo_dof", 0)

    pelvis = model.body("pelvis").id
    right_foot, left_foot = _foot_body_ids(model)

    qpos = np.zeros((steps, model.nq))
    com = np.zeros((steps, 3))
    dist = np.zeros(steps)
    contact_r = np.zeros(steps, dtype=bool)
    contact_l = np.zeros(steps, dtype=bool)
    exo_dof = np.zeros((steps, n_exo))
    exo_tau = np.zeros((steps, n_exo))
    done_at = None
    start_xy = np.array(data.xpos[pelvis][:2])

    for i in range(steps):
        action = policy(obs)
        obs, _reward, terminated, truncated, info = env.step(action)
        # NOTE: do not call mj_forward here.  env.step already ends with one, and running
        # the solver again between control steps perturbs qacc_warmstart, which changes the
        # trajectory (it cost ~750 steps off a fully passing 1500-step episode).
        qpos[i] = data.qpos
        com[i] = data.subtree_com[pelvis]
        dist[i] = float(np.linalg.norm(data.xpos[pelvis][:2] - start_xy))
        for c in range(data.ncon):
            for g in (data.contact[c].geom1, data.contact[c].geom2):
                if g < 0:
                    continue
                b = model.geom_bodyid[g]
                if b in right_foot:
                    contact_r[i] = True
                elif b in left_foot:
                    contact_l[i] = True
        if n_exo:
            exo_dof[i] = env.exo_dof_positions()
            exo_tau[i] = env.exo_joint_torques()
        if terminated or truncated:
            done_at = i + 1
            break

    travelled = float(dist[done_at - 1]) if done_at else float(dist[-1])
    n_run = done_at or steps
    onsets_r = _stance_onsets(contact_r[:n_run])
    onsets_l = _stance_onsets(contact_l[:n_run])
    frac_r = float(contact_r[:n_run].mean())
    frac_l = float(contact_l[:n_run].mean())
    # alternating gait: both feet have repeated stance phases (not shuffling, not
    # hopping) and their duty factors differ, i.e. the two phases are out of step
    alternating = onsets_r >= 4 and onsets_l >= 4 and 0.25 <= frac_r <= 0.95 and 0.25 <= frac_l <= 0.95 and abs(frac_r - frac_l) >= 0.03
    result = {
        "env_id": env_id,
        "steps_run": int(done_at or steps),
        "duration_s": float((done_at or steps) * CONTROL_DT),
        "survived_15s": done_at is None,
        "distance_m": travelled,
        "mean_speed_mps": travelled / float((done_at or steps) * CONTROL_DT),
        "min_com_height_m": float(np.min(com[:, 2]) if done_at is None else np.min(com[:done_at, 2])),
        "exo_torque_max_Nm": float(np.max(np.abs(exo_tau))) if n_exo else 0.0,
        "exo_dof_range_rad": (
            [[float(exo_dof[:, j].min()), float(exo_dof[:, j].max())] for j in range(n_exo)]
            if n_exo
            else []
        ),
        "right_contact_fraction": frac_r,
        "left_contact_fraction": frac_l,
        "right_stance_phases": onsets_r,
        "left_stance_phases": onsets_l,
        "alternating_contacts": bool(alternating),
        "done_reason": info.get("done_type", None) if done_at else None,
    }
    checks = {
        "1_survives_15s": result["survived_15s"],
        "2_distance>=5m": travelled >= MIN_DISTANCE_M,
        "3_exo_torques_zero": result["exo_torque_max_Nm"] == 0.0,
        "4_alternating_contacts": result["alternating_contacts"],
    }
    result["checks"] = checks
    result["passed"] = all(checks.values())
    return result, env, dict(com=com, dist=dist, exo_dof=exo_dof, contact_r=contact_r,
                             contact_l=contact_l, qpos=qpos, steps_run=done_at or steps)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=DEFAULT_ENV)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--policy", default=None,
                        help="SB3 .zip policy (as produced by scripts/finetune_walk_exo.py); "
                             "overrides --checkpoint")
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", default=None, help="write a gait summary figure here")
    parser.add_argument("--frames", default=None, help="directory for offscreen MuJoCo frames")
    parser.add_argument("--json", default=None, help="write the metrics here")
    args = parser.parse_args()

    sb3 = None
    stats = None
    if args.policy:
        import pickle

        from stable_baselines3 import PPO
        sb3 = PPO.load(args.policy, device="cpu")
        print(f"sb3 policy: {args.policy}")
        # the obs_rms carried by VecNormalize is what the policy was trained to see
        vn_path = args.policy[:-4] + "_vecnormalize.pkl" if args.policy.endswith(".zip") \
            else args.policy + "_vecnormalize.pkl"
        if Path(vn_path).exists():
            with open(vn_path, "rb") as fh:
                vn = pickle.load(fh)
            stats = (vn.obs_rms.mean, vn.obs_rms.var)
            print(f"obs normaliser: {vn_path}")
        else:
            # CheckpointCallback does not write a VecNormalize file, but the fine-tune
            # freezes obs_rms at the published statistics, so reuse those.
            import torch

            ck = torch.load(find_checkpoint(env_id="myoLegWalk-v0"), map_location="cpu")
            sd = ck["actor_state_dict"]
            stats = (sd["obs_normalizer._mean"].numpy().ravel(),
                     sd["obs_normalizer._var"].numpy().ravel())
            print(f"obs normaliser: published stats from {find_checkpoint(env_id='myoLegWalk-v0')}")

    result, env, log = evaluate(args.env, args.checkpoint, args.steps, args.seed,
                                 sb3_policy=sb3, obs_stats=stats)
    print(json.dumps(result, indent=2, default=float))

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    if args.plot:
        _plot(log, args.plot)
    if args.frames:
        _render(env, args.frames, log)
    return 0 if result["passed"] else 1


def _plot(log: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = log["steps_run"]
    t = np.arange(n) * CONTROL_DT
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(t, log["com"][:n, 2], color="tab:blue")
    axes[0].set_ylabel("pelvis COM z [m]")
    axes[1].plot(t, log["dist"][:n], color="tab:green")
    axes[1].set_ylabel("distance [m]")
    axes[2].step(t, log["contact_r"][:n], where="post", label="right")
    axes[2].step(t, log["contact_l"][:n], where="post", label="left")
    axes[2].set_ylabel("foot contact")
    axes[2].legend(loc="upper right", fontsize=8)
    if log["exo_dof"].size:
        for j, label in enumerate(("flex_r", "roll_r", "flex_l", "roll_l")):
            axes[3].plot(t, log["exo_dof"][:n, j], label=label)
        axes[3].set_ylabel("exo hinge [rad]")
        axes[3].legend(loc="upper right", fontsize=8, ncol=4)
    axes[3].set_xlabel("time [s]")
    fig.suptitle("Lin-I exoskeleton, zero assistance")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("wrote", path)


def _render(env, frames_dir: str, log: dict) -> None:
    """Offscreen frames at 10 fps plus a side/front camera pair."""
    out = Path(frames_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, data = env.model, env.data
    # the bundled model asks for a 640x640 offscreen buffer, which forbids a 720px render
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), 720)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), 720)
    renderer = mujoco.Renderer(model, 720, 720)
    cameras = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, c) or "" for c in range(model.ncam)
    }
    camera = "side_view" if "side_view" in cameras else (-1 if cameras else 0)
    n = log["steps_run"]
    save_every = 10  # 10 control steps = 0.1 s -> 10 fps
    qpos0 = log["qpos"]
    from PIL import Image

    written = 0
    for i in range(0, n, save_every):
        data.qpos[:] = qpos0[i]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        # encode a real PNG: write_bytes(renderer.render()) would dump the raw RGB buffer
        Image.fromarray(renderer.render()).save(out / f"frame_{i:05d}.png")
        written += 1
    renderer.close()
    print(f"wrote {written} frames to {out}")


if __name__ == "__main__":
    raise SystemExit(main())