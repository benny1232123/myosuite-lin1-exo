"""Zero-assistance 15 s gait check for the 'Lin-I' exoskeleton walking envs.

Runs the published ``myoLegWalk-v0`` baseline policy (no re-training) on a
Lin-I-equipped walker and reports the four acceptance criteria:

1. 1500 control steps (15.0 s at 0.01 s/step) without a termination,
2. forward displacement >= 5 m (mean speed >= 0.33 m/s),
3. both exoskeleton motors unpowered (actuator force at the exo hinges == 0),
4. alternating left/right foot ground contact.

``--assist-torque`` fits the hip drive and the phase-locked assist controller instead, in
which case criterion 3 becomes "the hinge torque never exceeds the drive's own cap" and the
report adds what the controller asked for against what the machine delivered.  Pass
``--assist-torque 0`` for the zero-assistance control: same drive, same controller, same
electrical behaviour, no requested torque -- that is the number an assist gain has to beat.
The delivered column is read at the hip through the transfer *measured for the interface this
run built*: ~0.12 of the hinge torque through the point-connected cuff (so a request bigger
than ~2 N.m saturates and says so), ~1.0 through a clamped one.  The report names the table it
used, so the hip number can be checked against the bench.

Because the exoskeleton changes the physics (3.1 kg of load, a strap-coupled passive
chain) the zero-shot policy may not survive 15 s; the script reports the numbers either
way and exits non-zero when a criterion fails, so it doubles as a regression check::

    python scripts/eval_walk_exo.py                       # myoLegWalkExo-v0, baseline ckpt
    python scripts/eval_walk_exo.py --env myoLegWalkExoLocked-v0
    python scripts/eval_walk_exo.py --checkpoint logs/.../final_model.zip
    python scripts/eval_walk_exo.py --frames frames --plot gait.png
    python scripts/eval_walk_exo.py --policy walk_exo_policy.zip --assist-torque 0
    python scripts/eval_walk_exo.py --policy walk_exo_policy.zip --assist-torque -6
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

import myosuite  # noqa: F401  (registers the CPU envs)
from myosuite import make_env
from myosuite.envs.gymnasium_env import CpuEnvAccessor
from myosuite.envs.myo.tasks.basic.leg.lin1_exo import (
    EXO_PARAMS_PATH,
    hip_torque_transfer,
    load_exo_params,
    style_muscle_geoms,
)
from myosuite.utils.checkpoint_utils import find_checkpoint, load_policy

DEFAULT_ENV = "myoLegWalkExo-v0"
BASELINE_ENV = "myoLegWalk-v0"
CONTROL_DT = 0.01
N_STEPS = 1500
MIN_DISTANCE_M = 5.0
HEIGHT_THRESHOLD_M = 0.8
SIDES = ("r", "l")


def _hip_torque_transfer(env) -> np.ndarray:
    """Measured hip moment per unit hinge torque, in drive-slot order (right first).

    Two numbers exist and they are not interchangeable: the point-connected cuff passes ~0.12
    of the hinge torque (it carries force and no couple), a clamped one passes ~1.0.  The one
    this run is referred through is the one the env actually built, never a hand-picked
    constant - asking the drive for the wrong end of an 8x range either saturates it or wastes
    it.  See ``assist.strap_interface`` in ``lin1_exo_params.json`` for the bench behind it.
    """
    clamped = bool(getattr(env, "_exo_meta", {}).get("strap"))
    table = hip_torque_transfer(load_exo_params(EXO_PARAMS_PATH), clamped)
    return np.asarray([table[side] for side in SIDES], dtype=np.float64)


def _assist_interface(env) -> str:
    return "clamped strap" if getattr(env, "_exo_meta", {}).get("strap") else "point-connected cuff"


def _assist_report(trace: dict, dt: float, transfer: np.ndarray, interface: str) -> dict:
    """What the assist controller asked for, and what the machine actually delivered.

    ``delivered`` is measured at the hinge (``qfrc_actuator``).  The hip column is that torque
    referred through the *measured* transfer of the interface that was built; it is an
    estimate of the moment at the hip, and ``assist_transfer_*`` is reported with it so the
    number can be checked against the bench rather than trusted.
    """
    if len(trace.get("phase", [])) == 0:
        return {}
    hip = trace["commanded_hip_torque"]
    hinge = trace["delivered_hinge_torque"]
    # what the joint received, through the measured interface
    at_hip = hinge * transfer
    speed = trace["joint_speed"]
    # the drive can only be judged where it was asked for something real
    asked = np.abs(hip) > 0.5 * float(np.abs(hip).max())
    return {
        "assist_transfer_used": [float(v) for v in transfer],
        "assist_transfer_interface": interface,
        "assist_peak_command_Nm": float(np.abs(hip).max()),
        "assist_mean_abs_command_Nm": float(np.abs(hip).mean()),
        "assist_peak_hinge_torque_Nm": float(np.abs(hinge).max()),
        "assist_peak_delivered_Nm": float(np.abs(at_hip).max()),
        "assist_mean_abs_delivered_Nm": float(np.abs(at_hip).mean()),
        "assist_tracking_rms_Nm": float(np.sqrt(np.mean((at_hip - hip)[asked] ** 2))),
        "assist_saturated_fraction": float(trace["saturated"].mean()),
        "assist_peak_current_A": float(np.abs(trace["current"]).max()),
        "assist_peak_voltage_V": float(np.abs(trace["voltage"]).max()),
        # the work the machine did, at the hinge where its torque and speed are measured
        "assist_mechanical_work_J": float(np.abs(hinge * speed).sum() * dt),
        "assist_active_fraction": float((np.abs(hip) > 1e-6).any(axis=1).mean()),
    }


def perturbed_initial_state(
    env,
    seed: int,
    qpos_noise: float,
    qvel_noise: float,
) -> dict[str, np.ndarray]:
    """A seeded opening state: the env's own reset, then Gaussian jitter.

    ``--seed`` on its own does nothing here.  The registered envs default to
    ``reset_type="init"``, which restores one keyframe, and a deterministic policy on a
    deterministic env has no other random source - so the "three-seed" checks that produced
    byte-identical distances were one rollout measured three times.  The perturbation is
    opt-in (0.0 reproduces the old behaviour exactly) because that keyframe is what the
    fine-tune trained on: MyoSuite's own ``reset_type="random"`` jitters qpos by 0.02, which
    throws this particular walker over inside a few steps.
    """
    env.reset(seed=seed)
    # drawn from env.np_random, not a fresh Generator: the CPU env contract requires that the
    # reset seed alone determines the episode
    qpos = env.data.qpos + env.np_random.normal(0.0, qpos_noise, size=env.data.qpos.shape)
    # the free joint's height and quaternion are left alone exactly as the env's own
    # randomiser leaves them: sinking the pelvis into the floor is a termination, not a seed
    qpos[2] = env.data.qpos[2]
    qpos[3:7] = env.data.qpos[3:7]
    qvel = env.data.qvel + env.np_random.normal(0.0, qvel_noise, size=env.data.qvel.shape)
    return {"qpos": np.asarray(qpos), "qvel": np.asarray(qvel), "act": env.data.act.copy()}


def force_initial_state(env, state: dict[str, np.ndarray]) -> np.ndarray:
    """Reset *env*, impose a captured opening state, and return the matching observation.

    This is what makes two conditions comparable: the same ``qpos/qvel/act`` is put on every
    env, so the only difference between them is what the env was built to do.
    """
    env.reset(seed=0)
    env.data.qpos[:] = state["qpos"]
    env.data.qvel[:] = state["qvel"]
    env.data.act[:] = state["act"]
    # required here: the state was written from outside the pipeline, so the derived
    # quantities (site positions, muscle lengths) are stale until the next mj_forward
    mujoco.mj_forward(env.model, env.data)
    env._accessor = CpuEnvAccessor(env.model, env.data, env._ctrl_dt)
    return env._obs_dict_to_vec(env._get_obs_dict(env._accessor))


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


def make_scaled_env(env_id: str, force_scale: float = 1.0, **kwargs: Any):
    """Build an env, optionally with the wearer's muscles weakened to human authority.

    The published myoLeg model is far stronger than the person this exoskeleton is fitted to:
    on the published policy its peak net muscle moment is ~3-4 N.m/kg at the hip and knee and
    ~4.7 N.m/kg at the ankle, with ankle positive power ~12 W/kg, against normative walking at
    1.2 m/s near 1 N.m/kg and 4-6 W/kg.  A 17 N.m module is therefore ~2% of what this model's
    hip can produce and nearer 15% of a real one, which is why a whole-body cost metric reads
    the device as inert.

    ``SarcopeniaWrapper`` is the repo's own knob for exactly this: it multiplies
    ``actuator_gainprm[:, 2]``, the maximum isometric force, and the exo's own actuator rows
    are zero in that column, so weakening the wearer provably does not rescale the drive.  Use
    it as a fraction; 1.0 leaves the registered env untouched.
    """
    env = make_env(env_id, **kwargs)
    if float(force_scale) != 1.0:
        from myosuite.envs.wrappers import SarcopeniaWrapper

        env = SarcopeniaWrapper(env, force_scale=float(force_scale))
    return env


def _sagittal_joints(model) -> tuple[np.ndarray, list[str]]:
    """Dof addresses of the wearer's three primary sagittal joints, both sides.

    Only hip_flexion / knee_angle / ankle_angle: the model's redundant knee chain carries
    equality-constrained helper dofs whose generalized forces are reaction-dominated, and the
    exo's own hinges are excluded because their work is what is being audited, not the yardstick.
    """
    wanted = ("hip_flexion", "knee_angle", "ankle_angle")
    adr, names = [], []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if not name.startswith("exo_") and name.rsplit("_", 1)[0] in wanted:
            adr.append(int(model.jnt_dofadr[j]))
            names.append(name)
    return np.asarray(adr, dtype=int), names


def evaluate(
    env_id: str = DEFAULT_ENV,
    checkpoint: str | Path | None = None,
    steps: int = N_STEPS,
    seed: int = 0,
    sb3_policy: Any = None,
    obs_stats: Any = None,
    env_kwargs: dict | None = None,
    init_noise: float = 0.0,
    vel_noise: float = 0.0,
    force_scale: float = 1.0,
) -> dict:
    env = make_scaled_env(env_id, force_scale, **(env_kwargs or {})).unwrapped
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
    if init_noise or vel_noise:
        state = perturbed_initial_state(env, seed, init_noise, vel_noise)
        obs = force_initial_state(env, state)
    model, data = env.model, env.data
    n_exo = getattr(env, "n_exo_dof", 0)
    # a fitted drive is never silent: even held at zero volts it brakes the hinge through
    # back-EMF, so the "torques must be exactly zero" criterion only applies without one
    powered = hasattr(env, "exo_motor_max_torques") and float(
        np.max(env.exo_motor_max_torques())
    ) > 0.0

    pelvis = model.body("pelvis").id
    right_foot, left_foot = _foot_body_ids(model)
    # the band's own ceiling: what moment the pads hold before they slide on the thigh
    slip = env.exo_strap_slip_torques() if hasattr(env, "exo_strap_slip_torques") else np.zeros(0)

    qpos = np.zeros((steps, model.nq))
    com = np.zeros((steps, 3))
    dist = np.zeros(steps)
    contact_r = np.zeros(steps, dtype=bool)
    contact_l = np.zeros(steps, dtype=bool)
    # sized off the report itself: n_exo_dof counts every dof the exo adds, including the pad
    # slides, which exo_dof_positions holds back because they are clamp forces, not hinges
    n_dof = int(env.exo_dof_positions().size) if n_exo else 0
    exo_dof = np.zeros((steps, n_dof))
    exo_tau = np.zeros((steps, n_dof))
    strap_tau = np.zeros((steps, len(SIDES))) if powered else None
    # one row per pad: the band puts two pads diametrically opposite on each thigh
    pad_n = np.zeros((steps, 2 * slip.size)) if slip.size else None
    activation = np.zeros(steps)
    # the wearer's own joint loading: the yardstick the device's torque is a fraction of, and
    # the number that says whether this model's muscles are human-strength
    joint_adr, joint_names = _sagittal_joints(model)
    # whole-model mass, wearer and device together: normative joint moments are quoted per kg
    # of body mass, and a pelvis-subtree figure would quietly drop the torso
    wearer_mass = float(model.body_subtreemass[0])
    j_moment = np.zeros((steps, joint_adr.size))
    j_power = np.zeros((steps, joint_adr.size))
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
        j_moment[i] = data.qfrc_actuator[joint_adr]
        j_power[i] = j_moment[i] * data.qvel[joint_adr]
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
        activation[i] = float(np.abs(data.act).mean()) if data.act.size else 0.0
        if powered:
            # the moment crossing each cuff: what the wearer feels, one row per side
            strap_tau[i] = np.linalg.norm(env.exo_strap_torques(), axis=1)
        if slip.size:
            pad_n[i] = env.exo_preload_forces()
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
        "exo_strap_torque_peak_Nm": float(np.abs(strap_tau).max()) if powered else 0.0,
        "strap_slip_torque_Nm": [float(v) for v in slip],
        "strap_pad_force_range_N": (
            [[float(pad_n[:n_run, k].min()), float(pad_n[:n_run, k].max())]
             for k in range(2 * slip.size)] if slip.size else []
        ),
        "mean_muscle_activation": float(activation[:n_run].mean()),
        "exo_dof_range_rad": (
            [[float(exo_dof[:, j].min()), float(exo_dof[:, j].max())] for j in range(n_dof)]
            if n_dof
            else []
        ),
        "right_contact_fraction": frac_r,
        "left_contact_fraction": frac_l,
        "right_stance_phases": onsets_r,
        "left_stance_phases": onsets_l,
        "alternating_contacts": bool(alternating),
        "wearer_mass_kg": wearer_mass,
        # normative walking at this speed is ~1 N.m/kg at the hip and knee and ~1.2-1.7 N.m/kg
        # at the ankle, with ankle positive power ~4-6 W/kg; these are the numbers to compare
        # against, and they are what says whether the assist magnitude is being judged against a
        # human or against a much stronger model
        "wearer_joint_peak_Nm_per_kg": {
            n: float(np.abs(j_moment[:n_run, k]).max()) / wearer_mass
            for k, n in enumerate(joint_names)},
        "wearer_joint_peak_W_per_kg": {
            n: [float(j_power[:n_run, k].max()) / wearer_mass,
                float(j_power[:n_run, k].min()) / wearer_mass]
            for k, n in enumerate(joint_names)},
        "done_reason": info.get("done_type", None) if done_at else None,
    }
    # the fidelity number itself: how much of the wearer's own hip moment the device supplies.
    # A real hip module is quoted at 10-15% of peak hip moment, so this ratio, not the absolute
    # newton-metres, is what says whether the simulated assist is the size of the physical one.
    # The numerator is the *drive's* hinge torque, not the strap peak: a clamped cuff carries up to
    # 20 N.m of passive friction with the motor held at zero, and reading that as an assist share
    # reported 9.7% for a zero-torque control. Under the 50 N clamp the strap transfers ~1.0 of the
    # hinge torque (assist_transfer_used), so the drive number is also what reaches the wearer.
    hip_peak = max((float(np.abs(j_moment[:n_run, k]).max())
                    for k, n in enumerate(joint_names) if n.startswith("hip_flexion")), default=0.0)
    result["exo_assist_share_of_wearer_hip_peak_pct"] = (
        100.0 * result["exo_torque_max_Nm"] / hip_peak if hip_peak > 0 else 0.0)
    checks = {
        "1_survives_15s": result["survived_15s"],
        "2_distance>=5m": travelled >= MIN_DISTANCE_M,
        "4_alternating_contacts": result["alternating_contacts"],
    }
    if powered:
        # the drive is allowed to push; what it is not allowed to do is exceed its own cap
        result.update(_assist_report(
            env.assist_summary(), CONTROL_DT, _hip_torque_transfer(env), _assist_interface(env)))
        cap = float(np.max(env.exo_motor_max_torques()))
        checks["3_exo_torques_within_the_drive_cap"] = bool(
            result["exo_torque_max_Nm"] <= cap + 1e-6
        )
    else:
        checks["3_exo_torques_zero"] = result["exo_torque_max_Nm"] == 0.0
    result["checks"] = checks
    result["passed"] = all(checks.values())
    return result, env, dict(com=com, dist=dist, exo_dof=exo_dof, contact_r=contact_r,
                             contact_l=contact_l, qpos=qpos, steps_run=done_at or steps)


def load_sb3_policy(path: str) -> tuple[Any, tuple[Any, Any]]:
    """The fine-tuned SB3 policy and the observation normaliser it was trained behind.

    A VecNormalize-wrapped policy must see normalised observations.  The checkpoint callback
    does not write a normaliser file, but the fine-tune freezes ``obs_rms`` at the published
    statistics, so those are the fallback.
    """
    import pickle

    from stable_baselines3 import PPO

    sb3 = PPO.load(path, device="cpu")
    vn_path = path[:-4] + "_vecnormalize.pkl" if path.endswith(".zip") else path + "_vecnormalize.pkl"
    if Path(vn_path).exists():
        with open(vn_path, "rb") as fh:
            vn = pickle.load(fh)
        print(f"sb3 policy: {path}\nobs normaliser: {vn_path}")
        return sb3, (vn.obs_rms.mean, vn.obs_rms.var)
    import torch

    ck = find_checkpoint(env_id=BASELINE_ENV)
    sd = torch.load(ck, map_location="cpu")["actor_state_dict"]
    print(f"sb3 policy: {path}\nobs normaliser: published stats from {ck}")
    return sb3, (sd["obs_normalizer._mean"].numpy().ravel(),
                 sd["obs_normalizer._var"].numpy().ravel())


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
    parser.add_argument("--init-noise", type=float, default=0.0,
                        help="std of a seeded qpos jitter applied after the reset, rad/m. "
                             "Without it --seed cannot change the rollout: the env resets to "
                             "one keyframe (reset_type='init') and nothing else in the loop is "
                             "random.  Keep it small - the fine-tune only ever started from "
                             "that keyframe, and 0.02 (the env's own randomiser) topples it")
    parser.add_argument("--vel-noise", type=float, default=0.0,
                        help="std of the seeded qvel jitter")
    parser.add_argument("--assist-torque", type=float, default=None,
                        help="fit the hip drive and install the phase-locked assist "
                             "controller at this peak torque, N.m at the hip (negative "
                             "extends, so a push-off assist is negative; the cuff transfers "
                             "about 0.12 of the hinge torque, so ~2 N.m is already the "
                             "module's full 17 N.m); pass 0.0 for the zero-assistance "
                             "control, which keeps the controller and the drive's own "
                             "electrical behaviour and asks for nothing")
    parser.add_argument("--assist-phase-lead", type=float, default=0.15,
                        help="fraction of a gait cycle the pulse is moved earlier than the "
                             "side's own extension peak")
    parser.add_argument("--assist-width", type=float, default=0.30,
                        help="full width of the assist pulse, as a fraction of a gait cycle")
    parser.add_argument("--drive-friction", type=float, default=0.0,
                        help="no-load drag of the hip module, N.m at the joint (0 = the "
                             "model the published policies were trained against)")
    parser.add_argument("--thermal-derate", action="store_true",
                        help="let the winding heat up and derate the drive over long runs")
    parser.add_argument("--strap-preload", type=float, default=0.0,
                        help="clamp the cuff to the thigh with a preloaded friction band, N of "
                             "normal force per pad (0 keeps the passive point-connected cuff "
                             "the published transfer numbers were measured on)")
    parser.add_argument("--muscle-force-scale", type=float, default=1.0,
                        help="weaken the wearer's muscles to this fraction of their published "
                             "max isometric force (MyoSuite's sarcopenia knob, applied to "
                             "gainprm[:,2]; the exo's own actuator rows are zero there, so the "
                             "drive is untouched).  1.0 keeps the registered model.")
    parser.add_argument("--plot", default=None, help="write a gait summary figure here")
    parser.add_argument("--frames", default=None, help="directory for offscreen MuJoCo frames")
    parser.add_argument("--json", default=None, help="write the metrics here")
    args = parser.parse_args()

    env_kwargs: dict[str, Any] = {}
    if args.assist_torque is not None:
        env_kwargs = {
            "exo_motor": True,
            "exo_assist_torque": args.assist_torque,
            "exo_assist_phase_lead": args.assist_phase_lead,
            "exo_assist_width": args.assist_width,
        }
    if args.drive_friction:
        env_kwargs["exo_drive_friction_Nm"] = float(args.drive_friction)
    if args.thermal_derate:
        env_kwargs["exo_thermal_derate"] = True
    if args.strap_preload:
        # the band is a property of the wearing interface, not of the drive: it has to be
        # buildable with no motor at all, which is the regression case for the passive gait
        env_kwargs["exo_strap_preload_N"] = float(args.strap_preload)

    sb3 = None
    stats = None
    if args.policy:
        sb3, stats = load_sb3_policy(args.policy)

    result, env, log = evaluate(args.env, args.checkpoint, args.steps, args.seed,
                                 sb3_policy=sb3, obs_stats=stats, env_kwargs=env_kwargs,
                                 init_noise=args.init_noise, vel_noise=args.vel_noise,
                                 force_scale=args.muscle_force_scale)
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
    # the same thin red muscle lines the interactive viewer uses (view-only)
    style_muscle_geoms(model, 0.008)
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