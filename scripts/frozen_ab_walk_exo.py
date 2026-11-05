"""Frozen-controller A/B: what the assist does to effort when the policy cannot retrain.

``eval_walk_exo.py`` compares two *trained* policies, and that comparison is worthless for
answering "does the exoskeleton help".  The gait policy is a closed-loop controller: fit it
with the assist active and it re-plans around the extra torque, so distance comes out flat
(20.79 m assisted vs 20.84 m not, on the same weights) no matter what the machine is worth.
This script pins the controller instead - one set of weights, the same initial state, and
only the assist setting changes between conditions - which is the simulation analogue of the
"immediate response" trial in exoskeleton experiments (the first steps before adaptation).
Anything that moves here is the device, not the training.

What is measured is effort, not distance.  The registered ``myoLegWalk`` reward
(``WalkEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS``) has no activation or action-penalty term, so
muscle effort is *not* something the policy was taught to minimise - it is an outcome the
optimiser never touched, which is exactly what makes it usable here.  Effort is read per
muscle group around the assisted joint so the answer says where the work went: a hip assist
that lowers hip-flexor activation while leaving the ankle the same has moved work onto the
machine, and one that lowers nothing has not.

Conditions share their opening state by construction: a reference env is reset once per seed
and a seeded perturbation is added to that pose, and the resulting ``qpos/qvel/act`` is forced
onto every condition's env before its rollout.  The conditions then differ only in
``exo_assist_torque`` and the excitation gain, so the per-seed difference is a paired one.

The perturbation is deliberately small.  MyoSuite's own ``reset_type="random"`` (std 0.02 on
qpos) throws this walker over within a few steps, because ``scripts/finetune_walk_exo.py``
trains with the registered default, ``reset_type="init"`` - every episode in the fine-tune
started from the same keyframe, so the controller has never been asked to recover from a
different one.  Repeating it here would measure the policy's lack of domain randomisation,
not the exoskeleton.

    python scripts/frozen_ab_walk_exo.py --policy finetuned_clamp50_assist_best.zip \
        --strap-preload 50 --assist-torque 0 -3 -6 -9 --seeds 0 1 2 3 4 5

``--act-scale`` adds a second condition axis: every muscle excitation the frozen policy emits
is multiplied by that gain before it reaches the env.  It answers a question no assist setting
can -- whether the walker's effort is discretionary at all.  A gain of 0.9 that costs little
distance means the muscles are over-driven and an effort penalty has slack to collect; a gain
of 0.9 that topples the walker means the activation the gait carries is what walking at this
speed demands, and no reweighting of the objective can buy it back.

Writes a table to stdout and the full record to ``--out`` as JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import mujoco

import myosuite  # noqa: F401  (registers the CPU envs)

from eval_walk_exo import (
    CONTROL_DT,
    N_STEPS,
    force_initial_state,
    load_sb3_policy,
    make_scaled_env,
    perturbed_initial_state,
)

DEFAULT_ENV = "myoLegWalkExo-v0"

# Muscles that span the hip in the sagittal plane, by the actuator names the myoLeg model uses.
HIP_FLEXORS = ("iliacus", "psoas", "recfem", "sart")
HIP_EXTENSORS = ("glmax1", "glmax2", "glmax3", "semimem", "semiten", "bflh")
# The two joints the walker can shift work onto if the hip gets help from outside.
KNEE_EXTENSORS = ("vasint", "vaslat", "vasmed")
ANKLE_PLANTARFLEXORS = ("soleus", "gasmed", "gaslat")
ANKLE_DORSIFLEXORS = ("tibant", "perlong", "perbrev")

GROUPS = {
    "hip_flexor": HIP_FLEXORS,
    "hip_extensor": HIP_EXTENSORS,
    "knee_extensor": KNEE_EXTENSORS,
    "ankle_plantar": ANKLE_PLANTARFLEXORS,
    "ankle_dorsi": ANKLE_DORSIFLEXORS,
}


def _label(torque: float, scale: float) -> str:
    """Condition name.  A unity gain keeps the plain torque label so runs stay comparable."""
    return f"{torque:+.0f}" if scale == 1.0 else f"{torque:+.0f}/s{scale:.2f}"


def _muscle_columns(model) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Actuator-state column for every muscle group, the full muscle block, and its names.

    ``data.act`` is ordered by actuator index, and the myoLeg model puts its 80 muscles first
    with the exoskeleton's own actuators appended after them - so the muscle block is the
    leading slice and group selection is a name lookup, not a hardcoded range.
    """
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
    ]
    muscle_cols = np.asarray(
        [i for i, n in enumerate(names) if n is not None and not n.startswith("exo_")],
        dtype=int,
    )
    if muscle_cols.size != model.na:
        raise RuntimeError(
            f"expected the {model.na} activation states to be the muscle actuators, "
            f"found {muscle_cols.size} non-exo actuators"
        )
    # each muscle exists as a left and right actuator sharing a stem: "iliacus_r", "iliacus_l"
    stems = [n[: -2] if n.endswith(("_r", "_l")) else n for n in names]
    # a group indexes columns of the *muscle block*, not actuator ids: if the model ever
    # interleaves the exo actuators the two differ, and picking the wrong one is silent
    column_of = {act_idx: pos for pos, act_idx in enumerate(muscle_cols)}
    columns: dict[str, np.ndarray] = {}
    for group, stems_in_group in GROUPS.items():
        cols = [column_of[i] for i, s in enumerate(stems)
                if s in stems_in_group and i in column_of]
        if not cols:
            raise RuntimeError(f"the model has no muscle matching group {group}")
        columns[group] = np.asarray(cols, dtype=int)
    return columns, muscle_cols, [names[i] for i in muscle_cols]


def _human_dof_mask(model) -> np.ndarray:
    """Which qvel entries are the wearer's own joints, excluding every hinge the exo adds.

    Muscle force reaches the skeleton through MuJoCo's ``qfrc_actuator`` exactly like the
    drive's hinge torque does, so the two have to be separated by joint name before their work
    can be compared -- otherwise the assist's own joules are counted twice, once as saving and
    once as cost.
    """
    mask = np.ones(model.nv, dtype=bool)
    n_dof = {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if name.startswith("exo_"):
            n = n_dof.get(model.jnt_type[j], 1)
            mask[model.jnt_dofadr[j]:model.jnt_dofadr[j] + n] = False
    return mask


def _rollout(env, policy, obs: np.ndarray, cols: dict[str, np.ndarray], muscle: np.ndarray,
             steps: int, human_dofs: np.ndarray | None = None) -> dict[str, Any]:
    """One deterministic pass of the frozen controller, collecting the effort traces."""
    act = np.zeros((steps, muscle.size), dtype=np.float64)
    act_mag = np.zeros(steps)
    joint_work = np.zeros(steps)
    pelvis = env.model.body("pelvis").id
    start_xy = np.array(env.data.xpos[pelvis][:2])
    dist = np.zeros(steps)
    done_at = None
    rew = 0.0
    for i in range(steps):
        action = policy(obs)
        obs, r, terminated, truncated, _info = env.step(action)
        act[i] = env.data.act[muscle]
        # the exact term the effort-aware objective penalises, recomputed here so the trade-off
        # can be read off the gain sweep instead of inferred
        act_mag[i] = float(np.linalg.norm(env.data.act[muscle]) / muscle.size)
        if human_dofs is not None:
            joint_work[i] = float(np.abs(
                env.data.qfrc_actuator[human_dofs] * env.data.qvel[human_dofs]
            ).sum())
        rew += float(r)
        dist[i] = float(np.linalg.norm(env.data.xpos[pelvis][:2] - start_xy))
        if terminated or truncated:
            done_at = i + 1
            break
    n = done_at or steps
    a = act[:n]
    out: dict[str, Any] = {
        "steps_run": int(n),
        "survived": done_at is None,
        "distance_m": float(dist[n - 1]),
        "mean_speed_mps": float(dist[n - 1]) / (n * CONTROL_DT),
        "act_mean": float(np.abs(a).mean()),
        "act_sq_mean": float(np.square(a).mean()),
        "act_mag_term": float(act_mag[:n].mean()),
        # |tau . omega| summed over the wearer's joints: the mechanical work the muscles
        # actually delivered, which is the denominator the assist's own joules need
        "joint_work_J": float(joint_work[:n].sum() * CONTROL_DT),
        # CAUTION: the env's dense carries the terminal -100, so this mean is only comparable
        # between conditions that both survived
        "dense_per_step": rew / n,
    }
    for group, c in cols.items():
        # sum across the group's heads and both sides, then average over time: the group's
        # total drive, which is what a human's EMG envelope for that muscle pair would show
        out[f"{group}_act"] = float(np.abs(a[:, c]).sum(axis=1).mean())
    out["hip_cocontraction"] = float(
        np.minimum(np.abs(a[:, cols["hip_flexor"]]).sum(axis=1),
                   np.abs(a[:, cols["hip_extensor"]]).sum(axis=1)).mean()
    )
    # the same integral of |act| the COT-style surrogates use, per metre walked
    integral = out["act_mean"] * n * CONTROL_DT
    out["activation_per_metre"] = float(integral / out["distance_m"]) if out["distance_m"] > 0.1 else None
    # the whole muscle block, not only the groups above: the group means say effort moved, and
    # only the individual muscles say which one carried it.  The myoLeg model has biarticular
    # heads in it - rectus femoris flexes the hip and extends the knee - so a hip assist that
    # "unloads the hip" could be spending the saving at the knee, and that is visible only here
    out["per_muscle"] = [float(v) for v in a.mean(axis=0)]
    return out


def _device(env) -> dict[str, float]:
    """What the drive actually did during the rollout that just finished."""
    trace = env.assist_summary()
    if len(np.asarray(trace.get("phase", []))) == 0:
        return {}
    hinge = np.asarray(trace["delivered_hinge_torque"])
    speed = np.asarray(trace["joint_speed"])
    return {
        "exo_peak_hinge_Nm": float(np.abs(hinge).max()),
        "exo_work_J": float(np.abs(hinge * speed).sum() * CONTROL_DT),
        "exo_saturated_fraction": float(np.asarray(trace["saturated"]).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=DEFAULT_ENV)
    parser.add_argument("--policy", required=True, help="SB3 .zip whose weights stay fixed")
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--strap-preload", type=float, default=50.0)
    parser.add_argument("--assist-torque", type=float, nargs="+", default=[0.0],
                        help="peak hip torque per condition, N.m (0 = the control)")
    parser.add_argument("--assist-phase-lead", type=float, default=0.30)
    parser.add_argument("--assist-width", type=float, default=0.30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--init-noise", type=float, default=0.004,
                        help="std of the per-seed qpos perturbation, rad/m (the free-joint "
                             "height and quaternion are left untouched)")
    parser.add_argument("--vel-noise", type=float, default=0.004,
                        help="std of the per-seed qvel perturbation")
    parser.add_argument("--act-scale", type=float, nargs="+", default=[1.0],
                        help="muscle-excitation gain per condition; conditions are the cross "
                             "product with --assist-torque")
    parser.add_argument("--muscle-force-scale", type=float, default=1.0,
                        help="scale every muscle's peak force by this factor (SarcopeniaWrapper). "
                             "1.0 is the registered model, whose muscles are 2-4x human; "
                             "~0.35 brings the hip/ankle peak moments to normative scale.")
    parser.add_argument("--out", default=None, help="write the full record here")
    args = parser.parse_args()

    sb3, stats = load_sb3_policy(args.policy)
    mean, var = stats
    mean = np.asarray(mean, dtype=np.float32).ravel()
    # rsl_rl EmpiricalNormalization: eps is added to std, not inside the sqrt, and the whole
    # pipeline is float32 - this gait is sensitive to the rounding
    scale = np.sqrt(np.asarray(var, dtype=np.float32).ravel()) + np.float32(1e-2)

    def policy(obs, gain=1.0):
        x = (np.asarray(obs, dtype=np.float32).ravel() - mean) / scale
        action, _ = sb3.predict(x, deterministic=True)
        if gain != 1.0:
            # the action *is* the muscle excitation, and the block is bounded [0, 1]: the gain
            # is a controller-side dial, not a change to what the policy weights are
            action = np.clip(action * gain, 0.0, 1.0)
        return action

    def build(torque: float):
        env = make_scaled_env(
            args.env,
            args.muscle_force_scale,
            exo_motor=True,
            exo_assist_torque=torque,
            exo_assist_phase_lead=args.assist_phase_lead,
            exo_assist_width=args.assist_width,
            exo_strap_preload_N=args.strap_preload,
        ).unwrapped
        return env

    conds = [(_label(t, g), t, g)
             for g in args.act_scale for t in args.assist_torque]
    # one env per condition, kept alive: rebuilding a compiled model per rollout would dominate
    envs = {lab: build(t) for lab, t, _g in conds}
    cols, muscle, muscle_names = _muscle_columns(envs[conds[0][0]].model)
    human_dofs = _human_dof_mask(envs[conds[0][0]].model)
    reference = envs[conds[0][0]]

    metric_keys = ("survived", "distance_m", "mean_speed_mps", "act_mean", "act_sq_mean",
                   "act_mag_term", "joint_work_J", "dense_per_step",
                   "hip_flexor_act", "hip_extensor_act", "hip_cocontraction",
                   "knee_extensor_act", "ankle_plantar_act", "ankle_dorsi_act",
                   "activation_per_metre")
    # averaged with the rest but printed on their own line: these describe the machine, and a
    # condition that reports zero work here is not an assist that failed to help, it is an
    # assist that never ran
    device_keys = ("exo_peak_hinge_Nm", "exo_work_J", "exo_saturated_fraction")
    labels = [lab for lab, _t, _g in conds]
    gains = {lab: g for lab, _t, g in conds}
    control = next((lab for lab, t, g in conds if t == 0.0 and g == 1.0), conds[0][0])
    record: dict[str, Any] = {"policy": args.policy, "seeds": {}, "conditions": {}}
    for seed in args.seeds:
        state = perturbed_initial_state(reference, seed, args.init_noise, args.vel_noise)
        per_seed = {}
        for lab, env in envs.items():
            obs = force_initial_state(env, state)
            m = _rollout(env, lambda o: policy(o, gains[lab]), obs, cols, muscle,
                         args.steps, human_dofs)
            m.update(_device(env))
            per_seed[lab] = m
        record["seeds"][str(seed)] = {k: v for k, v in per_seed.items()}
        print(f"seed {seed}: " + "  ".join(
            f"{lab:>9s} {per_seed[lab]['distance_m']:5.2f}m "
            f"hipFlex {per_seed[lab]['hip_flexor_act']:.3f}"
            for lab in labels))

    for lab in labels:
        rows = [record["seeds"][str(s)][lab] for s in args.seeds]
        record["conditions"][lab] = {
            k: float(np.mean([r[k] for r in rows if r[k] is not None]))
            for k in metric_keys + device_keys
            if any(r[k] is not None for r in rows)
        } | {"n_survived": int(sum(bool(r["survived"]) for r in rows)), "n_seeds": len(rows)}

    print(f"\nfrozen policy, mean over {len(args.seeds)} initial states "
          f"(control = {control} N.m)")
    header = f"{'metric':22s}" + "".join(f"{lab:>18s}" for lab in labels)
    print(header)
    for k in metric_keys:
        line = f"{k:22s}"
        for lab in labels:
            c = record["conditions"][lab]
            b = record["conditions"][control]
            if k not in c:
                line += f"{'-':>18s}"
            elif lab == control or k == "survived":
                line += f"{c[k]:>18.4f}"
            else:
                line += f"{c[k]:>12.4f}{(c[k] - b[k]) / abs(b[k]) * 100:>+6.1f}%"
        print(line)
    for lab in labels:
        c = record["conditions"][lab]
        # the assist's joules against the wearer's own, per rollout: a torque the machine
        # delivers is only "help" relative to the work it is standing in for
        share = (c.get("exo_work_J", 0.0) / c["joint_work_J"] * 100
                 if c.get("joint_work_J") else 0.0)
        print(f"  {lab:>9s}: survived {c['n_survived']}/{c['n_seeds']}  "
              f"exo peak {c.get('exo_peak_hinge_Nm', 0):.2f} N.m  "
              f"work {c.get('exo_work_J', 0):.1f} J ({share:.1f}% of joint work)  "
              f"saturated {c.get('exo_saturated_fraction', 0) * 100:.0f}%")

    # The percentages above are means of single-number rollouts, which is exactly the mistake
    # the earlier three-seed check made.  These are paired: each seed contributes one
    # difference between the condition and the control, and the scatter of that difference is
    # what says whether a -0.7% is an effect or is the jitter itself.
    print(f"\npaired per-seed difference against {control} N.m, mean +/- sd over "
          f"{len(args.seeds)} initial states")
    paired = ("distance_m", "act_mean", "act_mag_term", "joint_work_J",
              "hip_flexor_act", "hip_extensor_act",
              "hip_cocontraction", "knee_extensor_act", "ankle_plantar_act")
    print(f"{'metric':22s}" + "".join(f"{lab:>18s}" for lab in labels if lab != control))
    for k in paired:
        line = f"{k:22s}"
        for lab in labels:
            if lab == control:
                continue
            d = np.array([
                record["seeds"][str(s)][lab][k] - record["seeds"][str(s)][control][k]
                for s in args.seeds
            ], dtype=np.float64)
            line += f"{d.mean():>11.4f}{d.std(ddof=1):>6.3f}"
        print(line)

    # Which single muscles moved, ranked by how consistently they moved.  Sorting on the
    # t-score rather than the percentage is the point: a 0.4% shift that happens on all six
    # initial states is a mechanism, and a 3% shift that only happens on the one that fell is
    # not.
    print("\nlargest per-muscle responses (paired, ranked by |t| against the control)")
    for lab in labels:
        if lab == control:
            continue
        rows = []
        for j, name in enumerate(muscle_names):
            d = np.array([
                record["seeds"][str(s)][lab]["per_muscle"][j]
                - record["seeds"][str(s)][control]["per_muscle"][j]
                for s in args.seeds
            ], dtype=np.float64)
            sd = d.std(ddof=1)
            base = np.mean([record["seeds"][str(s)][control]["per_muscle"][j]
                            for s in args.seeds])
            rows.append((abs(d.mean() / (sd / np.sqrt(len(args.seeds)))) if sd > 0 else 0.0,
                         d.mean(), base, name))
        rows.sort(reverse=True)
        picked = [r for r in rows if r[0] > 2.0][:8]
        print(f"  {lab:>9s}: " + (", ".join(
            f"{n} {m:+.4f} ({m / abs(b) * 100:+.1f}% on {b:.3f})"
            for tv, m, b, n in picked) or "none above |t| = 2"))

    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
