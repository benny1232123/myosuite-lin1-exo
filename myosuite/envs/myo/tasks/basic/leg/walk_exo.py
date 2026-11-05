# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Walking with the 'Lin-I' hip exoskeleton attached (zero-assistance baseline).

``myoLegWalkExo-v0`` reuses the ``myoLegWalk-v0`` task (same model, same observation
schema, same reward) and only adds the exoskeleton to the physics model, see
:mod:`myosuite.envs.myo.tasks.basic.leg.lin1_exo`.  The exoskeleton contributes
2.6 kg of rigid load on the pelvis plus 0.24 kg per leg and adds two passive
degrees of freedom per side (hip flexion hinge + thigh-link roll hinge) that are
rigidly coupled to the human thigh through a soft cuff weld.

Two model variants are registered:

* ``myoLegWalkExo-v0``       - physical chain: hinges + cuff welds.
* ``myoLegWalkExoLocked-v0`` - exo DoFs removed: a rigid, purely inertial load on the
  pelvis and thighs (reference condition without any constraint coupling).

Both are *unassisted*: the device hangs off the body and taxes it, but nothing drives it.
Pass ``exo_motor=True`` to ``myoLegWalkExo-v0`` to hang a native MuJoCo ``<dcmotor>`` on
each exo hip flexion hinge, so the device delivers torque through a real electromechanical
relation (``tau = N*K*(V - K*N*omega)/R``, capped by the rotor's saturation limit) instead of
not at all.  The drive is commanded in volts through :meth:`LegWalkExoEnvV0.set_exo_voltage`,
not through the policy action: a real exo runs its own controller, and keeping the two apart
leaves the published ``myoLegWalk-v0`` baseline policy transferable.  The electrical
parameters in ``lin1_exo_params.json`` are fitted from the drive's datasheet - one RobStride
RS02 per hip, 17 N.m peak and 410 rpm no-load at 48 V - so the envelope is the device's, while
rotor inertia and everything thermal are still estimates.

On top of that, ``exo_assist_torque=<N.m>`` runs a phase-locked hip-assist pulse instead of
hand-held voltages: the environment converts the request into voltages itself, timed off the
same gait clock the task's cyclic reward uses and referred to the hinge through the measured
coupling in the ``assist`` block of ``lin1_exo_params.json`` - the two sides turn opposite
ways, which is why the controller reads that table rather than trusting the joint name.
``exo_assist_torque=0.0`` installs the controller and commands nothing, which is the
zero-assistance control of the comparison, with the same trace still recorded.

The observation vector is bit-identical to ``myoLegWalk-v0`` (the four exo
coordinates are stripped from ``qpos_without_xy``/``qvel``), so the published
``myoLegWalk-v0`` baseline policy can be loaded into this environment unchanged.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import mujoco
import numpy as np

from myosuite.core.model_builder import ModelBuilder
from myosuite.envs.gymnasium_env import CpuEnvAccessor
from myosuite.envs.myo.tasks.basic.leg.lin1_exo import (
    ExoMotorConfig,
    attach_lin1_exo,
    clock_exo_endcap,
    hip_torque_transfer,
    load_exo_params,
    pitch_exo,
    place_strap_on_muscle,
    seat_cuff_on_rod,
    style_muscle_geoms,
)
from myosuite.envs.myo.tasks.basic.leg.walk import LegWalkEnvV0


# The CAD pose stands with the feet 176 mm apart, which leaves the two thighs almost
# touching.  Opening the stance a little reads better and gives the thigh straps room.
_STANCE_WIDEN_RAD = 0.085

# The rod is bolted to the drum's clevis at the hip, so it can only turn about that bolt.
# Any extra swing, lift or standoff of the drawn link tips the fork out of the drum's
# lugs, which is what kept making the joint read as "not plugged in".  The whole device is
# instead pitched about the hip axis - one rigid turn, so the drums stay concentric in
# their housings and the fork stays in the lugs - until the back plate lies on the lumbar
# spine; the link, hanging 350 mm below the hips, comes out of that about 90 mm in front of
# the thigh, and the hinge turn about the clevis bolt tucks it back onto the thigh.
_ASSY_PITCH_DEG = 0.0
# the plate is slid this far towards the body after the pitch, together with the rest of
# the device, so the pad rests on the lower back
_ASSY_FORWARD_MM = 0.0
# the rod's lower end swings about the clevis point (user: 以铰链为圆心…摆动, then 应该向反方向
# 摆).  Positive turns the far end towards the exo's centre line, negative outwards; shipped
# negative.  Only the rod turns - the strap ring stays clamped on the thigh (0.1 mm), and
# the rod-to-femur clearance at -10 deg is the same 1 mm as the CAD pose, so nothing is
# pressed into the bone.  Positive 20 deg and up swings the rods across each other.
_ROD_SWING_DEG = 0.0
_STRAP_OUT_MM = 0.0
# roll spins the rod about its own axis, which twists its root out of the clevis (the user
# saw exactly that and rejected it), and it cannot close the gap anyway: the waist frame
# holds the hip module 115 mm out from the hip joint.  Keep it at zero and fix the pin.
_ROD_ROLL_DEG = 0.0
_ROD_LIFT_DEG = 0.0
_ROD_STANDOFF_MM = 0.0
# a turn about the clevis bolt keeps the fork seated; it swings the rod's far end forward
# along the thigh, which walks the rod from the thigh's outer edge onto the middle of its
# front face.  15 deg is where the bar covers the quadriceps without crossing the midline
# (25 deg puts it on the inner thigh, 35 deg over the knee).  Only the rod turns - the
# strap ring stays clamped on the leg.
_ROD_HINGE_DEG = 15.0
_MUSCLE_WIDTH = 0.008
# the module end caps clock onto the fork by default (see clock_exo_endcap); this is only a
# residual nudge added on top of that
_ENDCAP_CLOCK_DEG = 0.0
# walk.py resets from keyframe 2 ("init", the default) or draws 2 and 3 ("random"), so that is
# the crank angle the fork has to be seated at -- the standing keys 0/1 are 18-24 deg away
_ENDCAP_CLOCK_KEY = 2
# the buckle's plate faces sideways in CAD, so it reads as a sliver from the front; turn
# rod and strap together about the line from the pin through the strap to face it ahead
_PLATE_TURN_DEG = 0.0

# One entry per control step, recorded while the assist controller runs: what was asked for
# (at the hip and at the hinge), what the physics produced, and what it cost electrically.
_ASSIST_LOG_KEYS = (
    "phase",
    "commanded_hip_torque",
    "commanded_hinge_torque",
    "delivered_hinge_torque",
    "joint_speed",
    "voltage",
    "current",
    "saturated",
)



def _widen_stance(spec: mujoco.MjSpec, amount: float) -> None:
    """Open the stance in every keyframe: the hip adduction joints carry the leg width.

    The exo hangs off the pelvis, so its own abduction hinge follows the leg instead of
    setting the width; widening the legs is therefore a change of the human pose only.
    """
    model = spec.compile()
    limits = {}
    for name in ("hip_adduction_r", "hip_adduction_l"):
        jid = model.joint(name).id
        limits[int(model.jnt_qposadr[jid])] = model.jnt_range[jid]
    for key in spec.keys:
        qpos = key.qpos
        if qpos is None:
            continue
        for adr, (low, high) in limits.items():
            qpos[adr] = float(np.clip(qpos[adr] - amount, low, high))


def _extension_pulse(phase: float, center: float, width: float) -> float:
    """Raised-cosine window on the gait clock: 1 at *center*, 0 at ``+-width/2`` from it."""
    offset = (phase - center + 0.5) % 1.0 - 0.5
    half = 0.5 * width
    if abs(offset) >= half:
        return 0.0
    return float(0.5 * (1.0 + np.cos(np.pi * offset / half)))


@dataclasses.dataclass(frozen=True)
class ExoAssistConfig:
    """Phase-locked hip assist, expressed at the *human hip*.

    Attributes:
        hip_flexion_torque: Peak assist torque wanted at the hip, N*m, in the same sign
            convention as the hip joint itself - so **positive flexes and negative extends**.
            A push-off assist is therefore a negative number.  It is a prescription, not a
            promise: the request is referred to the hinge through the ``hip_torque_transfer``
            measured for the interface the model built - about 0.12 N*m at the hip per N*m at
            the hinge for the point-connected cuff (it carries force but no moment, so 2 N*m
            here already asks the module for its whole 17 N*m), about 1.0 for a clamped one.
            Anything above what the drive can hold is reported as saturation rather than
            quietly clipped.
        phase_lead: Fraction of the cycle the pulse is moved *earlier* than the side's own
            extension peak.  ``0`` pushes through maximum extension, ``0.15`` pushes in
            late stance and stops at toe-off, which is the usual prescription.
        width: Full width of the raised-cosine pulse, as a fraction of the cycle.
    """

    hip_flexion_torque: float = -2.0
    phase_lead: float = 0.15
    width: float = 0.30

    def __post_init__(self) -> None:
        if not 0.0 < self.width <= 1.0:
            raise ValueError("assist width must be in (0, 1] of a gait cycle")
        if abs(self.phase_lead) >= 0.5:
            raise ValueError("assist phase_lead must be within half a cycle")


class LegWalkExoEnvV0(LegWalkEnvV0):
    """``LegWalkEnvV0`` + Lin-I exoskeleton bodies, hinges and cuff welds."""

    def __init__(
        self,
        model_path: str,
        obsd_model_path: str | None = None,
        exo_params_path: str | None = None,
        exo_locked: bool = False,
        exo_weld: bool = True,
        exo_damping: float = 0.05,
        exo_armature: float = 0.0,
        exo_weld_solref: tuple[float, float] = (0.01, 1.0),
        exo_weld_solimp: tuple[float, ...] = (0.9, 0.95, 0.001, 0.5, 2.0),
        exo_visual_style: str = "mesh",
        exo_mesh_dir: str | None = None,
        exo_thigh_radius: float = None,
        exo_strap_preload_N: float = 0.0,
        exo_strap_friction: float = 0.5,
        exo_strap_radius: float = 0.067,
        exo_motor: bool = False,
        exo_motor_overrides: dict[str, Any] | None = None,
        exo_assist_torque: float | None = None,
        exo_assist_phase_lead: float = 0.15,
        exo_assist_width: float = 0.30,
        **kwargs: Any,
    ) -> None:
        if exo_motor and exo_locked:
            raise ValueError(
                "exo_motor drives the exo hip flexion hinge, which exo_locked removes"
            )
        if exo_assist_torque is not None and not exo_motor:
            raise ValueError(
                "exo_assist_torque commands the hip drive; build with exo_motor=True"
            )
        self._exo_assist = (
            None
            if exo_assist_torque is None
            else ExoAssistConfig(
                hip_flexion_torque=float(exo_assist_torque),
                phase_lead=float(exo_assist_phase_lead),
                width=float(exo_assist_width),
            )
        )
        self._exo_coupling: dict[str, Any] = {}
        self._exo_params: dict[str, Any] = {}
        self._exo_motor_names: list[str] = []
        self._exo_assist_transfer: np.ndarray = np.zeros(0)
        self._exo_assist_extension: np.ndarray = np.zeros(0)
        self.assist_log: dict[str, list] = {
            key: [] for key in _ASSIST_LOG_KEYS
        }
        self._exo_kwargs: dict[str, Any] = {
            "exo_params_path": exo_params_path,
            "exo_locked": bool(exo_locked),
            "exo_weld": bool(exo_weld),
            "exo_damping": float(exo_damping),
            "exo_armature": float(exo_armature),
            "exo_thigh_radius": exo_thigh_radius,
            "exo_strap_preload_N": float(exo_strap_preload_N),
            "exo_strap_friction": float(exo_strap_friction),
            "exo_strap_radius": float(exo_strap_radius),
            "exo_weld_solref": tuple(exo_weld_solref),
            "exo_weld_solimp": tuple(exo_weld_solimp),
            "exo_visual_style": str(exo_visual_style),
            "exo_mesh_dir": exo_mesh_dir,
            "exo_motor": bool(exo_motor),
            "exo_motor_overrides": dict(exo_motor_overrides or {}),
        }
        self._exo_meta: dict[str, Any] = {}
        self._exo_qpos_adr: np.ndarray = np.zeros(0, dtype=int)
        self._exo_dof_adr: np.ndarray = np.zeros(0, dtype=int)
        self._exo_ctrl_adr: np.ndarray = np.zeros(0, dtype=int)
        self._exo_n_ctrl: int = 0
        self._exo_motor_dof_adr: np.ndarray = np.zeros(0, dtype=int)
        self._exo_motor: ExoMotorConfig | None = None
        super().__init__(
            model_path,
            obsd_model_path,
            exo_params_path=exo_params_path,
            exo_locked=exo_locked,
            exo_weld=exo_weld,
            exo_damping=exo_damping,
            exo_armature=exo_armature,
            exo_weld_solref=tuple(exo_weld_solref),
            exo_weld_solimp=tuple(exo_weld_solimp),
            exo_visual_style=str(exo_visual_style),
            exo_mesh_dir=exo_mesh_dir,
            exo_motor=exo_motor,
            exo_motor_overrides=dict(exo_motor_overrides or {}),
            **kwargs,
        )
        self._index_exo_dofs()
        self._index_exo_motors()
        self._index_exo_assist()

    # -- model -----------------------------------------------------------------------
    def _build_model(self, model_path: str) -> tuple[mujoco.MjModel, mujoco.MjSpec]:
        params = load_exo_params(self._exo_kwargs["exo_params_path"])
        meta: dict[str, Any] = {}
        # how the exo hinges are coupled to the human hips, measured (see the params file)
        self._exo_coupling = dict(params.get("assist", {}).get("hip_coupling", {}))
        self._exo_params = params
        motor: ExoMotorConfig | None = None
        if self._exo_kwargs["exo_motor"]:
            motor = ExoMotorConfig.from_params(params)
            overrides = self._exo_kwargs["exo_motor_overrides"]
            if overrides:
                motor = dataclasses.replace(motor, **overrides)
        self._exo_motor = motor

        def _attach(spec: mujoco.MjSpec) -> mujoco.MjSpec:
            _widen_stance(spec, _STANCE_WIDEN_RAD)
            attach_kwargs = {}
            if self._exo_kwargs["exo_thigh_radius"] is not None:
                attach_kwargs["thigh_radius"] = float(self._exo_kwargs["exo_thigh_radius"])
            meta.update(
                attach_lin1_exo(
                    spec,
                    params,
                    locked=self._exo_kwargs["exo_locked"],
                    weld=self._exo_kwargs["exo_weld"],
                    damping=self._exo_kwargs["exo_damping"],
                    armature=self._exo_kwargs["exo_armature"],
                    motor=motor,
                    weld_solref=self._exo_kwargs["exo_weld_solref"],
                    weld_solimp=self._exo_kwargs["exo_weld_solimp"],
                    strap_preload=self._exo_kwargs["exo_strap_preload_N"],
                    strap_friction=self._exo_kwargs["exo_strap_friction"],
                    strap_radius=self._exo_kwargs["exo_strap_radius"],
                    **attach_kwargs,
                    visual_style=self._exo_kwargs["exo_visual_style"],
                    mesh_dir=self._exo_kwargs["exo_mesh_dir"],
                )
            )
            return spec

        model, spec = ModelBuilder.from_xml_file(model_path).apply_transform(_attach).build()
        style_muscle_geoms(model, _MUSCLE_WIDTH)
        place_strap_on_muscle(model, _ROD_SWING_DEG, _STRAP_OUT_MM, align_front=False,
                              roll_deg=_ROD_ROLL_DEG, lift_deg=_ROD_LIFT_DEG,
                              standoff_mm=_ROD_STANDOFF_MM, turn_deg=_PLATE_TURN_DEG,
                              hinge_deg=_ROD_HINGE_DEG)
        # the strap ring is worn where the rod ends, not at the CAD station
        seat_cuff_on_rod(model)
        pitch_exo(model, _ASSY_PITCH_DEG, _ASSY_FORWARD_MM)
        meta["endcap_clock_deg"] = clock_exo_endcap(
            model, _ENDCAP_CLOCK_DEG, _ENDCAP_CLOCK_KEY)
        self._exo_meta = meta
        return model, spec

    def _index_exo_dofs(self) -> None:
        """Record the qpos/dof addresses of the exo hinges (appended after the human DoFs).

        The strap pads' slide joints are exo coordinates too, and they are counted in
        ``added_dof`` because the observation has to drop them, but they are deliberately left
        out of these addresses: ``exo_joint_torques`` reads ``qfrc_actuator``, and the preload
        actuator of a pad reports its *clamp force* in newtons.  Counting that among the hinge
        torques made a healthy 100 N strap look like a 100 N.m drive, which tripped the
        drive-envelope check for reasons that have nothing to do with the motor.
        """
        strap = set(self._exo_meta.get("strap_joints", []))
        qadr, dadr = [], []
        for name in self._exo_meta.get("joints", []):
            if name in strap:
                continue
            jid = self.model.joint(name).id
            qadr.append(int(self.model.jnt_qposadr[jid]))
            dadr.append(int(self.model.jnt_dofadr[jid]))
        self._exo_qpos_adr = np.asarray(qadr, dtype=int)
        self._exo_dof_adr = np.asarray(dadr, dtype=int)
        n = int(self._exo_meta.get("added_dof", 0))
        if n and self._exo_qpos_adr.size:
            first = int(self._exo_qpos_adr.min())
            if first != self.model.nq - n:
                raise RuntimeError(
                    "exo DoFs are expected to be the last "
                    f"{n} qpos entries (found at {first} of {self.model.nq})"
                )

    def _index_exo_motors(self) -> None:
        """Record the ctrl slots of the hip drive and narrow the policy action to the muscles.

        A real exoskeleton runs its own controller, so its voltages are not part of the
        policy action; they live in the ctrl slots the drive appended and are set through
        :meth:`set_exo_voltage`.  The strap preload sits in the same appended block: it is a
        buckled strap, not an input, so it is excluded from the action along with the drive.
        """
        names = list(self._exo_meta.get("actuators", []))
        strap = list(self._exo_meta.get("strap_actuators", []))
        self._exo_motor_names = names
        self._exo_ctrl_adr = np.asarray(
            [int(self.model.actuator(name).id) for name in names], dtype=int
        )
        exo_ctrl = [int(self.model.actuator(name).id) for name in names + strap]
        self._exo_n_ctrl = len(exo_ctrl)
        self._exo_motor_dof_adr = np.asarray(
            [
                int(self.model.jnt_dofadr[self.model.actuator_trnid[idx, 0]])
                for idx in self._exo_ctrl_adr
            ],
            dtype=int,
        )
        if exo_ctrl:
            expected = np.arange(self.model.nu - len(exo_ctrl), self.model.nu)
            if sorted(exo_ctrl) != expected.tolist():
                raise RuntimeError(
                    f"exo actuators expected in ctrl slots {expected.tolist()}, "
                    f"found {sorted(exo_ctrl)}"
                )
            self.action_space = self._muscle_action_space(int(expected[0]))

    def _index_exo_assist(self) -> None:
        """Line the measured hip coupling up with the drive slots, in slot order.

        The actuator names end in the side letter, which is the only safe way to pair a
        slot with a number: the two sides' hinge coordinates turn opposite ways, so a
        mistake here silently assists one leg while resisting the other.

        The torque transfer is the one measured for the interface this model actually built:
        ~0.12 through the point-connected cuff, ~1.0 through a clamped one.  They differ by
        8x, so a controller would either saturate the drive or leave most of it unused if the
        two were interchangeable.
        """
        if self._exo_assist is None:
            return
        if self._exo_ctrl_adr.size == 0:
            raise RuntimeError("the assist controller found no hip drive to command")
        columns = {
            "extension_peak_clock_phase": "_exo_assist_extension",
        }
        missing = [key for key in columns if key not in self._exo_coupling]
        if missing:
            raise RuntimeError(
                f"lin1_exo_params.json records no {missing[0]!r}; the assist controller "
                "needs the measured coupling, not a guessed sign"
            )
        sides = [name.rsplit("_", 1)[-1] for name in self._exo_motor_names]

        for key, attr in columns.items():
            table = self._exo_coupling[key]
            try:
                setattr(self, attr, np.asarray([table[side] for side in sides],
                                               dtype=np.float64))
            except KeyError as exc:
                raise RuntimeError(
                    f"{key} has no entry for side {exc}; known: {sorted(table)}"
                ) from exc
        clamped = bool(self._exo_meta.get("strap"))
        try:
            transfer = hip_torque_transfer(self._exo_params, clamped)
        except KeyError as exc:
            raise RuntimeError(str(exc)) from exc
        try:
            self._exo_assist_transfer = np.asarray(
                [transfer[side] for side in sides], dtype=np.float64)
        except KeyError as exc:
            raise RuntimeError(
                f"the measured hip torque transfer has no entry for side {exc}; "
                f"known: {sorted(transfer)}") from exc
        if np.any(np.abs(self._exo_assist_transfer) < 0.02):
            raise RuntimeError(
                f"the recorded hip_torque_transfer {self._exo_assist_transfer.tolist()} is "
                "so small that referring a hip prescription to the hinge would divide it away"
            )

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start an episode: the drive is unpowered and the assist trace is empty."""
        out = super().reset(seed=seed, options=options, **kwargs)
        self.assist_log = {key: [] for key in _ASSIST_LOG_KEYS}
        return out

    # -- assist control ----------------------------------------------------------------
    def _step_physics(self, nstep: int | None = None) -> None:
        """Command the assist voltage, integrate, then log what the drive actually did.

        The hook is :meth:`_step_physics` rather than ``step`` because the base task applies
        the muscle command immediately before it: the exo controller therefore reads the same
        state the muscles were computed from, exactly one control period before the physics
        sees either.
        """
        if self._exo_assist is None:
            super()._step_physics(nstep)
            return
        # One voltage per control step is not enough.  The hinge picks up a few rad/s inside
        # 10 ms, and since back-EMF rides on that speed, a held voltage delivers roughly 70%
        # of the requested torque by the end of the step: the shortfall would be a property
        # of the sample rate, not of the machine.  Re-solving it every physics step is the
        # 1 kHz analogue of the current loop a real QDD closes for the same reason.
        steps = self.frame_skip if nstep is None else int(nstep)
        request = None
        for _ in range(steps):
            request = self._apply_exo_assist()
            super()._step_physics(1)
        if request is not None:
            self._log_assist(request)

    def _apply_exo_assist(self) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Turn the configured assist into terminal voltages for this control step.

        The pulse is written at the human hip and referred to the hinge through the measured
        ``hip_torque_transfer``, so the sign convention of the joint name never reaches the
        configuration.  The voltage comes from :meth:`ExoMotorConfig.voltage_for_torque`,
        which is the inverse of the same line MuJoCo integrates: the request is honoured
        where the machine can keep it, and the bus limit or the torque cap is what shows up in
        the log otherwise.
        """
        cfg = self._exo_assist
        motor = self._exo_motor
        phase = (self.steps / self.hip_period) % 1.0
        centers = (self._exo_assist_extension - cfg.phase_lead) % 1.0
        shape = np.array([_extension_pulse(phase, c, cfg.width) for c in centers])
        hip_torque = cfg.hip_flexion_torque * shape
        hinge_torque = hip_torque / self._exo_assist_transfer
        speed = self.data.qvel[self._exo_motor_dof_adr]
        voltage = np.array(
            [motor.voltage_for_torque(t, w) for t, w in zip(hinge_torque, speed)]
        )
        self.set_exo_voltage(voltage)
        reachable = np.array(
            [motor.envelope(v, w) for v, w in zip(voltage, speed)]
        )
        saturated = np.abs(reachable) + 1e-9 < np.abs(hinge_torque)
        return phase, hip_torque, hinge_torque, voltage, saturated

    def _log_assist(self, request: tuple) -> None:
        phase, hip_torque, hinge_torque, voltage, saturated = request
        state = self.exo_motor_state()
        log = self.assist_log
        log["phase"].append(phase)
        log["commanded_hip_torque"].append(hip_torque)
        log["commanded_hinge_torque"].append(hinge_torque)
        log["delivered_hinge_torque"].append(
            self.data.qfrc_actuator[self._exo_motor_dof_adr].copy()
        )
        # read with the torque, from the same end-of-step state: the two are the power the
        # drive actually delivered over this control step
        log["joint_speed"].append(self.data.qvel[self._exo_motor_dof_adr].copy())
        log["voltage"].append(voltage)
        log["current"].append(state["current"].copy())
        log["saturated"].append(saturated)

    def assist_summary(self) -> dict[str, np.ndarray]:
        """This episode's assist trace, one row per control step (empty when unused)."""
        return {key: np.asarray(value) for key, value in self.assist_log.items()}

    # -- observation -----------------------------------------------------------------
    @property
    def n_exo_dof(self) -> int:
        return int(self._exo_meta.get("added_dof", 0))

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        obs = super()._get_obs_dict(accessor)
        n = self.n_exo_dof
        if n:
            obs["qpos_without_xy"] = obs["qpos_without_xy"][:-n]
            obs["qvel"] = obs["qvel"][:-n]
        # ``muscle_length/velocity/force`` are read off ``data.actuator_*``, which is indexed
        # by actuator, not by muscle: the exo's own ctrl slots would otherwise append numbers
        # the baseline policy has never seen and silently break the transfer.  The count comes
        # from the build metadata rather than ``_exo_n_ctrl`` because the base env computes
        # ``observation_space`` from this method *before* ``_index_exo_motors`` runs, and an
        # inflated space (403 -> 415) makes DummyVecEnv reject the first reset.
        meta = getattr(self, "_exo_meta", {})
        d = len(meta.get("actuators", [])) + len(meta.get("strap_actuators", []))
        if d:
            for key in ("muscle_length", "muscle_velocity", "muscle_force"):
                obs[key] = obs[key][:-d]
        return obs

    # -- exoskeleton helpers ---------------------------------------------------------
    def exo_joint_names(self) -> list[str]:
        return list(self._exo_meta.get("joints", []))

    def exo_body_names(self) -> list[str]:
        return list(self._exo_meta.get("bodies", []))

    def exo_weld_names(self) -> list[str]:
        return list(self._exo_meta.get("welds", []))

    def exo_dof_positions(self) -> np.ndarray:
        return self.data.qpos[self._exo_qpos_adr].copy()

    def exo_dof_velocities(self) -> np.ndarray:
        return self.data.qvel[self._exo_dof_adr].copy()

    def exo_joint_torques(self) -> np.ndarray:
        """Actuator forces applied at the exo hinges (zero while unpowered)."""
        if self._exo_dof_adr.size == 0:
            return np.zeros(0)
        return self.data.qfrc_actuator[self._exo_dof_adr].copy()

    def exo_motor_max_torques(self) -> np.ndarray:
        """Stalled torque the drive chain can hold at each powered hinge, N·m at the joint."""
        if self._exo_motor is None:
            return np.zeros(self._exo_dof_adr.size)
        return np.full(
            self._exo_ctrl_adr.size, float(self._exo_motor.peak_joint_torque)
        )

    # -- hip drive ---------------------------------------------------------------------
    def set_exo_voltage(self, voltage: np.ndarray | float) -> None:
        """Command the hip-module terminal voltages, in volts, one per powered side.

        Takes effect from the next physics step and holds until the next call; ``reset()``
        leaves the drive unpowered, because MuJoCo re-zeroes ``ctrl`` from the actuator
        defaults.
        """
        if self._exo_ctrl_adr.size == 0:
            raise RuntimeError("this environment was built without the hip drive")
        volt = np.broadcast_to(np.asarray(voltage, dtype=np.float64),
                              self._exo_ctrl_adr.shape).astype(np.float64)
        cr = self.model.actuator_ctrlrange[self._exo_ctrl_adr]
        self.data.ctrl[self._exo_ctrl_adr] = np.clip(volt, cr[:, 0], cr[:, 1])

    def exo_motor_state(self) -> dict[str, np.ndarray]:
        """Electrical-side readout of each hip module: voltage, current, rotor torque, speed.

        ``ACTUATORFRC`` is reported by MuJoCo *before* the gear ratio is applied, so
        ``rotor_torque`` is the motor-shaft number; the torque reaching the hinge is that
        times the ratio, which :meth:`exo_joint_torques` reads off the joint directly.
        ``current`` is recovered as ``rotor_torque / K``, the same quantity the terminal
        voltage element uses to produce torque, so a current limit is checkable.
        """
        if self._exo_ctrl_adr.size == 0 or self._exo_motor is None:
            return {}
        rotor = self._exo_sensor_col("exo_motor_torque")
        speed = self._exo_sensor_col("exo_hip_speed")
        return {
            "voltage": self.data.ctrl[self._exo_ctrl_adr].copy(),
            "current": rotor / self._exo_motor.torque_constant,
            "rotor_torque": rotor,
            "joint_speed": speed,
        }

    def exo_strap_torques(self) -> np.ndarray:
        """Moment inside the exo's own kinematic chain at the cuff frame, per side, N*m.

        Not the assistance the wearer receives.  The cuff is joined to the femur by a
        point-connect equality, which is outside the tree the sensor integrates over, so this
        reads the cuff's own residual dynamics and stays around 0.04 N*m whether the drive is
        pushing 0 or its whole 17.  The wearer-facing number is the hip moment in
        :meth:`assist_summary`, referred through ``hip_torque_transfer``.
        """
        return self._exo_sensor_stack("exo_strap_torque")

    def exo_strap_forces(self) -> np.ndarray:
        """Strap pressure on the limb, one 3-vector per side, N.

        The force half of the same interface wrench; it is what a load cell in the strap
        would report, and what decides whether the cuff slips or digs in.
        """
        return self._exo_sensor_stack("exo_strap_force")

    def exo_strap_slip_torques(self) -> np.ndarray:
        """Hip moment each clamped band passes before its pads slide, N*m, per banded side.

        ``2*mu*N*r`` - the interface's own ceiling, which is the number the worn exoskeleton
        is sized by while the drive is unpowered.  Empty when the model was built with no
        preload, i.e. when the cuff is still only a point connection.
        """
        table = self._exo_meta.get("strap_slip_torque_Nm", {})
        sides = list(self._exo_meta.get("strap_sides", []))
        if not table or not sides:
            return np.zeros(0)
        return np.array([float(table[side]) for side in sides])

    def exo_preload_forces(self) -> np.ndarray:
        """Normal force the band is holding on the thigh, per pad, N (read from the joints)."""
        names = list(self._exo_meta.get("strap_actuators", []))
        if not names:
            return np.zeros(0)
        adr = np.asarray([int(self.model.actuator(name).id) for name in names], dtype=int)
        return self.data.actuator_force[adr].copy()

    def _exo_sensor_stack(self, prefix: str) -> np.ndarray:
        names = [n for n in self._exo_meta.get("sensors", []) if n.startswith(prefix)]
        if not names:
            return np.zeros((0, 3))
        return np.stack([self._exo_sensor(name, 3) for name in names])

    def _exo_sensor(self, name: str, width: int) -> np.ndarray:
        sid = self.model.sensor(name).id
        adr = int(self.model.sensor_adr[sid])
        return self.data.sensordata[adr : adr + width].copy()

    def _exo_sensor_col(self, prefix: str) -> np.ndarray:
        return np.array([
            self._exo_sensor(f"{prefix}_{name.rsplit('_', 1)[-1]}", 1)[0]
            for name in self._exo_meta["actuators"]
        ])

    def exo_interface_loads(self) -> np.ndarray:
        """Cuff reaction torque at each exo hinge, N·m — the load that crosses to the leg.

        On the hinge equation of motion the motor term and the cuff-weld term differ by
        whatever it takes to swing the exo's own link, so this is the torque the device
        delivers *through the strap* rather than the torque the motor shaft produces;
        :meth:`exo_joint_torques` is the latter.  Any contact on the exo frame itself is
        folded into the same number.
        """
        if self._exo_dof_adr.size == 0:
            return np.zeros(0)
        return self.data.qfrc_constraint[self._exo_dof_adr].copy()

    def exo_frame_state(self) -> dict[str, np.ndarray]:
        """World pose of every exo body (for logging/visualisation)."""
        out: dict[str, np.ndarray] = {}
        for name in self.exo_body_names():
            bid = self.model.body(name).id
            out[name] = np.concatenate([self.data.xpos[bid], self.data.xquat[bid]])
        return out
