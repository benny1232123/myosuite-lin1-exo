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

The observation vector is bit-identical to ``myoLegWalk-v0`` (the four exo
coordinates are stripped from ``qpos_without_xy``/``qvel``), so the published
``myoLegWalk-v0`` baseline policy can be loaded into this environment unchanged.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from myosuite.core.model_builder import ModelBuilder
from myosuite.envs.gymnasium_env import CpuEnvAccessor
from myosuite.envs.myo.tasks.basic.leg.lin1_exo import (
    attach_lin1_exo,
    clock_exo_endcap,
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
        **kwargs: Any,
    ) -> None:
        self._exo_kwargs: dict[str, Any] = {
            "exo_params_path": exo_params_path,
            "exo_locked": bool(exo_locked),
            "exo_weld": bool(exo_weld),
            "exo_damping": float(exo_damping),
            "exo_armature": float(exo_armature),
            "exo_thigh_radius": exo_thigh_radius,
            "exo_weld_solref": tuple(exo_weld_solref),
            "exo_weld_solimp": tuple(exo_weld_solimp),
            "exo_visual_style": str(exo_visual_style),
            "exo_mesh_dir": exo_mesh_dir,
        }
        self._exo_meta: dict[str, Any] = {}
        self._exo_qpos_adr: np.ndarray = np.zeros(0, dtype=int)
        self._exo_dof_adr: np.ndarray = np.zeros(0, dtype=int)
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
            **kwargs,
        )
        self._index_exo_dofs()

    # -- model -----------------------------------------------------------------------
    def _build_model(self, model_path: str) -> tuple[mujoco.MjModel, mujoco.MjSpec]:
        params = load_exo_params(self._exo_kwargs["exo_params_path"])
        meta: dict[str, Any] = {}

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
                    weld_solref=self._exo_kwargs["exo_weld_solref"],
                    weld_solimp=self._exo_kwargs["exo_weld_solimp"],
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
        """Record the qpos/dof addresses of the exo hinges (appended after the human DoFs)."""
        qadr, dadr = [], []
        for name in self._exo_meta.get("joints", []):
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
        return np.zeros(self._exo_dof_adr.size)

    def exo_frame_state(self) -> dict[str, np.ndarray]:
        """World pose of every exo body (for logging/visualisation)."""
        out: dict[str, np.ndarray] = {}
        for name in self.exo_body_names():
            bid = self.model.body(name).id
            out[name] = np.concatenate([self.data.xpos[bid], self.data.xquat[bid]])
        return out
