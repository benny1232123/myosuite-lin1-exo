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
    load_exo_params,
    style_muscle_geoms,
)
from myosuite.envs.myo.tasks.basic.leg.walk import LegWalkEnvV0


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
        style_muscle_geoms(model)
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