# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""'Lin-I' hip exoskeleton (simplified assembly) attachment for the MyoLeg models.

The inertial parameters, group layout and kinematic landmarks of the 'Lin-I' hip
exoskeleton robot development platform (V0.1.1 simplified assembly) were extracted
from the native STEP file and are stored in
``myosuite/envs/myo/assets/leg/lin1_exo/lin1_exo_params.json``.  All quantities in
that file are already SI (metres, kilograms) and expressed in the CAD frame

    +X anterior, +Y left, +Z superior, origin at the pelvis centre.

Frame convention used here
--------------------------
Every exo body is created as a child of a human body and keeps the *human* frame
orientation (identity ``quat``), so all exo geometry is expressed directly in
``pelvis``-local coordinates:

    pelvis local  x = anterior,  y = superior,  z = right

The CAD frame is mapped onto it with a fixed permutation

    R0: CAD x -> pelvis x,  CAD y -> pelvis -z,  CAD z -> pelvis y

Kinematic placement (per side)
------------------------------
* the CAD hip-axis point is mapped onto the human hip joint centre (HJC), which is
  the origin of the ``femur_l/r`` body, i.e. ``pelvis``-local ``femur.pos``;
* the CAD hip axis (CAD +-Y) is mapped onto the hip medial-lateral axis, i.e. the
  human ``hip_flexion_*`` axis (``pelvis``-local +-Z, outward per side);
* the resulting rotation about that lateral axis (the CAD assembly is drawn with the
  thigh ~42 deg flexed, the human reference pose is straight) maps the CAD thigh-link
  direction exactly onto the human thigh direction ``femur -> tibia``;
* consequently the distal cuff lands exactly on the thigh at
  ``hip_to_cuff`` / ``|femur->tibia|`` (~87 % of the femur, ~50 mm above the knee),
  which makes the closed kinematic chain track the human hip exactly.

Degrees of freedom (two per side, mirroring the device)
------------------------------------------------------
``exo_hip_flexion_{r,l}``  active sagittal flexion/extension about the hip
                           medial-lateral axis (the exo motor torque lives here; it
                           stays absent while the device is unpowered);
``exo_axial_roll_{r,l}``   passive rotation of the thigh link about its own
                           longitudinal axis - the pin-in-slot transmission of the
                           real device, which lets the exo follow hip internal
                           rotation without driving it.

Human/exo coupling
------------------
The waist/back frame is a joint-free child of ``pelvis`` (a rigid 6-DoF attachment),
and the distal cuff is welded to ``femur_{r,l}`` (rigid 6-DoF attachment).  With two
exo DoFs per side and three human hip DoFs per side the closed chain carries one
redundant constraint (it corresponds to hip adduction/abduction, which the device
does not act on); a finite-stiffness weld absorbs it, standing in for the slot
play and the strap/cushion compliance.

Modes
-----
``locked=True``   every exo DoF is removed: the exoskeleton becomes a rigid,
                  purely inertial extension of the pelvis and thigh (no
                  constraint, zero coupling error).  Useful as an inert-load
                  reference and for policy transfer.
``locked=False``  the physical chain: 2 hinges per side + soft cuff welds.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import mujoco
import numpy as np

__all__ = [
    "EXO_PARAMS_PATH",
    "SIDES",
    "attach_lin1_exo",
    "load_exo_params",
]

# Asset location: <repo>/myosuite/envs/myo/assets/leg/lin1_exo/lin1_exo_params.json
EXO_ASSET_DIR = pathlib.Path(__file__).resolve().parents[3] / "assets" / "leg" / "lin1_exo"
EXO_PARAMS_PATH = EXO_ASSET_DIR / "lin1_exo_params.json"

SIDES = ("r", "l")

# CAD (+X anterior, +Y left, +Z superior) -> pelvis local (x anterior, y superior, z right)
_R_CAD_TO_PELVIS = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ]
)

# Bodies welded rigidly to the waist frame (they share the back frame and the hip axis).
_FRAME_GROUPS = ("frame", "bracket_r", "bracket_l", "hipmod_r", "hipmod_l")

# The module's end cover ships as its own solid, ``hipcap_<side>.stl`` (the thin disc outboard of
# the drum - connected component #12614/#12667 of the CAD ``hipmod_<side>_0.stl``).  It is the one
# housing piece that turns: ``clock_exo_endcap`` rotates it about the flexion axis so its relief
# keeps tracking the crank fork, while the finned motor cylinder and the mounting arm stay bolted
# to the waist frame exactly like the bracket.

# The back plate, the two arms and the hip modules are one rigid CAD part, drawn in the
# CAD's own assembly pose: the arms' end plates carry the drums' axles, so the frame is
# never shifted on its own.  ``pitch_exo`` tips and slides the whole device instead.

# The crank pin the long link hangs off, measured off the SolidWorks per-part export: the
# link's proximal end sits 44 mm from the hip axis point, in CAD millimetres.
_CRANK_PIN_MM = {
    "r": (148.9, -184.1, -81.1),
    "l": (149.7, 184.9, -77.8),
}
# The pin is a real clevis hinge: the rod swings on it as the strap drags it along the
# thigh, and on the hardware it rotates visibly through the gait (user: 这个铰链部分不是
# 刚性连接的 可以旋转的).  The centring spring only picks the assembly branch, so it is
# soft - the hardware's return spring, not a weld.  (A stiffer k=20 with half the travel
# made the joint read as rigidly bolted.)
_CRANK_PIN_RANGE = (-0.45, 0.45)
_CRANK_PIN_FRICTIONLOSS = 0.0
_CRANK_PIN_STIFFNESS = 2.0

# The hip it is worn on is not a single-axis joint, so the exo's hip is two DOF: the
# module's axle gives flexion, and the assembly also swings in the frontal plane.  Without
# that second DOF the whole chain is trapped in the sagittal plane and cannot follow the
# thigh, and the cuff strap tore open by up to 70 mm every stride.
_HIP_ABDUCTION_RANGE = (-0.35, 0.35)
_ABDUCTION_SIGN = {"r": -1.0, "l": 1.0}   # sign per side, so positive = outward

# The crank fork and the drum's end cover turn together about the motor axis (the user:
# 外壳表壳和曲折杆以整个电机的中轴进行旋转).  The cover's clock is already tied to the flexion
# key, so the fork angle alone sets both.  The seed turns the fork by this angle and the
# pin/slot absorb the weld, so the link and the thigh ring keep their pose while the fork
# and the cover visibly clock.  Sign per side so both forks turn the same way relative to
# their hip.
_CRANK_CLOCK_DEG = 10.0
_CRANK_CLOCK_SIGN = {"r": 1.0, "l": -1.0}

# walk.py resets from keyframe 2 ("init", the default) or draws 2 and 3 ("random"), so the
# strap anchor is re-clamped on keyframe 2.
_RESET_KEY = 2

# The CAD places the waist frame where its own back plate sits behind the user.  The
# back plate, the arms and the hip modules are one rigid part, so the frame cannot be
# moved on its own without pulling the arm ends out of the hip modules' slots.

# Deliberately avoids red: MyoSuite draws the muscle wrap geoms in red, and a red
# exoskeleton merges into them in every render.  The five groups are also ordered by
# brightness so the back frame stays readable against the body.
_RGBA = {
    "frame": (0.30, 0.32, 0.36, 1.0),
    "bracket": (0.62, 0.65, 0.70, 1.0),
    "hipmod": (0.16, 0.42, 0.85, 1.0),
    "crank": (0.06, 0.17, 0.50, 1.0),
    "thigh": (0.90, 0.55, 0.10, 1.0),
    "cuff": (0.55, 0.85, 0.25, 1.0),
}


def load_exo_params(path: str | pathlib.Path | None = None) -> dict[str, Any]:
    """Return the CAD-derived exoskeleton parameters."""
    with open(path or EXO_PARAMS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------------------
# small linear-algebra helpers
# --------------------------------------------------------------------------------------
def _unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    return vec / np.linalg.norm(vec)


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix of ``angle`` rad about the unit ``axis``."""
    axis = _unit(axis)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _principal_inertia(inertia: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Express ``inertia`` (CAD frame, about the group COM) in a rotated body frame.

    Returns ``(principal_moments, iquat)`` ready for ``MjsBody.inertia/iquat``.
    """
    rot = np.eye(3) @ np.asarray(rotation, dtype=float)
    moments, axes = np.linalg.eigh(rot @ np.asarray(inertia, dtype=float) @ rot.T)
    if np.linalg.det(axes) < 0:  # keep a proper rotation
        axes = axes.copy()
        axes[:, 0] *= -1.0
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(axes, dtype=np.float64).ravel())
    return np.maximum(moments, 1e-9), quat


def _parallel_axis(mass: float, offset: np.ndarray) -> np.ndarray:
    d = np.asarray(offset, dtype=float)
    return float(mass) * (float(d @ d) * np.eye(3) - np.outer(d, d))


def _combine_groups(
    parts: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]],
    rotation: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse CAD groups into a single body mass/inertia in the rotated body frame.

    ``parts`` items are ``(mass, com, inertia, rotation_i)`` in CAD coordinates.
    Returns ``(mass, com_body, inertia_body, rotation_body)``.
    """
    mass = float(sum(part[0] for part in parts))
    com = sum(part[0] * np.asarray(part[1], float) for part in parts) / mass
    inertia = np.zeros((3, 3))
    for part_mass, part_com, part_inertia, part_rot in parts:
        rot = rotation @ np.asarray(part_rot, dtype=float)
        inertia += (
            rot @ np.asarray(part_inertia, dtype=float) @ rot.T
            + _parallel_axis(part_mass, np.asarray(part_com, float) - com)
        )
    return mass, rotation @ com, inertia, rotation


# --------------------------------------------------------------------------------------
# model building blocks
# --------------------------------------------------------------------------------------
def _get_body(spec: mujoco.MjSpec, name: str) -> Any:
    body = spec.body(name)
    if body is None:
        raise KeyError(f"body '{name}' not found in the spec")
    return body


def _axis_sign(body: Any, joint_name: str, direction: np.ndarray) -> float:
    """+1/-1 mapping a human joint angle onto an exo hinge that shares its axis.

    ``direction`` is expressed in the pelvis frame, while the human joint axis lives in
    the (femur) body frame, so the body quaternion is applied first.
    """
    quat = np.asarray(body.quat if body.quat is not None else [1.0, 0.0, 0.0, 0.0], dtype=float)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.ascontiguousarray(quat, dtype=np.float64))
    axis = rot.reshape(3, 3) @ np.asarray(_get_joint_axis(body, joint_name), dtype=float)
    return float(np.sign(axis @ np.asarray(direction, dtype=float)) or 1.0)


def _get_joint_axis(body: Any, joint_name: str) -> np.ndarray:
    for jnt in body.joints:
        if jnt.name == joint_name:
            return np.asarray(jnt.axis, dtype=float)
    raise KeyError(f"joint '{joint_name}' not found on body '{body.name}'")


def _mat_to_quat(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix -> MuJoCo ``(w, x, y, z)`` quaternion."""
    quat = np.zeros(4, dtype=np.float64)
    mat = np.ascontiguousarray(np.asarray(rotation, dtype=np.float64).ravel())
    mujoco.mju_mat2Quat(quat, mat)
    return quat


def _add_box(
    body: Any,
    center: np.ndarray,
    half_extents: np.ndarray,
    quat: np.ndarray,
    rgba: tuple[float, ...],
    group: int,
    name: str,
) -> Any:
    """Visual-only bounding box (no mass, no contact)."""
    return body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=np.asarray(half_extents, dtype=np.float64),
        pos=np.asarray(center, dtype=np.float64),
        quat=np.asarray(quat, dtype=np.float64),
        mass=0.0,
        contype=0,
        conaffinity=0,
        rgba=np.asarray(rgba, dtype=np.float64),
        group=group,
        friction=np.array([0.0, 0.005, 1e-4]),
    )


def _add_mesh_geom(
    spec: mujoco.MjSpec,
    body: Any,
    mesh_path: pathlib.Path,
    mesh_names: dict[pathlib.Path, str],
    rotation: np.ndarray,
    anchor: np.ndarray,
    rgba: tuple[float, ...],
    group: int,
    name: str,
) -> Any:
    """Visual-only CAD mesh (no mass, no contact), placed at ``rotation @ (p - anchor)``.

    The STL is authored in CAD coordinates (metres), so the whole CAD->body mapping is
    carried by the geom transform alone: ``x_body = pos + R @ p_cad`` with
    ``pos = -R @ anchor`` gives ``R @ (p_cad - anchor)``.
    """
    mesh_path = pathlib.Path(mesh_path)
    mesh_name = mesh_names.get(mesh_path)
    if mesh_name is None:
        mesh_name = f"lin1_{mesh_path.stem}"
        spec.add_mesh(name=mesh_name, file=str(mesh_path))
        mesh_names[mesh_path] = mesh_name
    return body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh_name,
        pos=np.asarray(-rotation @ anchor, dtype=np.float64),
        quat=_mat_to_quat(rotation),
        mass=0.0,
        contype=0,
        conaffinity=0,
        rgba=np.asarray(rgba, dtype=np.float64),
        group=group,
        friction=np.array([0.0, 0.005, 1e-4]),
    )


def _group_visual(
    spec: mujoco.MjSpec,
    body: Any,
    name: str,
    group: dict[str, Any],
    rotation: np.ndarray,
    anchor: np.ndarray,
    rgba: tuple[float, ...],
    group_id: int,
    mesh_names: dict[pathlib.Path, str],
    mesh_dir: pathlib.Path | None,
    visual_style: str,
) -> str:
    """Add one visual geom for a CAD group: the real mesh, or its bounding box.

    Returns the geom name.  ``mesh_dir`` points at ``<group>.stl`` files produced by
    ``cad_exo_mesh.py``; when a file is missing the bounding box is used instead so the
    model still builds (boxes are only a fallback - the mesh is the readable one).
    """
    if visual_style == "mesh" and mesh_dir is not None:
        # The SolidWorks export splits each group across <group>_0.stl, <group>_1.stl, ...
        # because one mesh would exceed MuJoCo's 200,000-face limit; add every file.
        paths = sorted(pathlib.Path(mesh_dir).glob(f"{name}_*.stl"))
        single = pathlib.Path(mesh_dir) / f"{name}.stl"
        if single.is_file():
            paths.append(single)
        if paths:
            last = ""
            for index, path in enumerate(paths):
                last = _add_mesh_geom(
                    spec, body, path, mesh_names, rotation, anchor, rgba, group_id,
                    f"exo_mesh_{name}_{index}",
                ).name
            return last
        if name.startswith("bracket"):
            # the bracket solids now live inside the ``frame`` mesh - draw nothing
            return ""
    lo = np.asarray(group["bbox_min_m"], dtype=float)
    hi = np.asarray(group["bbox_max_m"], dtype=float)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return _add_box(
        body, rotation @ (center - anchor), half, _mat_to_quat(rotation), rgba,
        group_id, f"exo_box_{name}",
    ).name


# --------------------------------------------------------------------------------------
# attachment
# --------------------------------------------------------------------------------------
def attach_lin1_exo(
    spec: mujoco.MjSpec,
    params: dict[str, Any] | None = None,
    *,
    locked: bool = True,
    weld: bool = True,
    damping: float = 0.05,
    armature: float = 0.0,
    flexion_range: tuple[float, float] = (-1.6, 1.6),
    roll_range: tuple[float, float] = (-1.2, 1.2),
    slot_range: tuple[float, float] = (-0.06, 0.06),
    # The anchor is where the strap weld closes onto; the strap pad itself is
    # rebuilt with a radial floor bent to 45 mm about the femur axis, so the pad
    # hovers ~5-12 mm over the bare bone in its band no matter where the anchor
    # sits, and the outer bracket arm rides clear at its own radii.
    # The radial offset therefore does not belong here: mjEQ_CONNECT drives the
    # cuff-site origin, which sits on the roll axis, onto the anchor, so a
    # non-zero radius displaces the whole ring off the femur axis instead of
    # sizing it.  The only joint that can pay for that is hip abduction, and it
    # is the fork's seat on the motor end-cap -- 60 mm of anchor offset costs
    # 12.5 deg of abduction and lifts the fork 7 mm clear of the cover.
    thigh_radius: float = 0.0,
    slot_mass: float = 0.01,
    weld_solref: tuple[float, float] = (0.01, 1.0),
    weld_solimp: tuple[float, ...] = (0.9, 0.95, 0.001, 0.5, 2.0),
    geom_group: int = 1,
    add_visuals: bool = True,
    visual_style: str = "mesh",
    mesh_dir: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Attach the 'Lin-I' exoskeleton to an existing MyoLeg ``spec`` (in place).

    Args:
        spec: compiled-pending spec of ``myolegs_with_torso.xml`` (or any MyoLeg model
            exposing ``pelvis``, ``femur_{r,l}``, ``tibia_{r,l}``).
        params: CAD parameter dict; defaults to :data:`EXO_PARAMS_PATH`.
        locked: drop the exo degrees of freedom (rigid inertial load, no welds).
        weld: add the cuff/femur welds (only used when ``locked=False``).
        damping: passive viscous damping on both exo hinges (Nm s/rad).
        armature: rotor/structural inertia added to the exo hinges.
        flexion_range: hip flexion/extension limits (human hip flexion range plus a small
            margin, so the strap stretch cannot push the exo hinge into its stop).
        roll_range: passive link roll limits (pin-in-slot travel, rad).
        weld_solref: thigh strap constraint ``(timeconst, dampratio)``.  The time constant
            is kept near the solver timestep: with the hip abduction DOF the cuff target is
            kinematically reachable (the reset gap is exactly zero), so the strap tracks the
            thigh instead of lagging 20-40 mm behind it on every stride.
        weld_solimp: thigh strap impedance parameters - a stiff, padded strap.
        geom_group: MuJoCo geom group used for the exo visuals.
        add_visuals: add the exo visuals at all.
        visual_style: ``"mesh"`` draws the real CAD tessellation (one STL per rigid-body
            group, produced by ``cad_exo_mesh.py``); ``"box"`` draws the group bounding
            boxes.  ``"mesh"`` silently falls back to boxes for any group whose STL is
            missing, so the model always builds.
        mesh_dir: where the ``<group>.stl`` files live; defaults to
            ``EXO_ASSET_DIR / "meshes"``.

    Returns:
        Metadata dict (body/joint/weld names, per-side frames) for tests and envs.
    """
    if params is None:
        params = load_exo_params()
    groups = params["groups"]
    landmarks = params["landmarks"]
    if mesh_dir is None:
        mesh_dir = EXO_ASSET_DIR / "meshes"
    mesh_dir = pathlib.Path(mesh_dir)

    pelvis = _get_body(spec, "pelvis")
    frames = {}
    seed_map: dict[str, dict[str, float]] = {}
    attach_sites: dict[str, Any] = {}
    quat_id = np.array([1.0, 0.0, 0.0, 0.0])
    mesh_names: dict[pathlib.Path, str] = {}
    added_dof = 0

    # ---- waist frame: rigid child of pelvis (carries frame + brackets + hip modules) ----
    frame_parts = []
    for name in _FRAME_GROUPS:
        if name.startswith("hipmod"):
            # the motor housing is drawn here, but its lumped mass rides the crank body
            # (the whole module's centroid sits 6 mm off the hip axis)
            continue
        grp = groups[name]
        frame_parts.append(
            (float(grp["mass_kg"]), np.asarray(grp["com_m"], float),
             np.asarray(grp["inertia_kgm2"], float), np.eye(3))
        )
    # ---- where the whole assembly sits on the body ------------------------
    # The CAD frame is much wider than a human pelvis: its hip axes are 191 mm off the
    # CAD midline, the human hip joints only 77 mm off theirs.  So the assembly is *not*
    # centred on the body - each hip module ends up 114 mm outboard of the hip joint, with
    # the back frame arching behind the user and its waist pad (the frame's front face,
    # CAD x = +40 mm) landing on the lumbar surface.  This single translation reproduces
    # that, and it is self-checking: with the exo axis there, the CAD link length of
    # 352.2 mm lands 351.5 mm from the thigh-surface strap point, i.e. the CAD was designed
    # for exactly this mounting.
    shifts = []
    for side in SIDES:
        hip = np.asarray(_get_body(spec, f"femur_{side}").pos, dtype=float)
        hip_cad = np.asarray(landmarks[side]["hip_point_mm"], dtype=float) * 1e-3
        exo_local = _R_CAD_TO_PELVIS @ hip_cad
        # keep the hip module's height and its fore-aft position, take the CAD's width
        target = np.array([hip[0], hip[1], np.sign(hip[2]) * abs(exo_local[2])])
        shifts.append(target - exo_local)
    frame_origin = np.mean(shifts, axis=0)

    mass, com, inertia, _ = _combine_groups(frame_parts, _R_CAD_TO_PELVIS)
    moments, iquat = _principal_inertia(inertia, np.eye(3))
    exo_frame = pelvis.add_body(
        name="exo_frame", pos=np.asarray(frame_origin, dtype=np.float64), quat=quat_id)
    exo_frame.mass = mass
    exo_frame.ipos = np.asarray(com, dtype=np.float64)
    exo_frame.inertia = np.asarray(moments, dtype=np.float64)
    exo_frame.iquat = np.asarray(iquat, dtype=np.float64)
    if add_visuals:
        # every frame chunk is drawn in its CAD pose: the arms' end plates carry the drums'
        # axles, so a shifted frame would sit off the drums
        for name in _FRAME_GROUPS:
            kind = "frame" if name == "frame" else (
                "bracket" if name.startswith("bracket") else "hipmod")
            _group_visual(
                spec, exo_frame, name, groups[name], _R_CAD_TO_PELVIS, np.zeros(3),
                _RGBA[kind], geom_group, mesh_names, mesh_dir, visual_style,
            )
        for side in SIDES:
            # no bounding-box fallback: a box cannot nest on the drum, so a missing cover
            # asset just leaves the housing as one un-turnable piece
            cap = mesh_dir / f"hipcap_{side}.stl"
            if visual_style == "mesh" and cap.is_file():
                _add_mesh_geom(
                    spec, exo_frame, cap, mesh_names, _R_CAD_TO_PELVIS, np.zeros(3),
                    _RGBA["hipmod"], geom_group, f"exo_mesh_hipcap_{side}",
                )

    # ---- per side: exo flexion hinge -> thigh link -> pin-in-slot slide -> roll -> cuff --
    #
    # The exo is placed in its *own* CAD assembly pose: one global rigid transform puts the
    # waist frame behind the body and leaves the hip modules where the CAD puts them, about
    # 114 mm outboard of and 70 mm ahead of the human hip joint.  That is what the machine
    # looks like in reality - back frame against the back, hip modules beside the hips,
    # thigh link down the *outside* of the thigh - and it is why the exo must not be
    # anchored to the human hip centre.  Per side there is therefore
    #
    #   exo_hip_flexion  hinge, driven, about the CAD hip axis (zero assistance here); it
    #                    turns the crank the motor output is bolted to
    #   exo_crank_pin    hinge, passive, where the long link hangs off that crank
    #   exo_slot         prismatic, the pin travelling its short travel in the long slot
    #   exo_axial_roll   hinge, passive, about the link's own axis
    #
    # Four exo DOF against the three rows of the cuff strap.  The crank pin is co-axial with
    # the motor, so it is a slop joint: the motor hinge does the visible rotation and the pin
    # absorbs only what the strap cannot follow.
    for side in SIDES:
        femur = _get_body(spec, f"femur_{side}")
        tibia = _get_body(spec, f"tibia_{side}")

        hip = np.asarray(femur.pos, dtype=float)           # HJC in the pelvis frame
        thigh = _unit(np.asarray(tibia.pos, dtype=float))  # HJC -> knee, femur frame
        lateral = np.array([0.0, 0.0, float(np.sign(hip[2]))])  # outward hip axis
        hip_cad = np.asarray(landmarks[side]["hip_point_mm"], dtype=float) * 1e-3
        link_cad = _unit(np.asarray(landmarks[side]["link_axis_unit"], dtype=float))
        axis_cad = _unit(np.asarray(landmarks[side]["hip_axis_mm"], dtype=float))
        hip_to_cuff = float(landmarks[side]["hip_to_cuff_mm"]) * 1e-3

        # Only the global CAD->pelvis rotation is applied - no per-side twist, so the
        # assembly keeps the shape it was designed with.
        exo_axis = _R_CAD_TO_PELVIS @ hip_cad            # exo hinge, exo-frame coords
        exo_lateral = _R_CAD_TO_PELVIS @ axis_cad        # exo hinge axis, pelvis coords
        # the abduction hinge turns about the fore-aft axis: the CAD +X is anterior, and the
        # sign is mirrored per side so a positive joint value always swings the leg outward
        exo_forward = _ABDUCTION_SIGN[side] * (_R_CAD_TO_PELVIS @ np.array([1.0, 0.0, 0.0]))
        # The CAD link direction carries a 12.5 deg medial tilt: the link leaves the hip
        # module (114 mm outboard) and converges on the thigh surface at the cuff, which is
        # how the machine is worn.  Keeping the tilt is what lands the cuff on the leg; the
        # slot takes up the few millimetres of length this costs.
        link_dir = _unit(_R_CAD_TO_PELVIS @ link_cad)     # slot / roll axis
        cuff_local = link_dir * hip_to_cuff              # CAD cuff centre, link coords
        # a CAD point p maps to exo_axis + pin_local + R @ (p - hip_cad) inside ``link``;
        # the cuff body sits hip_to_cuff down the link from the hip, so its anchor is
        # hip_cad + R^T @ cuff_local
        cuff_anchor = hip_cad + _R_CAD_TO_PELVIS.T @ cuff_local

        # The hip module's bent rod is bolted to the motor output, so it turns with the hip
        # hinge.  The long link does not hinge at the hip: it hangs off the rod's pin, 44 mm
        # out from the hip axis, which is the pivot its strap geometry is dimensioned around.
        pin_cad = np.asarray(_CRANK_PIN_MM[side], dtype=float) * 1e-3
        pin_local = _R_CAD_TO_PELVIS @ (pin_cad - hip_cad)   # pin, relative to the hip
        crank = exo_frame.add_body(name=f"exo_crank_{side}", pos=exo_axis, quat=quat_id)
        link = crank.add_body(name=f"exo_link_{side}", pos=pin_local, quat=quat_id)
        if locked:
            # pure inertial load: the chain is welded rigid, no slot carrier, no DOF
            slot = link
        else:
            # MjSpec joint anchors are child-local and do NOT move the body origin, so the
            # slot carrier itself is created at the CAD cuff point and its slide joint is
            # anchored at its own origin.
            slot = link.add_body(
                name=f"exo_slot_{side}",
                pos=np.asarray(cuff_local - pin_local, dtype=np.float64), quat=quat_id)
            # the slot carrier is a bare kinematic body; MuJoCo needs a non-zero mass on a
            # moving body, so it carries the (neglected) pin-and-slot hardware explicitly
            slot.mass = slot_mass
            slot.ipos = np.zeros(3)
            slot.inertia = np.full(3, slot_mass * 1e-4)
            slot.iquat = np.array([1.0, 0.0, 0.0, 0.0])
        cuff = slot.add_body(name=f"exo_cuff_{side}", pos=np.zeros(3), quat=quat_id)

        crank_grp = groups[f"hipmod_{side}"]
        cmass, ccom, cinertia, _ = _combine_groups(
            [(float(crank_grp["mass_kg"]), np.asarray(crank_grp["com_m"], float),
              np.asarray(crank_grp["inertia_kgm2"], float), np.eye(3))],
            _R_CAD_TO_PELVIS,
        )
        cmoments, ciquat = _principal_inertia(cinertia, np.eye(3))
        crank.mass = cmass
        crank.ipos = np.asarray(ccom - exo_axis, dtype=np.float64)
        crank.inertia = np.asarray(cmoments, dtype=np.float64)
        crank.iquat = np.asarray(ciquat, dtype=np.float64)
        if add_visuals:
            # The drum is the motor output: it sits on the hip axis and turns with the hip
            # hinge.  Drawn on the crank body at its CAD anchor it stays concentric inside
            # the housing, and its clevis lugs never leave the long rod's root.
            _group_visual(
                spec, crank, f"crank_{side}", groups[f"hipmod_{side}"],
                _R_CAD_TO_PELVIS, hip_cad, _RGBA["crank"], geom_group,
                mesh_names, mesh_dir, visual_style,
            )

        bodies = {
            f"thigh_{side}": (link, _RGBA["thigh"], pin_cad),
            f"cuff_{side}": (cuff, _RGBA["cuff"], cuff_anchor),
        }
        joint_names: list[str] = []
        weld_names: list[str] = []

        for name, (body, rgba, anchor) in bodies.items():
            grp = groups[name]
            origin = np.zeros(3) if name.startswith("thigh") else cuff_local - pin_local
            mass, com, inertia, _ = _combine_groups(
                [(float(grp["mass_kg"]), np.asarray(grp["com_m"], float),
                  np.asarray(grp["inertia_kgm2"], float), np.eye(3))],
                _R_CAD_TO_PELVIS,
            )
            moments, iquat = _principal_inertia(inertia, np.eye(3))
            body.mass = mass
            body.ipos = np.asarray(com - exo_axis - pin_local - origin, dtype=np.float64)
            body.inertia = np.asarray(moments, dtype=np.float64)
            body.iquat = np.asarray(iquat, dtype=np.float64)
            if add_visuals:
                _group_visual(
                    spec, body, name, grp, _R_CAD_TO_PELVIS, anchor, rgba, geom_group,
                    mesh_names, mesh_dir, visual_style,
                )

        if not locked:
            # declared first, so its anchor and axis stay in the pelvis frame: the hip
            # module swings with the leg instead of levering the strap off the thigh
            jab = crank.add_joint(
                name=f"exo_hip_abduction_{side}",
                type=mujoco.mjtJoint.mjJNT_HINGE,
                pos=np.zeros(3),
                axis=exo_forward,
                range=np.asarray(_HIP_ABDUCTION_RANGE, dtype=np.float64),
                limited=True,
                damping=float(damping),
                armature=float(armature),
                frictionloss=0.0,
            )
            j1 = crank.add_joint(
                name=f"exo_hip_flexion_{side}",
                type=mujoco.mjtJoint.mjJNT_HINGE,
                pos=np.zeros(3),     # MjSpec joint anchors are body-local
                axis=exo_lateral,
                range=np.asarray(flexion_range, dtype=np.float64),
                limited=True,
                damping=float(damping),
                armature=float(armature),
                frictionloss=0.0,
            )
            # the long link hangs off the crank's pin: a real clevis hinge which the thigh
            # strap drags around, so it turns as the leg swings.  The spring only picks the
            # assembly branch, so the turn stays visible instead of wandering.
            j1b = link.add_joint(
                name=f"exo_crank_pin_{side}",
                type=mujoco.mjtJoint.mjJNT_HINGE,
                pos=np.zeros(3),
                axis=exo_lateral,
                range=np.asarray(_CRANK_PIN_RANGE, dtype=np.float64),
                limited=True,
                stiffness=float(_CRANK_PIN_STIFFNESS),
                springref=0.0,
                damping=max(2.0, float(damping)),
                armature=float(armature),
                frictionloss=float(_CRANK_PIN_FRICTIONLOSS),
            )
            j2 = slot.add_joint(
                name=f"exo_slot_{side}",
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                pos=np.zeros(3),     # child-local anchor = the CAD cuff point
                axis=link_dir,
                range=np.asarray(slot_range, dtype=np.float64),
                limited=True,
                damping=float(damping),
                armature=float(armature),
                frictionloss=0.0,
            )
            j3 = cuff.add_joint(
                name=f"exo_axial_roll_{side}",
                type=mujoco.mjtJoint.mjJNT_HINGE,
                pos=np.zeros(3),
                axis=link_dir,
                range=np.asarray(roll_range, dtype=np.float64),
                limited=True,
                # this hinge rotates the cuff about the link axis, so it carries no strap
                # load and nothing drives it; with the shared 0.05 damping it free-spun to
                # its stop (the visible wobble), so it is damped hard to hold still
                damping=max(2.0, float(damping)),
                armature=float(armature),
                frictionloss=0.0,
            )
            joint_names += [jab.name, j1.name, j1b.name, j2.name, j3.name]
            added_dof += 5

            if weld:
                # the strap sits on the thigh surface, one thigh radius outboard of the
                # femur axis, and spans only the few millimetres to the cuff - the cuff
                # lands on the thigh because link_dir keeps its CAD medial tilt
                attach = hip_to_cuff * thigh + thigh_radius * lateral
                attach_sites[side] = femur.add_site(name=f"exo_attach_{side}", pos=attach)
                # The cuff site sits ON the roll axis.  Offsetting it one thigh radius to
                # give the strap a third translating DOF was tried and rejected: correcting
                # tens of millimetres of gap that way needs >1.2 rad of roll, which pins the
                # joint at its stop and visibly spins the cuff.  The residual strap gap is
                # reported honestly instead of being papered over.
                cuff.add_site(name=f"exo_cuff_site_{side}", pos=np.zeros(3))
                eq = spec.add_equality(name=f"exo_weld_{side}")
                # A thigh strap presses the cuff *against* the thigh but lets it rotate, so
                # the coupling is a point (3 rows), not a weld (6).  A weld over-constrains
                # this chain: the cuff has only 3 kinematic DOF, so the 3 extra orientation
                # rows cannot be satisfied, which drove the joints into their travel stops
                # and tore the cuff up to 93 mm off the thigh.  mjEQ_CONNECT with site
                # objects uses the site origins as the anchors, so eq.data is unused.
                eq.type = mujoco.mjtEq.mjEQ_CONNECT
                eq.objtype = mujoco.mjtObj.mjOBJ_SITE
                eq.name1 = f"exo_attach_{side}"
                eq.name2 = f"exo_cuff_site_{side}"
                eq.data = np.zeros(11)
                eq.solref = np.asarray(weld_solref, dtype=np.float64)
                eq.solimp = np.asarray(weld_solimp, dtype=np.float64)
                weld_names.append(eq.name)

        frames[side] = {
            "hip_pelvis": hip,
            "thigh_dir": thigh,
            "lateral": lateral,
            "exo_axis": exo_axis,
            "exo_lateral": exo_lateral,
            "link_dir": link_dir,
            "cuff_local": cuff_local,
            "hip_to_cuff": hip_to_cuff,
            "joint_names": joint_names,
            "weld_names": weld_names,
        }
        if not locked:
            # Neutral exo pose: the flexion hinge is pre-rotated until the link points along
            # the human thigh, and the slot takes up whatever length is left over.  The
            # coordinate descent in _refine_cuff_gap then closes the weld.
            flex0 = float(np.arctan2(float(np.cross(link_dir, thigh) @ exo_lateral),
                                     float(link_dir @ thigh)))
            seed_map[side] = {
                "joints": list(joint_names),
                "values": {j1.name: flex0},
            }


    if added_dof:
        _seed_keyframe_exo_dofs(spec, added_dof, seed_map, attach_sites)

    return {
        "locked": bool(locked),
        "weld": bool(weld and not locked),
        "added_dof": added_dof,
        "bodies": ["exo_frame"]
        + [f"exo_crank_{s}" for s in SIDES]
        + [f"exo_link_{s}" for s in SIDES]
        + [f"exo_slot_{s}" for s in SIDES if not locked]
        + [f"exo_cuff_{s}" for s in SIDES],
        "joints": [n for side in SIDES for n in frames[side]["joint_names"]],
        "welds": [n for side in SIDES for n in frames[side]["weld_names"]],
        "frames": frames,
        "exo_mass_kg": float(sum(groups[n]["mass_kg"] for n in groups)),
    }


_MUSCLE_WRAP_RADIUS = 0.012
"""Radius pinned on the muscle wrap cylinders.

MuJoCo routes a muscle path around the surface of its wrap geom, so the radius feeds
back into the muscle lengths and therefore the dynamics.  12 mm is the value the walking
policy was trained and verified with; these geoms are hidden, so it is never seen.
"""


def place_strap_on_muscle(model: mujoco.MjModel, swing_deg: float = 0.0,
                          out_mm: float = 0.0, align_front: bool = True,
                          roll_deg: float = 0.0, lift_deg: float = 0.0,
                          standoff_mm: float = 0.0, turn_deg: float = 0.0,
                          hinge_deg: float = 0.0) -> float:
    """Set the thigh rod and strap in the pose they are worn in, clear of the flesh.

    The strap is drawn in the CAD's assembly pose, hanging on the *side* of the thigh.  The
    rod is bolted to the drum through a clevis, so the only motion that keeps its root in
    the drum's lugs is a turn about the clevis bolt itself; ``hinge_deg`` turns the whole
    link - rod and strap together - about that bolt, lifting the buckle off the muscle and
    dragging it forward until the rod rides on the muscle's surface.  The other turns
    (roll about the leg, swing about the crank pin, lift about the pelvis lateral, standoff
    and turn about the pin) let a walk be measured on other poses; they skew the clevis, so
    the shipped build only uses the hinge.  The pose it is measured on is the static
    (standing) one, so the strap is placed where the machine is worn; the strap is left
    unfastened from the rod for now.

    Returns the angle that was taken out of the strap, in degrees.
    """
    data = mujoco.MjData(model)
    if model.nkey:
        data.qpos[:] = model.key_qpos[0]
    mujoco.mj_kinematics(model, data)
    pelvis_id = model.body("pelvis").id
    pelvis_mat = data.xmat[pelvis_id].reshape(3, 3)
    anterior = pelvis_mat @ np.array([1.0, 0.0, 0.0])
    lateral = pelvis_mat @ np.array([0.0, 0.0, 1.0])
    taken = 0.0
    for side in SIDES:
        hip = data.xpos[model.body(f"femur_{side}").id].copy()
        knee = data.xpos[model.body(f"tibia_{side}").id].copy()
        thigh = _unit(knee - hip)
        pin = data.xpos[model.body(f"exo_link_{side}").id].copy()
        if roll_deg:
            # the rod hangs off the drum on the outside of the thigh; roll it about the
            # leg's centreline, through the crank pin, until the long link runs down the
            # middle of the thigh's front face.  The pin point is on the axis, so the rod's
            # root stays in the drum's lugs.
            spin = _rodrigues(thigh, -np.deg2rad(roll_deg))
            for gid in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
                if not name.startswith(f"exo_mesh_thigh_{side}_"):
                    continue
                body = model.geom_bodyid[gid]
                body_mat = data.xmat[body].reshape(3, 3)
                world = pin + spin @ (data.geom_xpos[gid] - pin)
                model.geom_pos[gid] = body_mat.T @ (world - data.xpos[body])
                model.geom_quat[gid] = _mat_to_quat(
                    body_mat.T @ spin @ data.geom_xmat[gid].reshape(3, 3))
            mujoco.mj_kinematics(model, data)
        if swing_deg:
            # the rod's lower end is turned in towards the exo's centre line about the
            # clevis point, in the frontal plane.  The strap ring is a separate part clamped
            # on the thigh (unfastened from the rod), so only the rod turns.
            inward = lateral * (1.0 if (data.xpos[pelvis_id] - pin) @ lateral > 0.0 else -1.0)
            arm = data.xpos[model.body(f"exo_cuff_{side}").id] - pin
            # positive swing_deg always turns the far end inwards, negative outwards
            angle = np.deg2rad(abs(swing_deg))
            if (_rodrigues(anterior, angle) @ arm) @ inward < arm @ inward:
                angle = -angle
            if swing_deg < 0:
                angle = -angle
            spin = _rodrigues(anterior, angle)
            for gid in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
                if not name.startswith(f"exo_mesh_thigh_{side}_"):
                    continue
                body = model.geom_bodyid[gid]
                body_mat = data.xmat[body].reshape(3, 3)
                world = pin + spin @ (data.geom_xpos[gid] - pin)
                model.geom_pos[gid] = body_mat.T @ (world - data.xpos[body])
                model.geom_quat[gid] = _mat_to_quat(
                    body_mat.T @ spin @ data.geom_xmat[gid].reshape(3, 3))
            mujoco.mj_kinematics(model, data)
        if hinge_deg:
            # the rod hangs in the fork's clevis and the only turn that leaves its root in
            # the lugs is about the bolt the clevis swings on.  The strap ring is a separate
            # part clamped on the thigh (unfastened from the rod), so only the rod turns:
            # the ring stays on the leg.  Positive hinge_deg swings the rod's far end
            # forward, onto the middle of the thigh's front face.
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"exo_crank_pin_{side}")
            bolt = _unit(data.xmat[model.jnt_bodyid[jid]].reshape(3, 3) @ model.jnt_axis[jid])
            arm = data.xpos[model.body(f"exo_cuff_{side}").id] - pin
            # pick the bolt's sense once per side so that a positive hinge_deg always swings
            # the far end forward and a negative one tucks it back
            angle = np.deg2rad(abs(hinge_deg))
            if (_rodrigues(bolt, angle) @ arm) @ anterior < arm @ anterior:
                angle = -angle
            if hinge_deg < 0:
                angle = -angle
            spin = _rodrigues(bolt, angle)
            for gid in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
                if not name.startswith(f"exo_mesh_thigh_{side}_"):
                    continue
                body = model.geom_bodyid[gid]
                body_mat = data.xmat[body].reshape(3, 3)
                world = pin + spin @ (data.geom_xpos[gid] - pin)
                model.geom_pos[gid] = body_mat.T @ (world - data.xpos[body])
                model.geom_quat[gid] = _mat_to_quat(
                    body_mat.T @ spin @ data.geom_xmat[gid].reshape(3, 3))
            mujoco.mj_kinematics(model, data)
        if lift_deg:
            # the strap buckle stands off the thigh in the CAD; with the rod swung in to the
            # middle of the thigh its lower end comes close to the knee, so lift the whole
            # link - rod and strap together, still turning about the hinge - until it clears
            # the bone and rides in front of it
            arm = data.xpos[model.body(f"exo_cuff_{side}").id] - pin
            angle = np.deg2rad(lift_deg)
            spin = _rodrigues(lateral, angle)
            if (spin @ arm) @ anterior < arm @ anterior:
                spin = _rodrigues(lateral, -angle)
            for gid in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
                if not name.startswith((f"exo_mesh_thigh_{side}_",
                                        f"exo_mesh_cuff_{side}_")):
                    continue
                body = model.geom_bodyid[gid]
                body_mat = data.xmat[body].reshape(3, 3)
                world = pin + spin @ (data.geom_xpos[gid] - pin)
                model.geom_pos[gid] = body_mat.T @ (world - data.xpos[body])
                model.geom_quat[gid] = _mat_to_quat(
                    body_mat.T @ spin @ data.geom_xmat[gid].reshape(3, 3))
            mujoco.mj_kinematics(model, data)
        if standoff_mm or turn_deg:
            # the buckle sinks into the thigh where the CAD puts it, so swing the whole
            # link - rod and strap, bolted together - out about the crank pin until it
            # stands clear of the leg; then turn the pair about the line from the pin
            # through the strap until the plate faces ahead instead of edge-on.  Both
            # turns pass through the pin, so the rod's root stays in the drum's lugs and
            # the pair keeps its CAD relation.
            cloud = []
            for gid in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
                if not name.startswith(f"exo_mesh_cuff_{side}_"):
                    continue
                mesh = int(model.geom_dataid[gid])
                first, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
                cloud.append(
                    data.geom_xpos[gid]
                    + model.mesh_vert[first:first + count] @ data.geom_xmat[gid].reshape(3, 3).T)
            if cloud:
                centre = np.vstack(cloud).mean(0) - pin
            else:
                centre = data.xpos[model.body(f"exo_cuff_{side}").id] - pin
            spin = np.eye(3)
            if standoff_mm:
                radial = _unit((pin + centre - hip) - ((pin + centre - hip) @ thigh) * thigh)
                angle = standoff_mm * 1e-3 / max(np.linalg.norm(centre), 1e-6)
                axis = _unit(np.cross(centre, radial))
                out = _rodrigues(axis, angle)
                if (out @ centre) @ radial < centre @ radial:
                    out = _rodrigues(axis, -angle)
                spin = out @ spin
            if turn_deg:
                spin = _rodrigues(_unit(centre), np.deg2rad(turn_deg)) @ spin
            for gid in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
                if not name.startswith((f"exo_mesh_thigh_{side}_",
                                        f"exo_mesh_cuff_{side}_")):
                    continue
                body = model.geom_bodyid[gid]
                body_mat = data.xmat[body].reshape(3, 3)
                world = pin + spin @ (data.geom_xpos[gid] - pin)
                model.geom_pos[gid] = body_mat.T @ (world - data.xpos[body])
                model.geom_quat[gid] = _mat_to_quat(
                    body_mat.T @ spin @ data.geom_xmat[gid].reshape(3, 3))
            mujoco.mj_kinematics(model, data)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"exo_slot_{side}")
        slide = data.qpos[model.jnt_qposadr[jid]] if jid >= 0 else 0.0
        if slide:
            # the strap hangs on the pin-and-slot carriage, which slides along the link to
            # take up the CAD-to-thigh length error; at the worn pose that slide walks the
            # strap off the rod's bolt holes.  Put the strap back on the rod's CAD station
            # so the two stay bolted together.
            axis = data.xmat[model.jnt_bodyid[jid]].reshape(3, 3) @ model.jnt_axis[jid]
            for gid in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
                if not name.startswith(f"exo_mesh_cuff_{side}_"):
                    continue
                body = model.geom_bodyid[gid]
                body_mat = data.xmat[body].reshape(3, 3)
                world = data.geom_xpos[gid] - slide * axis
                model.geom_pos[gid] = body_mat.T @ (world - data.xpos[body])
            mujoco.mj_kinematics(model, data)
        ahead = _unit(anterior - (anterior @ thigh) * thigh)
        side_axis = np.cross(thigh, ahead)
        points = []
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if not name.startswith(f"exo_mesh_cuff_{side}_"):
                continue
            mesh = int(model.geom_dataid[gid])
            first, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            points.append(
                data.geom_xpos[gid]
                + model.mesh_vert[first:first + count] @ data.geom_xmat[gid].reshape(3, 3).T
            )
        if not points:
            continue
        radial = np.vstack(points) - hip
        radial -= (radial @ thigh)[:, None] * thigh
        radii = np.linalg.norm(radial, axis=1)
        direction = radial[radii > 0.88 * radii.max()].mean(0)
        direction -= (direction @ thigh) * thigh
        direction /= np.linalg.norm(direction)
        azimuth = float(np.arctan2(direction @ side_axis, direction @ ahead))
        # the strap is pushed straight out of the thigh, along the line from the leg's
        # centre to the strap's own centre
        outward = radial.mean(0)
        outward -= (outward @ thigh) * thigh
        outward /= np.linalg.norm(outward)
        spin = _rodrigues(thigh, -azimuth) if align_front else np.eye(3)
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if not name.startswith(f"exo_mesh_cuff_{side}_"):
                continue
            body = model.geom_bodyid[gid]
            body_mat = data.xmat[body].reshape(3, 3)
            world = hip + spin @ (data.geom_xpos[gid] - hip)
            world = world + out_mm * 1e-3 * (spin @ outward)
            model.geom_pos[gid] = body_mat.T @ (world - data.xpos[body])
            model.geom_quat[gid] = _mat_to_quat(
                body_mat.T @ spin @ data.geom_xmat[gid].reshape(3, 3))
        taken += np.degrees(azimuth)
    mujoco.mj_forward(model, data)
    return float(taken / max(len(SIDES), 1))


def seat_cuff_on_rod(model: mujoco.MjModel, sides: tuple[str, ...] = SIDES) -> float:
    """Slide each thigh strap down the leg until it meets the rod's distal buckle.

    The rod ends in a buckle that clamps the strap, but the strap is drawn at its CAD
    station, ~8 cm short of where the rod ends once the rod is turned onto the muscle.  The
    band is a separate part on the cuff body, so it can be slid along the leg without
    touching the weld: it lands on the buckle and still wraps the thigh.

    Returns the mean slide distance in millimetres.
    """
    data = mujoco.MjData(model)
    if model.nkey:
        # seated on the pose the machine is reset into, so the buckle reads right there
        data.qpos[:] = model.key_qpos[min(_RESET_KEY, model.nkey - 1)]
    mujoco.mj_kinematics(model, data)
    model_pose = data
    moved = 0.0
    for side in sides:
        hip = model_pose.xpos[model.body(f"femur_{side}").id].copy()
        knee = model_pose.xpos[model.body(f"tibia_{side}").id].copy()
        thigh = _unit(knee - hip)

        def _verts(prefix: str) -> np.ndarray:
            out = []
            for gid in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
                if not name.startswith(prefix):
                    continue
                lo = int(model.mesh_vertadr[int(model.geom_dataid[gid])])
                n = int(model.mesh_vertnum[int(model.geom_dataid[gid])])
                v = np.asarray(model.mesh_vert[lo:lo + n], float)
                out.append(model_pose.geom_xpos[gid] + (
                    model_pose.geom_xmat[gid].reshape(3, 3) @ v.T).T)
            return np.vstack(out)

        rod = _verts(f"exo_mesh_thigh_{side}_")
        band = _verts(f"exo_mesh_cuff_{side}_")
        tip = rod[np.argmax((rod - hip) @ thigh)]
        shift = tip - band[int(np.argmin(np.linalg.norm(band - tip, axis=1)))]
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if not name.startswith(f"exo_mesh_cuff_{side}_"):
                continue
            body = model.geom_bodyid[gid]
            body_mat = model_pose.xmat[body].reshape(3, 3)
            world = model_pose.geom_xpos[gid] + shift
            model.geom_pos[gid] = body_mat.T @ (world - model_pose.xpos[body])
        moved += float(np.linalg.norm(shift)) * 1000.0
    mujoco.mj_forward(model, data)
    return moved / max(len(sides), 1)


def pitch_exo(model: mujoco.MjModel, pitch_deg: float = 0.0,
              forward_mm: float = 0.0,
              groups: tuple[str, ...] = ("exo_mesh_",)) -> None:
    """Tip the whole exoskeleton about the hip axis so the back plate lies on the lumbar spine.

    The frame is drawn in the CAD's assembly pose, which leaves the back plate leaning off
    the back by about 15 degrees, its lower edge standing out behind the body.  Pitching it
    - plate, arms, motor housings, drums, thigh links and straps - about the lateral axis
    through the hips lays the plate onto the back, and ``forward_mm`` then slides the whole
    device towards the body until the pad rests on the spine.

    The whole device is turned and slid as one rigid unit, so every CAD fit survives: the
    drums stay concentric in their housings, the arms stay on the drums' axles and the link
    roots stay in the drums' lugs.  The price is that the links, hanging some 350 mm below
    the hips, come forward with it; that is taken out afterwards about the clevis bolt by
    ``place_strap_on_muscle(..., hinge_deg=...)``, which turns each link about its own pin
    axis and therefore keeps the root seated.  Call this after that function.
    """
    if not pitch_deg and not forward_mm:
        return
    data = mujoco.MjData(model)
    if model.nkey:
        data.qpos[:] = model.key_qpos[0]
    mujoco.mj_kinematics(model, data)
    pelvis_mat = data.xmat[model.body("pelvis").id].reshape(3, 3)
    axis = pelvis_mat @ np.array([0.0, 0.0, 1.0])
    hip = 0.5 * (data.xpos[model.body("femur_r").id] + data.xpos[model.body("femur_l").id])
    spin = _rodrigues(axis, np.deg2rad(pitch_deg))
    shift = pelvis_mat @ np.array([forward_mm * 1e-3, 0.0, 0.0])
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if not name.startswith(groups):
            continue
        body = model.geom_bodyid[gid]
        body_mat = data.xmat[body].reshape(3, 3)
        world = hip + spin @ (data.geom_xpos[gid] - hip) + shift
        model.geom_pos[gid] = body_mat.T @ (world - data.xpos[body])
        model.geom_quat[gid] = _mat_to_quat(
            body_mat.T @ spin @ data.geom_xmat[gid].reshape(3, 3))
    mujoco.mj_kinematics(model, data)


def clock_exo_endcap(model: mujoco.MjModel, extra_deg: float = 0.0, key: int = 0,
                     sides: tuple[str, ...] = SIDES) -> dict[str, float]:
    """Turn each motor module's end cover about the drum axis so its relief sits under the fork.

    The CAD assembles the chain on a flexed hip, so the cranks never sit at flexion 0 in the sim:
    the walking keyframes hold them near -60 deg / +60 deg.  The cover's relief, though, is drawn
    where the fork's ear sits at flexion 0 - measured on the meshes, the ear rests on the relief
    floor 0.27 mm there and walks off to 12.1 mm (right) / 9.2 mm (left) at the reset flexion,
    which is why the fork read as "out of the notch".  Re-clocking the cover by ``key``'s own
    flexion, about that side's own flexion axis through the drum centre, recovers the CAD fit; the
    two walking keyframes the env can reset from are 4 deg apart, so both stay well inside the
    relief, while the standing keyframes land on its edge.

    Only ``exo_mesh_hipcap_{side}`` moves - the chamfered cover and nothing else.  The rest of the
    module (the finned motor cylinder and the mounting arm, ``exo_mesh_hipmod_{side}_0``) sits on a
    pad welded to the waist frame and must stay exactly where the bracket stays, so the housing
    keeps its shipped pose whatever angle is applied here.  The cover's own seat is an annular lip
    at R 36-41 mm spanning ~270 deg of azimuth and concentric with the hinge to within 1.3 mm, so
    turning it costs little but not nothing: at the 61 deg reset clock the lip's clearance to the
    module goes from 0.69 mm median to 1.26 mm (2.1 mm at the 90th percentile) over its 12.6k
    seated vertices.  The cover geom is visual-only and its body inertia comes from the params
    file, so this changes neither the mesh nor the dynamics.

    Returns the angle applied per side, in degrees.
    """
    if key >= model.nkey:
        return {}
    hinges = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
              for j in range(model.njnt)}
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[key]
    mujoco.mj_kinematics(model, data)
    turned: dict[str, float] = {}
    for side in sides:
        name_j = f"exo_hip_flexion_{side}"
        if name_j not in hinges:
            # the locked variant welds the chain at joint zero, which is where the CAD already
            # nests the fork on the cover, so its cap needs no clock
            continue
        joint = model.joint(name_j)
        angle = float(model.key_qpos[key][int(joint.qposadr[0])]) + np.deg2rad(extra_deg)
        if not angle:
            continue
        carrier = model.jnt_bodyid[joint.id]
        axis = data.xmat[carrier].reshape(3, 3) @ model.jnt_axis[joint.id]
        axis = _unit(axis)
        centre = data.xpos[carrier]
        spin = _rodrigues(axis, angle)
        want = f"exo_mesh_hipcap_{side}"
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name != want:
                continue
            body = model.geom_bodyid[gid]
            body_mat = data.xmat[body].reshape(3, 3)
            world = centre + spin @ (data.geom_xpos[gid] - centre)
            model.geom_pos[gid] = body_mat.T @ (world - data.xpos[body])
            model.geom_quat[gid] = _mat_to_quat(
                body_mat.T @ spin @ data.geom_xmat[gid].reshape(3, 3))
        turned[side] = float(np.degrees(angle))
    mujoco.mj_kinematics(model, data)
    return turned


def style_muscle_geoms(model: mujoco.MjModel, muscle_width: float = 0.012) -> int:
    """Show the muscles as tissue, or hide them for the bare-skeleton look.

    MyoSuite represents each muscle twice: as a wrap geom (a cylinder along the belly, a
    sphere at a joint) and as a spatial tendon.  The tendon is what shows.  With
    ``muscle_width > 0`` the tendons are drawn at that width in their default muscle
    red - at 12 mm (the wrap radius the walking policy was verified with) they read as
    muscle bellies riding the bones instead of a tangle of 5 mm blood vessels - and the
    wrap cylinders stay fully transparent: an invisible wrap still routes its muscle
    path and moves the dynamics, so the pinned 12 mm radius and transparent colour are
    what keep the verified dynamics while the tissue is drawn from the tendon paths
    alone.  With ``muscle_width == 0`` the tendons are hidden too.

    Returns the number of wrap geoms hidden.
    """
    model.tendon_width[:] = float(muscle_width)
    n = 0
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if "_wrap_" not in name:
            continue
        geom_type = model.geom_type[gid]
        if geom_type in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            model.geom_size[gid][0] = _MUSCLE_WRAP_RADIUS
        elif geom_type != mujoco.mjtGeom.mjGEOM_SPHERE:
            continue
        model.geom_rgba[gid] = np.array([0.0, 0.0, 0.0, 0.0])
        n += 1
    return n


def _pad_keyframes(spec: mujoco.MjSpec, added_dof: int) -> None:
    """Extend every keyframe by ``added_dof`` zeros (new hinges, at rest)."""
    for key in spec.keys:
        if key.qpos is not None:
            key.qpos = np.concatenate([np.asarray(key.qpos, dtype=float), np.zeros(added_dof)])
        if key.qvel is not None:
            key.qvel = np.concatenate([np.asarray(key.qvel, dtype=float), np.zeros(added_dof)])


def _seed_keyframe_exo_dofs(
    spec: mujoco.MjSpec, added_dof: int, seed_map: dict[str, dict[str, float]],
    attach_sites: dict[str, Any] | None = None,
) -> None:
    """Pad the keyframes and place the exo hinges so the cuff welds start closed.

    Plain zero padding leaves the exoskeleton in its CAD (spec-neutral) pose while the
    keyframe has the thigh flexed, which opens every cuff weld by up to 80 mm the moment
    the episode starts.  The exo flexion hinge turns about the same axis through the same
    point as the human hip flexion and the exo roll hinge about the same axis as the human
    hip rotation, so seeding them with the human angles already closes the loop to within
    the hip's adduction angle (~2 cm).  A short coordinate descent then lets the passive
    roll hinge absorb the rest, so the reset pose starts with the welds closed instead of
    kicking the walker on the first control step.
    """
    _pad_keyframes(spec, added_dof)
    if not seed_map:
        return

    model = spec.compile()
    data = mujoco.MjData(model)

    def _qadr(name: str) -> int:
        return int(model.joint(name).qposadr[0])

    def _dadr(name: str) -> int:
        return int(model.joint(name).dofadr[0])

    def _gap(qpos: np.ndarray, side: str) -> float:
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        cuff = model.site(f"exo_cuff_site_{side}").id
        attach = model.site(f"exo_attach_{side}").id
        return float(np.linalg.norm(data.site_xpos[cuff] - data.site_xpos[attach]))

    def _clamp(qpos: np.ndarray, names: list[str]) -> None:
        for name in names:
            lo, hi = model.jnt_range[model.joint(name).id]
            adr = _qadr(name)
            if lo < hi:
                qpos[adr] = float(np.clip(qpos[adr], lo + 1e-6, hi - 1e-6))

    limits = {
        n: (model.jnt_range[model.joint(n).id], _qadr(n))
        for side in seed_map for n in seed_map[side]["joints"]
    }

    for key in spec.keys:
        if key.qpos is not None:
            qpos = np.array(key.qpos, dtype=float)
            # keyed by name, not position: joint_names is [abduction, flexion, crank pin,
            # slot, roll], so a positional list silently seeds the wrong hinge
            for side, seed in seed_map.items():
                for n, val in seed["values"].items():
                    qpos[_qadr(n)] = val
            key.qpos = qpos
        if key.qvel is not None:
            qvel = np.array(key.qvel, dtype=float)
            for side, seed in seed_map.items():
                for n in seed["joints"]:
                    qvel[_dadr(n)] = 0.0
            key.qvel = qvel

    def _fit_names(side: str, freeze_flexion: bool) -> list[str]:
        names = list(seed_map[side]["joints"])
        # abduction is the fork's seat on the motor's end cap: never spend it on cuff reach.
        # Once the crank/drum clock is seeded, flexion is frozen too and the pin absorbs it.
        drop = ("exo_hip_abduction", "exo_hip_flexion") if freeze_flexion else ("exo_hip_abduction",)
        return [n for n in names if not n.startswith(drop)]

    def _refit_keys(freeze_flexion: bool = False) -> None:
        for key in spec.keys:
            if key.qpos is None:
                continue
            qpos = np.array(key.qpos, dtype=float)
            for side in seed_map:
                fit = _fit_names(side, freeze_flexion)
                adrs = [_qadr(n) for n in fit]
                _refine_cuff_gap(qpos, fit, adrs, side, _gap, limits)
                _clamp(qpos, list(seed_map[side]["joints"]))
            key.qpos = qpos

    _refit_keys()

    if attach_sites and _CRANK_CLOCK_DEG:
        # clock the crank (and with it the drum's end cover, whose clock reads this same
        # flexion key) about the motor axis on the reset keyframe only; the pin then absorbs
        # the turn so the link and the thigh ring keep their pose.
        key = spec.keys[_RESET_KEY if _RESET_KEY < len(spec.keys) else 0]
        if key.qpos is not None:
            qpos = np.array(key.qpos, dtype=float)
            for side in seed_map:
                name = f"exo_hip_flexion_{side}"
                lo, hi = model.jnt_range[model.joint(name).id]
                adr = _qadr(name)
                shift = float(np.deg2rad(_CRANK_CLOCK_DEG * _CRANK_CLOCK_SIGN[side]))
                qpos[adr] = float(np.clip(qpos[adr] + shift, lo + 1e-6, hi - 1e-6))
            key.qpos = qpos
        _refit_keys(freeze_flexion=True)

    if attach_sites:
        # close the residual strap gap by moving the anchor onto the ring at the reset key
        model = spec.compile()
        data = mujoco.MjData(model)
        data.qpos[:] = model.key_qpos[_RESET_KEY if _RESET_KEY < model.nkey else 0]
        mujoco.mj_kinematics(model, data)
        for side, site in attach_sites.items():
            femur = model.body(f"femur_{side}").id
            cuff = model.site(f"exo_cuff_site_{side}").id
            local = np.asarray(data.xmat[femur]).reshape(3, 3).T @ (
                data.site_xpos[cuff] - data.xpos[femur])
            site.pos = local
        model = spec.compile()
        data = mujoco.MjData(model)
        _refit_keys(freeze_flexion=True)


def _refine_cuff_gap(
    qpos: np.ndarray,
    names: list[str],
    adrs: list[int],
    side: str,
    gap_fn: Any,
    limits: dict[str, tuple[Any, int]] | None = None,
    tol: float = 1e-4,
) -> float:
    """Coordinate descent on the exo DOF; returns the residual cuff gap.

    Candidates are clamped to the joint ranges: without that the descent happily runs the
    slot out to 0.36 m, which closes the weld geometrically but is not a real mechanism.
    """
    def _fit(adr: int, val: float) -> float:
        if limits is None:
            return val
        rng, _a = limits[names[adrs.index(adr)]]
        lo, hi = float(rng[0]), float(rng[1])
        if lo >= hi:
            return val
        return float(np.clip(val, lo + 1e-6, hi - 1e-6))

    best = gap_fn(qpos, side)
    step = 0.2
    for _ in range(60):
        moved = False
        for adr in adrs:
            base = qpos[adr]
            for cand in (_fit(adr, base + step), _fit(adr, base - step)):
                qpos[adr] = cand
                val = gap_fn(qpos, side)
                if val < best - 1e-12:
                    best, base, moved = val, cand, True
            qpos[adr] = base
        if not moved:
            step *= 0.5
            if step < tol:
                break
    return best
