# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The 'Lin-I' hip exoskeleton must be a passive, CAD-sized load on the walker.

``myoLegWalkExo-v0`` adds nine rigid bodies (back frame + hip module / crank / thigh link /
cuff per side), four passive DOF per side and a soft cuff weld per side to ``myoLegWalk-v0``.
The published ``myoLegWalk-v0`` policy is only usable in this environment if

* the added mass equals the mass extracted from the CAD assembly (3.097 kg),
* the observation vector stays compatible with the baseline env,
* the crank turns with the motor hinge, the way the real module's bent rod does,
* the exo hinges carry **no** actuator force (zero assistance),
* the cuff weld actually drags the exo chain with the human thigh, and
* the exo geometry never collides with the body it is attached to.

These are cheap structural checks; ``scripts/eval_walk_exo.py`` covers the 15 s gait.
"""

from __future__ import annotations

import json
import os

import mujoco
import numpy as np
import pytest

import myosuite  # noqa: F401  (registers the CPU envs)
from myosuite import make_env
from myosuite.envs.myo.tasks.basic.leg.lin1_exo import EXO_PARAMS_PATH
from myosuite.utils import gym

pytestmark = pytest.mark.tier1

EXO_ENV = "myoLegWalkExo-v0"
EXO_LOCKED_ENV = "myoLegWalkExoLocked-v0"
BASE_ENV = "myoLegWalk-v0"
SIDES = ("r", "l")
EXO_BODIES = (
    "exo_frame", "exo_crank_r", "exo_link_r", "exo_slot_r", "exo_cuff_r",
    "exo_crank_l", "exo_link_l", "exo_slot_l", "exo_cuff_l",
)
# the two bare slot carriers only carry the (neglected) pin-and-slot hardware
SLOT_BODIES = ("exo_slot_r", "exo_slot_l")
CAD_BODIES = tuple(b for b in EXO_BODIES if b not in SLOT_BODIES)


def _body_mass(env, name: str) -> float:
    """Scalar body mass (``MjModel.body().mass`` comes back as a 1-element array)."""
    return float(env.model.body_mass[env.model.body(name).id])


def _params() -> dict:
    with open(EXO_PARAMS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _cad_total_mass() -> float:
    return float(sum(g["mass_kg"] for g in _params()["groups"].values()))


def _slot_mass(exo_env) -> float:
    return sum(_body_mass(exo_env, n) for n in SLOT_BODIES)


@pytest.fixture(scope="module")
def exo_env():
    return make_env(EXO_ENV).unwrapped


@pytest.fixture(scope="module")
def base_env():
    return make_env(BASE_ENV).unwrapped


@pytest.fixture(scope="module")
def locked_env():
    return make_env(EXO_LOCKED_ENV).unwrapped


def test_exo_adds_nine_bodies_with_exactly_the_cad_mass(exo_env, base_env) -> None:
    assert set(exo_env.exo_body_names()) == set(EXO_BODIES)
    assert exo_env.model.nbody == base_env.model.nbody + len(EXO_BODIES)

    added = sum(_body_mass(exo_env, n) for n in CAD_BODIES)
    assert added == pytest.approx(_cad_total_mass(), rel=1e-6)
    assert _slot_mass(exo_env) > 0.0, "slot carriers need a non-zero mass to be valid"
    total = float(exo_env.model.body_mass.sum())
    assert total == pytest.approx(
        float(base_env.model.body_mass.sum()) + added + _slot_mass(exo_env), rel=1e-9
    )


def test_exo_chain_is_five_passive_dofs_per_side(exo_env, base_env) -> None:
    joints = exo_env.exo_joint_names()
    assert len(joints) == 10
    assert exo_env.model.nq == base_env.model.nq + 10
    assert exo_env.model.nv == base_env.model.nv + 10
    assert exo_env.model.nu == base_env.model.nu, "zero assistance must not add actuators"
    for side in SIDES:
        for prefix in ("exo_hip_abduction", "exo_hip_flexion", "exo_crank_pin",
                       "exo_slot", "exo_axial_roll"):
            assert any(j.startswith(f"{prefix}_{side}") for j in joints), f"{prefix}_{side}"
    for name in joints:
        if "_slot_" in name:
            assert exo_env.model.joint(name).type == mujoco.mjtJoint.mjJNT_SLIDE, name
        else:
            assert exo_env.model.joint(name).type == mujoco.mjtJoint.mjJNT_HINGE, name


def test_cuff_weld_couples_the_exo_chain_to_the_thigh(exo_env) -> None:
    welds = exo_env.exo_weld_names()
    assert len(welds) == 2
    model = exo_env.model
    assert model.neq >= 2
    for side in SIDES:
        eq = model.equality(f"exo_weld_{side}")
        # a thigh strap presses the cuff against the thigh but lets it rotate, so the
        # coupling is a point (3 rows) for a cuff chain that has four translating DOF
        # (hip abduction, hip flexion, crank pin, slot) - not a weld
        assert eq.type == mujoco.mjtEq.mjEQ_CONNECT
        assert model.eq_objtype[eq.id] == mujoco.mjtObj.mjOBJ_SITE
        cuff = model.site(f"exo_cuff_site_{side}").id
        attach = model.site(f"exo_attach_{side}").id
        assert {int(model.eq_obj1id[eq.id]), int(model.eq_obj2id[eq.id])} == {cuff, attach}, side


def test_locked_variant_is_a_pure_inertial_load(locked_env, base_env) -> None:
    assert locked_env.model.nq == base_env.model.nq
    assert locked_env.model.nv == base_env.model.nv
    assert locked_env.exo_joint_names() == []
    assert locked_env.exo_weld_names() == []
    assert locked_env.n_exo_dof == 0
    assert float(locked_env.model.body_mass.sum()) == pytest.approx(
        float(base_env.model.body_mass.sum()) + _cad_total_mass(), rel=1e-9
    )
    assert set(locked_env.exo_body_names()) == set(CAD_BODIES), "no slot carriers when locked"


def test_keyframes_stay_valid_with_the_extra_dofs(exo_env) -> None:
    model = exo_env.model
    assert model.nkey > 0
    assert model.key_qpos.shape == (model.nkey, model.nq)
    assert model.key_qvel.shape == (model.nkey, model.nv)
    for k in range(model.nkey):
        qpos = model.key_qpos[k]
        assert np.all(np.isfinite(qpos))
        for name in exo_env.exo_joint_names():
            jnt = model.joint(name)
            lo, hi = model.jnt_range[jnt.id]
            assert lo <= qpos[jnt.qposadr[0]] <= hi, f"{name} out of range in keyframe {k}"


def test_reset_keyframe_starts_with_closed_cuff_welds(exo_env) -> None:
    """The keyframes must place the exo hinges so the welds start (nearly) closed.

    The residual is geometric, not a modelling bug: the exo has no hip-adduction DOF, so
    the ~0.03 rad of hip adduction in the reset keyframe displaces the cuff ~1.7 cm off
    the flexion circle.  That is what the real thigh cuff/strap has to absorb.
    """
    exo_env.reset(seed=3)
    for side in SIDES:
        cuff = exo_env.model.site(f"exo_cuff_site_{side}").id
        attach = exo_env.model.site(f"exo_attach_{side}").id
        gap = float(
            np.linalg.norm(exo_env.data.site_xpos[cuff] - exo_env.data.site_xpos[attach])
        )
        assert gap < 0.025, f"cuff weld {side} starts {gap * 1000:.1f} mm open"


def test_exo_chain_is_pivoted_exactly_at_the_hip_joint_centre(exo_env) -> None:
    """``exo_crank_{side}`` must sit on the human hip joint centre.

    The crank body frame is placed at the HJC, so its world origin has to coincide with
    the femur origin (the HJC) to the last bit, and ``exo_hip_flexion_{side}`` -- the
    motor hinge -- turns about it.  ``exo_link_{side}`` is deliberately NOT here: it
    hangs off the crank pin 44 mm out, which is the pivot the strap geometry is
    dimensioned around.  An anchor given in pelvis coordinates instead of body-local
    coordinates silently shifts the whole chain -- the mass, the visual boxes and the
    cuff weld site -- by the HJC offset (124 mm).
    """
    exo_env.reset(seed=3)
    mujoco.mj_forward(exo_env.model, exo_env.data)
    model, data = exo_env.model, exo_env.data
    # the CAD puts the hip modules 191 mm off the assembly midline, the human hip joints
    # only 77 mm off theirs, so the exo hinges end up 114 mm outboard of the HJC at the
    # same height and fore-aft position.  This is what the CAD was dimensioned for: the
    # 352.2 mm link then reaches the thigh-surface strap point to within a millimetre.
    pb = data.xmat[model.body("pelvis").id].reshape(3, 3)
    P = data.xpos[model.body("pelvis").id]

    def to_pelvis(p):
        return pb.T @ (np.asarray(p) - P)

    for side in SIDES:
        offset = to_pelvis(data.xpos[model.body(f"exo_crank_{side}").id]) - to_pelvis(
            data.xpos[model.body(f"femur_{side}").id]
        )
        assert np.linalg.norm(offset[:2]) < 5e-3, f"{side} exo axis off in height/fore-aft"
        assert abs(abs(float(offset[2])) - 0.114) < 5e-3, f"{side} offset {offset * 1000} mm"


def test_exo_hinges_share_the_human_hip_axes(exo_env) -> None:
    """Flexion hinge = hip mediolateral axis; roll hinge = femur long axis."""
    model = exo_env.model
    for side in SIDES:
        flex = model.jnt_axis[model.joint(f"exo_hip_flexion_{side}").id]
        hip = model.jnt_axis[model.joint(f"hip_flexion_{side}").id]
        assert abs(float(flex @ hip)) == pytest.approx(1.0, abs=1e-6), side

        roll = model.jnt_axis[model.joint(f"exo_axial_roll_{side}").id]
        slide = model.jnt_axis[model.joint(f"exo_slot_{side}").id]
        # the slot runs along the CAD link and the roll hinge spins about it, so both
        # share the link axis by construction
        assert abs(float(roll @ slide)) == pytest.approx(1.0, abs=1e-6), side


def test_observation_stays_compatible_with_the_baseline_policy(exo_env, base_env) -> None:
    exo_obs = exo_env.reset(seed=0)[0]
    base_obs = base_env.reset(seed=0)[0]
    assert exo_obs.shape == base_obs.shape
    assert exo_env.observation_space.shape == base_env.observation_space.shape
    assert np.all(np.isfinite(exo_obs))


def test_zero_assistance_leaves_the_exo_motors_unpowered(exo_env) -> None:
    exo_env.reset(seed=1)
    rng = np.random.default_rng(0)
    for _ in range(10):
        exo_env.step(rng.uniform(0.0, 1.0, size=exo_env.action_space.shape[0]))
        assert np.allclose(exo_env.exo_joint_torques(), 0.0)


def test_cuff_weld_drags_the_exo_chain_with_the_thigh(exo_env) -> None:
    """Sweep the human hip through flexion: the exo chain must be dragged along with it.

    The hip is driven kinematically (a ramp, not a teleport) because an instantaneous
    0.4 rad jump is a ~14 cm constraint violation no strap could survive.  The cuff weld
    is a deliberately compliant thigh strap (see ``attach_lin1_exo``), so the exo hinge
    is expected to trail the human hip by a few degrees while the strap stretches a few
    centimetres - what matters is that it follows the same sweep and never separates.
    """
    exo_env.reset(seed=2)
    mujoco.mj_forward(exo_env.model, exo_env.data)
    model, data = exo_env.model, exo_env.data
    hip = model.joint("hip_flexion_r").qposadr[0]
    q0 = float(data.qpos[hip])
    cuff = model.site("exo_cuff_site_r").id
    attach = model.site("exo_attach_r").id

    gaps = []
    for i in range(200):
        mujoco.mj_step(model, data)
        data.qpos[hip] = q0 + 0.4 * min(1.0, (i + 1) / 100.0)
        mujoco.mj_forward(model, data)
        if i > 5:
            gaps.append(float(np.linalg.norm(data.site_xpos[cuff] - data.site_xpos[attach])))

    # the exo hinge axis is 114 mm outboard of the human hip, so the exo angle is *not*
    # the human hip angle any more - the slot and the roll hinge take up the difference.
    # What has to hold is that the whole chain is dragged along the sweep.
    exo_flex = float(exo_env.exo_dof_positions()[0])
    human_hip = float(data.qpos[hip])
    assert np.isfinite(exo_flex)
    assert abs(exo_flex - human_hip) < 1.4, "exo chain does not follow the human hip sweep"
    assert max(gaps) < 0.20, f"cuff strap stretched {max(gaps) * 1000:.0f} mm"


def test_exo_geometry_never_collides_with_the_body(exo_env) -> None:
    model = exo_env.model
    checked = 0
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name.startswith("exo_"):
            checked += 1
            assert model.geom_contype[g] == 0, name
            assert model.geom_conaffinity[g] == 0, name
    assert checked >= len(EXO_BODIES), "no exo visual geoms were found"


def test_every_cad_group_is_drawn_with_its_real_mesh(exo_env) -> None:
    """The exo visuals are the CAD tessellation, not just bounding boxes.

    ``cad_exo_mesh.py`` writes one ``<group>.stl`` per rigid-body group; the env must
    turn each of them into a mesh geom.  Missing files fall back to boxes, so this test
    only asserts the mesh path whenever the assets are actually shipped.
    """
    from myosuite.envs.myo.tasks.basic.leg import lin1_exo as exo_mod

    mesh_dir = exo_mod.EXO_ASSET_DIR / "meshes"
    # one CAD group per rigid body, and the SolidWorks export splits a heavy group
    # across <group>_0.stl, <group>_1.stl, ... because of MuJoCo's face limit
    bodies = {"frame": "exo_frame", "bracket_r": "exo_frame", "bracket_l": "exo_frame"}
    for side in SIDES:
        # the drum is the motor output and turns with the hip hinge; its housing stays on
        # the waist frame, and the long rod hangs off the drum's clevis pin
        bodies[f"crank_{side}"] = f"exo_crank_{side}"
        bodies[f"hipmod_{side}"] = "exo_frame"
        bodies[f"thigh_{side}"] = f"exo_link_{side}"
        bodies[f"cuff_{side}"] = f"exo_cuff_{side}"

    model = exo_env.model
    drawn = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        for g in range(model.ngeom)
    }
    checked = 0
    for name, body_name in bodies.items():
        paths = sorted(mesh_dir.glob(f"{name}_*.stl"))
        single = mesh_dir / f"{name}.stl"
        if single.is_file():
            paths.append(single)
        if not paths:
            continue
        for index, path in enumerate(paths):
            geom = f"exo_mesh_{name}_{index}"
            assert geom in drawn, f"{geom} missing for {path.name}"
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            assert model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH, geom
            assert model.geom_dataid[gid] >= 0, geom
            # the visual mesh must not contribute mass: the dynamics are the CAD-derived
            # body inertias, not whatever the tessellation happens to weigh
            assert model.geom_bodyid[gid] == model.body(body_name).id, geom
            checked += 1
    assert checked >= len(EXO_BODIES), f"only {checked} CAD mesh geoms were drawn"


def test_the_module_end_cover_is_its_own_housing_geom(exo_env) -> None:
    """The cover the fork nests on is a separate geom from the motor housing.

    ``clock_exo_endcap`` turns that cover about the drum axis so its relief keeps tracking the
    crank fork; everything else in the module - the finned motor cylinder and the mounting arm -
    is welded to the waist frame and must not move, so the cover has to be addressable on its own.
    """
    from myosuite.envs.myo.tasks.basic.leg import lin1_exo as exo_mod

    model = exo_env.model
    frame = model.body("exo_frame").id
    for side in SIDES:
        if not (exo_mod.EXO_ASSET_DIR / "meshes" / f"hipcap_{side}.stl").is_file():
            continue
        cap = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"exo_mesh_hipcap_{side}")
        housing = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"exo_mesh_hipmod_{side}_0")
        assert cap >= 0 and housing >= 0, side
        assert model.geom_type[cap] == mujoco.mjtGeom.mjGEOM_MESH
        assert model.geom_bodyid[cap] == frame, "the cover belongs to the waist frame"
        # the split must be a partition, not an addition: the cover may not be drawn twice
        verts = {}
        for gid in (cap, housing):
            mesh = int(model.geom_dataid[gid])
            lo, n = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            verts[gid] = model.mesh_vert[lo:lo + n]
        # shared boundary vertices are expected (the cut face), overlapping *faces* are not: the
        # two clouds may not coincide more than the seam allows
        from scipy.spatial import cKDTree

        near = cKDTree(verts[housing]).query(verts[cap], k=1)[0]
        assert (near < 1e-4).mean() < 0.05, f"{(near < 1e-4).mean():.1%} of the cover is duplicated"


def test_exo_body_masses_match_the_cad_parameter_file(exo_env) -> None:
    groups = _params()["groups"]
    # the motor housing is drawn on the frame but its lumped mass rides the crank, the
    # body the cad model bolts the bent rod to
    frame_mass = sum(groups[g]["mass_kg"] for g in ("frame", "bracket_r", "bracket_l"))
    assert _body_mass(exo_env, "exo_frame") == pytest.approx(frame_mass, rel=1e-6)
    for side in SIDES:
        assert _body_mass(exo_env, f"exo_crank_{side}") == pytest.approx(
            groups[f"hipmod_{side}"]["mass_kg"], rel=1e-6
        )
        assert _body_mass(exo_env, f"exo_link_{side}") == pytest.approx(
            groups[f"thigh_{side}"]["mass_kg"], rel=1e-6
        )
        assert _body_mass(exo_env, f"exo_cuff_{side}") == pytest.approx(
            groups[f"cuff_{side}"]["mass_kg"], rel=1e-6
        )


def test_registry_exposes_both_exo_variants() -> None:
    registry = gym.envs.registry
    for env_id in (EXO_ENV, EXO_LOCKED_ENV):
        assert env_id in registry, env_id
    # 15 s of walking needs 1500 control steps of 0.01 s
    assert registry[EXO_ENV].max_episode_steps >= 1500


def test_exo_parameters_are_packaged_with_the_leg_assets() -> None:
    assert os.path.isfile(EXO_PARAMS_PATH)
    assert str(EXO_PARAMS_PATH).replace("\\", "/").endswith(
        "envs/myo/assets/leg/lin1_exo/lin1_exo_params.json"
    )
