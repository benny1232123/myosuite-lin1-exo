"""Render the Lin-I exoskeleton geoms in isolation and measure their world extent.

Diagnosis for "the model looks wrong": are the exo meshes actually in the model, at the
right scale, and where do they sit relative to the skeleton?

Run from the repo root:  python scripts/render_exo_only.py
"""

from __future__ import annotations

import os

import mujoco
import numpy as np
from PIL import Image

import myosuite  # noqa: F401
from myosuite import make_env

OUT = r"E:\exo algorithm"
W, H = 1100, 800

env = make_env("myoLegWalkExo-v0").unwrapped
env.reset(seed=0)
sim = getattr(env, "sim", env)
model, data = sim.model, sim.data
mujoco.mj_forward(model, data)

model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
model.vis.global_.offheight = max(model.vis.global_.offheight, H)

name = mujoco.mj_id2name
exo_bodies = [i for i in range(model.nbody)
              if (name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("exo_")]
print("exo bodies:", [name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in exo_bodies])
exo_geoms = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in exo_bodies]
print(f"exo geoms : {len(exo_geoms)} of {model.ngeom}")
for g in exo_geoms:
    print(f"  geom {g} body={name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))} "
          f"type={int(model.geom_type[g])} rgba={np.round(model.geom_rgba[g], 2).tolist()}")

# world-space extent of every exo geom
lo = np.full(3, np.inf)
hi = np.full(3, -np.inf)
for g in exo_geoms:
    if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
        continue
    mid = int(model.geom_dataid[g])
    adr, num = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    v = model.mesh_vert[adr:adr + num]
    R = data.geom_xmat[g].reshape(3, 3)
    w = v @ R.T + data.geom_xpos[g]
    lo = np.minimum(lo, w.min(axis=0))
    hi = np.maximum(hi, w.max(axis=0))
print("exo world bbox min:", np.round(lo, 4).tolist())
print("exo world bbox max:", np.round(hi, 4).tolist())
print("exo world size   :", np.round(hi - lo, 4).tolist(), "m")

pelvis = model.body("pelvis").id
print("pelvis xpos:", np.round(data.xpos[pelvis], 4).tolist())
for i in exo_bodies:
    print(f"  body {name(model, mujoco.mjtObj.mjOBJ_BODY, i):<16} xpos "
          f"{np.round(data.xpos[i], 4).tolist()}")

orig = model.geom_rgba.copy()
renderer = mujoco.Renderer(model, H, W)


def shot(path, keep_exo_only, lookat, distance, azimuth, elevation):
    for g in range(model.ngeom):
        show = (g in exo_geoms) if keep_exo_only else True
        model.geom_rgba[g, 3] = orig[g, 3] if show else 0.0
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    renderer.update_scene(data, camera=cam)
    Image.fromarray(renderer.render()).save(path)
    print("wrote", path)


# the exo on its own, framed on the pelvis, from the front and the side
shot(os.path.join(OUT, "exo_only_front.png"), True, [0.0, 0.0, 0.9], 1.6, 90, -5)
shot(os.path.join(OUT, "exo_only_side.png"), True, [0.0, 0.0, 0.9], 1.6, 0, -5)
# whole model, same camera, for reference
shot(os.path.join(OUT, "exo_with_body_front.png"), False, [0.0, 0.0, 0.9], 1.9, 90, -5)
shot(os.path.join(OUT, "exo_with_body_side.png"), False, [0.0, 0.0, 0.9], 1.9, 0, -5)
print("done")
