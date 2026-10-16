"""Export the built exoskeleton-wearing model to a standalone MJCF file.

Why: ``mujoco.viewer.launch_passive`` opens a window but its camera handling is fiddly on
this build, and the compiled ``simulate`` viewer that comes with a MuJoCo install cannot
load a model that was only ever built in memory.  This script does the one missing step -
dumping the finished MjSpec to disk - so the exact model that was validated here can be
opened by any MuJoCo front end.

Usage:
    python scripts/export_walk_exo_xml.py [--env myoLegWalkExo-v0] [--out <path>]
"""

from __future__ import annotations

import argparse
import pathlib

import mujoco

import myosuite  # noqa: F401  (registers the CPU envs)
from myosuite import make_env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="myoLegWalkExo-v0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-room", action="store_true",
                    help="keep the myosuite scene (walls, pictures, furniture); by default "
                         "only the floor is exported so the walker never leaves the room")
    args = ap.parse_args()

    env = make_env(args.env).unwrapped
    spec = mujoco.MjSpec()
    spec = env._mj_spec.copy() if hasattr(env, "_mj_spec") else spec

    out = pathlib.Path(args.out or f"{args.env}.xml")

    if not args.keep_room:
        # the myosuite scene geoms live on the worldbody; move them to geom group 5 so
        # they can be switched off in any viewer, which drops the room entirely
        room = mujoco.MjModel.from_xml_string(spec.to_xml())
        for gid in range(room.ngeom):
            name = mujoco.mj_id2name(room, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if (room.geom_bodyid[gid] == 0
                    and name not in ("floor", "terrain", "viz_ground")):
                room.geom_group[gid] = 5
        mujoco.mj_saveLastXML(str(out), room)
    else:
        out.write_text(spec.to_xml(), encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(out))
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  nq={model.nq} nv={model.nv} nbody={model.nbody} nu={model.nu} "
          f"nmesh={model.nmesh} neq={model.neq} mass={float(model.body_mass.sum()):.3f} kg")
    print(f"  nkey={model.nkey}")
    exo = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
           for i in range(model.nbody)
           if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("exo")]
    print(f"  exo bodies: {exo}")
    print("open with:  mujoco simulate " + str(out) + "   (or open in any MuJoCo viewer)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
