"""Interactive viewer for the Lin-I exoskeleton walking in MyoSuite.

Opens a real-time MuJoCo window and plays the exo walk policy on
``myoLegWalkExo-v0`` (zero assistance) so the coupling can be watched - and the camera
dragged around - instead of only read off a rendered PNG.  Without ``--policy`` the
``walk_exo_policy.zip`` fine-tune next to the repo is used when it exists, which is the
same policy ``scripts/eval_walk_exo.py`` renders the walk video with; otherwise the
published baseline is used (that one falls at ~4 s).

    python scripts/view_walk_exo.py                 # exo policy, real time
    python scripts/view_walk_exo.py --camera side_view   # the walk-video camera
    python scripts/view_walk_exo.py --no-policy     # free camera, no motion
    python scripts/view_walk_exo.py --speed 0.2     # slow motion
    python scripts/view_walk_exo.py --hide-exo      # see the same run without the exo

Controls: drag with the left mouse button to orbit, right button to pan, wheel to zoom,
space to pause, r to reset, Esc or q to quit.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

import myosuite  # noqa: F401  (registers the CPU envs)
from myosuite import make_env
from myosuite.envs.myo.tasks.basic.leg.lin1_exo import style_muscle_geoms
from myosuite.utils.checkpoint_utils import find_checkpoint, load_policy

# control period of the walk envs: 1 ms physics x 10 frame skip
CONTROL_DT = 0.01


def default_policy(policy: str | None) -> str | None:
    """Fill in the exo fine-tune when the caller passes no ``--policy``.

    The published ``myoLegWalk-v0`` baseline used otherwise falls at ~4 s, so it never
    shows the walk the video does.  Both this script and ``eval_walk_exo.py`` normalise
    the observations with the published checkpoint statistics, so they agree.
    """
    if policy:
        return policy
    local = Path(__file__).resolve().parents[1] / "walk_exo_policy.zip"
    if local.exists():
        print(f"policy: {local} (exo fine-tune; pass --policy to override)")
        return str(local)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="myoLegWalkExo-v0")
    ap.add_argument("--baseline", default="myoLegWalk-v0",
                    help="env whose published checkpoint drives the muscles")
    ap.add_argument("--policy", default=None,
                    help="SB3 .zip controller (e.g. walk_exo_policy.zip); without it the "
                         "published baseline of --baseline is used (that one falls at ~4 s)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback rate; 0.2 is slow motion")
    ap.add_argument("--no-policy", action="store_true", help="do not step the policy")
    ap.add_argument("--hide-exo", action="store_true",
                    help="render the exo geoms invisible, to compare against the bare walker")
    ap.add_argument("--closed-scene", action="store_true",
                    help="keep the model's uneven terrain heightfield; the default is an "
                         "open scene with only the flat floor plane as ground")
    ap.add_argument("--hide-muscles", action="store_true",
                    help="hide the 12 mm red muscle lines and show the bare skeleton; the "
                         "muscles are drawn by default")
    ap.add_argument("--distance", type=float, default=3.2,
                    help="tracking-camera distance in metres")
    ap.add_argument("--camera", default=None,
                    help="use a model camera by name (e.g. side_view, the walk-video "
                         "view) instead of the tracking camera.  A named camera is FIXED, "
                         "so dragging the mouse does nothing - omit it to get the draggable "
                         "tracking camera")
    ap.add_argument("--lookat", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="aim a FREE camera at this point instead of tracking the torso; "
                         "the mouse then orbits around it")
    ap.add_argument("--azimuth", type=float, default=180.0,
                    help="starting azimuth in degrees (drag changes it)")
    ap.add_argument("--elevation", type=float, default=-10.0,
                    help="starting elevation in degrees (drag changes it)")
    args = ap.parse_args()

    env = make_env(args.env).unwrapped
    model, data = env.model, env.data

    if args.hide_exo:
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name.startswith("exo_"):
                model.geom_rgba[gid][3] = 0.0

    if not args.closed_scene:
        # open scene: hide the terrain heightfield so the flat floor plane is the only
        # ground the eye sees (physics is untouched - the hfield stays collidable)
        hidden = 0
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name == "terrain" or model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_HFIELD:
                model.geom_rgba[gid][3] = 0.0
                hidden += 1
        print(f"open scene: hid {hidden} terrain geom(s)")

    if args.hide_muscles:
        # the env draws the 12 mm red muscle lines; this strips them for the bone-only look
        print(f"muscle style: hid {style_muscle_geoms(model, 0.0)} wrap geom(s)")

    obs, _ = env.reset(seed=0)
    args.policy = default_policy(args.policy)
    policy = None
    if not args.no_policy:
        if args.policy:
            import torch
            from stable_baselines3 import PPO

            sb3 = PPO.load(args.policy, device="cpu")
            sd = torch.load(find_checkpoint(env_id=args.baseline),
                            map_location="cpu")["actor_state_dict"]
            # rsl_rl EmpiricalNormalization in float32: (x - mean) / (std + 1e-2)
            mean = sd["obs_normalizer._mean"].numpy().astype(np.float32).ravel()
            scale = np.sqrt(sd["obs_normalizer._var"].numpy()
                            .astype(np.float32).ravel()) + np.float32(1e-2)

            def policy(obs):
                x = (np.asarray(obs, dtype=np.float32).ravel() - mean) / scale
                return sb3.predict(x, deterministic=True)[0]

            print(f"policy: {args.policy} (SB3, rsl_rl normalisation)")
        else:
            ckpt = find_checkpoint(env_id=args.baseline)
            policy = load_policy(env, ckpt)
            print(f"policy: {ckpt}")

    dt = CONTROL_DT / args.speed
    print(f"{args.env}: {model.nq} qpos, {model.nu} actuators, "
          f"{env.n_exo_dof} exo DOF - drag to orbit, space to pause, r to reset, q to quit")

    paused = {"on": False}

    def on_key(key: int) -> None:
        # mujoco.mjtKey is not exposed in this build; mjKEY_SPACE is 32
        if key == 32:
            paused["on"] = not paused["on"]
            print("paused" if paused["on"] else "running")

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key,
                                     show_left_ui=False,
                                     show_right_ui=False) as viewer:
        if args.camera:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
            if cid < 0:
                raise SystemExit(f"no camera named {args.camera!r}")
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = cid
            print(f"camera: {args.camera} (fixed - the mouse cannot orbit it)")
        elif args.lookat is not None:
            # a FREE camera pointed at one spot: the mouse orbits around that point
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = np.asarray(args.lookat, dtype=float)
            viewer.cam.distance = args.distance
            viewer.cam.azimuth = args.azimuth
            viewer.cam.elevation = args.elevation
            print(f"camera: free, looking at {list(args.lookat)} - drag to orbit")
        else:
            # aim at the walker: the default free camera looks at the origin, which is the
            # floor, and a hand-set lookat gets overridden by the viewer, so a tracking
            # camera is used - it follows the torso whatever the pose
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = model.body("torso").id
            viewer.cam.distance = args.distance
            viewer.cam.azimuth = args.azimuth
            viewer.cam.elevation = args.elevation
        while viewer.is_running():
            step_start = time.perf_counter()
            if policy is not None and not paused["on"]:
                obs, _r, terminated, truncated, _info = env.step(policy(obs))
                if terminated or truncated:
                    obs, _ = env.reset()
            with viewer.lock():
                pass
            viewer.sync()
            remaining = dt - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
