"""Visual report for the zero-assistance 'Lin-I' exoskeleton walk.

Renders the ``myoLegWalkExo-v0`` rollout (baseline ``myoLegWalk-v0`` policy, no
re-training, both exo motors unpowered) into a small set of figures plus an HTML
dashboard:

* ``01_views.png``    - sagittal / human-only / coronal / top / hip close-up / exo-only
* ``02_stride.png``   - one gait cycle sampled every ~100 ms in the sagittal plane
* ``03_walk_full.mp4`` (+ .gif) - 15 s sagittal tracking animation
* ``04_walk_hip.mp4``          - 15 s right-hip close-up (passive strap coupling)
* ``05_gait.png``     - COM height, distance, contacts, exo hinges vs human hips
* ``06_strap.png``    - weld slack at the thigh cuff
* ``index.html``      - dashboard with the acceptance table and every figure

Usage::

    python scripts/viz_walk_exo.py                       # default outputs in viz/
    python scripts/viz_walk_exo.py --outdir viz --no-anim
    python scripts/viz_walk_exo.py --env myoLegWalkExoLocked-v0 --open
"""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
from pathlib import Path

# must be set before mujoco is imported: Windows only offers wgl for offscreen GL
os.environ.setdefault("MUJOCO_GL", "wgl")

import mujoco
import numpy as np

import myosuite  # noqa: F401  (registers the CPU envs)
from myosuite import make_env
from myosuite.utils.checkpoint_utils import find_checkpoint, load_policy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_walk_exo  # noqa: E402  (shared constants + acceptance logic)

CONTROL_DT = eval_walk_exo.CONTROL_DT
BASELINE_ENV = "myoLegWalk-v0"
WORKDIR = Path(r"E:\exo algorithm")

# geom groups: 0 = body skin, 1 = exo CAD bounding boxes, 2 = studio ground (viz only),
# 3 = muscle wrap geoms, 4 = bones, 5 = simulation set dressing (finite floor, room
# meshes) which every panel switches off.  Toggling group 1 isolates the exoskeleton.
GROUP_STUDIO = 2
GROUP_HIDDEN = 5
GROUP_SKIN = 0
GROUP_EXO = 1
GROUP_ANATOMY = 3
# Only the skin and the exoskeleton are left visible by default: a walking figure drawn in
# full muscle detail reads as a red tangle that swallows the exoskeleton, and the point of
# these figures is the exo.  studio_model() enforces this by geom name.
GROUPS_ALL = (1, 1, 1, 0, 0, 0)
GROUPS_MUSCLES = (1, 1, 1, 1, 1, 0)
GROUPS_NO_EXO = (0, 0, 1, 0, 0, 0)
GROUPS_EXO_ONLY = (0, 1, 1, 0, 0, 0)
SKY_TOP = np.array([0.93, 0.95, 0.97])   # light studio backdrop instead of a black sky
SKY_BOTTOM = np.array([0.80, 0.84, 0.88])

# view = (unit direction from target towards the eye, distance [m]).  The walker
# travels along world +y, so +-x is the sagittal (side) direction and +y is frontal.
VIEWS = {
    "sagittal": ((1.0, 0.0, 0.09), 3.1),
    "sagittal_far": ((1.0, 0.0, 0.09), 4.2),
    "coronal": ((0.0, 1.0, 0.13), 3.2),
    "rear": ((0.0, -1.0, 0.16), 1.9),      # from behind: the waist/back frame
    "rear_far": ((0.0, -1.0, 0.12), 3.0),
    "top": ((0.26, -0.13, 0.95), 3.4),
    "hip": ((0.80, -0.55, 0.19), 1.25),
}


# ----------------------------------------------------------------------------------
# rollout (superset of eval_walk_exo.evaluate: adds sites, human hip angles)
# ----------------------------------------------------------------------------------
def _sid(model, objtype, name: str) -> int:
    idx = mujoco.mj_name2id(model, objtype, name)
    if idx < 0:
        raise KeyError(f"{name} missing from the model")
    return idx


def rollout(env_id: str, steps: int, seed: int = 0) -> dict:
    """Run the zero-shot baseline policy and record everything the figures need."""
    env = make_env(env_id).unwrapped
    ckpt = find_checkpoint(env_id=BASELINE_ENV)
    if ckpt is None:
        raise SystemExit("no baseline checkpoint found; download the myoLegWalk-v0 weights first")
    policy = load_policy(env, Path(ckpt))
    print(f"policy: {ckpt}")

    obs, _ = env.reset(seed=seed)
    model, data = env.model, env.data
    pelvis = model.body("pelvis").id
    right_foot, left_foot = eval_walk_exo._foot_body_ids(model)
    n_exo = getattr(env, "n_exo_dof", 0)

    hip_qadr = {
        name: model.jnt_qposadr[_sid(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in ("hip_flexion_r", "hip_flexion_l", "hip_rotation_r", "hip_rotation_l")
    }
    sites = {
        side: (
            _sid(model, mujoco.mjtObj.mjOBJ_SITE, f"exo_attach_{side}"),
            _sid(model, mujoco.mjtObj.mjOBJ_SITE, f"exo_cuff_site_{side}"),
        )
        for side in ("r", "l")
    }

    log = {
        "env_id": env_id,
        "qpos": np.zeros((steps, model.nq)),
        "com": np.zeros((steps, 3)),
        "dist": np.zeros(steps),
        "contact_r": np.zeros(steps, dtype=bool),
        "contact_l": np.zeros(steps, dtype=bool),
        "exo_dof": np.zeros((steps, max(n_exo, 1))),
        "exo_tau": np.zeros((steps, max(n_exo, 1))),
        "hip": np.zeros((steps, 4)),
        "strap": np.zeros((steps, 2)),
        "steps_run": steps,
    }
    start_xy = np.array(data.xpos[pelvis][:2])
    for i in range(steps):
        obs, _r, terminated, truncated, _info = env.step(policy(obs))
        mujoco.mj_forward(model, data)
        log["qpos"][i] = data.qpos
        log["com"][i] = data.subtree_com[pelvis]
        log["dist"][i] = float(np.linalg.norm(data.xpos[pelvis][:2] - start_xy))
        log["hip"][i] = [data.qpos[a] for a in hip_qadr.values()]
        log["exo_dof"][i] = env.exo_dof_positions() if n_exo else 0.0
        log["exo_tau"][i] = env.exo_joint_torques() if n_exo else 0.0
        for k, side in enumerate(("r", "l")):
            a, b = sites[side]
            log["strap"][i, k] = float(np.linalg.norm(data.site_xpos[a] - data.site_xpos[b]))
        for c in range(data.ncon):
            for g in (data.contact[c].geom1, data.contact[c].geom2):
                if g < 0:
                    continue
                body = model.geom_bodyid[g]
                if body in right_foot:
                    log["contact_r"][i] = True
                elif body in left_foot:
                    log["contact_l"][i] = True
        if terminated or truncated:
            log["steps_run"] = i + 1
            break
    log["model"] = model
    log["env"] = env
    log["viz"] = studio_model(env)
    log["viz_anatomy"] = studio_model(env, keep_anatomy=True)
    return log


def studio_model(env, keep_anatomy: bool = False) -> mujoco.MjModel:
    """Copy the env's model and turn it into a neutral studio (render only).

    Two changes: a large ground plane (geom group 2) replaces the simulation floor, whose
    finite extent would leave a walker that covered ~20 m floating in a black void; and the
    original world-body geoms (floor, room meshes) are moved to geom group 5, which every
    panel switches off, so the cluttered room does not show up in the close-ups.  The copy
    is never stepped: the figures replay recorded qpos values into it.  ``MjSpec.copy`` is
    used because a compiled model cannot be turned back into a spec (``MjSpec.from_model``
    does not exist in mujoco 3.16).
    """
    spec = env.unwrapped._mj_spec.copy()
    spec.worldbody.add_geom(
        name="viz_ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[200.0, 200.0, 0.1],
        pos=[0.0, 0.0, -0.004],
        rgba=[0.58, 0.60, 0.64, 1.0],
        contype=0,
        conaffinity=0,
        group=2,
    )
    model = spec.compile()
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == 0 and model.geom_group[gid] != GROUP_STUDIO:
            model.geom_group[gid] = GROUP_HIDDEN
    # Decisive pass, by name: the exo is the only thing named ``exo_*``, so everything else
    # except the bare skin is parked in the always-hidden group.  Group toggles alone are
    # not enough here - the muscle wrap and the bones are what turn a walking figure into a
    # red tangle that swallows the exoskeleton, and they survive the studio-ground shuffle.
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if keep_anatomy:
            model.geom_group[gid] = GROUP_ANATOMY if model.geom_group[gid] in (3, 4) \
                else model.geom_group[gid]
            continue
        if name.startswith("exo_"):
            model.geom_group[gid] = GROUP_EXO
        elif model.geom_group[gid] == GROUP_STUDIO:
            pass
        elif model.geom_bodyid[gid] != 0 and model.geom_group[gid] == 0:
            model.geom_group[gid] = GROUP_SKIN          # limb / torso skin
        else:
            model.geom_group[gid] = GROUP_HIDDEN
    return model


def ghost_body_model(model, out: Path) -> mujoco.MjModel:
    """Copy of *model* with the human skin faded, so the exo reads through it.

    Only the skin geoms (group 0, non-hidden) are made translucent; the exo (group 1) and
    the studio ground (group 2) are left alone.  ``rgba`` lives in the model, so this has
    to happen on a copy - the physics is untouched either way, the figure is only replayed.
    """
    tmp = out / "_ghost.mjb"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(model, str(tmp))
    ghost = mujoco.MjModel.from_binary_path(str(tmp))
    tmp.unlink(missing_ok=True)
    for gid in range(ghost.ngeom):
        if ghost.geom_group[gid] == GROUP_SKIN:
            ghost.geom_rgba[gid][3] = 0.22
    return ghost


# ----------------------------------------------------------------------------------
# offscreen rendering
# ----------------------------------------------------------------------------------
class Shotter:
    """Offscreen renderer with a MuJoCo free camera and geom-group toggles.

    MuJoCo's free camera is placed with ``lookat + distance`` along
    ``(cos el cos az, cos el sin az, sin el)``; the bindings expose azimuth,
    elevation, distance and lookat but not the field of view (fixed at 45 deg), so
    zoom is done by moving the camera.
    """

    def __init__(self, model, width: int, height: int):
        self.model = model
        self.width, self.height = width, height
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        self.renderer = mujoco.Renderer(model, height, width)
        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.opt)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.groups = GROUPS_ALL

    def set_groups(self, groups: tuple[int, ...]) -> None:
        self.groups = groups

    def set_model(self, model: mujoco.MjModel) -> None:
        """Swap the rendered model (used for the translucent-body figure)."""
        if model is self.model:
            return
        self.renderer.close()
        self.model = model
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), self.width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), self.height)
        self.renderer = mujoco.Renderer(model, self.height, self.width)

    def shot(self, data, view, target, distance: float | None = None) -> np.ndarray:
        """Render ``data``; ``view`` is (direction, distance) from ``VIEWS``."""
        direction, dist = view
        d = np.asarray(direction, dtype=float)
        d /= np.linalg.norm(d)
        self.camera.lookat[:] = np.asarray(target, dtype=float)
        self.camera.distance = float(distance if distance is not None else dist)
        self.camera.azimuth = float(np.degrees(np.arctan2(-d[1], -d[0])))
        self.camera.elevation = float(-np.degrees(np.arcsin(np.clip(d[2], -1.0, 1.0))))
        self.opt.geomgroup[:] = self.groups
        self.renderer.update_scene(data, camera=self.camera, scene_option=self.opt)
        return _paint_sky(self.renderer.render())

    def close(self) -> None:
        self.renderer.close()


def _paint_sky(img: np.ndarray) -> np.ndarray:
    """Replace MuJoCo's near-black sky with a soft vertical gradient.

    The renderer gives no hook for a clear colour, and its built-in skybox goes to black
    at the zenith, which reads as a hole in the figures.  Only very dark, near-neutral
    pixels are repainted, so shadows and coloured geoms survive.
    """
    dark = img.max(axis=2) < 45
    neutral = (img.max(axis=2).astype(int) - img.min(axis=2).astype(int)) < 20
    sky = dark & neutral
    if not sky.any():
        return img
    h = img.shape[0]
    ramp = np.linspace(0.0, 1.0, h)[:, None]
    grad = (SKY_TOP[None, :] * (1.0 - ramp) + SKY_BOTTOM[None, :] * ramp)
    out = img.copy()
    out[sky] = np.broadcast_to((grad[:, None, :] * 255.0).astype(np.uint8), img.shape)[sky]
    return out


def set_step(log, i: int):
    """Load control step *i* into the render data and return (data, pelvis target)."""
    data = log.get("viz_data")
    if data is None:
        data = log["viz_data"] = mujoco.MjData(log["viz"])
    data.qpos[:] = log["qpos"][i]
    mujoco.mj_forward(log["viz"], data)
    pelvis = log["viz"].body("pelvis").id
    return data, np.array(data.xpos[pelvis], dtype=float)


def _plt():
    """Agg backend with a CJK-capable font (matplotlib's default has no hanzi)."""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    return plt


# ----------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------
def figure_views(log, out: Path, step: int) -> None:
    plt = _plt()
    shot = Shotter(log["viz"], 560, 760)
    panels = [
        ("矢状面 · 人体 + 林-Ⅰ 外骨骼", "sagittal", 0.0, GROUPS_ALL),
        ("矢状面 · 隐藏外骨骼（仅人体）", "sagittal", 0.0, GROUPS_NO_EXO),
        ("冠状面 · 正面", "coronal", 0.0, GROUPS_ALL),
        ("后面 · 腰背靠背板", "rear", (0.0, 0.0, -0.12), GROUPS_ALL),
        ("肌肉 / 骨骼（外骨骼隐藏）", "coronal", 0.0, GROUPS_MUSCLES),
        ("右髋特写 · 髋模块 + 贴合垫", "hip", (0.0, 0.05, -0.12), GROUPS_ALL),
        ("外骨骼高亮 · 人体半透明", "sagittal", 0.0, GROUPS_ALL),
        ("外骨骼 · 腰背框 + 双侧髋模块（背面）", "rear", (0.0, 0.0, -0.12), GROUPS_EXO_ONLY),
        ("外骨骼 · 整体（背面）", "rear_far", 0.0, GROUPS_EXO_ONLY),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(17.5, 9.2))
    for ax, (title, view, offset, groups) in zip(axes.ravel(), panels):
        data, target = set_step(log, step)
        target = target + np.asarray(offset)
        if "半透明" in title:
            shot.set_model(ghost_body_model(log["viz"], out))
        elif "肌肉" in title:
            shot.set_model(log["viz_anatomy"])
        else:
            shot.set_model(log["viz"])
        shot.set_groups(groups)
        ax.imshow(shot.shot(data, VIEWS[view], target))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(
        f"林-Ⅰ 髋关节外骨骼 + MyoLeg 人体（MuJoCo {mujoco.__version__}）"
        f"   t = {step * CONTROL_DT:.2f} s，零助力",
        fontsize=13,
    )
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "01_views.png", dpi=120)
    plt.close(fig)
    shot.close()
    print("wrote", out / "01_views.png")


def figure_stride(log, out: Path, t_start: float = 6.0, n: int = 8, span: float = 0.8) -> None:
    plt = _plt()
    shot = Shotter(log["viz"], 420, 620)
    shot.set_groups(GROUPS_ALL)
    steps = [int(round((t_start + span * k / (n - 1)) / CONTROL_DT)) for k in range(n)]
    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    for ax, step in zip(axes.ravel(), steps):
        step = min(step, log["steps_run"] - 1)
        data, target = set_step(log, step)
        ax.imshow(shot.shot(data, VIEWS["sagittal"], target))
        contact = "R" if log["contact_r"][step] else "·"
        contact += "L" if log["contact_l"][step] else "·"
        ax.set_title(
            f"t={step * CONTROL_DT:.2f} s   触地 {contact}   "
            f"exo flex_r {log['exo_dof'][step, 0]:+.2f} rad",
            fontsize=9,
        )
        ax.axis("off")
    fig.suptitle("矢状面步态循环（一个周期 ≈ 0.8 s，零助力）", fontsize=13)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "02_stride.png", dpi=120)
    plt.close(fig)
    shot.close()
    print("wrote", out / "02_stride.png")


def figure_gait(log, out: Path) -> None:
    plt = _plt()
    n = log["steps_run"]
    t = np.arange(n) * CONTROL_DT
    fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(t, log["com"][:n, 2], color="tab:blue")
    axes[0].axhline(0.8, color="crimson", ls="--", lw=1, label="done 阈值 0.8 m")
    axes[0].set_ylabel("骨盆 COM 高度 [m]")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].plot(t, log["dist"][:n], color="tab:green")
    axes[1].set_ylabel("累计位移 [m]")
    axes[2].step(t, log["contact_r"][:n], where="post", label="右足")
    axes[2].step(t, log["contact_l"][:n], where="post", label="左足", alpha=0.6)
    axes[2].set_ylabel("足底接触")
    axes[2].legend(loc="upper right", fontsize=8, ncol=2)
    for j, label in enumerate(("exo flex_r", "exo roll_r", "exo flex_l", "exo roll_l")):
        axes[3].plot(t, log["exo_dof"][:n, j], lw=1.2, label=label)
    axes[3].set_ylabel("外骨骼铰链角 [rad]")
    axes[3].legend(loc="upper right", fontsize=8, ncol=4)
    axes[4].plot(t, log["hip"][:n, 0], color="k", lw=1.6, label="人 hip_flexion_r")
    axes[4].plot(t, log["exo_dof"][:n, 0], color="tab:orange", lw=1.0, ls="--",
                 label="exo 屈伸_r")
    axes[4].plot(t, log["hip"][:n, 2], color="gray", lw=1.6, label="人 hip_rotation_r")
    axes[4].plot(t, log["exo_dof"][:n, 1], color="tab:red", lw=1.0, ls="--",
                 label="exo 内旋_r")
    axes[4].set_ylabel("被动跟随对比 [rad]")
    axes[4].set_xlabel("时间 [s]")
    axes[4].legend(loc="upper right", fontsize=8, ncol=2)
    fig.suptitle("零助力步行：外骨骼被动跟随人体髋关节（电机力矩恒为 0）", fontsize=13)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "05_gait.png", dpi=120)
    plt.close(fig)
    print("wrote", out / "05_gait.png")

    fig, ax = plt.subplots(figsize=(10, 3.0))
    for k, side in enumerate(("右", "左")):
        ax.plot(t, 1000.0 * log["strap"][:n, k], lw=1.2, label=f"{side}腿贴合垫焊缝间隙")
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("间隙 [mm]")
    ax.set_title("贴合垫处 weld 松弛量（对应有衬垫束带的柔度）", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "06_strap.png", dpi=120)
    plt.close(fig)
    print("wrote", out / "06_strap.png")


def animate(log, out: Path, view: str, name: str, width: int, height: int,
            stride: int = 5, fps: float = 20.0, gif_stride: int = 10) -> None:
    """Write an MP4 (and a small GIF) of the whole 15 s rollout."""
    import imageio.v2 as imageio

    out.mkdir(parents=True, exist_ok=True)
    shot = Shotter(log["viz"], width, height)
    shot.set_groups(GROUPS_ALL)
    n = log["steps_run"]
    frames: list[np.ndarray] = []
    gif_frames: list[np.ndarray] = []
    for i in range(0, n, stride):
        data, target = set_step(log, i)
        img = shot.shot(data, VIEWS[view], target)
        frames.append(img)
        if i % gif_stride == 0:
            gif_frames.append(img[::2, ::2])
    shot.close()

    mp4 = out / f"{name}.mp4"
    writer = imageio.get_writer(str(mp4), fps=fps, codec="libx264", quality=8,
                                macro_block_size=None, ffmpeg_log_level="error")
    for img in frames:
        writer.append_data(img)
    writer.close()
    print("wrote", mp4, f"({len(frames)} frames, {len(frames) / fps:.1f} s)")

    from PIL import Image

    gif = out / f"{name}.gif"
    pal = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=128) for f in gif_frames]
    pal[0].save(gif, save_all=True, append_images=pal[1:], loop=0,
                duration=int(1000 * stride * CONTROL_DT / fps), optimize=False)
    print("wrote", gif)


# ----------------------------------------------------------------------------------
# dashboard
# ----------------------------------------------------------------------------------
NAMES = {
    "myoLegWalk-v0": "基线（无外骨骼）",
    "myoLegWalkExo-v0": "林-Ⅰ 外骨骼（铰链 + 贴合垫，零助力）",
    "myoLegWalkExoLocked-v0": "林-Ⅰ 外骨骼（刚体惯性负载）",
}


def dashboard(out: Path, metrics: dict[str, dict], params: dict) -> None:
    rows = []
    for env_id, m in metrics.items():
        chk = m.get("checks", {})

        def mark(key):
            v = chk.get(key)
            return "—" if v is None else ("✅" if v else "❌")

        rows.append(
            "<tr><td>{name}<br><small>{env}</small></td><td>{steps} 步 / {dur:.1f} s</td>"
            "<td>{dist:.2f} m<br><small>{spd:.2f} m/s</small></td><td>{tau}</td>"
            "<td>{sr} / {sl}</td><td>{c1}{c2}{c3}{c4}</td></tr>".format(
                name=html.escape(NAMES.get(env_id, env_id)),
                env=html.escape(env_id),
                steps=m["steps_run"],
                dur=m["duration_s"],
                dist=m["distance_m"],
                spd=m["mean_speed_mps"],
                tau=("—" if env_id == BASELINE_ENV else f"{m['exo_torque_max_Nm']:.3g}"),
                sr=m["right_stance_phases"],
                sl=m["left_stance_phases"],
                c1=mark("1_survives_15s"), c2=mark("2_distance>=5m"),
                c3=mark("3_exo_torques_zero"), c4=mark("4_alternating_contacts"),
            )
        )

    groups = params.get("groups", {})
    prows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{g['n_parts']}</td><td>{g['mass_kg']:.3f}</td>"
        f"<td>{g['com_m'][0]*1000:.0f}, {g['com_m'][1]*1000:.0f}, {g['com_m'][2]*1000:.0f}</td></tr>"
        for name, g in groups.items()
    )
    total = sum(g["mass_kg"] for g in groups.values())
    n_parts = sum(g["n_parts"] for g in groups.values())

    page = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>林-Ⅰ 外骨骼 · MyoSuite 零助力步行</title>
<style>
 body{{font-family:"Microsoft YaHei",Segoe UI,sans-serif;margin:24px 32px;color:#222;background:#fafafa}}
 h1{{margin-bottom:4px}} h2{{margin-top:32px;border-bottom:2px solid #ddd;padding-bottom:4px}}
 table{{border-collapse:collapse;margin:10px 0;font-size:13px;background:#fff}}
 th,td{{border:1px solid #ccc;padding:6px 10px;text-align:center}}
 th{{background:#eee}}
 img{{max-width:100%;border:1px solid #ddd;border-radius:4px;margin:6px 0}}
 video{{max-width:520px;border:1px solid #ddd;border-radius:4px;margin:6px 12px 6px 0}}
 small{{color:#666}}
 pre{{background:#fff;border:1px solid #ddd;padding:10px;font-size:12px;overflow:auto}}
</style></head><body>
<h1>“林-Ⅰ”髋关节外骨骼 · MyoSuite 零助力 15 s 连续步行</h1>
<p><small>人体模型：MyoLeg（29 DOF / 80 肌）· 外骨骼：CAD 自动提取（{total:.3f} kg，{len(groups)} 个刚体组 / {n_parts} 个零件）
· 策略：<code>myoLegWalk-v0</code> 基线零样本，无微调 · 外骨骼电机力矩恒为 0</small></p>

<h2>验收结果</h2>
<table><tr><th>环境</th><th>时长</th><th>位移 / 均速</th><th>外骨骼电机力矩 [N·m]</th><th>右/左支撑相</th>
<th>①1500 步不终止</th><th>②位移 ≥ 5 m</th><th>③电机恒 0</th><th>④左右交替触地</th></tr>
{''.join(rows)}
</table>
<p><small>判据：①连续 1500 个控制步（0.01 s/步 = 15.0 s）无 done；②前进位移 ≥ 5 m；③两台外骨骼电机
actuator 力矩恒等于 0；④双足均出现 ≥ 4 个支撑相且占空比不同（交替触地）。</small></p>

<h2>模型与装配</h2>
<img src="01_views.png" alt="views">
<img src="cad_groups.png" alt="cad">

<h2>步态周期（矢状面）</h2>
<img src="02_stride.png" alt="stride">

<h2>15 s 连续步行动画</h2>
<div>
<video src="03_walk_full.mp4" autoplay loop muted playsinline></video>
<video src="04_walk_hip.mp4" autoplay loop muted playsinline></video>
</div>
<p><small>左：全身矢状面；右：右髋特写。髋模块随人体髋屈伸摆动、贴合垫跟随大腿下段，电机全程不发力。
（<a href="03_walk_full.gif">GIF</a> · <a href="03_walk_full.mp4">MP4</a> · <a href="04_walk_hip.mp4">髋部 MP4</a>）</small></p>

<h2>运动学数据</h2>
<img src="05_gait.png" alt="gait">
<img src="06_strap.png" alt="strap">

<h2>外骨骼刚体参数（CAD 自动提取）</h2>
<table><tr><th>刚体组</th><th>零件数</th><th>质量 [kg]</th><th>CAD 质心 [mm]（X 前 / Y 左 / Z 上）</th></tr>
{prows}
<tr><th>合计</th><th>{n_parts}</th><th>{total:.3f}</th><th></th></tr>
</table>

<h2>原始指标</h2>
<pre>{html.escape(json.dumps(metrics, indent=2, ensure_ascii=False, default=float))}</pre>
</body></html>"""

    (out / "index.html").write_text(page, encoding="utf-8")
    print("wrote", out / "index.html")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="myoLegWalkExo-v0")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default="viz")
    parser.add_argument("--no-anim", action="store_true")
    parser.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    log = rollout(args.env, args.steps, args.seed)
    last = log["steps_run"] - 1
    print(f"rollout: {log['steps_run']} steps, {log['dist'][last]:.2f} m, "
          f"|tau|max = {np.max(np.abs(log['exo_tau'])):.3g} N·m")

    hero_step = int(min(7.0, log["steps_run"] * CONTROL_DT - 0.3) / CONTROL_DT)
    figure_views(log, out, hero_step)
    figure_stride(log, out)
    figure_gait(log, out)
    if not args.no_anim:
        animate(log, out, "sagittal", "03_walk_full", 400, 620)
        animate(log, out, "hip", "04_walk_hip", 460, 520)

    metrics = {}
    for env_id in (BASELINE_ENV, "myoLegWalkExo-v0", "myoLegWalkExoLocked-v0"):
        path = WORKDIR / f"metrics_{env_id}.json"
        if path.exists():
            metrics[env_id] = json.loads(path.read_text(encoding="utf-8"))
    if not metrics:
        metrics = {args.env: eval_walk_exo.evaluate(args.env, steps=args.steps, seed=args.seed)[0]}

    params_path = (
        Path(__file__).resolve().parents[1]
        / "myosuite/envs/myo/assets/leg/lin1_exo/lin1_exo_params.json"
    )
    params = json.loads(params_path.read_text(encoding="utf-8"))
    cad = WORKDIR / "cad_groups.png"
    if cad.exists():
        (out / cad.name).write_bytes(cad.read_bytes())
    dashboard(out, metrics, params)

    if args.open:
        subprocess.Popen(["cmd", "/c", "start", "", str((out / "index.html").resolve())])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())