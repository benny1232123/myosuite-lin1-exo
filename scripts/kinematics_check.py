"""Kinematic verification of the Lin-I exo chain against the human leg.

Draws the actual model geometry (no rendering): a coronal slice through the cuff
height and a sagittal slice through the hip axis, showing the femur axis, the exo
hinge axis, the link, the cuff and the strap attachment.  This is the check that
answers "is the exo where it should be" without eyeballing a shaded render.
"""

import os

os.environ.setdefault("MUJOCO_GL", "wgl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mujoco  # noqa: E402

import myosuite  # noqa: F401,E402
from myosuite import make_env  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

env = make_env("myoLegWalkExo-v0").unwrapped
m, d = env.model, env.data
env.reset(seed=0)
mujoco.mj_forward(m, d)

pb = d.xmat[m.body("pelvis").id].reshape(3, 3)
P = d.xpos[m.body("pelvis").id]
tp = lambda p: pb.T @ (np.asarray(p) - P)     # pelvis frame

fig, axes = plt.subplots(1, 2, figsize=(13, 7))
R2 = np.array([[1.0, 0.0], [0.0, 1.0]])

for side, tag in (("r", "右"), ("l", "左")):
    femur = m.body(f"femur_{side}").id
    tibia = m.body(f"tibia_{side}").id
    hip = tp(d.xpos[femur]) * 1000            # mm, pelvis frame
    knee = tp(d.xpos[tibia]) * 1000
    axis = tp(d.xpos[m.body(f"exo_link_{side}").id]) * 1000
    cuff = tp(d.xpos[m.body(f"exo_cuff_{side}").id]) * 1000
    att = tp(d.site_xpos[m.site(f"exo_attach_{side}").id]) * 1000

    thigh = knee - hip
    L = np.linalg.norm(thigh)
    u = thigh / L                              # along the femur
    lat = np.array([-u[1], u[0]])              # 90 deg CCW in the z-y plane

    # coronal slice at the cuff height: z horizontal, y vertical
    ax = axes[0]
    s_c = np.dot(cuff - hip, u)
    c = hip + s_c * u
    for rr, lab in ((78, ""),):
        th = np.linspace(0, 2 * np.pi, 120)
        ax.plot(c[0] + rr * np.cos(th), c[1] + rr * np.sin(th),
                color="0.85", lw=8, zorder=1)
    ax.plot([hip[0], knee[0]], [hip[1], knee[1]], "k-", lw=3, zorder=2,
            label="股骨轴" if side == "r" else None)
    ax.plot(*hip, "ko", ms=7, zorder=4, label="髋关节 HJC" if side == "r" else None)
    ax.plot(*knee, "k^", ms=8, zorder=4, label="膝" if side == "r" else None)
    ax.plot(*axis, "s", color="#1f5fd0", ms=11, zorder=5,
            label="外骨髋轴" if side == "r" else None)
    ax.plot([axis[0], cuff[0]], [axis[1], cuff[1]], "-", color="#e07b00", lw=6,
            zorder=3, label="外骨连杆" if side == "r" else None)
    ax.plot(*cuff, "o", color="#55aa22", ms=11, zorder=5,
            label="贴合垫(外骨)" if side == "r" else None)
    ax.plot(*att, "x", color="#aa2222", ms=12, mew=3, zorder=5,
            label="贴合垫(人体)" if side == "r" else None)
    ax.annotate("", xy=cuff[:2], xytext=att[:2],
                arrowprops=dict(arrowstyle="-", color="#aa2222", lw=1.5,
                                linestyle="--"))
    off = np.linalg.norm(cuff[:2] - att[:2])
    d_axis = abs(axis[2] - hip[2])
    d_lat = np.linalg.norm(cuff[:2] - att[:2])
    print(f"{tag}侧: HJC z={hip[2]:.1f}  外骨髋轴 z={axis[2]:.1f}  "
          f"(外移 {d_axis:.1f} mm)")
    print(f"     连杆长={np.linalg.norm(cuff-axis):.1f} mm  "
          f"连杆与大股轴夹角={np.degrees(np.arccos(abs(np.dot((cuff-axis)/np.linalg.norm(cuff-axis), u)))):.2f} deg")
    print(f"     贴合垫偏移(横向)={d_lat:.1f} mm   束带跨度={off:.1f} mm")

axes[0].set_title("冠状面切片（贴合垫高度）— 外骨骼在人体外侧", fontsize=13)
axes[0].set_xlabel("x [mm]（前）")
axes[0].set_ylabel("y [mm]（上）")
axes[0].set_aspect("equal")
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=9, loc="upper left")

# sagittal slice through the right hip axis: x horizontal, y vertical
ax = axes[1]
for side, tag in (("r", "右"), ("l", "左")):
    femur = m.body(f"femur_{side}").id
    tibia = m.body(f"tibia_{side}").id
    hip = tp(d.xpos[femur]) * 1000
    knee = tp(d.xpos[tibia]) * 1000
    axis = tp(d.xpos[m.body(f"exo_link_{side}").id]) * 1000
    cuff = tp(d.xpos[m.body(f"exo_cuff_{side}").id]) * 1000
    col = "#1f5fd0" if side == "r" else "#888888"
    ax.plot([hip[0], knee[0]], [hip[1], knee[1]], "-", color=col,
            lw=3 if side == "r" else 1.5, zorder=2,
            label=f"{tag} 股骨轴" if side == "r" else f"{tag} 股骨轴")
    ax.plot(*axis, "s", color="#1f5fd0", ms=10, zorder=5)
    ax.plot([axis[0], cuff[0]], [axis[1], cuff[1]], "-", color="#e07b00",
            lw=6 if side == "r" else 2, zorder=3)
    ax.plot(*cuff, "o", color="#55aa22", ms=10, zorder=5)
    if side == "r":
        ax.plot(*hip, "ko", ms=7)
        ax.plot(*knee, "k^", ms=8)
        ax.annotate("外骨髋轴", xy=axis[:2], xytext=(axis[0] + 12, axis[1] - 30),
                    fontsize=9, arrowprops=dict(arrowstyle="->", color="#1f5fd0"))
ax.set_title("矢状面切片（过右髋轴）— 连杆与大腿的关系", fontsize=13)
ax.set_xlabel("x [mm]（前）")
ax.set_ylabel("y [mm]（上）")
ax.set_aspect("equal")
ax.grid(alpha=0.3)

fig.suptitle("林-Ⅰ 外骨骼运动学校验（数值取自 myoLegWalkExo-v0 模型）", fontsize=14)
fig.tight_layout()
out = r"E:\exo algorithm\kinematics_check.png"
fig.savefig(out, dpi=130)
print("wrote", out)
