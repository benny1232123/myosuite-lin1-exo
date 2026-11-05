"""三张力矩图：输出、能力、位置。

主线是"仿真里的力矩输出像不像真机"，所以只保留三张：
  tor_curve.png     一个步态周期的力矩曲线（命令 vs 实际），配电流、电压
  tor_envelope.png  力矩-转速能力线：厂家曲线、17 N·m 上限、实测运行点
  tor_rating.png    12 N·m 相对峰值 17 / 连续 6 的位置

坐标轴一律用白话：A 不写 Arms、rpm 不写 rad/s 的换算、不出现相位。

Usage: python scripts/make_figures.py [--out DIR]

数据在 docs/data/trace_a12_lead.json（一次 12 N·m、lead 0.05 的 1500 步驱动日志）。
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

for f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        break
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "axes.titlesize": 12.5,
                     "axes.labelsize": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 200, "legend.frameon": False})

BLUE, ORANGE, RED, GREY = "#0f6f8c", "#c9741f", "#a33d6b", "#7c8b99"
STRIDE_S = 1.0
DATA = Path(__file__).resolve().parent.parent / "docs" / "data" / "trace_a12_lead.json"
RPM = 60.0 / (2 * np.pi)
TAU_PEAK, TAU_CONT = 17.0, 6.0
RPM_KNEE, RPM_NOLOAD = 225.0, 410.0
STALL = RPM_KNEE / (RPM_NOLOAD - RPM_KNEE) * TAU_PEAK   # 37.7 N.m at 0 rpm, from the same line


def load(name):
    return json.load(open(name))


def curve(out):
    c = load(DATA)["conds"]["lead005"]
    n = c["steps_run"]
    sps = int(round(c["hip_period"]))
    end = (n // sps) * sps
    sl = slice(end - sps, end)
    tr = c["trace"]
    t = np.asarray(tr["phase"], float)[:n][sl] * STRIDE_S
    cmd = np.asarray(tr["commanded_hinge_torque"], float)[:n][sl]
    got = np.asarray(tr["delivered_hinge_torque"], float)[:n][sl]
    cur = np.asarray(tr["current"], float)[:n][sl]
    vol = np.asarray(tr["voltage"], float)[:n][sl]

    fig, axs = plt.subplots(3, 1, figsize=(7.4, 6.2), sharex=True,
                            gridspec_kw={"height_ratios": [1.7, 1, 1], "hspace": 0.16})
    ax = axs[0]
    ax.plot(t, got[:, 0], color=BLUE, lw=2.2, label="左腿 实际输出")
    ax.plot(t, got[:, 1], color=ORANGE, lw=2.2, label="右腿 实际输出")
    ax.plot(t, cmd[:, 0], ls="--", color=BLUE, lw=0.9, alpha=0.95, label="左腿 命令")
    ax.plot(t, cmd[:, 1], ls="--", color=ORANGE, lw=0.9, alpha=0.95, label="右腿 命令")
    ax.axhline(TAU_PEAK, ls=":", color=RED, lw=1.1)
    ax.text(0.01, TAU_PEAK + 0.6, "电机峰值 17 N·m", fontsize=9.5, color=RED)
    ax.set_ylim(-15, 20)
    ax.set_ylabel("髋关节力矩 (N·m)")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.set_title("力矩曲线：命令多少，就交出多少（最大差 0.56 N·m）", pad=8)

    ax = axs[1]
    ax.plot(t, cur[:, 0], color=BLUE, lw=1.8)
    ax.plot(t, cur[:, 1], color=ORANGE, lw=1.8)
    lim = 23 / np.sqrt(2)
    for y in (lim, -lim):
        ax.axhline(y, ls=":", color=RED, lw=1)
    ax.text(0.01, lim + 0.8, "驱动器上限 16 A", fontsize=9.5, color=RED,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.text(0.01, -lim + 0.9, "实际峰值 10.9 A（用了 67%）", fontsize=9.5, color="#41586c",
            bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_ylim(-20, 21)
    ax.set_ylabel("电流 (A)")

    ax = axs[2]
    ax.plot(t, vol[:, 0], color=BLUE, lw=1.8)
    ax.plot(t, vol[:, 1], color=ORANGE, lw=1.8)
    ax.axhline(48, ls=":", color=RED, lw=1)
    ax.text(0.01, 48 - 6, "电源 48 V（实际峰值 26.6 V）", fontsize=9.5, color=RED,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_ylim(-32, 56)
    ax.set_xlim(0, STRIDE_S)
    ax.set_ylabel("电压 (V)")
    ax.set_xlabel("时间（秒）—— 1 秒是一个完整步态周期，左右腿各推一次")
    fig.savefig(out / "tor_curve.png", bbox_inches="tight")
    plt.close(fig)


def envelope(out):
    c = load(DATA)["conds"]["lead005"]
    n = c["steps_run"]
    tr = c["trace"]
    tau = np.abs(np.asarray(tr["delivered_hinge_torque"], float)[:n]).max(axis=1)
    spd = np.abs(np.asarray(tr["joint_speed"], float)[:n]).max(axis=1) * RPM

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    kn = np.linspace(0, RPM_KNEE, 60)
    fall = np.linspace(0, RPM_NOLOAD, 200)
    ax.plot(np.r_[kn, fall], np.r_[np.full(len(kn), TAU_PEAK),
                                   TAU_PEAK * (1 - fall / RPM_NOLOAD) / (1 - RPM_KNEE / RPM_NOLOAD)],
            color=RED, lw=2.0, label="电机能给的力矩（厂家曲线）")
    ax.plot([0, 0], [0, STALL], color=RED, lw=2.0, ls=":")     # 被限流截掉的部分
    ax.scatter(spd, tau, s=9, color=ORANGE, alpha=0.6, label="仿真里的实际运行点")
    ax.set_xlim(0, RPM_NOLOAD * 1.06)
    ax.set_ylim(0, 20)
    ax.set_xlabel("髋关节转速（转/分）—— 越快越抬不动")
    ax.set_ylabel("能输出的力矩 (N·m)")
    ax.set_title("转速越快，能给的力矩越小；我们的运行点都在左下角低速区", pad=8)
    ax.annotate("17 N·m 是驱动器限流\n不是电机极限", xy=(60, TAU_PEAK), xytext=(120, 18.2),
                fontsize=9.5, color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1))
    ax.annotate("410 转/分时\n一个力矩都出不了", xy=(RPM_NOLOAD, 0.4), xytext=(300, 6.5),
                fontsize=9.5, color="#41586c", arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(out / "tor_envelope.png", bbox_inches="tight")
    plt.close(fig)


def rating(out):
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    cats = ["电机峰值\n（只能短时）", "我们的处方", "连续工作上限\n（超过会发热）"]
    vals = [TAU_PEAK, 12.0, TAU_CONT]
    cols = ["#dfe7ee", ORANGE, "#f0d5dc"]
    y = [2, 1, 0]
    ax.barh(y, vals, color=cols, edgecolor=[GREY, ORANGE, RED], lw=1.4, height=0.55)
    for yy, v, c in zip(y, vals, [GREY, ORANGE, RED]):
        ax.text(v + 0.4, yy, f"{v:g} N·m", va="center", fontsize=12, fontweight="bold", color=c)
    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=10.5)
    ax.set_xlim(0, 21)
    ax.set_xlabel("髋关节力矩 (N·m)")
    ax.set_title("处方 12 N·m：是峰值的七成，但已是连续上限的两倍", pad=8)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(out / "tor_rating.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "docs" / "images"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    curve(out)
    envelope(out)
    rating(out)
    print("wrote 3 torque figures to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
