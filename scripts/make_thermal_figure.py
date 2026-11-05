"""图：绕组温度上去以后，驱动器允许的力矩会掉下来。

左图用 17 N·m 的请求跑 120 秒（`_thermal_run.py 120 ... 17`）：请求一直是 17，但可用上限从 17 掉到
13.5，实际能出的力矩跟着掉。同一套模型下 12 N·m 处方跑 60 秒只到 33.6 °C，说明走路那条处方本身
不触发热降额。

Usage: python scripts/make_thermal_figure.py [--data docs/data/thermal_17nm_120s.json]
       [--out docs/images/thermal.png]
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "docs" / "data" / "thermal_17nm_120s.json"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "docs" / "images" / "thermal.png"))
    ap.add_argument("--assist", type=float, default=17.0)
    a = ap.parse_args()
    rec = json.load(open(a.data))
    t = np.asarray(rec["t"])
    temp = np.asarray(rec["temp"])[:, 0]
    cap = np.asarray(rec["cap"])[:, 0]

    fig, axs = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True,
                            gridspec_kw={"height_ratios": [1, 1], "hspace": 0.18})
    ax = axs[0]
    ax.plot(t, temp, color=RED, lw=2.2)
    ax.axhline(rec["max_C"], ls=":", color=RED, lw=1.1)
    ax.text(1.5, rec["max_C"] - 7, f"绕组上限 {rec['max_C']:.0f} °C", fontsize=9.5, color=RED)
    ax.set_ylim(20, 110)
    ax.set_ylabel("绕组温度 (°C)")
    ax.set_title("一直要 17 N·m：热起来以后，能给的力矩会掉", pad=8)

    ax = axs[1]
    ax.axhline(a.assist, ls="--", color=GREY, lw=1.2)
    ax.text(1.5, a.assist + 0.35, f"请求 {a.assist:.0f} N·m", fontsize=9.5, color="#41586c")
    ax.plot(t, cap, color=ORANGE, lw=2.4)
    ax.text(t[-1] * 0.62, cap[-1] - 1.1, f"可用上限 {cap[-1]:.1f} N·m", fontsize=10, color=ORANGE)
    ax.set_ylim(10, 19)
    ax.set_ylabel("允许的力矩 (N·m)")
    ax.set_xlabel(f"时间（秒）—— 连续走 {rec['seconds']:.0f} 秒")
    fig.savefig(a.out, bbox_inches="tight")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
