"""Run the drive long enough for the winding to heat, and log the derating.

The thermal state resets with every episode (so a reset still determines the rollout), which
means a normal 15 s acceptance run never shows it: at the walking duty the winding rises a few
degrees and the cap barely moves.  This drives one episode for a minute with the same 12 N.m
prescription and records temperature and the torque cap the drive is allowed.

Usage: python scripts/thermal_demo.py [SECONDS] [OUT.json] [ASSIST_Nm]

默认 120 秒、17 N·m。写成 out.json 后可以交给 scripts/make_thermal_figure.py 出图。
"""
import json
import sys

import numpy as np

import myosuite  # noqa: F401
from myosuite import make_env

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
TORQUE = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
OUT = sys.argv[2] if len(sys.argv) > 2 else "thermal_run.json"
ENV = "myoLegWalkExo-v0"

env = make_env(ENV, exo_motor=True, exo_assist_torque=TORQUE, exo_assist_phase_lead=0.05,
               exo_assist_width=0.30, exo_strap_preload_N=50.0, exo_thermal_derate=True).unwrapped
env.reset(seed=0)
action = np.zeros(env.action_space.shape, dtype=np.float32)
steps = int(round(SECONDS / env.dt)) if hasattr(env, "dt") else int(SECONDS * 100)
for _ in range(steps):
    env.step(action)

log = env.assist_summary()
t = np.arange(len(log["phase"])) * float(env.model.opt.timestep) * env.frame_skip
rec = {
    "seconds": float(t[-1]),
    "dt": float(env.model.opt.timestep) * env.frame_skip,
    "ambient_C": env._exo_motor.ambient_C,
    "max_C": env._exo_motor.winding_max_C,
    "tau_s": env._exo_motor.thermal_time_constant_s,
    "t": t.tolist(),
    "temp": np.asarray(log["winding_temp"]).tolist(),
    "cap": np.asarray(log["torque_cap"]).tolist(),
    "delivered": np.abs(np.asarray(log["delivered_hinge_torque"])).max(axis=1).tolist(),
}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(rec, fh)
print(f"{rec['seconds']:.0f} s, {len(t)} steps -> {OUT}")
print(f"  winding  {rec['temp'][0][0]:.1f} -> {rec['temp'][-1][0]:.1f} C "
      f"(limit {rec['max_C']:.0f})")
print(f"  cap      {rec['cap'][0][0]:.1f} -> {rec['cap'][-1][0]:.1f} N.m "
      f"from a {rec['cap'][0][0]:.1f} N.m drive")
