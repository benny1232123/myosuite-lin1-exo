# Lin-I — a passive hip exoskeleton in MuJoCo / MyoSuite

A CAD-accurate simulation of the **Lin-I passive hip exoskeleton** worn by the MyoSuite
musculoskeletal walker, plus the tooling to wear it, watch it, and teach a controller to walk in it
with the device completely unpowered.

The device is a hip module on each side of a waist frame: a drum with a flexion and an abduction
hinge, a crank fork that seats on the drum's end cover, a parallel thigh link hanging off the
clevis pin, a pin-in-slot slide and a thigh strap that couples the link to the femur through a
compliant weld. Nothing actuates it — the joints are passive, and the strap drags the link along
the thigh as the human walks.

---

## What it does

| | |
|---|---|
| **Env** | `myoLegWalkExo-v0` — `myoLegWalk-v0` + Lin-I; `myoLegWalkExoLocked-v0` draws the same device rigidly for comparison |
| **Device** | 5 passive DOF per side (abduction, flexion, crank pin, slot, link roll), 3 kg lumped onto the CAD mass distribution, one compliant point weld per side (thigh strap) |
| **Policy** | `walk_exo_policy.zip` — PPO fine-tune of the published `myoLegWalk-v0` policy, zero assistance |
| **Acceptance** | 1500 / 1500 control steps (15 s), 21.9 m, 0 Nm at both exo hinges, alternating foot contacts |
| **Tests** | `myosuite/tests/test_lin1_exo.py` — 17 checks (CAD groups, weld coupling, clevis seating, reset welds, strap drag) |

## Quick start

```bash
pip install -e .                       # MyoSuite, editable

python scripts/view_walk_exo.py                       # watch the shipped policy walk
python scripts/view_walk_exo.py --no-policy           # hold the reset pose, drag to orbit
python scripts/eval_walk_exo.py --policy walk_exo_policy.zip   # the 15 s acceptance run
pytest myosuite/tests/test_lin1_exo.py                # the device's own tests
```

Training a new controller (fine-tune of the published baseline onto the exo env):

```bash
python scripts/finetune_walk_exo.py --resume walk_exo_policy.zip \
       --steps 1500000 --n-envs 4 --target-kl 0.08 --eval-every 50000
python scripts/eval_walk_exo.py --policy finetune_ckpt/ppo_1400000_steps.zip
```

Two gotchas learned the hard way: evaluate SB3 checkpoints with `--policy` (the `--checkpoint`
loader is for the rsl_rl format and silently produces a 43-step fall), and the **final**
checkpoint of a run can regress — sweep `finetune_ckpt/ppo_*_steps.zip` and keep the first one
that passes.

## How the wearing geometry is built

The CAD gives the device's assembly pose, not the pose it is worn in. `lin1_exo.py` places it with
a handful of geometry-only moves (all documented next to their constants):

* **flexion / abduction hinges** are seeded per keyframe so the strap weld starts closed instead of
  kicking the walker on the first control step;
* the **crank fork and the drum's end cover clock together about the motor axis** — the cover's
  clock reads the same flexion value, so one seed turns both;
* the **thigh rod turns about the clevis bolt** (`_ROD_HINGE_DEG`) — the only rotation that keeps
  its knuckle in the fork's lugs — and the **strap ring is seated on the rod's distal buckle**
  (`seat_cuff_on_rod`), so the strap is worn where the bar reaches;
* the stance is narrowed in every keyframe so the legs stand the way the device is worn.

These are mesh placements: the collision model, the joint set and the weld are untouched, so the
physics stays that of the passive device.

## Layout

```
myosuite/envs/myo/tasks/basic/leg/lin1_exo.py   device: CAD parts, hinges, strap weld, placement
myosuite/envs/myo/tasks/basic/leg/walk_exo.py   the two exo envs + wearing knobs
myosuite/envs/myo/assets/leg/lin1_exo/meshes/   STL parts exported from SolidWorks
myosuite/tests/test_lin1_exo.py                 device tests
scripts/view_walk_exo.py                        interactive viewer
scripts/eval_walk_exo.py                        15 s acceptance run
scripts/finetune_walk_exo.py                    PPO fine-tune onto the exo env
scripts/diag_exo_*.py                           diagnostics used while tuning
walk_exo_policy.zip                             the shipped 15 s controller
```

## Upstream

This is a fork of [MyoSuite](https://github.com/MyoHub/myosuite) (Apache-2.0, see `LICENSE`).
Everything outside `lin1_exo.py`, `walk_exo.py`, the `lin1_exo` assets, the exo scripts/tests and
this README is upstream MyoSuite; see its documentation for the musculoskeletal models, tasks and
the RL tooling this work builds on.

---

<sub>中文摘要：Lin-I 被动髋外骨骼在 MyoSuite 里的 CAD 级仿真——每侧 5 个被动自由度、大腿绑带以柔性焊缝耦合，电机零输出。仓库自带能稳定行走 15 s（1500 步 / 21.9 m）的策略、验收脚本、交互查看器和 17 项设备测试。</sub>
