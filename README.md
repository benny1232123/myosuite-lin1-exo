# Lin-I 髋关节外骨骼 · MuJoCo / MyoSuite 仿真

<img src="docs/images/device_front.png" width="420" alt="Lin-I 佩戴在肌骨模型上">

本仓库将 Lin-I 髋关节外骨骼集成到 MyoSuite 人体肌骨模型中。每侧髋关节安装一台 **RobStride RS02** 驱动器，提供被动（无电机）与辅助两种构型。

驱动按电气关系建模：关节力矩由母线电压、转速、绕组电阻与驱动器电流限幅共同决定，而非在关节上设定固定力矩常数。因此，转速升高、电流达到限幅或母线电压下降时，可用力矩随之变化。

以下四组结果依次描述该驱动在仿真中的表现：**力矩输出 → 力矩-转速能力 → 额定力矩 → 发热**。本页仅报告仿真结果，不涉及与真机的一致性验证。

## 一、力矩输出

<img src="docs/images/tor_curve.png" width="760" alt="一个步态周期的力矩、电流、电压">

一个步态周期（1 s）内的驱动量如下：

- **上**：电机实际输出力矩；左右腿各推进一次，峰值 12 N·m。
- **中**：对应电流；峰值 **10.9 A**，为驱动器上限 16 A 的 **67%**。
- **下**：端电压；峰值 **26.6 V**，约为 48 V 母线电压的 55%。

## 二、力矩-转速能力

<img src="docs/images/tor_envelope.png" width="700" alt="力矩-转速能力线">

可用力矩随转速升高而下降。**17 N·m 的平台由驱动器电流限幅决定，并非电机本体极限**：取消电流限幅后，静止状态可输出 37.7 N·m。本仿真仅工作在 0–100 转/分、12 N·m 以内的区域（图中虚线框），远离高速力矩衰减段；因此该工况下的力矩约束来自电流限幅，而非转速。

## 三、额定力矩

<img src="docs/images/tor_rating.png" width="700" alt="12 N·m 与电机峰值、连续上限的关系">

电机峰值力矩 17 N·m 仅可短时输出，连续输出上限为 6 N·m。**处方值 12 N·m 为峰值的 71%，但达到连续上限的两倍**：短时行走可行，长时间运行将导致绕组发热。

## 四、发热

<img src="docs/images/thermal.png" width="700" alt="绕组温度与可用力矩上限">

驱动器持续输出 17 N·m 时，绕组温度自 25 °C 起上升，2 min 后达到 53 °C，可用力矩由 17 N·m 降至 **13.5 N·m**。在 12 N·m 行走处方下，运行 1 min 后绕组温度为 33.6 °C，模型预测稳态约 59 °C，不会触发热降额。

热模型为一阶模型，时间常数 175 s 由说明书中的两个工作点反推（6 N·m 可长期运行；13 N·m 持续 42 s）；温升与力矩平方成正比，6 N·m 时的稳态温度恰为设定的 100 °C 绕组上限。温度在每次 reset 时恢复至室温，因此常规 15 s 验收流程无法观察到该效应，需延长运行时间。

温升同时改变电气参数：铜绕组电阻温度系数为 0.00393/K，100 °C 时电阻为冷态的 1.30 倍；零速下输出 12 N·m 所需电压由 15.3 V 升至 19.8 V。仿真中该电阻在每个仿真步写回 `<dcmotor>`。

## 并排对比

左侧为被动构型（无电机），右侧为 12 N·m 辅助构型；两段视频采用相同机位与同一时间轴：

<img src="docs/videos/walk_compare_loop.gif" width="760" alt="被动与驱动并排">

完整 15 s 对比（左右并排）：[docs/videos/walk_compare_15s.mp4](docs/videos/walk_compare_15s.mp4)

## 电气模型的实现

- 采用 MuJoCo 原生 `<dcmotor>` 模型，包含反电动势、绕组电阻、力矩常数、电流限幅，并将转子惯量折算至关节。
- 力矩常数与绕组电阻取自厂家参数曲线，并通过第二个电压工作点交叉验证（两种路径结果一致度 99.2%）。
- 17 N·m 的上限并非固定写入：输出 17 N·m 时电流达到驱动器上限的 93%，该上限由电流限幅产生。
- 每个仿真步记录电压、电流、饱和标志与实际输出力矩，可用于查询任意处方的电气工作点。

相位前移对受扰起步状态的影响如下：将辅助力矩**提前 0.05 个步态周期**后，12 个受扰起步状态中可完成 15 s 行走的数量由 3 个增至 10 个。将扰动放大 4 倍（换用难度更高的一批状态）后，最优点移至 0.06，增益降至 7 个（对照组 4 个）；若按训练相位 0.30 施加，完成数仍为 **0 个**。这表明相位不匹配的负作用比可恢复状态数更稳定。两批数据见 `docs/data/phase_noise_sweep.json`，可通过 `scripts/frozen_ab_walk_exo.py` 复现。

## 运行

```bash
pip install -e .        # 依赖 mujoco >= 3.16（3.16 与 3.17 均通过全部测试）

# 被动构型行走演示
python scripts/view_walk_exo.py

# 重新生成前三张图（数据位于 docs/data/，为一次 12 N·m 处方、1500 步的驱动日志）
python scripts/make_figures.py

# 热降额演示（120 s、17 N·m）并生成图像
python scripts/thermal_demo.py 120 thermal_run.json 17
python scripts/make_thermal_figure.py --data thermal_run.json

# 12 N·m 辅助构型的验收
python scripts/eval_walk_exo.py --policy finetuned_cal070_a12.zip \
       --assist-torque 12 --assist-phase-lead 0.05 --strap-preload 50 \
       --muscle-force-scale 0.70 --steps 1500

# 在同一批初始状态上比较两种处方（策略权重保持不变）
python scripts/frozen_ab_walk_exo.py --policy finetuned_cal070_a12.zip \
       --assist-torque 0 12 --assist-phase-lead 0.05 --seeds 0 1 2 3 4 5
```

## 目录

```
myosuite/envs/myo/tasks/basic/leg/lin1_exo.py   装置几何 + 电机
myosuite/envs/myo/tasks/basic/leg/walk_exo.py   环境（佩戴 / 驱动 / 辅助旋钮）
myosuite/tests/                                 273 项测试
scripts/                                        查看、验收、A/B、训练、出视频
finetuned_cal070_a12.zip                        本页所有结果用的策略（12 N·m 训练、肌力 0.70）
```

## 已知限制

- 热模型仅时间常数取自说明书；100 °C 绕组上限依据绝缘等级假设，尚未实测。
- 电机空载摩擦提供开关（`exo_drive_friction_Nm`，按 0.5 Arms 折算约 0.56 N·m），默认关闭：现有策略在未启用该摩擦的模型上训练。启用后原策略仍可完成 15 s 行走（21.3 m；关闭时为 21.9 m）。
- 未建模电感。
- 电机参数取自厂家参数表；本仿真不包含真机测量。

---

<sub>English: a CAD-accurate Lin-I hip exoskeleton in MyoSuite with one RobStride RS02 per hip behind
MuJoCo's native DC-motor model, so torque comes out of a voltage command and the driver's current limit
rather than a constant: the delivered hinge torque tracks the command to within 0.56 N·m, drawing 67% of
the driver's current and 55% of the 48 V bus. The 17 N·m plateau is the current limit, not the motor —
an unrestricted stall would reach 37.7 N·m — and a 12 N·m prescription is seven tenths of that peak but
twice the module's continuous rating.</sub>
