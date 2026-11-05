# Lin-I 髋外骨骼 · MuJoCo / MyoSuite 仿真

<img src="docs/images/device_front.png" width="420" alt="Lin-I 佩戴在肌骨模型上">

把 **Lin-I 髋关节外骨骼**搬进 MyoSuite 的人体肌骨模型。装置有两个版本：不带电机的**被动版**，
和每侧髋装一台 **RobStride RS02** 电机的**驱动版**。

驱动版的重点是电气：电机不是"直接给一个力矩"，而是给**电压**，由反电动势、绕组电阻和驱动器
23 A 限流算出力矩——这样仿真里的力矩输出才有真机的脾性。

## 三个结论

**1. 辅助要在对的时刻给。** 同样是 12 N·m，脉冲落点提前 0.25 个步态周期，能走满 15 秒的初态就从
3/12 变成 10/12，而策略一个字都没改。专门在那个时刻再训 15 万步，反而掉回 6/12——缺的不是训练。

<img src="docs/images/fig_phase_lead.png" width="600" alt="相位超前与存活数">

**2. 力矩再大也不会更好。** 0 → 17 N·m 扫下来，走得最好的反而是 0 N·m。瓶颈不在电机而在绑带：
夹紧式袖带能把约 1.0 的力矩传到髋，原来的点焊只有 0.12，电机给的力矩大部分没进到人身上。

<img src="docs/images/fig_torque_ceiling.png" width="580" alt="指令力矩与存活数">

**3. 12 N·m 只用掉电机三分之二的能力。** 峰值电流 15.5 A（上限 23 A）、关节 RMS 力矩 4.0 N·m
（连续额定 6 N·m）、母线电压 26.6 V（48 V 母线）。换成 17 N·m 就到了 93% / 95%，只能短时用。

<img src="docs/images/fig_rs02_envelope.png" width="100%" alt="处方与 RS02 电气、热、电压边界">

## 并排看

左边不带电机，右边 12 N·m 辅助（lead 0.05）。同一机位、同一条时间轴，只有驱动不同：

<img src="docs/videos/walk_compare_loop.gif" width="760" alt="被动与驱动并排">

完整 15 秒（左右并排）：[docs/videos/walk_compare_15s.mp4](docs/videos/walk_compare_15s.mp4)

## 电气模型怎么做的

- 用 MuJoCo 原生 `<dcmotor>`：反电动势、绕组电阻、力矩常数、电流限幅、转子惯量折合到关节。
- 力矩常数 K 和绕组电阻 R 都是**从说明书曲线拟合**的，并用第二个电压点交叉验证（两条路径 99.2% 一致）。
- 所以 17 N·m 这个"峰值"不是写死的常数：仿真跑到 17 N·m 时峰值电流 21.5 A，正好是 23 A 的 93%，
  是限流造出来的。
- 每一步都记录电压、电流、是否饱和、实际输出力矩，任何一条处方都能拿去和真机边界对账。

## 怎么跑

```bash
pip install -e .

# 看被动版行走
python scripts/view_walk_exo.py

# 带电机、12 N·m @ lead 0.05 的验收
python scripts/eval_walk_exo.py --policy finetuned_cal070_a12.zip \
       --assist-torque 12 --assist-phase-lead 0.05 --strap-preload 50 \
       --muscle-force-scale 0.70 --steps 1500

# 在同一批初态上比较两种处方（策略权重不动）
python scripts/frozen_ab_walk_exo.py --policy finetuned_cal070_a12.zip \
       --assist-torque 0 12 --assist-phase-lead 0.05 --seeds 0 1 2 3 4 5
```

## 目录

```
myosuite/envs/myo/tasks/basic/leg/lin1_exo.py   装置几何 + 电机
myosuite/envs/myo/tasks/basic/leg/walk_exo.py   环境（佩戴 / 驱动 / 辅助旋钮）
myosuite/tests/                                 266 项测试
scripts/                                        查看、验收、A/B、训练、出视频
```

## 还没做的

- **发热没建模**。上面说的"连续 6 N·m"是额定值；走久了会不会过热，仿真里看不到。
- 空载摩擦（关节上约 0.6 N·m）和电感没有建模。
- 电气参数目前只有说明书这一条验证路径，还没有真机台架对照。

---

<sub>English: a CAD-accurate Lin-I hip exoskeleton in MyoSuite, passive and with one RobStride RS02 per hip
behind MuJoCo's native DC-motor model, so torque comes out of a voltage command and a 23 A limit rather
than a constant — the simulated 17 N·m peak draws 21.5 A, 93% of the driver's limit. Moving the assist
pulse from the training default to lead 0.05 takes the same frozen policy from 3/12 to 10/12 perturbed
initial states, while more torque buys nothing because the cuff, not the motor, is the ceiling.</sub>
