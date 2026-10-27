# 原速、原夹爪的统一参考重构

2026-10-23。用户明确要求保留原始20Hz时序和原夹爪命令，不以S2增加训练帧数和执行步数，不增加夹爪接口。后续以原速统一动作/未来state引导为优先路线。S2数据仍是已验证历史产物，不代表当前原速方案已经解决。

所有新实验复用维护中的LIBERO recorder、原生500Hz物理、严格IPOPT/B10完整7维关节参考，task kp=150、null kp=10，容差关闭。没有中途状态重置、OSC失败回退或把离线关节见证传入在线控制器。只改离线末端参考，随后差分为物理米/弧度动作，参考独立累积。原夹爪命令逐帧保持不变，完整执行N-1个原50ms区间，不删除等待、不额外settle。

## 固定六条诊断示范

`plans/original_speed_guidance6.json`：Spatial04/demo_0、Spatial04/demo_22、Object00/demo_0、Goal00/demo_23、Goal05/demo_11、Long08/demo_1。有意覆盖困难抓取、推压和一条原OSC失败示范；不是总体随机成功率样本。

## 方法

1. S1实际state相邻FK参考作为基线。
2. 固定前视：每帧取150ms更远的实际末端位姿，即从常规next-state再向前3格，不延长总时长。
3. PD前馈：从实际轨迹估计速度、加速度，采用r=p+(kd/kp)v+a/kp；姿态作相应SO(3)处理。这只是局部连续PD模型的近似，不是接触动力学的精确逆解。
4. 全局原动作意图投影：每一帧从原完整state FK和原动作构造反馈相对目标（位置0.05m、旋转0.5rad缩放；零旋转保持目标历史），统一拟合有界关节解并取其FK作为新参考。没有接触阶段分类。由于完整state FK与原生控制缓存有时差、初始保持方向从FK初始化，不声称与录制时内部OSC目标严格等效。

前视/PD探针：`probe_original_speed_reference.py`。全局原动作投影：`probe_projected_action_reference.py`。每个方法都采用同一规则处理全部六条，而不是只在失败示范上启用。

## 已完成首轮

- S1实际state基线：1/6成功，无拒解（四条复用完全相同时序的已有记录，两条本轮重录）。
- 固定前视：2/6成功，无拒解。Spatial04/demo_22由失败转成功；Spatial04/demo_0仍失败。
- PD前馈：1/6成功，无拒解。推盘即便实际轨迹误差仅几毫米，仍可能任务失败，说明运动预瞄不自动保留接触意图。
- 全局原动作意图投影、极弱构型正则1e-5：4/6成功，两条Spatial04后段IPOPT拒解。Object00、Goal00、Goal05、Long08均成功，平均相邻源实际state位置误差分别约0.39、0.86、3.01、0.85mm。

上述4/6没有使用不同候选逐条拼接。拒解前缀的小误差也不能算成整段成功。

## 拒解诊断与后续

两条Spatial04的投影关节见证未贴模型硬限位，但靠近伸展奇异构型，q4约-0.467rad。拒解目标见证的6x7几何雅可比最小奇异值分别约8.23e-7、1.80e-6；对应源state约0.0624、0.00180。最近几何投影可能把不可达虚拟目标投到可达域边界，虽有合法q存在，仍难以被在线局部IPOPT稳定求出。

本轮追加统一构型正则0.01，全部六条重新执行，6/6成功且无拒解。冻结参数后新增Spatial04/demo_2、Goal02/demo_1、Spatial07/demo_6、Long06/demo_45、Long00/demo_10，4/5成功且无拒解。Long00/demo_10全程完成但任务从未成功；失败记录保留。合计同一方法10/11，不是逐条挑选候选后的恢复率。新增样本没有参与本轮0.01调参，但部分是历史已知困难示范，不是从未查看过的随机测试集。

目标仍由原动作意图定义，正则只在离线拟合时鼓励靠近源示范构型（具体seed为下一源state的q），不修改在线IPOPT目标精度和求解预算。统一每帧最小化的残差为：

```text
[ FK_position(q)-OSC_intent_position,
  0.1 * Log(FK_rotation(q) * OSC_intent_rotation.T),
  0.01 * (q-q_source_next) ]
```

有界最小二乘之后取FK作为在线严格目标。此处0.1、0.01是不同单位残差的缩放，整体目标为残差平方和；不是声称目标严格等于原OSC目标，也没有给出避免奇异的保证。

## 跟踪指标必须分开

10条成功示范共1825帧。相同时间下实际末端对原示范下一实际state的误差：位置均值1.66mm、p95 8.22mm，姿态均值0.307度、p95 1.286度。各成功示范平均位置误差0.35–5.63mm。

实际末端对新控制参考的误差仍为位置均值24.56mm、p95 44.15mm，姿态均值2.23度。这是包含运动前馈和接触预载的控制目标，不能把“原示范轨迹恢复到毫米级”误称为“对控制参考的毫米级跟踪”。原速任务效果恢复显著，但若需求是实际末端紧贴控制参考，仍需单独改控制接口/前馈；本轮没有改底层控制器。

## 数据与核验

统一原速数据包：`data/libero_tasktol_state/20261023_original_speed_reg001_11_v1/manifest.json`。11次尝试、10条成功、9任务尝试、8任务有成功数据。失败只留在attempts，不进入训练data。复用原生重录图像和观测，训练状态仍按现有打包器对齐，不挪用源图像。所有11条均通过`analyze_state_reference.py`核验动作、原始夹爪、源hash、严格求解、关节见证FK、物理时间和视频帧数。

成功动作冷启动回放：`results/original_speed_reg001_10_cold_v1/`。10/10成功，完整动作前物理state、最终state、IPOPT完整关节参考均逐位一致。每次在新进程初始化后只给客户端保存动作、初始state和参考锚点，不提供源未来state或离线关节见证。原夹爪值保持不变；已有通用解码器对原±1没有改变。数据包`validation.json`另行核验物理时序、观测、动作参考重建及回放证据。此结论是保存动作复现，不是学习后VLA成功率或扰动泛化。

所有物理录制、参考、逐帧误差、失败、视频、初末帧、源文件快照与hash保留在：

- `results/original_speed_guidance6_v1/`
- `results/original_speed_projected_action6_v1/`
- `results/original_speed_projected_action6_reg001_v1/`
- `results/original_speed_projected_action_new5_reg001_v1/`

文献依据：UMI提供未来相对轨迹和延迟匹配先例（https://arxiv.org/html/2402.10329v3）；实际轨迹差分参考Bridge/OpenVLA的Bridge重标注（https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/rlds/utils/data_utils.py）。这些先例不构成本文原速闭环成功保证。

## 2026-10-24 后续更新

已按IRLC和PhysicsGen/MJPC思路增加实际闭环state反馈优化。原速同11条诊断示范在离线试录后11/11成功，综合state代价下降约48.5%；关节位置误差基本未改善，物体轨迹改善明显。详细固定规则、成本和适用范围见`CLOSED_LOOP_STATE_REFINEMENT_METHOD.md`，勿将离线搜索结果写作单次或训练后策略成功率。

## 2026-10-24 用户约束修正后的当前版本

上述综合state代价包含原机器人关节跟踪项，现已按用户要求废止该验收目标。当前外层只评分实际末端与任务物体轨迹，不处罚机械臂/夹爪关节偏差。另发现历史LIBERO适配层将IPOPT预算扩大为60次/0.5秒；当前显式调用未修改的库默认`IpoptSettings()`（20次/0.095秒），默认目标和共享源码保留原样。

历史动作候选经任务代价重新选择后，用默认IPOPT重新录制11/11成功、零拒解；独立冷回放11/11成功且物理state/关节参考逐位一致。新数据包为`data/libero_tasktol_state/20261024_task_pose_default_ipopt11_v1/`，当前方法及初始化限制见`TASK_POSE_DEFAULT_IPOPT_METHOD.md`。历史结果和旧目标文件仅作为实验记录保留。
