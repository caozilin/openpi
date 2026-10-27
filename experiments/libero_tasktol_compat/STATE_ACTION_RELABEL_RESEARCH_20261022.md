# 以实际state重标注动作：文献与本机证据

调研日期：2026-10-22。这里只做文献、官方代码核对与既有结果复查，没有启动新的重录批次，也没有修改数据或控制器。

## 初步结论

这种方向已有直接工程先例与研究支撑。最接近的先例是BridgeData V2及OpenVLA的Bridge预处理：前6维运动动作由相邻已到达state差分替换，夹爪动作另行保留/处理。它证明这种标签定义可以用于机器人模仿学习，不保证把同样差分直接输入任意低层控制器就能复现相同转移。

我们应把两件事分开：一是从已实现state定义新的运动目标，二是在目标控制器中找出能按时实现这些目标的命令。几何可达性、有限时间跟踪能力、接触过程一致性需要分别验证。

## 主要一手来源

### 1. BridgeData V2 / OpenVLA：直接的state差分重标注

Bridge官方数据加载器默认启用relabel_actions，用next_observations与observations的前6维proprio差构造运动标签，夹爪单独二值化。OpenVLA的relabel_bridge_actions同样使用相邻state前6维差，保留原夹爪列并裁去无下一状态的末帧；Bridge原始格式和OXE格式的转换都调用该函数。这是本次问题最直接的先例，属于Bridge数据处理，不是LIBERO重录器对原动作的处理。

来源：[Bridge官方代码，_process_actions](https://github.com/rail-berkeley/bridge_data_v2/blob/main/jaxrl_m/data/bridge_dataset.py#L274)、[OpenVLA重标注函数](https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/rlds/utils/data_utils.py#L148)、[OpenVLA Bridge转换调用](https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/rlds/oxe/transforms.py#L28)。本次已核对源码，不仅依据issue讨论。

### 2. UMI（RSS 2024）：从实测轨迹产生目标，并处理延迟

UMI从手持示范恢复末端运动轨迹，以当前末端为共同锚点表示未来相对位姿序列；训练数据不依赖原机器人的电机动作。部署时标定观测与执行延迟、提前发送未来目标。动态抛掷实验中，去掉延迟匹配后成功率从87.5%降到57.5%。这支持轨迹监督的可行性，也表明不能忽略跟踪时序。UMI并非“相邻差分直接作为任意控制器命令”的证明。

来源：[论文III-B与V-B](https://arxiv.org/html/2402.10329v3#S3.SS2)、[官方数据集实现](https://github.com/real-stanford/universal_manipulation_interface/blob/main/diffusion_policy/dataset/umi_dataset.py)。

### 3. HYDRA（CoRL 2023）：支持部分重标注，保留接触动作

HYDRA在稀疏路点阶段，用部署控制器根据示范state和未来waypoint重新生成动作；精细交互的dense阶段保留原动作。论文明确没有给出dense阶段的统一重标注办法，并指出离线换动作可能改变在线状态分布。它支持按阶段逐步重标注，不能作为全阶段简单差分已获保证的证据。

来源：[论文4.1、附录B与D.4](https://arxiv.org/html/2306.17237v2#S4.SS1)。

### 4. BCO（IJCAI 2018）：从状态转移反推动作

Behavioral Cloning from Observation先通过环境交互学习逆动力学模型，再由示范的相邻state推断动作并做行为克隆。它研究了完全缺少原动作的情形，在Reacher、Ant等MuJoCo任务上验证；不是LIBERO接触操作的直接验证。与裸差分相比，这条路线显式学习目标环境的动作响应。

来源：[论文第4节](https://arxiv.org/html/1805.01954v2#S4)。

### 补充：AWE（CoRL 2023）

Automatic Waypoint Extraction按轨迹重建误差从示范提取少量waypoint，支持以未来目标代替逐步低层命令的思路。其重点是路点提取与策略学习，不是保证50ms内到达相邻state。

来源：[论文](https://proceedings.mlr.press/v229/shi23b/shi23b.pdf)、[作者实现](https://github.com/lucys0/awe)。

## 对本项目应采用的标签定义

令p_t、R_t来自原始动作前state的关节FK，第一版只替换前6维，保留原始夹爪g_t：

```text
Δp_t = p_(t+1) - p_t
Δr_t = Log(R_(t+1) R_t^T)
a_state_t = [Δp_t, Δr_t, g_t]
```

这表示“下一实际位姿目标”，不是恢复原OSC操纵命令。当前TaskTol接口接受米/弧度物理增量，不需要除以dt；若另用速度接口才需要。LIBERO旋转必须按SO(3)差分，不能照搬Bridge中前6维向量直接相减的实现。

用state[0]作为参考锚点累加这些增量，离散参考可重建到FK(q_(t+1))。若q_(t+1)满足IPOPT硬关节限位，它就是该目标的现成存在性见证；但不保证求解器局部收敛、连续插值可达、50ms到位或接触过程相同。

仍需正确对齐原始state与图像。原LIBERO的obs是动作后观测，末端缓存还有一个2ms物理步时差，而state是动作前完整物理状态。本项目应继续用完整state的FK。缺少最终完整state时裁去末帧，不能把同帧obs误当下一帧state。静止或近零运动的接触/等待样本仍有物理意义，不能因为新差分接近零就自动删掉其时间。

## 为什么不要求零延迟，但必须匹配响应

以下是本项目的简化分析，不是论文中的实验结论。假设一维控制响应为：

```text
x_(t+1) = x_t + α (r_t - x_t),   0 < α < 1
```

若把已实现下一状态直接设为新目标r'_t=x_(t+1)，则同周期实际位移变成α[x_(t+1)-x_t]，比示范位移再缩小一次。例如原目标前进10mm、实际走3mm，直接用3mm作新目标，同样响应可能只走0.9mm。真实系统更复杂，但这个例子说明“实现位移”不自动等于“能再次实现该位移的控制命令”。

可选择的处理是：测量跟踪滞后并对未来轨迹做前视/时序补偿；用目标环境的逆动力学或短窗优化反推动作；或者在允许改变任务节奏的阶段使用带反馈的路点跟踪。不能仅依赖提高关节增益，动态接触与夹爪时序也要一起验证。UMI提供延迟匹配的实证，BCO提供逆动力学路线的先例。

接触时尤其要区分位移与控制作用：末端几乎不动，也可能正在顶压、夹持或推开约束。把这些动作替换为近零位移可能丢失目标偏置对应的力。只有机器人末端位姿标签，也不足以唯一确定物体转移和接触力。

## 本机已有证据与新增静态检查

回查旧recorded_motion分支，参考来自重录数据中相邻实际关节state的FK差分，夹爪沿用原动作，没有直接回放源关节指令：

| 旧实验增益kp | 最终成功 |
|---|---:|
| 150 | 37/40 |
| 300 | 37/40 |
| 1000 | 38/40 |
| 2000 | 36/40 |
| 4000 | 39/40 |

来源为results/ipopt_source_context_recorded_kp150_all40_v3、recorded_gain_all40_v4和recorded_gain_all40_v5的原有summary/report。它们是OpenVLA重录state、旧关节控制链、容差关闭的离线轨迹跟踪，不是当前原始state＋B10，也不是VLA策略准确率。增益结果不单调，不能据此推荐kp4000作为正式默认。

本次只新增了原始state限位静态检查：上轮12条原始示范共有2560个state，12帧的第4关节轻微超出模型标称关节范围，最大约0.396度。其余2548帧满足标称限位。这意味着实际仿真状态不能无条件作为“满足IPOPT硬限位”的见证。超限state的末端目标仍可能有其他合法逆解，应标记并另求解，不能直接判不可达，也不能直接裁剪q冒充原位姿。

MuJoCo采用软约束，允许数值状态有约束违反；这与IPOPT的硬关节不等式语义不同。参见[MuJoCo约束模型说明](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)。

原始检查记录：[original_state_joint_bounds.json](results/state_action_relabel_research_20261022/original_state_joint_bounds.json)、[具体超限采样](results/state_action_relabel_research_20261022/out_of_bounds_state_details.json)。

## 建议的最小验证顺序

1. 在当前12条配对示范上，只替换运动标签为state-next增量，保持原夹爪；先固定控制器、初始化和时序，以免把多种改变混在一起。逐帧记录合法q见证与超限例外。
2. 分别用OSC和IPOPT/B10执行同一冻结实际轨迹，保持完整7维IPOPT关节参考。对比严格求解通过率、实际跟踪时滞、物体/接触变化和任务成功率。
3. 单独比较50/100/200ms未来目标或小段轨迹的时序补偿，不把单帧差分有效作为默认假设；若使用未来数据，只用于离线标签或策略预测的未来chunk，不能给在线客户端提供真值未来state。
4. 按自由运动与接触阶段分别分析。必要时采用HYDRA式部分重标注，或对接触段进行目标控制器下的逆动力学/轨迹优化。是否舍弃全部原始运动动作，由这些对照决定。

文献支撑的是这种标签与控制接口路线值得验证；当前LIBERO＋B10闭环成功率仍须通过上述独立对照确认。本次调研没有把原始目标重映射12条的4/12结果误当成state差分结果：两者的目标定义不同。
