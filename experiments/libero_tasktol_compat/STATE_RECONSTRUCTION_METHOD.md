# 适用于IPOPT/B10的LIBERO数据重构方法

日期：2026-10-23。方法已实现并在40任务的48条示范、冻结规则后另取的8条示范上完成重构。当前主张是**离线数据重构可行**，不等同于训练后VLA的任务成功率，也不承诺所有原始示范均可恢复。

后续用户约束已更新：正式方案优先保持原速20Hz、原始夹爪命令，不通过S2增加训练帧和执行时长，也不增加夹爪脉冲/目标接口。本文件保留已验证S2版本作为历史基线；新的原速统一参考实验见`ORIGINAL_SPEED_RECONSTRUCTION_PROGRESS.md`，尚不能将S2结果作为原速方法的成功率。

## 方法

每个源时间格为50ms。读取动作前完整MuJoCo state，不用有2ms缓存时差的末端观测来计算FK。对源关节路径作S2线性插值，每100ms经过一个原路径节点，执行指令仍为20Hz。末端参考为插值关节路径的FK；动作采用相邻参考的平移差及SO(3)对数差：

```text
delta_p[k] = p_ref[k+1] - p_ref[k]
delta_r[k] = Log(R_ref[k+1] @ R_ref[k].T)
action[k] = [delta_p[k], delta_r[k], gripper[k]]
```

参考从初始实际FK锚定并独立累加；每帧共享IPOPT求出完整7维关节参考，再由B10复用原生LIBERO执行循环。没有跟随原始关节目标、动态重置state或用OSC替代失败求解。关节插值只用于离线定义末端标签，执行端不读取该关节路径。

源state缺少最终完整后继时裁去最后一个未知转移；保留其余全部时间格和等待段。额外settle为0，避免开始时已经偏离源状态。

## 固定三候选离线搜索

按以下顺序，每次从同一初始state重新重录整段；第一次整段IPOPT均可执行、最终LIBERO判定成功的版本才进入训练组。

1. **state运动参考＋夹爪积分时序适配。** 原生Panda每2ms按sign(command)累加夹爪目标；减速后在新增20Hz时间格发送0保持，保留每个源动作周期25次有效积分更新。
2. **相同state运动参考＋原夹爪意图持续执行。** 某些接触需要持续闭合，该候选保留原±1。仅夹爪时序不同。
3. **state运动参考＋持续张手接触偏置＋夹爪时序适配。** 由原完整state重新计算夹爪与外部物体接触，筛出前后各4个源格均保持张手的接触，向前后3格渐变。在这些窗口才保留原动作的局部位姿偏置；其他阶段仍由state定义参考。用有界关节拟合获得见证，再以其FK定义新目标，解决原偏置目标可能超出硬关节范围的问题。

第三项有意改变参考，不声称与原OSC内部goal等效；零旋转不复刻原OSC历史保持状态。默认严格关节边界内缩1e-4rad，投影位置/旋转残差保存在参考文件。投影过程中没有放宽在线IPOPT任务精度。

所有候选使用同一task kp=150、null kp=10、容差关闭的B10客户端。选择发生在离线数据生成时，最终客户端不做基于成功谓词的候选切换。三次均失败时保留失败记录并拒绝接纳该示范；没有把失败录成成功。

## 证据与正确指标

| 范围 | 第一个候选成功 | 最多三候选后的恢复 | 完整重录次数 |
|---|---:|---:|---:|
| 主批：40任务48条示范 | 44/48 | 48/48 | 55 |
| 冻结规则后新增：4个较难任务各demo_0、demo_1 | 4/8 | 8/8 | 16 |
| 合计 | 48/56 | 56/56 | 71 |

主批有44条选第一候选、1条选第二候选、3条选第三候选；新增批4条选第一候选、4条选第三候选。最终成功重录共19416帧，每帧都通过严格IPOPT，共享客户端明确传输完整7维q_d。第三候选意味着**部分保留原动作的接触意图**，不能宣称完全舍弃原始动作已解决所有任务。

主批代表示范原先经过任务可视化选择，故不是2000条源数据的随机总体估计；新增批不根据这些示范的成功状态挑选。两批没有重复示范，但新增批是同任务的新示范，不是新任务泛化。

数据构造和客户端冷启动分别验证。冷启动只读保存动作、初始完整state和参考锚点；原记录及关节目标只在执行后用于比较。56/56冷启动成功，实际完整state、最终state和完整q_d均逐位复现。数据包`validation.json`确认原始输入hash、观测时序、动作重建、训练状态对齐和所有71次尝试的归档；35项相关测试通过。

56条中仅7条采用接触候选，共756/19416个目标帧（约3.9%）混入原运动偏置，其余目标主要由实际state构造。这里统计目标帧，不把包含夹爪意图的数据称为完全无原动作。

## 数据与训练接口

- 主批：`data/libero_tasktol_state/20261023_reconstruction48_v1/manifest.json`。
- 新增批：`data/libero_tasktol_state/20261023_reconstruction_new8_v1/manifest.json`。
- 每任务一个`dataset.hdf5`。`data/demo_*`只含成功重录，`attempts`与`reconstruction_trials`保留失败候选；`references`含离线参考，`controller_targets`含执行时的q_d供核验。
- `obs`为该IPOPT执行产生的新图像/原生观测，不挪用原示范图像。`states`和`tasktol/pre_action_states`为实际动作前完整物理state。
- `policy_state`为8维`[末端xyz, axis-angle, 两指qpos]`，来自相同原生动作前观测。原生关节/末端缓存滞后最终物理state一个2ms步，首帧forward除外；这与客户端观测一致。
- 原重录器为上游兼容保留了不同来源的首帧`robot_states`；训练视图改用实际`pre_action_robot_states`，旧数组留作`legacy_robot_states`。原录制分片不修改。
- 前6维动作单位是米和弧度，已是独立参考差分。关闭额外`DeltaActions`，不要再按原生OSC乘0.05/0.5；对新数据重新计算归一化统计。
- 夹爪有−1/0/+1三种意图。策略输出反归一化后，用`examples/libero/tasktol_actions.py`的0.5死区解码，保留0保持。直接将近零连续值交给原生sign积分器会错误地变成全速开合。注入±0.2保持噪声的冷启动单例已逐位复现。
- 图像保留LIBERO原方向；导出给VLA时统一旋转180度一次。HDF5外链保留数据分片原地址，资产也继续使用原地址。

## 从新示范重构到验收

准备JSON计划，每项给出`case`（selection_manifest的0–39索引）、`suite`、`task_id`、`demo`。输入源HDF5保持只读，现有重录ledger提供可复现的初始化历史；不需要部署模型或重装环境。

```bash
cd /media/czl/sata/franka_my_code/openpi
TASKTOL_PY=/media/czl/sata/franka_my_code/franka_mujoco/experiments/libero_closed_loop/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

"$TASKTOL_PY" experiments/libero_tasktol_compat/reconstruct_state_dataset.py \
  --plan /absolute/path/new_plan.json --output /absolute/path/new_reconstruction --workers 3

"$TASKTOL_PY" experiments/libero_tasktol_compat/build_state_reference_dataset.py \
  --reconstruction /absolute/path/new_reconstruction --output /absolute/path/new_dataset

"$TASKTOL_PY" experiments/libero_tasktol_compat/replay_state_dataset_batch.py \
  --dataset /absolute/path/new_dataset --output /absolute/path/new_cold_replay --workers 3

"$TASKTOL_PY" experiments/libero_tasktol_compat/validate_state_dataset.py \
  --dataset /absolute/path/new_dataset --cold /absolute/path/new_cold_replay
```

中断后对同一重构命令加`--resume`，先确认原进程已终止；它校验候选实现、复用完整结果、归档未完成候选。冷启动不依赖训练数据里的未来state；参考和关节见证是离线标签与诊断信息。

## 适用范围

已验证的是LIBERO原生环境中，重构后的参考经严格IPOPT/B10闭环控制可执行，并能生成重新对齐的训练数据。没有验证训练后的VLA成功率、真实机器人性能或整个LIBERO源数据集恢复率。

20Hz指仿真指令时序，不是硬实时保证。56条选中重录在并行批次中的求解中位15.23ms、p95 53.94ms、最大288.96ms；1467/19416帧超过50ms。在线真实墙钟时限仍需单独做时序设计，不能把仿真完成率当作实时调度验收。
