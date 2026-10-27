# LIBERO OSC 参考目标差分重标注

本实验只使用原生 OSC，不运行 IPOPT，也不修改容差算法或原始 HDF5。
运行环境是 `franka_mujoco/experiments/libero_closed_loop/.venv`，固定 MuJoCo 3.2.3；
LIBERO 代码和资产仍从 OpenPI 的 `third_party/libero` 原地址加载。

## 目标和动作定义

对每个原始保留动作 `a[t]`，从 `tasktol/pre_action_states[t]` 的完整实际 qpos
计算末端位姿。动作经过原生 OSC 的缩放后，位置目标为
`G_p[t] = actual_p[t] + scaled_action[t, :3]`。
旋转动作非零时，姿态目标为
`G_R[t] = Exp(scaled_action[t, 3:6]) @ actual_R[t]`；旋转动作全零时保留上一姿态目标。
第一帧的保持姿态来自原生控制器 reset，不能用 settling 后的实际姿态代替。

保存明确的初始实际位姿锚点 `P0, R0`，重标注动作定义为：

- 首帧：`dp[0] = G_p[0] - P0`，`dr[0] = Log(G_R[0] @ R0.T)`。
- 后续：`dp[t] = G_p[t] - G_p[t-1]`，`dr[t] = Log(G_R[t] @ G_R[t-1].T)`。
- 夹爪动作原样保留，时间索引与原始 pre-action 图像/state 对齐。

动作单位是米和世界坐标系旋转向量弧度，不能作为原生归一化 OSC delta 动作直接发送。
姿态使用旋转矩阵组合和 SO(3) 对数，不相减欧拉角或绝对旋转向量。

## 成对仿真

`native_alignment.py` 提供两种诊断输入模式，均通过同一个原生 `env.step()` 执行：

- `feedback_delta`：原动作交给原生 OSC，目标相对当前实际反馈生成。
- `accumulated_reference`：从保存的锚点出发，每周期调用原 TaskTol
  `advance_reference_pose` 累加重标注动作，再通过原生 OSC 的绝对目标模式执行。
  累加分支不由新执行的实际位姿重置，不查询未来动作。

目标统一采用 TaskTol 工具姿态；发送 OSC 时使用标定的固定工具旋转偏移还原 grip-site 姿态。
保持相同实际初态、原生 reset/RNG/fixture 历史、夹爪累积器、kp/kd、20Hz 控制与 25 个原生物理步。
原生绝对姿态接口内部的 float32 四元数转换会带来微小数值误差，日志单独记录目标角度误差。

这是固定演示动作驱动的原生状态反馈控制实验，不是训练后的 VLA 策略闭环评估。

## 文件

- `osc_reference.py`：目标解码、差分重标注和共享参考积分器调用。
- `osc_reference_test.py`：反馈相对目标、零旋转保持、首帧锚点和非交换旋转的重建测试。
- `native_alignment.py`：两种输入模式的原生仿真，保存重标注数组、实际状态、视频和初末图片。
- `compare_osc_reference.py`：成对检查与末端/关节/完整状态差异的 JSON 汇总。
- `relabel_osc_dataset.py`：批量导出全部保留演示的独立 HDF5 标注文件，仅执行正运动学。

新 HDF5 的 `data/demo_*/actions` 是新定义的物理参考增量，`anchor_*` 是显式初始参考，
`goal_*` 保存原解码目标。`source_demo` 是到原演示的 HDF5 外部链接，可读取原图像、state 和动作，
不复制图像、不改写源数据。文件属性 `action_contract` 和 manifest 记录语义与输入指纹。
批量重标注依赖原生机器人初始化噪声为零；代码遇到非零噪声会停止，要求逐 episode 恢复初始保持姿态。

这批文件来自已有 OSC 演示的离线动作重标注，不是 IPOPT 重录训练数据。
后续训练/客户端需明确使用新动作语义及对应归一化，不能沿用旧 OSC 动作解释。

## 运行示例

在 OpenPI 根目录设置 `PYTHONPATH` 包含 TaskTol 和 OpenPI，并使用上述固定虚拟环境的 Python。
仿真同时设置 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONDONTWRITEBYTECODE=1`。

```bash
python experiments/libero_tasktol_compat/native_alignment.py \
  --output experiments/libero_tasktol_compat/results/osc_reference_feedback40_v1 \
  --workers 2 --refresh-controller --source-reset-state --source-reset-history \
  --osc-reference-mode feedback_delta

python experiments/libero_tasktol_compat/native_alignment.py \
  --output experiments/libero_tasktol_compat/results/osc_reference_accumulated40_v1 \
  --workers 2 --refresh-controller --source-reset-state --source-reset-history \
  --osc-reference-mode accumulated_reference

python experiments/libero_tasktol_compat/relabel_osc_dataset.py \
  --output data/libero_reference_actions/20261020_osc_reference_v1 --workers 2

python experiments/libero_tasktol_compat/compare_osc_reference.py \
  --feedback experiments/libero_tasktol_compat/results/osc_reference_feedback40_v1 \
  --accumulated experiments/libero_tasktol_compat/results/osc_reference_accumulated40_v1 \
  --output experiments/libero_tasktol_compat/results/osc_reference_comparison_v1.json
```

实验不创建或打开 HTML 报告；过程证据保存在本地数组、图片、视频和 JSON 中。

## 2026-10-20 验证结果

全部 40 个任务文件、1699 条保留演示、275336 帧已生成独立标注。
读回后的 8495 项逐 episode 检查全部通过，40 条仿真代表演示的批量标注与仿真输入也全部匹配。
参考重建最大位置误差为 2.78e-17 m，最大角度误差为 5.23e-15 rad。

40 条固定代表演示共 6544 个执行周期：原始反馈相对 OSC 与重标注后的累加参考 OSC 均为 39/40，
共同失败任务为 `libero_10/03`，成功判定没有变化。
两条实际执行轨迹的末端位置平均差异为 0.04224 mm，p95 为 0.21457 mm，最大为 3.92729 mm；
最大差异发生在共同失败的长任务。姿态平均差异为 0.01166 度，最大为 3.06748 度。
80 段视频全部完整解码通过；成对初始化、增益、目标重建、物理步数等 360 项检查全部通过。

本次标注的目标与此前 `source_goals` IPOPT 实验中的名义目标在数值精度内一致：
覆盖 40 个任务的 6444 个尝试帧，最大位置差异 1.39e-17 m、最大姿态差异 7.43e-15 rad。
因此该实验确认的是“重标注 + 独立名义参考累加”在原生 OSC 下可行；
它不会自动消除既有 IPOPT 可行性与关节位置控制执行差异。

机器可读证据位于 `results/osc_reference_comparison_v1.json`、
`results/osc_reference_dataset_validation_v1.json`、`results/osc_reference_video_validation_v1.json`
和 `results/osc_reference_vs_existing_ipopt_targets_v1.json`。

## 直接切换到 IPOPT

MuJoCo 项目的正式入口现支持 `--reference-mode relabelled_actions` 和
`--reference-actions-path`，直接读取上述 HDF5 的动作和初始参考，不重新解码源 OSC 动作。
读取器位于 `tasktol/benchmark/libero_reference_actions.py`，会核对源文件、state/action 指纹、
夹爪、SO(3) 旋转、初始锚点和完整参考重建。每帧仍由 `LiberoTaskTolSession.step()`
调用共享参考积分、原 IPOPT 及原生 LIBERO 关节执行链，没有增加物理循环或 OSC 回退。

在 OpenPI 根目录使用锁定虚拟环境的 Python，可运行固定 kp150、关闭容差的 40 条对照：

```bash
python experiments/libero_tasktol_compat/run_ipopt_alignment.py \
  --output experiments/libero_tasktol_compat/results/ipopt_relabelled_kp150_all40_v1 \
  --kp 150 --workers 3 --reference-mode relabelled_actions \
  --reference-root data/libero_reference_actions/20261020_osc_reference_v1

python experiments/libero_tasktol_compat/compare_ipopt_reference.py \
  --osc experiments/libero_tasktol_compat/results/osc_reference_accumulated40_v1 \
  --ipopt experiments/libero_tasktol_compat/results/ipopt_relabelled_kp150_all40_v1/kp_150 \
  --dataset data/libero_reference_actions/20261020_osc_reference_v1 \
  --output experiments/libero_tasktol_compat/results/ipopt_relabelled_comparison_v1.json
```

kp150 是显式的统一对照参数，不代表与 OSC 的末端阻抗等效；原生默认 kp50 没有改变。
轨迹差异仅对实际执行帧比较；被拒绝的目标和提前停止后未执行的尾段单独计数。

本次直接读取 HDF5 后的结果为：40 条代表演示中，OSC 末帧成功 39 条，IPOPT 末帧成功 34 条。
IPOPT 共尝试 6444 帧，实际执行 6441 帧；三个目标未通过严格求解，停止时不推进物理。
实际执行帧相对累加参考 OSC 的末端位置差异均值 4.0725 mm，p95 为 10.5427 mm，最大 45.5414 mm。
已接受 IK 解的最大 FK 位置残差仅 2.80e-6 mm，不能将实际执行偏差等同于 IK 数值残差。
直接动作读取、参考重建、初始化、原生执行等 480 项配对检查全部通过；40 段视频完整解码通过。

同时保留第二种成功口径：IPOPT 有 36 条在某个实际执行帧达到原生成功判定。
`libero_spatial/06` 在帧 91–98 成功，`libero_goal/05` 在帧 132–142 成功，
继续完整示范动作后均失去成功条件。原始客户端 `examples/libero/main.py` 会在 done 时退出；
本次诊断为了与之前的完整轨迹比较保持一致，未中途停止，因此同时报告“曾成功 36/40”和“末帧成功 34/40”。
未曾成功的四条分别为三个求解被拒任务 `libero_spatial/04`、`libero_goal/02`、`libero_10/08`，
以及执行完成但任务未成功的 `libero_10/00`。

推盘任务 `libero_goal/05` 用同一参数额外重跑一次，终帧失败和中途成功均复现；
重跑与主批次的关节目标及实际末端轨迹逐元素一致。它与较早 source_goals 运行的关节目标最大仅相差约 1.04e-11 rad，
实际末端轨迹最大相差 0.93 mm，但末帧判定发生改变；不据此宣称输入语义变化或优化器算法退化。
主批次结果保持原样，未按重跑结果择优。

机器可读结果为 `results/ipopt_relabelled_comparison_v1.json`、
`results/ipopt_relabelled_validation_v1.json` 和 `results/ipopt_relabelled_repeatability_v1.json`。

## 关节执行动态诊断

保持完整 IPOPT 关节参考和当前原生关节控制器，新增只读 `trace_joint_tracking.py`，
包裹原生 `run_controller()` 调用以记录 q、qdot、目标、质量矩阵、Jacobian、补偿力矩及限幅前后力矩。
没有新增控制/物理调用，也未改变控制模式。对 Spatial00、Goal05、Long00 的两条执行链共采样 26100 次；
六条带采样轨迹与原主批次实际关节、末端轨迹逐元素一致。

每控制周期确实执行 25 次原生控制，物理周期 2ms。关节链 13050 次采样中仅 18 次力矩限幅；
三个首帧均无机器人与外物接触且不限幅。全40任务已执行帧的关节目标追踪 RMS 为 0.03433rad（1.97度）。
三条轨迹按现有帧索引对齐，最佳目标/实际关节轨迹偏移均为3帧；这是有限带宽响应的相位滞后，
不能解释成程序队列中额外等待了150ms。

以 Spatial00 首帧为例，第四关节目标阶跃为 0.12729rad，50ms 后实际移动 0.01673rad，约13.1%。
在 kp150、kd=2sqrt(150) 的理想无外力临界阻尼模型中，零初速50ms阶跃响应仅完成12.6%；
带实际初速的预测与这一步实际位置最大相差0.000707rad，说明当前关节环的响应速度与其参数相符。

在完全相同初始 q、qdot、M、J 和补偿力矩下，两种控制律在接触前已给出不同的瞬时响应。
Spatial00 的 `J M^-1 (tau-bias)` 绕世界Y轴角加速度贡献：关节控制为 +5.231rad/s²，OSC为 -5.886rad/s²。
原生OSC的分块位置/姿态惯量映射所产生的耦合项贡献约 -11.109rad/s²，
有限关节误差到末端误差的非线性项约 -0.008rad/s²，两者与实际差值吻合至数值精度。
该量是原始控制力矩的瞬时加速度贡献，不包含共同的 Jdot*qdot 和外接触力；不是测得的全时段实际角加速度。

这些结果具体指出了当前跟踪器的有限带宽和原生控制律的动态映射差异。
它们并不支持“完整关节参考本身不可行”，也尚未将全部40任务的误差或失败逐项完全归因。
证据为 `results/joint_tracking_diagnosis_v1.json`、
`results/joint_tracking_instrumentation_validation_v1.json` 和 `results/joint_tracking_video_validation_v1.json`。

## 同状态直接力矩对比

`compare_native_torques.py` 只比较每条轨迹首个物理步，避免将已经不同的后续实际状态混为同状态。
三条样本的 q、qdot、M、J 和补偿力矩逐元素相同，目标位姿在原生姿态转换精度内一致，且无机器人外接触、无力矩限幅。
Spatial00 的实际原生力矩（Nm）如下，差值方向为关节控制减OSC：

| 关节 | 原生OSC | IPOPT关节目标 + 原生关节控制 | 差值 |
| --- | ---: | ---: | ---: |
| 1 | 7.565 | 11.882 | 4.317 |
| 2 | -13.971 | 14.200 | 28.170 |
| 3 | 7.911 | 7.406 | -0.505 |
| 4 | 30.521 | 50.967 | 20.446 |
| 5 | -3.504 | -4.484 | -0.980 |
| 6 | 8.081 | -9.383 | -17.465 |
| 7 | -2.705 | -1.709 | 0.996 |

采用动态一致的力矩任务/零空间投影，将差值精确分成三项：
原OSC分块惯量与完整任务惯量映射差异、有限关节误差到末端误差的任务项差异、冗余方向跟踪差异。
第二关节的28.170Nm差值分别为27.614Nm、0.559Nm、-0.00256Nm，重构残差小于1e-14Nm。
因此该样本首步的大力矩差主要来自惯量映射，不能归咎于保留关节参考或额外冗余姿态要求。
具体实现差异位于原生 `joint_pos.py:243` 的 `M @ joint_PD + bias` 与
`osc.py:336-351` 的分块任务惯量、Jacobian转置映射及零空间项。

Goal05 和 Long00 的首步七关节力矩差向量范数分别为21.441Nm和4.068Nm；
21项状态、目标、无接触/不限幅及力矩分解检查全部通过。
该对照定位的是同状态首步力矩差来源，不将三个采样点宣称为全任务误差的完整归因。
完整向量与逐项分解保存于 `results/native_torque_comparison_v1.json`。

## 完整关节参考的惯量适配实验（2026-10-20）

在 MuJoCo 项目增加 `tasktol/simulation/libero_joint_controller.py`，通过
`LiberoTaskTolSession(..., joint_control_mode="osc_inertia_matched")` 或 CLI
`--joint-control-mode osc_inertia_matched` 显式启用。默认仍为 `native`，本次结果不支持将候选设为默认。
原生 JointPositionController 父类负责目标解码、插值、状态更新和清理，子类仅修正返回力矩前的惯量映射；
初始化时接管原控制器状态，没有额外 reset、forward 或物理步。七维 IPOPT 目标全部保留。

```text
Lambda_full = pinv(J M^-1 J^T)
Lambda_split = blockdiag(pinv(Jp M^-1 Jp^T), pinv(Jr M^-1 Jr^T))
M_eff = M + J^T (Lambda_split - Lambda_full) J
tau = M_eff [kp (q_d - q) - kd qdot] + bias
```

直接复用 robosuite `opspace_matrices` 伪逆语义，仅接受固定、均匀的 kp/kd。
该修正匹配局部任务误差惯量关系；没有消除有限关节误差到位姿误差的非线性，也未将关节姿态项改为 OSC 的初始姿态项。
不声称全局稳定性或与 OSC 完全等价。遇到非有限输出、非正定有效矩阵立即失败，不静默换控制器。

实现后重新采集 Spatial00、Goal05、Long00 的原生物理步力矩。首步 q、qdot、M、J、bias
及 IPOPT 关节目标与此前基线逐元素相同，实际力矩符合上式至 1.1e-14 Nm；
相对 OSC 的力矩差范数分别由 39.211、21.441、4.068 Nm 降为 4.051、0.749、2.482 Nm。
三条只读采样运行与本次主批次实际关节、末端轨迹逐元素一致。

固定 kp150、kd=2sqrt(150)，无容差，40 条相同示范对照如下：

| 指标 | 原生关节控制 | 惯量适配候选 | 原生累加参考 OSC |
| --- | ---: | ---: | ---: |
| 末帧成功 | 34/40 | 32/40 | 39/40 |
| 执行过程中曾成功 | 36/40 | 33/40 | 39/40 |
| 相对 OSC 末端位置平均差 | 4.073 mm | 2.029 mm | — |
| 相对 OSC 末端位置 p95 差 | 10.543 mm | 5.370 mm | — |
| 相对 OSC 末端位置最大差 | 45.541 mm | 60.549 mm | — |

两条关节链都实际执行 6441 帧、尝试 6444 帧，3 个 IPOPT 拒绝目标仍在相同任务和帧索引：
Spatial04/105（CPU 时限）、Goal02/75（局部不可行）、Long08/371（迭代上限）。
误差统计使用两条链都执行的相同 6441 帧，不因提前结束选择更短有利片段。
求解器实现及目标函数指纹未变，也没有放宽约束或接收不可行输出。

新增成功为 Spatial06、Goal05，新增末帧失败为 Long03、Long04、Long06、Goal09。
Long03 原本也是 OSC 的失败案例；Goal09 在中途多次成功，末帧失去条件。
Long06 的相对 OSC 最大位置误差由 7.589 mm 增至 45.416 mm；不能仅用全局平均误差下降判断闭环改善。
这些是观测结果，尚未将具体物体接触/成功谓词失败完整归因。

161025 次物理步控制均产生有限输出和正定有效矩阵；最小特征值约 3.19e-7，
这只覆盖已采样状态。共 259 次原生力矩限幅，其中 Long03 为225次，Long07为18次。
保留完整关节参考不等于逐时刻精确跟踪；修正后的全物理步关节跟踪 RMS 为0.05595rad，
最大单关节误差0.51378rad。它与此前按20Hz记录的关节误差指标采样时点不同，不直接比较。

验证完成：原生锁定环境17项相关测试通过；主 MuJoCo 环境中通用工具坐标测试1项通过；
compileall 和 Ruff F/E9 通过。40任务640/640配对检查通过，43条视频全部完整解码并保存过程图片索引。
本轮没有生成或打开 HTML 报告，没有连接训练好的 VLA 模型。

证据：`results/inertia_matched_comparison_summary_v1.json`、
`results/ipopt_inertia_matched_comparison_v1.json`、`results/inertia_matched_evidence_validation_v1.json`、
`results/inertia_matched_provenance_validation_v1.json`。
候选控制器保留供后续逐项诊断，当前结论是平均跟踪改善但任务成功退化，闭环问题尚未解决。

## 后续任务/零空间消融与B10试验

完整方法、输入协议和结果见同目录 `FACTORIZED_CONTROL_EXPERIMENT.md`。
最终32个短时分叉使用与完整回放逐元素相同的IPO​PT关节参考；
288/288检查通过，B10比仅使用线性关节任务反馈更接近原生OSC。
但40任务结果为末帧35/40、曾成功35/40，原关节控制为34/40、36/40，
并出现Long00最大163.848mm偏差。因此默认模式保留，尚不能宣称迁移问题已解决。
