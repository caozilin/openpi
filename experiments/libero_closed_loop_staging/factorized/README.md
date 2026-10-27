# LIBERO 原生执行 + TaskTol/IPOPT 运动生成器

客户端代码在 `franka_mujoco/tasktol`。本目录提供隔离的运行环境、批量调度和审计材料。LIBERO 源码、场景和资产直接使用原 OpenPI 子项目路径。

## 执行边界

依据用户 2026-10-20 明确要求，仅替换运动生成器：

```text
示范名义目标 / 后续策略的物理位姿增量
                ↓
原有 TaskTol/IPOPT 与阶段旋转容差实现
                ↓
7 维绝对关节目标 q_target
                ↓
原生 JOINT_POSITION 输入编码 + 夹爪命令
                ↓
LIBERO OffScreenRenderEnv.step(native_action)
                ↓
所选关节力矩映射、原生力矩限制、夹爪、物理步进、观测和成功判定
```

LIBERO 原生关节控制器接收归一化关节增量。本客户端只将其输出缩放配置为各关节的完整跨度，用 `(q_target - q_current) / span` 编码；原生控制器解码后的目标与 IPOPT 目标逐帧核对，误差必须小于 `1e-10 rad`。环境输入为 8 维：7 维关节输入与 1 维夹爪；它不同于原 OSC 的 6+1 输入。

默认 `joint_control_mode=native` 继续使用原生 robosuite 关节控制算法。可选 `osc_inertia_matched` 仅修改关节误差到力矩的惯量映射；物理循环、夹爪逻辑和任务判定保持原生。没有额外的速度前馈、重力补偿、手工力矩写入、状态强制回放或接触参数改动。原 `tasktol/simulation/preset_runtime.py` 已恢复原文件，LIBERO 路径不使用它；这遵循用户对本次原生执行边界的明确要求。此前额外物理后端实验保留并标为 superseded，不能用来代表当前方案。

两个核心入口：

- `tasktol/simulation/libero.py`：`LiberoTaskTolSession.step()`，接收物理位姿增量 `[m, rotation-vector rad, gripper]`，调用原 IPOPT，最后只调用一次原生 `env.step()`。
- `tasktol/benchmark/libero.py`：示范参考驱动的录制 CLI，生成视频、实际轨迹、关节目标、原生动作和逐帧求解记录。

物理点为 LIBERO grip_site，姿态为物理 right_hand 工具坐标。固定旋转从原生模型现场计算并记录；FK/Jacobian 使用该工具约定，保证语义 mask 的物理夹持轴一致。该适配不修改模型或资产。

## 2026-10-20 初始化对齐修复

当前 `tasktol.benchmark.libero` 默认使用 `--initialization-mode source_context`：根据处理文件的逐示范随机状态与原生硬重置历史恢复固定物体布局，核对随机状态，恢复夹爪累加状态并使控制器缓存失效。未增加物理步进。实现位于 `tasktol/benchmark/libero_replay_state.py`。`snapshot_only` 明确保留旧初始化用于对照；旧记录未覆盖。

最新逐任务对照与过程证据位于 `/media/czl/sata/franka_my_code/openpi/experiments/libero_tasktol_compat/results/alignment_report_v1/RESULTS.md`，HTML 同目录 `index.html`。同一批40条示范、关闭容差：修复后源控制器目标参考 kp50 为34/40、kp150为35/40；实际运动参考 kp4000为39/40，平均末端位置差1.58mm、最大4.65mm。不同参考和增益不混为一个成功率。默认kp50保留，未按任务挑选配置。

实际运动参考不能表达源OSC所有接触加载；相同kp1000的中层抽屉对照中，实际运动参考约打开5mm，源控制器目标参考约打开160mm。三个严格目标拒绝也不能通过调高关节增益解决。当前仍未连接VLA模型，不宣称达到全时段20Hz墙钟实时性。

## 运行

在本目录执行；使用已有锁定环境，不触发项目自动同步：

```bash
PYTHONPATH=../.. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --no-sync python -m tasktol.benchmark.libero \
  --suite libero_spatial --task-id 0 --output results/my_native_replay
```

默认使用 LIBERO 原生 JOINT_POSITION 默认增益 `kp=50`；可用 `--kp` 显式配置原生关节控制器，报告记录实际值，不会自动挑选成功配置。

默认 `--reference-mode source_goals` 离线解码并冻结源命令目标后执行 IPOPT。它不是直接把归一化 OSC 动作当成米或弧度。`source_context` 初始化恢复原控制器的初始零旋转保持方向；旧 `snapshot_only` 模式使用首个 FK 近似。`--reference-mode recorded_motion` 可另行诊断跟踪已观测机器人运动的情况；这些不同参考模式不能混为同一基线。

### 直接执行重标注参考动作

`--reference-mode relabelled_actions` 配合 `--reference-actions-path`，直接读取已验证的物理位姿增量 HDF5。
读取器核对源 HDF5、逐演示 state/action 指纹、夹爪动作、初始参考锚点和目标重建；不重解码 OSC 动作。
每个控制周期仍调用原 `LiberoTaskTolSession.step()`，独立累加参考、求解 IPOPT，再经原生关节控制器执行。
无容差时逐帧核对累加目标与标注目标一致，不接受不可行解，也不回退 OSC。

```bash
PYTHONPATH=../.. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --no-sync python -m tasktol.benchmark.libero \
  --suite libero_spatial --task-id 0 --output results/my_relabelled_ipopt \
  --reference-mode relabelled_actions --kp 150 \
  --reference-actions-path /media/czl/sata/franka_my_code/openpi/data/libero_reference_actions/20261020_osc_reference_v1/libero_spatial/00/reference_actions.hdf5
```

上述 kp150 是明确固定的对照参数，默认原生 kp50 保留。重标注动作单位为米和世界坐标旋转向量弧度；
夹爪使用源动作，不能将该 HDF5 动作按旧归一化 OSC 动作解释。每次报告记录 HDF5、读取器和执行实现指纹。

通过 `--tolerance-deg 5` 可验证共享容差执行链，叠加冻结的逐子任务语义 mask，并保持 ENG/DIS 严格。5° 只是显式诊断参数，不写入冻结标注，不作为搜索结果或数值容差真值。阶段时序来自源代表示范，尚未重新标注实际 IPOPT 接触事件。

批量与审计：

```bash
uv run --no-sync python run_batch.py --name native_nominal40 --workers 3
uv run --no-sync python run_batch.py --name native_tolerance5_four --task-ids 0 --workers 1 --tolerance-deg 5
uv run --no-sync python validate_and_report.py
```

已有结果请勿无意覆盖；新增实验应使用不同的 `--name` 或 `--output`。

## 环境与证据

隔离环境：Python 3.12、MuJoCo 3.2.3、robosuite 1.4.1、cyipopt 1.7.0。两个原虚拟环境保持不变。IPOPT 源码直接从现有 TaskTol 导入，没有复制求解算法。cyipopt wheel 复用已有 CPython 3.12 编译产物，原 IPOPT 动态库仍由 `.runtime/ipopt` 提供，来源见相邻 `libero_compatibility/artifacts/cyipopt_reuse_provenance.json`。

依赖通过 `pyproject.toml`、`uv.lock` 与 `requirements.lock.txt` 固定。整包同步停滞时复用已校验本地 wheel，补充下载缺失项并核对锁定 SHA256；最终用 `uv pip sync --require-hashes requirements.lock.txt` 和 `uv pip check` 审计通过。`fetch_locked_wheels.py` 及 `local_wheels.lock.txt` 留存该引导过程。

当前记录属于示范参考驱动的 IPOPT 闭环，不代表已连接某个训练 checkpoint 的 VLA 成功率。后续策略可向同一个 `step()` 提供物理动作；模型的归一化、状态维度及阶段来源须按对应 checkpoint 契约接入。

结果见 [RESULTS.md](RESULTS.md)、[逐任务视频报告](report.html)、[summary.json](summary.json) 和 [validation.json](validation.json)。每条轨迹保存：源索引、原始源动作、物理名义增量、IPOPT 关节目标、8 维原生关节动作、实际观测、求解状态、阶段与 mask、初始/终止 PNG、20Hz 主视角 MP4。失败记录保留，不计入成功示范。

### 完整关节参考的 OSC 惯量适配模式

显式添加 `--joint-control-mode osc_inertia_matched`。默认仍为 `native`。
`tasktol/simulation/libero_joint_controller.py` 继承原生 JointPositionController，接管已初始化的控制器状态，不额外调用构造器、reset 或 forward。
目标解码、关节插值、状态更新和清理由原生父类执行；七维关节目标全部保留。每个原生物理步更新

```text
Lambda_full = pinv(J M^-1 J^T)
Lambda_split = blockdiag(pinv(Jp M^-1 Jp^T), pinv(Jr M^-1 Jr^T))
M_eff = M + J^T (Lambda_split - Lambda_full) J
tau = M_eff [kp (q_d - q) - kd qdot] + bias
```

矩阵计算直接复用 robosuite `opspace_matrices` 的伪逆规则。仅接受 fixed impedance 与各关节统一的 kp/kd；
局部任务误差映射与原生分离位置/姿态惯量匹配，不代表有限姿态误差下与 OSC 完全等价。
原生 robot 继续限幅并施加力矩，不新增执行循环、补偿项或接触设置。非有限输出或非正定有效矩阵直接报错，不静默回退。
报告保存每个物理步的调用计数、最小有效特征值、对称性误差、修正量与力矩饱和计数。
固定 kp150、容差关闭的 40 任务对照结果保存在 OpenPI 的 `experiments/libero_tasktol_compat/results/ipopt_inertia_matched_kp150_all40_v1`。

本轮验证结果：固定 kp150、无容差、40 条示范，原生关节控制末帧34/40、曾成功36/40，
惯量适配候选末帧32/40、曾成功33/40。与 OSC 的共同6441帧位置平均差由4.073mm降至2.029mm，
但最大差由45.541mm升至60.549mm。因此本候选保持显式实验模式，不替换 `native` 默认模式。
三处 IPOPT 拒绝目标仍存在，未放宽约束。完整数字与失败案例见 OpenPI
`experiments/libero_tasktol_compat/results/inertia_matched_comparison_summary_v1.json`。
17项原生环境相关测试及1项主环境工具坐标测试通过，640/640配对检查、43/43视频与过程图片检查通过。

### OSC 任务反馈 + IPOPT 完整关节冗余参考（B10，实验选项）

显式启用 `--joint-control-mode osc_task_joint_null`，默认仍为 `native`。
此模式通过 `LiberoTaskTolSession` 的同一 IPOPT/容差/参考累加链求出完整七维 q_d，
将 q_d 原样传给共享 `FactorizedReferenceController.set_joint_reference()`；
任务目标使用 FK(q_d)，零空间姿态目标使用完整 q_d，固定零空间 kp=10、kd=2sqrt(10)。
任务增益在本轮实验固定150。它保留局部完整构型反馈，不声称全局关节收敛或原关节PD动态不变。

该选项使用原生 OSC 命令协议：绝对FK位姿六维 + 夹爪，以及同进程显式七维关节参考。
这不同于默认原生JOINT_POSITION的八维归一化关节/夹爪输入。
报告逐帧记录 `joint_reference_transport`，核对控制器保存的完整 q_d 等于IPO​PT输出；
物理步进、夹爪执行及力矩限幅仍由原生 LIBERO env.step 执行。不可行IPO​PT输出继续中止，
没有绕过求解器或回退纯OSC。

控制律为 `tau = J.T Lambda_split u_OSC + N.T M u_joint_null + bias`。
对照使用同一模块的任务反馈/冗余参考2×2组合，均固定null kp10；旧惯量修正null kp150另列基线。
四个任务26个相同状态的单周期分叉验证中，D恢复原生OSC，234/234检查通过；
B的50ms末端位置差均值0.002705mm，A10为0.390663mm。
以上不是完整任务成功率，完整回放结果单独记录。
证据在OpenPI `experiments/libero_tasktol_compat/results/factorized_short_forks_v1/analysis.json`。
`factorized_short_forks_pilot_v1`因诊断方法覆盖问题标为无效，未纳入任何结果。
