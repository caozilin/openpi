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
原生关节控制、力矩限制、夹爪、物理步进、观测和成功判定
```

LIBERO 原生关节控制器接收归一化关节增量。本客户端只将其输出缩放配置为各关节的完整跨度，用 `(q_target - q_current) / span` 编码；原生控制器解码后的目标与 IPOPT 目标逐帧核对，误差必须小于 `1e-10 rad`。环境输入为 8 维：7 维关节输入与 1 维夹爪；它不同于原 OSC 的 6+1 输入。

未复制或修改 robosuite 控制器算法、物理循环、夹爪逻辑和任务判定。没有额外的速度前馈、重力补偿、手工力矩写入、状态强制回放或接触参数改动。原 `tasktol/simulation/preset_runtime.py` 已恢复原文件，LIBERO 路径不使用它；这遵循用户对本次原生执行边界的明确要求。此前额外物理后端实验保留并标为 superseded，不能用来代表当前方案。

两个核心入口：

- `tasktol/simulation/libero.py`：`LiberoTaskTolSession.step()`，接收物理位姿增量 `[m, rotation-vector rad, gripper]`，调用原 IPOPT，最后只调用一次原生 `env.step()`。
- `tasktol/benchmark/libero.py`：示范参考驱动的录制 CLI，生成视频、实际轨迹、关节目标、原生动作和逐帧求解记录。

物理点为 LIBERO grip_site，姿态为物理 right_hand 工具坐标。固定旋转从原生模型现场计算并记录；FK/Jacobian 使用该工具约定，保证语义 mask 的物理夹持轴一致。该适配不修改模型或资产。

## 运行

在本目录执行；使用已有锁定环境，不触发项目自动同步：

```bash
PYTHONPATH=../.. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --no-sync python -m tasktol.benchmark.libero \
  --suite libero_spatial --task-id 0 --output results/my_native_replay
```

默认使用 LIBERO 原生 JOINT_POSITION 默认增益 `kp=50`；可用 `--kp` 显式配置原生关节控制器，报告记录实际值，不会自动挑选成功配置。

默认 `--reference-mode source_goals` 离线解码并冻结源命令目标后执行 IPOPT。它不是直接把归一化 OSC 动作当成米或弧度。源示范缺少完整控制器内部历史，初始零旋转保持方向以首个 FK 近似。`--reference-mode recorded_motion` 可另行诊断跟踪已观测机器人运动的情况；这些不同参考模式不能混为同一基线。

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
