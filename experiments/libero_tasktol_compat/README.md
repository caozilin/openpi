# LIBERO × TaskTol 独立联调实验

本目录用于验证现有 TaskTol 运行栈能否在同一进程加载、运行 LIBERO；不是新的正式执行后端。诊断直接调用 LIBERO 的 `OffScreenRenderEnv` 和 `env.step`，没有复制物理步进、控制器或容差优化算法。

正式实验目录：`/media/czl/sata/franka_my_code/franka_mujoco/experiments/libero_compatibility`。结论见 `RESULTS.md`，视频/图像见 `report.html`，机器可读检查见 `summary.json`。

## 环境与原地址

- 独立 `.venv`：Python 3.12.3、MuJoCo 3.10.0、NumPy 2.5.1、robosuite 1.4.1。原两个虚拟环境保持不变。
- LIBERO 源码直接从 `/media/czl/sata/franka_my_code/openpi/third_party/libero` 导入。
- LIBERO 资产直接使用 `/media/czl/sata/franka_my_code/openpi/third_party/libero/libero/libero/assets`。BDDL、原始 HDF5 与配置同样保留原路径。
- TaskTol 代码从 `/media/czl/sata/franka_my_code/franka_mujoco/tasktol` 导入；没有复制优化器。各次报告记录被导入核心文件的 SHA256。
- 原项目的 cyipopt 1.7.0 已编译扩展可用于相同 CPython 3.12 ABI；实验将它重新打包为本地 wheel 并在 `pyproject.toml` / `uv.lock` 声明。文件级来源指纹保存在 `artifacts/cyipopt_reuse_provenance.json`，wheel 内包含原许可证。
- IPOPT 动态库仍从 `/media/czl/sata/franka_my_code/franka_mujoco/.runtime/ipopt` 加载。该 wheel 是当前 Linux 主机的复用材料，不是跨平台发布包。当前本机没有完整 IPOPT 开发头文件，故不重新编译。

## 运行

在正式实验目录中，直接使用已验证环境，不触发自动重新安装：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONDONTWRITEBYTECODE=1 \
  uv run --no-sync python smoke_test.py --compat \
  --suite libero_spatial --task-id 0 --output results/my_spatial_probe
```

将 suite 改为 `libero_object`、`libero_goal`、`libero_10` 可运行对应代表任务。默认回放完整保留动作；`--steps N` 可限制步数。原始初始状态来自对应 HDF5 代表 demo 的首个保存状态，不使用需要 torch 的 benchmark `.pruned_init`。动作索引复用已有处理清单，统一跳过此前筛选的 noop，包含 10 次 settling。

`--compat` 只启用经过测试的 `mj_fullM` 局部签名桥接。去掉该参数可复现原 robosuite 在 MuJoCo 3.10 下构造控制器失败的错误。桥接使用当前控制器真实的 `MjData`，不改写全局 `mujoco` 模块，也不更改原控制器计算公式。

对照原环境：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONDONTWRITEBYTECODE=1 \
  /media/czl/sata/franka_my_code/openpi/examples/libero/.venv/bin/python \
  smoke_test.py --skip-tasktol --suite libero_spatial --output results/my_old_baseline
```

检查与汇总：

```bash
uv run --no-sync python -m pytest -q test_compatibility.py
uv pip check --python .venv/bin/python
uv run --no-sync python summarize.py
```

汇总器读取已保存的四对 `baseline_*` / `final_*` 结果，并对全部视频完整解码。环境 `status=passed` 表示加载、步进、渲染与接口检查通过；任务是否成功另见 `task_success`。不能将两者混为一谈。

## 安装可复现性

依赖均记录在本实验的 `pyproject.toml`、`uv.lock` 和导出的 `requirements.lock.txt`。`build` 组记录构建工具，`future` 补足 BDDL 实际需要但元数据未声明的依赖。

网络正常时可使用 `uv sync --locked --group build`。本次整包下载停滞，使用 `fetch_locked_wheels.py` 从锁文件中的官方 PyPI 地址分段下载，并逐一核对 SHA256；材料保留在 OpenPI 的 `experiments/libero_tasktol_compat/locked_wheels`。

实际安装最后以如下命令按导出清单审计完成，避免 uv 项目同步因 wheel 来源地址不同而重复下载同版本大包：

```bash
uv export --locked --all-groups --no-emit-project --format requirements-txt \
  --output-file requirements.lock.txt
uv pip sync --require-hashes --python .venv/bin/python requirements.lock.txt
```

本地 wheel 路径相对于实验目录，需在该目录执行。`local_wheels.lock.txt` 是下载引导材料清单；最终安装版本以 `uv.lock` 和 `requirements.lock.txt` 为准。

## 结论边界

四个任务集各测试一条示范；原环境 4/4 成功，新环境 3/4 成功。长任务出现约 6 cm 的最大末端回放偏差。已证明单进程可以运行，并可复用原 TaskTol FK/Jacobian；尚未证明物理等价，也尚未接入 TaskTol 容差控制器或搜索数值容差。

正式接入时仍须使用一个统一执行后端，不应将此环境诊断脚本当作另一套 TaskTol rollout。当前先保留两个正式环境，定位升级栈的回放漂移，再决定采用兼容物理版本或独立进程调用原优化器。
