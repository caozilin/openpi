# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

## 默认逐轴语义容差配置

`configs/libero40_semantic_tolerances_deg.json` 保存用户编辑表格中的40任务、49子任务
APP／EXEC三轴角度；顺序为 **xF、yF、zF**，单位为度、含义为正负对称半宽。
0为严格轴，其余数值直接转为弧度传入求解器；ENG／DIS固定三轴0，heading退化保护保留。
Long06白杯与巧克力布丁分别使用各自角度，按当前物体距离选择并在抓取／执行期间锁存。

再次编辑工作簿后运行以下命令更新默认配置（只读表格，不启动测评）：

```bash
python3 examples/libero/import_semantic_tolerances.py
```

准备新的容差测评run时，用 `confirmed_mask_policy.freeze_default_config(new_run_dir)`
生成该run的 `confirmed_masks.json` 并将返回字段合入新 `evaluation_request.json`。
请求同时冻结当前客户端、`confirmed_mask_policy.py`、`solver_failure_hold.py`、
`success_continuation.py` 等执行源码SHA。worker只读该run的冻结副本，拒绝角度或SHA不一致的续跑；
既有run里的旧表与结果保持原值。逐帧日志和回放保留当前请求的分轴角度，失败保持时另记上一接受参考边界。

新测评默认首次原生成功后继续闭环20帧（1秒），连续20帧求解失败提前终止策略仍有效。
新请求须合入 `success_continuation.request_fields()`；首次成功和延长后末态成功分别记录。

## 本地 benchmark 求解失败策略

本地重构模型评测客户端使用 `solver_failure_hold.py`：求解成功才提交新的名义位置、
姿态及容差释放状态；失败帧丢弃本帧增量，回滚到上一帧已接受参考，继续用上一帧的
关节目标和夹爪命令执行完整控制周期。首次求解尚未成功时保持初始实测关节位置，
夹爪沿用初始化的打开命令。不使用失败的关节候选，也不把失败动作累积到下一帧。

默认 **连续20帧失败** 才提前终止（20 Hz下为1秒仿真控制时间，不是求解墙钟耗时）；
任何一帧成功会把连续失败计数清零。终止的第20帧也执行50 ms保持控制。
若保持期间原生任务成功谓词变为真，仍记录任务成功。
可通过 `--max-consecutive-solver-failures` 修改阈值，必须与新run请求中记录的值一致。
模型仍按既有重规划周期消费动作，失败动作不会重放或排队。

每帧分别记录 `solver_failed`、`reference_held`、`nominal_reference_updated`、
`consecutive_solver_failures`、`solver_failure_limit_reached`；`aborted`仅表示实际提前终止。
失败候选另存为 `rejected_nominal_target` / `rejected_joint_target_rad`，
`nominal_target` / `joint_target_rad` 保持实际采用的参考。结果记录终止原因
`ik_failure_streak`、失败总帧数、最长连续失败数及配置阈值。

历史数据与冻结runtime不修改。新请求须记录
`solver_failure_policy="hold_last_accepted_reference"`、
`max_consecutive_solver_failures=20`，并把新适配器与客户端SHA加入 `source_sha256`。
旧请求无法使用新策略静默续跑。已标记配置错误的全111容差run保持禁用；此改动
不修正或启用其任务mask，也不启动评测、推理或自动监督。

验证：`.venv/bin/python -m pytest -q examples/libero/solver_failure_hold_test.py`。

## Benchmark测评回放（含最新400回合）

```bash
python3 examples/libero/serve_mask_annotation_viewer.py --benchmark --port 8768
```

打开 <http://127.0.0.1:8768>。当前终端持续运行，`Ctrl+C` 结束进程并释放端口。
目录可选“无容差 · 40任务×10次（含失败重测）”，每任务seed0–9；也可单独查看
新增seed6–9、原240和17条失败重测。回合详情标明实际记录来源，原数据不复制或覆盖。
旧 `--background` 参数保留兼容，但不再启动后台进程。

## 重构后的 Rollout 可视化（当前默认）

在项目根目录运行 `python3 examples/libero/serve_mask_annotation_viewer.py --port 8768`，
打开 <http://127.0.0.1:8768/index.html>。默认仅展示共享 MuJoCo 关节运动生成器
重构后的成功 rollout；任务显示已保存条数 / 10，不加载旧 B10/OSC 录制。

数据目录为 `data/libero_tasktol_joint_servo/20261025_success10_v1`。
新增成功数据后，先更新页面目录：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
../franka_mujoco/experiments/libero_closed_loop/.venv/bin/python \
examples/libero/build_reconstructed_rollout_viewer.py
```

页面保留逐帧视频、在线阶段、关节/动作曲线、参考修正记录，以及分别相对原始
state 和补偿参考的误差。启动查看器不会恢复数据录制。

默认并排显示左侧重构 rollout 与右侧同 demo 的原始数据集视频，播放、暂停、
倍速和定位共用时间轴；取消“并排对比原始数据”即可隐藏右侧。
右侧从原始 HDF5 的 `obs/agentview_rgb` 直接编码，保留全部原始帧，
仅按显示约定旋转180°；它不是 OSC 重录或 state 回放。原始分辨率128×128，
重构分辨率256×256。两侧按帧号对应，原始额外的最后一帧可用“打开原始视频”查看。

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85

## Regenerate original LIBERO demonstrations

`regenerate_libero_dataset.py` replays the original HDF5 demonstrations using the preprocessing rules from
[OpenVLA](https://github.com/openvla/openvla/blob/0ebc5e333ee7a916813bf50db240f1af7ba1d295/experiments/robot/libero/regenerate_libero_dataset.py).
It uses the local LIBERO simulation environment and does not load a VLA model. The unmodified upstream source,
its SHA256, and its MIT license are retained in `upstream/` for equivalence tests.

Install the additional HDF5 dependency into the **Python 3.8 simulation environment**:

```bash
uv pip install --python examples/libero/.venv/bin/python -r examples/libero/requirements-regenerate.txt
```

The default input is `third_party/libero/libero/datasets`. Select a stable output disk with room for the rendered
images. Outputs are separate from the originals. Lossless gzip level 1 is enabled by default; `--compression lzf`
and `--compression none` are also supported. Storage compression does not change the arrays used for training.

Start with two tasks and three demonstrations per task. Replace the example output path with a location on your disk:

```bash
examples/libero/.venv/bin/python examples/libero/regenerate_libero_dataset.py \
  --output-dir /path/to/stable-disk/libero-smoke \
  --suites libero_spatial --task-ids 0 1 \
  --max-demos-per-task 3 --workers 2
```

Use `--dry-run` to print the task plan without constructing simulation environments. Once the smoke test passes,
process all four suites with four workers:

```bash
examples/libero/.venv/bin/python examples/libero/regenerate_libero_dataset.py \
  --output-dir /path/to/stable-disk/libero-openvla \
  --workers 4 --resume
```

Workers use `spawn`, independent offscreen rendering contexts, and one BLAS thread each.
Two terminal progress bars show processed demonstrations (including unsuccessful replays) and finished tasks,
with throughput and estimated remaining time. `saved` counts successful demonstrations; resumed episodes are
included as their checkpoints are validated. Use `--no-progress` to disable the bars for log-only runs.

Each task owns its HDF5, lock, and metadata file. The `summary.json` describes the current invocation; per-task `.metainfo.json` files and
the HDF5 episode ledger are the persistent processing records. A failed task makes the command exit nonzero while
other tasks can finish. Only the source demonstrations selected by the current limits are considered complete.

### Replay and alignment rules

- Create the environment at 256×256 and call `env.seed(0)`.
- Restore the original `states[0]`, then execute ten `[0, 0, 0, 0, 0, 0, -1]` settling actions.
- Apply OpenVLA's `1e-4` no-op predicate, including its gripper and first-action special cases. Filtered actions are
  not executed in the simulator.
- Record observations before each retained action, replay the complete retained sequence, and save the episode
  only if its final `done` is true. Original `demo_N` names are preserved; successful episode indices may have gaps.
- Store raw, unrotated camera images. Apply the 180° rotation once in the eventual HDF5→LeRobot exporter, matching
  the official RLDS conversion and this project's LIBERO evaluation preprocessing.

Simulation retains LIBERO's default 20 Hz control rate. Source indices refer to the original sequence; replay
time advances only for the ten settling actions and retained actions. Preserve this simulator time when building
tolerance trajectories; a training export's declared image FPS should not silently change the physical time step.

The original OpenVLA script stores the **reset state** in legacy `states[0]`, while its first observation is from
after settling. This entry point preserves that compatibility behavior. The added `data/demo_N/tasktol` group
contains the original initial state, post-settling state, actual pre-action simulator/robot states, final state,
and indices of retained actions in the original demonstration. Use `tasktol/pre_action_states` to align states
with observations and actions. The `source_metadata` group and original demo attributes retain source environment
metadata and model XML.

Flattened simulator states do not serialize every controller or solver internal variable. To reconstruct a
mid-trajectory execution context, replay the same settling steps and retained action prefix from the original
initial state. The added states provide alignment and verification rather than a promise of arbitrary controller
state restoration.

### Resume and reproducibility

Ctrl+C or SIGTERM requests a graceful stop. An unfinished episode is replayed on the next invocation; committed
episodes remain in a `.hdf5.inprogress` file. Repeat the same command with `--resume`. Completed task files are
checked and skipped. NumPy and Python RNG checkpoints preserve the reset sequence when continuing a task.
An HDF5 that was damaged by a power loss or SIGKILL requires separate inspection; the script will not silently
overwrite it. Final `.hdf5` files are published only after the selected task's episodes have been committed and
flushed.

Resume checks include replay-input hashes, source demo IDs, preprocessing settings, runtime package versions,
this script's hash, local LIBERO Python source hashes, and asset size/mtime inventory. Changing these settings,
editing LIBERO, or expanding a three-demo smoke run into a full run requires a new output directory. These checks
protect the local replay configuration; outputs are not claimed to be byte-identical to a published RLDS release.

The processed HDF5 is ready for tolerance generation and annotation. The current OpenPI training loader still
requires a LeRobot export; this entry point does not perform that conversion or generate tolerance labels.

Run the CPU-only equivalence and checkpoint tests with the main project environment:

```bash
.venv/bin/python -m pytest -q examples/libero/regenerate_libero_dataset_test.py
```
