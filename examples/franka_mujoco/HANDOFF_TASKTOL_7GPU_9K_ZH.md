> **历史文档，原生13维训练入口已停用。** 2026-10-28已统一恢复原生32维；本文旧启动/恢复命令不再适用。当前配置与迁移说明见 `JOINT32_MIGRATION_ZH.md`。旧13维checkpoint保留归档，不可用新32维配置直接恢复。

# TaskTol 七卡训练交接：mix20、4 倍动作 loss、9k 步

更新日期：2026-10-27，时区 Asia/Shanghai。

本文供你完成数据修改后，自行转换、计算归一化统计、生成 Arrow 并在服务器后台训练。
所有服务器命令以 `czl` 用户执行。本文中的命令尚未替你启动转换或正式训练。

## 1. 已经确定的训练要求

| 项目 | 约定 |
|---|---|
| 服务器 | `ssh czl@100.73.28.99`；使用已有登录凭据 |
| 仓库 | `/home/czl/openpi`，分支 `task_tolerance`，已核验基础提交 `48b7b39` |
| 配置名 | `pi05_tasktol_mix20_4xloss_7gpu_9k` |
| GPU | 7 张 RTX 3090，编号 0～6，单进程 JAX 数据并行 |
| batch | 全局 **84**，每卡 **12**；`fsdp_devices=1` |
| 训练长度 | **9,000 步** |
| 学习率 | warmup **300** 步，峰值 **5e-5**，余弦衰减至第 9,000 步的 **5e-6** |
| 保存 | 每 **1,500** 步保存并保留：1500、3000、4500、6000、7500、9000 |
| 日志 | 每 **10** 步记录这 10 步的平均 loss 及其他指标 |
| SwanLab | 项目 **`openpi-tasktol`**，online，账号 `caozilin` 已登录并验证 |
| 默认实验名 | `TaskTol_mix20_4xloss_7gpu_bs84_9k`；新数据版本建议追加 `_v2` 等后缀 |
| 数据 | 已准备好的 nominal/tolerance 混合 LeRobot 数据集；mix20 表示轨迹配比约 80/20 |
| 动作权重 | nominal 轨迹 **1.0**，tolerance 执行轨迹 **4.0** |
| 模型 | pi0.5 joint13 LoRA，输出 7 维动作＋6 维旋转容差，horizon **16**，文本长度 **48** |
| 优化器 | AdamW，梯度裁剪 1.0，EMA 关闭，种子 42，8 个 DataLoader workers |
| 验证 | 默认未配置验证集；验证 batch 系数为 1.0，指定验证集后才启用 |

模型、冻结规则、图像变换、归一化方式、预训练权重加载逻辑均沿用
`pi05_franka_mujoco_joint13`。预训练权重为 GCS 上的 `pi05_base`；本配置仍训练视觉模块。

4 倍动作权重依据 `trajectory_is_tolerance == 1`，不是依据样本是否有容差标签：

```text
nominal:   loss = 1.0 × action_MSE + 0.5 × rotation_tolerance_MSE
tolerance: loss = 4.0 × action_MSE + 0.5 × rotation_tolerance_MSE
```

这里 MSE 是模型流匹配目标的误差。4 倍作用于容差轨迹样本的全部 7 个动作维度、整个
16 步 horizon；6 维容差预测项仍乘 0.5。batch 内取普通平均，不按权重和重新归一化。
混合数据已经在磁盘上准备，训练不会自动创建 20% 混合，也不会自动过采样容差轨迹。

参考训练量：原 batch 24 × 30,000 = 720,000 个样本；新 batch 84 × 9,000 = 756,000，
多约 5%。这是样本处理量换算，最终收敛仍结合曲线和任务评估判断。

## 2. 服务器现状和空间

依赖已通过锁文件同步和冲突检查，训练/转换/统计模块可导入。
Python 3.11.18，JAX/jaxlib 0.5.3，Flax 0.10.2，Torch 2.7.1，Transformers 4.53.2，
SwanLab 0.7.20，PyAV 14.2.0。已有 `uv`、`tmux`、`rsync`。

SwanLab key 保存在服务器用户凭据中，不放进代码、文档或 shell 脚本。
可运行 `.venv/bin/python -m swanlab verify` 确认账号。凭据是用户级，训练配置负责选择项目。

- `/home`：本地 Samsung SATA SSD，检查时剩约 **152 GiB**，使用率 96%。
- `/mnt/SharedData`：本地致态 2TB SATA SSD、ext4，检查时剩约 **710 GiB**，`czl` 可写。
- `/mnt/SharedData` 名称虽带 Shared，实际不是网络文件系统。
- 仓库和虚拟环境保留在 `/home/czl/openpi`；原始数据、LeRobot、Arrow、统计、checkpoint、日志放共享盘。
- 每份完整可恢复 checkpoint 的未压缩张量约 **10.69 GiB**：参数 7.21＋Adam 状态 3.48。
  六份约 **64.13 GiB**，建议预留 **80 GiB**，另加原始数据、Parquet、Arrow、日志和转换中间文件。
  七卡保存不会再乘七；实际磁盘大小以写出的文件为准。

本次看到的现有路径：

```text
/mnt/SharedData/czl/raw/franka_nominal80_tolerance20
/mnt/SharedData/czl/lerobot/caozilin/franka_nominal80_tolerance20
```

现有 LeRobot 元数据暂记 1,444 条轨迹、236,316 帧，schema `tasktol_mujoco/8.0`，joint 模式。
这个数字只是检查时快照，不代表你修改后的数据已经完成。转换完成后以最终元数据为准。

## 3. 先固定新数据版本，再保存路径配置

你修改原始轨迹、标签、mix 比例或样本集合后，使用新的 LeRobot repo_id、输出目录、
Arrow 缓存和归一化统计，并开启新实验。旧转换的 `--resume` 用于同一版本中断续转；
它会核对源数据和标签哈希，不能用来把修改后的数据混入旧版本。

以下示例使用 `_v2`。先完成原始数据修改，再检查 `RAW_DIR` 是否指向最终数据。
如果在本地转换，也使用相同的新 repo_id，之后按第 5 节上传。

连接服务器并进入持久会话：

```bash
ssh czl@100.73.28.99
tmux new -s tasktol-prepare
```

保存一次路径设置；重新连接、打开新终端或恢复任务时都先 `source` 这份文件：

```bash
cat > "$HOME/tasktol_mix20_v2.env" <<'ENV'
export PATH="$HOME/.local/bin:$PATH"
export REPO_DIR=/home/czl/openpi
export RUN_ROOT=/mnt/SharedData/czl

# 修改为最终原始数据目录和本次新版本名称。
export RAW_DIR=/mnt/SharedData/czl/raw/franka_nominal80_tolerance20
export REPO_ID=caozilin/franka_nominal80_tolerance20_v2
export EXP_NAME=TaskTol_mix20_4xloss_7gpu_bs84_9k_v2

export HF_ENDPOINT=https://hf-mirror.com
export HF_LEROBOT_HOME="$RUN_ROOT/lerobot"
export DATASET_ROOT="$HF_LEROBOT_HOME/$REPO_ID"
# 每个数据版本使用独立 Arrow 缓存根目录。
export HF_DATASETS_CACHE="$RUN_ROOT/hf_datasets/$REPO_ID"
export JOINT_ASSETS_BASE="$RUN_ROOT/assets/pi05_franka_mujoco_joint13"
export ACTION_ASSETS_BASE="$RUN_ROOT/assets/pi05_franka_mujoco"
export CHECKPOINT_BASE="$RUN_ROOT/checkpoints"
export RUN_DIR="$RUN_ROOT/runs/$EXP_NAME"
export SWANLAB_LOG_DIR="$RUN_ROOT/swanlog"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export PYTHONUNBUFFERED=1
ENV

source "$HOME/tasktol_mix20_v2.env"
cd "$REPO_DIR"
set -euo pipefail
mkdir -p "$HF_LEROBOT_HOME" "$HF_DATASETS_CACHE" "$JOINT_ASSETS_BASE" \
  "$ACTION_ASSETS_BASE" "$CHECKPOINT_BASE" "$RUN_DIR" "$SWANLAB_LOG_DIR"

git status --short --branch
uv sync --locked --check
uv pip check
.venv/bin/python -m swanlab verify
df -h "$HF_LEROBOT_HOME" "$CHECKPOINT_BASE"
```

分支应显示 `task_tolerance`，并且下列配置和脚本存在：

```bash
test -f examples/franka_mujoco/train_tasktol_mix20_4xloss_7gpu_9k.sh
rg -n 'pi05_tasktol_mix20_4xloss_7gpu_9k' src/openpi/training/config.py
```

目前新增配置和脚本是已同步到服务器的未提交改动。保持这份工作区；执行 git 清理、
硬重置或替换 checkout 之前先保存它们。

## 4. 在服务器转换 LeRobot

输入为 TaskTol schema 8.0：每任务有 `task_metadata.json`、`manifest.json`；成功 episode
有 `trajectory.json`、`main_rgb.mp4`、`wrist_rgb.mp4`，视频为 10Hz、224×224 RGB。
联合模式还必须有与当前 trajectory 哈希、帧数和 PRE/POST 覆盖一致的 `tolerance_labels.json`。
修改轨迹后应重新完成标签；转换器会拒绝陈旧或不完整的标签。

```bash
source "$HOME/tasktol_mix20_v2.env"
cd "$REPO_DIR"
set -euo pipefail

uv run --locked python examples/franka_mujoco/convert_franka_mujoco_data_to_lerobot.py \
  --raw-dir "$RAW_DIR" \
  --repo-id "$REPO_ID" \
  --joint-task-tolerance \
  --workers 12 \
  --image-writer-processes 0 \
  --image-writer-threads 2 \
  --progress-interval-seconds 30 \
  2>&1 | tee "$RUN_DIR/convert.log"
```

输出为 `$HF_LEROBOT_HOME/$REPO_ID`，不是 `--raw-dir`。`HF_LEROBOT_HOME` 必须在
启动 Python 前设置。当前转换器没有 `--output-dir` 参数。

转换器保留真实图像、7 维动作和 6 维容差，并把执行变体写为 `trajectory_is_tolerance`。
`execution_variant=nominal` 为 0，`tolerance` 为 1；字段冲突会报错，未声明的基础采集视为 nominal。

同版本转换中断后，使用上面完全相同的命令追加 `--resume`。源数据更改后用新版本目录。
`--overwrite` 会删除同名输出，先确认目标；不要让两个转换进程写同一目录。
完成标志为日志最后的 `Dataset complete: ... episodes at ...`，退出码为 0。

## 5. 如果在本地转换：上传完整 LeRobot 目录

本地转换应使用同一 repo_id，并设置本地的 `HF_LEROBOT_HOME`。转换完成且退出后，
在**本地终端**同步整个输出目录：

```bash
rsync -avh --partial --info=progress2 \
  /LOCAL/LEROBOT_ROOT/caozilin/franka_nominal80_tolerance20_v2/ \
  czl@100.73.28.99:/mnt/SharedData/czl/lerobot/caozilin/franka_nominal80_tolerance20_v2/
```

替换本地路径。远端父目录可先用 SSH `mkdir -p` 创建。等 rsync 成功退出后再计算统计或训练。
正常训练使用的是转换输出；原始目录可另外存档。Arrow 可以在服务器第一次加载时生成，
不必从本地传输。若在本地也已算好统计，两个 `norm_stats.json` 可以按第 6 节的目录一并上传。

## 6. 计算归一化统计：使用这份新数据

优先直接扫描 LeRobot Parquet 的数字列，避免为了统计先生成 Arrow 或解码图像：

```bash
source "$HOME/tasktol_mix20_v2.env"
cd "$REPO_DIR"
set -euo pipefail

uv run --locked python scripts/compute_franka_mujoco_norm_stats.py \
  --dataset-root "$DATASET_ROOT" \
  --source lerobot \
  --joint-task-tolerance \
  --action-horizon 16 \
  --batch-size 24 \
  --joint-output-dir "$JOINT_ASSETS_BASE/$REPO_ID" \
  --pi05-output-dir "$ACTION_ASSETS_BASE/$REPO_ID" \
  2>&1 | tee "$RUN_DIR/norm_stats.log"
```

这里 `--batch-size 24` 是统计时的流式聚合批次，和七卡训练 batch 84 独立；保留原统计默认值。
脚本按 horizon 16 构造动作块后统计，包含 episode 尾部处理。统计不会按 4 倍 loss 权重重加权。

应生成：

```text
$JOINT_ASSETS_BASE/$REPO_ID/norm_stats.json  # state=7，actions=13，当前联合训练用
$ACTION_ASSETS_BASE/$REPO_ID/norm_stats.json # state=7，actions=7，动作基线可用
```

原有配置的统计路径默认指向仓库 assets；本文训练命令显式传
`--data.assets.assets-dir "$JOINT_ASSETS_BASE"`，匹配共享盘上的新统计。
如果省略此覆盖，会去找另一处文件。该参数指向 `<repo_id>` 的**父目录**。

如果已经有本版本的完整 Arrow，也可把 `--source lerobot` 改成 `--source arrow` 并增加
`--arrow-cache-dir "$HF_DATASETS_CACHE"`；其他参数相同。首次流程使用上面的 Parquet 方案即可。

## 7. 检查数据、权重标志、统计，并预生成 Arrow

以下检查只读取数据、建立缓存、抽样执行训练输入变换；不创建模型、不登录新实验、不启动训练。
第一次 LeRobot 加载可能花较长时间创建 Arrow 并核对时间戳。等待命令成功退出。

```bash
source "$HOME/tasktol_mix20_v2.env"
cd "$REPO_DIR"
set -euo pipefail

JAX_PLATFORMS=cpu uv run --locked python - <<'PY' 2>&1 | tee "$RUN_DIR/preflight.log"
import collections
import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from openpi.shared import normalize
from openpi.training import config, data_loader

root = Path(os.environ['DATASET_ROOT'])
repo = os.environ['REPO_ID']
assets = Path(os.environ['JOINT_ASSETS_BASE'])
info = json.loads((root / 'meta/info.json').read_text())
assert info['source_schema'] == 'tasktol_mujoco/8.0'
assert info['joint_task_tolerance'] is True
assert info['fps'] == 10
assert info['total_frames'] >= 84
print('DATASET', root, 'episodes=', info['total_episodes'], 'frames=', info['total_frames'])

# 仅扫描数字列，确认所有行的执行类型，并检查帧数。
counts = collections.Counter()
episode_flags = {}
for path in sorted(root.glob('data/chunk-*/episode_*.parquet')):
    for batch in pq.ParquetFile(path).iter_batches(columns=['episode_index', 'trajectory_is_tolerance']):
        columns = batch.to_pydict()
        for episode, flag in zip(columns['episode_index'], columns['trajectory_is_tolerance'], strict=True):
            flag = np.asarray(flag).reshape(-1)
            assert flag.shape == (1,) and int(flag[0]) in (0, 1), (path, flag)
            flag = int(flag[0])
            episode = int(episode)
            assert episode_flags.setdefault(episode, flag) == flag
            counts[flag] += 1
assert sum(counts.values()) == info['total_frames'], counts
assert len(episode_flags) == info['total_episodes']
assert counts[1] > 0, '没有 tolerance 执行样本：检查转换后的 execution_variant 标志'
assert counts[0] > 0, '没有 nominal 样本：检查 mix20 数据'
print('FRAME_COUNTS nominal/tolerance=', dict(counts))
print('EPISODE_COUNTS nominal/tolerance=', dict(collections.Counter(episode_flags.values())))
print('TOLERANCE_FRAME_FRACTION', counts[1] / sum(counts.values()))

stats = normalize.load(assets / repo)
for key, dim in [('state', 7), ('actions', 13)]:
    for field in ['mean', 'std', 'q01', 'q99']:
        values = np.asarray(getattr(stats[key], field))
        assert values.shape == (dim,) and np.isfinite(values).all(), (key, field, values.shape)
    assert (np.asarray(stats[key].std) >= 0).all()
print('NORM_STATS_OK', assets / repo / 'norm_stats.json')

base = config.get_config('pi05_tasktol_mix20_4xloss_7gpu_9k')
cfg = dataclasses.replace(base, data=dataclasses.replace(base.data,
    repo_id=repo, lerobot_dataset_root=str(root),
    hf_datasets_cache_dir=os.environ['HF_DATASETS_CACHE'],
    assets=config.AssetsConfig(assets_dir=str(assets))))
data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
assert data_cfg.norm_stats is not None
assert cfg.data.tolerance_trajectory_action_loss_weight == 4.0
dataset = data_loader.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)
transformed = data_loader.transform_dataset(dataset, data_cfg)
for index in sorted({0, len(dataset) // 2, len(dataset) - 1}):
    sample = transformed[index]
    action = np.asarray(sample['actions'])
    assert action.shape == (16, 13) and np.isfinite(action).all(), action.shape
    assert np.asarray(sample['state']).shape == (13,)
    weight = float(sample['action_loss_weight'])
    assert weight in (1.0, 4.0)
    print('SAMPLE', index, 'actions=', action.shape, 'action_loss_weight=', weight,
          'images=', {k: np.asarray(v).shape for k, v in sample['image'].items()})
print('PREFLIGHT_OK; Arrow cache:', os.environ['HF_DATASETS_CACHE'])
PY
```

`JAX_PLATFORMS=cpu` 只作用于这条检查命令；不要把它长期 export 到训练终端。
检查帧/轨迹比例，是为了识别实际 mix；轨迹数量 20% 不保证帧数量也恰好 20%。
本检查包含形状、数值和标志检查；修改数据后仍应自行抽查画面、指令和动作语义。

Arrow 的两个参数区别：

| 参数 | 用途 |
|---|---|
| `--data.hf-datasets-cache-dir "$HF_DATASETS_CACHE"` | 正常 LeRobot 加载，首次按需生成 Arrow，后续复用；目录需存在 |
| `--data.lerobot-arrow-cache-dir "$HF_DATASETS_CACHE"` | 直接内存映射读取已经完成的 Arrow；要求 dataset root 和元数据，缓存缺失就报错 |

默认训练使用第一种。缓存已完成且想跳过常规加载扫描时，可以额外传第二种。
有多个匹配条目时，应传包含 `dataset_info.json` 和全部 `.arrow` 的精确目录。
直读模式只支持当前这种嵌入图像数据，不支持 LeRobot video features。
它不验证原始 Parquet 每个字节的来源；修改数据后请使用新缓存目录，别按相同帧数复用旧缓存。

## 8. 持久保存训练命令，退出 SSH 后继续运行

先完成第 6、7 节，确认 `PREFLIGHT_OK`。创建本实验专用脚本：

```bash
source "$HOME/tasktol_mix20_v2.env"
mkdir -p "$RUN_DIR"

cat > "$RUN_DIR/train.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
source "$HOME/tasktol_mix20_v2.env"
cd "$REPO_DIR"
trap 'status=$?; printf "\nTRAIN_EXIT=%s time=%s\n" "$status" "$(date -Is)"' EXIT

bash examples/franka_mujoco/train_tasktol_mix20_4xloss_7gpu_9k.sh \
  --data.repo-id "$REPO_ID" \
  --data.lerobot-dataset-root "$DATASET_ROOT" \
  --data.hf-datasets-cache-dir "$HF_DATASETS_CACHE" \
  --data.assets.assets-dir "$JOINT_ASSETS_BASE" \
  --checkpoint-base-dir "$CHECKPOINT_BASE" \
  --exp-name "$EXP_NAME" \
  "$@"
SH

# 先确认七张卡未被其他训练占用。
nvidia-smi

TRAIN_LOG="$RUN_DIR/train_$(date +%Y%m%d_%H%M%S).log"
nohup bash "$RUN_DIR/train.sh" > "$TRAIN_LOG" 2>&1 < /dev/null &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$RUN_DIR/train.pid"
printf '%s\n' "$TRAIN_LOG" > "$RUN_DIR/latest_log.txt"
printf 'PID=%s\nLOG=%s\n' "$TRAIN_PID" "$TRAIN_LOG"
tail -f "$TRAIN_LOG"
```

`tail -f` 中按 Ctrl+C 只结束日志查看，后台训练继续。nohup 可在 SSH 断开后保持运行。
tmux 中也可以按 Ctrl+B，再按 D 脱离会话；重新连接使用 `tmux attach -t tasktol-prepare`。

首次启动会加载数据、预训练权重，初始化模型并编译。看见 `Generating train split` 时
通常在生成 Arrow。首个训练步比后续慢；先检查日志是否推进，避免重复启动。
服务器 `huggingface.co` 曾超时，脚本已默认使用 `hf-mirror.com`；GCS checkpoint 可访问。

配置无需再重复输入 `--batch-size`、`--num-train-steps`、`--project-name` 等，它们已经固定在
新配置内。若用第二种 Arrow 直读，可在启动 `train.sh` 时追加：

```bash
bash "$RUN_DIR/train.sh" --data.lerobot-arrow-cache-dir "$HF_DATASETS_CACHE"
```

上面是前台运行形式；需要后台时把相同附加参数放进前面的 nohup 命令，二者选一种启动。
提交训练后保持本实验 env 文件不变，新实验另建 env 文件。

## 9. 查看进度、指标和磁盘

重新连接后：

```bash
source "$HOME/tasktol_mix20_v2.env"
tail -n 80 "$(cat "$RUN_DIR/latest_log.txt")"
ps -p "$(cat "$RUN_DIR/train.pid")" -o pid,etime,args
nvidia-smi
df -h "$CHECKPOINT_BASE" "$HF_DATASETS_CACHE"
du -sh "$DATASET_ROOT" "$HF_DATASETS_CACHE" "$CHECKPOINT_BASE" "$SWANLAB_LOG_DIR"
```

SwanLab 打开登录账号下的 `openpi-tasktol` 项目，按 `$EXP_NAME` 查找；初始化日志也会打印链接。
主要指标：

- `loss`、`loss/total`：优化使用的总 loss。
- `loss/action`：原始动作项；`loss_weighted/action`：含容差轨迹 4 倍权重的动作项。
- `loss/rotation_tolerance`：原始容差项；`loss_weighted/rotation_tolerance`：乘 0.5 后的项。
- `grad_norm`、`param_norm`：梯度与参数范数；梯度裁剪阈值仍为 1.0。

4 倍权重改变了优化目标，新的 total loss 数值不能直接与未加权旧实验横比。
曲线每 10 步更新，不是只在每第十步计算梯度；每步都做完整更新。

输出 checkpoint 的实际目录：

```text
$CHECKPOINT_BASE/pi05_tasktol_mix20_4xloss_7gpu_9k/$EXP_NAME/
  swanlab_id.txt
  1500/
  3000/
  4500/
  6000/
  7500/
  9000/
```

每个保存目录包含 `params`、`train_state`、`assets`。训练结尾会等待异步保存结束。
日志出现 9,000 步、保存完成且 `TRAIN_EXIT=0` 才算正常结束；不要仅凭第 9,000 步 loss 就关机。

## 10. 停止和恢复

需要停止时，先核对 PID 确实指向本实验，再发送中断：

```bash
source "$HOME/tasktol_mix20_v2.env"
TRAIN_PID=$(cat "$RUN_DIR/train.pid")
ps -p "$TRAIN_PID" -o pid,etime,args
ps -u czl -o pid,ppid,etime,args | rg 'scripts/train.py|train_tasktol_mix20'
```

pid 文件可能是启动 shell；以上面进程列表中本实验的 Python `scripts/train.py` PID 为准，
执行 `kill -TERM <PYTHON_PID>`。恢复点是最近一次**完整保存**的 checkpoint，间隔内进度会丢失。

确认原进程已退出且数据、统计、实验配置未变，再启动恢复：

```bash
source "$HOME/tasktol_mix20_v2.env"
TRAIN_LOG="$RUN_DIR/resume_$(date +%Y%m%d_%H%M%S).log"
nohup bash "$RUN_DIR/train.sh" --resume > "$TRAIN_LOG" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RUN_DIR/train.pid"
printf '%s\n' "$TRAIN_LOG" > "$RUN_DIR/latest_log.txt"
tail -f "$TRAIN_LOG"
```

`--resume` 的总目标仍为 9,000 步，例如从 3,000 恢复，再跑剩余 6,000 步。
online 模式会使用 checkpoint 根目录的 `swanlab_id.txt` 尝试恢复原云端实验。
若从未完成第一份 checkpoint，会重新初始化；不能恢复尚未保存的训练进度。
`--overwrite` 会删除同实验目录，用新实验名更容易保留旧结果。

## 11. 常见定位

| 现象 | 优先检查 |
|---|---|
| 缺少 `norm_stats` 或 stats 维度错误 | repo_id、`--data.assets.assets-dir`，应为 state 7 / actions 13；使用本次新数据重算 |
| 找不到数据或 Arrow 缓存 | dataset root 必须到 `meta`/`data` 的父目录；先建 HF cache；直读仅用于完整缓存 |
| 执行类型全为 nominal，4 倍没生效 | 检查源 `execution_variant`，转换后 `trajectory_is_tolerance`；重新转换新版本 |
| stale labels、哈希不符、resume 拒绝 | 修改轨迹后重做标签，用新 repo_id、新缓存、新统计 |
| 训练启动前看不到进度 | 区分 Arrow 生成、时间戳扫描、checkpoint 下载和 JIT 编译；看 CPU、磁盘和日志 |
| batch/device 不整除 | 本次必须 7 张可见卡，全局 84；启动脚本已设置 GPU 0～6 |
| GPU OOM | 确认模型还是 joint13/text48，每卡 12，其他进程未占显存；新增验证时系数保持 1.0 |
| SwanLab 登录失败 | 使用 `.venv/bin/python -m swanlab verify`；需要重登时用交互 login，勿将 key 写到仓库 |
| checkpoint 目录已存在 | 同实验续训加 `--resume`；新数据/新设置改实验名 |
| 七卡时一步变慢 | 查温度降频、最慢卡、真实 DataLoader 等待和 I/O；不要只看显存是否占满 |
| 磁盘增长快 | Parquet/Arrow/原始视频均占空间，六份全量可恢复 checkpoint 约 64 GiB；确认路径确在共享盘 |

七卡基准：每卡 12，真实 train_step，合成输入常驻 GPU，平均约 3.90 秒/步、21.51 样本/秒，
为首轮单卡约 5.62 倍。纯计时推算 9k 步约 9.8 小时；真实训练还会增加加载、验证、
日志与保存等时间。训练期间部分卡曾温度降频，不能把该吞吐当成保证值。

七卡时间线观察到约 0.9～1.06 秒/步的 NCCL 区间，包含传输及等慢卡；快卡确实会等待。
共享盘当前本地 SSD 空间充足，但真实数据读取速度尚未纳入前述基准。

## 12. 交接时已验证 / 等你完成

已验证：分支与依赖、SwanLab 账号认证、训练配置 CLI 解析、单卡 batch 12/16、
1/2/4/7 卡合成输入真实训练步和七卡通信时间线。

等你完成：最终原始数据修改与标签、完整转换、新版归一化、Arrow 和输入检查，随后启动正式训练。
本文使用 `_v2` 示例；如果本次数据已在其他名字下完整准备好，可以在 env 文件里替换路径和 repo_id，
并让所有步骤使用同一套变量。开始正式训练前确认 7 卡空闲、`PREFLIGHT_OK`、SwanLab 认证正常。
