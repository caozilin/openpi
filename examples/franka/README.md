# Franka LoRA Fine-Tuning Workflow

这份文档对应 `vla4desk` 采集的 Franka 真机数据，目标是：

- 先把原始 `data.json + cam1.mp4 + cam2.mp4` 转成 LeRobot
- 然后在云服务器上用原版 JAX 单卡完成 `pi05_franka_lora` 微调

当前使用的训练配置是：

- `pi05_franka_lora`

不使用：

- `pi05_franka_lora_discrete`

## 1. 原始数据目录格式

转换脚本当前假设你的原始数据目录长这样：

```text
/path/to/collected/
  task_a/
    epo_1/
      data.json
      cam1.mp4
      cam2.mp4
    epo_2/
      ...
  task_b/
    ...
```

也支持直接把 `--source_dir` 指到单个 `epo_x` 目录。

## 2. 转换成 LeRobot 数据集

在有原始数据的机器上执行：

```bash
uv run examples/franka/convert_vla4desk_data_to_lerobot.py \
  --source_dir /media/czl/sata/franka_my_code/vla4desk/collected \
  --repo_id caozilin/my_franka_dataset
```

这一步会：

- 扫描 `source_dir` 下的所有 `epo_x`
- 逐帧读取 `cam1.mp4`、`cam2.mp4`
- 从 `data.json` 中读取 `state` 和 `action`
- 将它们写成 LeRobot 的标准字段：
  - `image`
  - `wrist_image`
  - `state`
  - `actions`
  - `task`

说明：

- 现在脚本会把 `data.json` 里的 `action` 原样写入 LeRobot
- 不会再把前 6 维按 `action_scale` 缩放回去
- `task` 优先取 `prompt`，其次取 `task_name`，再不行就退回父目录名

输出目录不是原始数据目录，而是：

```text
HF_LEROBOT_HOME / caozilin/my_franka_dataset
```

也就是 LeRobot 默认数据根目录下的 `caozilin/my_franka_dataset`。

## 3. 把转换后的数据放到云服务器

如果转换在本地完成，而训练在云服务器完成，那么需要把整个 LeRobot 数据集目录同步到云服务器。

你需要同步的是：

```text
HF_LEROBOT_HOME / caozilin/my_franka_dataset
```

同步完成后，在云服务器上确认：

- 训练环境能访问这个 LeRobot 数据集目录
- `repo_id` 仍然对应 `caozilin/my_franka_dataset`

如果你选择在本地连 `norm stats` 也一起算完，那么云服务器上通常只需要把下面两个目录一起复制过去：

```text
HF_LEROBOT_HOME / caozilin/my_franka_dataset
./assets/pi05_franka_lora/caozilin/my_franka_dataset
```

也就是：

- 数据集目录
- 对应 config 的 `norm_stats.json` 目录

这样云服务器上就可以直接开始训练，不必重新计算 `norm stats`。

## 4. 在云服务器上计算 norm stats

在云服务器的 `openpi` 仓库目录中执行：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_franka_lora
```

这一步会读取 `pi05_franka_lora` 对应的数据配置，并计算：

- `state`
- `actions`

的归一化统计信息。

默认输出位置：

```text
<openpi_repo>/assets/pi05_franka_lora/caozilin/my_franka_dataset/norm_stats.json
```

如果这一步没跑，训练时会报缺少 `norm_stats`。

如果你已经在本地完成了这一步，并且已经把：

```text
<openpi_repo>/assets/pi05_franka_lora/caozilin/my_franka_dataset
```

完整复制到了云服务器同样的位置，那么这一节可以跳过。

## 5. 训练前检查最终送入模型的数据

在完成 LeRobot 转换和 `norm stats` 之后，建议先检查一次“真正送入模型训练”的数据是否正确，而不是只检查原始 JSON。

可以运行：

```bash
uv run examples/franka/inspect_training_inputs.py \
  --config-name pi05_franka_lora \
  --num-samples 16 \
  --output-dir ./debug/inspect_pi05_franka_lora
```

这一步走的是和训练一致的完整数据管线：

- LeRobot dataset loading
- repack transforms
- data transforms
- normalization
- model transforms

也就是说，它检查的是最终喂给模型的训练输入。

输出目录默认类似：

```text
<openpi_repo>/debug/inspect_pi05_franka_lora
```

里面会有：

- `overview.png`
  - 多个 sample 的三路图像拼图
- `samples/sample_0000.png`
  - 单个 sample 的可视化图
- `samples/sample_0000.npz`
  - 对应 sample 的 `state/actions/image/tokenized_prompt`
- `summary.json`
  - 每个 sample 的统计摘要
- `summary.csv`
  - 便于快速筛查数值范围

建议重点检查：

- 图像是否对应正确视角
- `state` 数值范围是否正常
- `actions` 数值范围是否符合预期
- `action_horizon` 是否正确
- prompt/token 长度是否正常
- image mask 是否合理

如果这一步看起来不对，不要直接开始训练，先修正转换脚本或数据配置。

## 6. 在云服务器上用原版 JAX 单卡训练

继续在云服务器的 `openpi` 仓库目录中执行：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_franka_lora --exp-name=franka_lora_v1 --overwrite
```

说明：

- 这是原版 JAX 训练入口，不是 PyTorch 入口
- 这是单卡命令
- `--exp-name` 可以换成你自己的实验名
- `--overwrite` 会覆盖同名实验目录

默认 checkpoint 输出目录：

```text
./checkpoints/pi05_franka_lora/franka_lora_v1
```

## 7. 最终你会得到什么

训练完成后，主要结果在：

```text
./checkpoints/pi05_franka_lora/franka_lora_v1
```

其中包括：

- 训练过程中的 checkpoint
- 最终模型参数
- checkpoint 内复制进去的 `assets`

对应的归一化统计仍在：

```text
<openpi_repo>/assets/pi05_franka_lora/caozilin/my_franka_dataset/norm_stats.json
```

## 8. 最短执行顺序

如果只看最短流程，就是这三步：

```bash
uv run examples/franka/convert_vla4desk_data_to_lerobot.py \
  --source_dir /media/czl/sata/franka_my_code/vla4desk/collected \
  --repo_id caozilin/my_franka_dataset

uv run scripts/compute_norm_stats.py --config-name pi05_franka_lora

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_franka_lora --exp-name=franka_lora_v1 --overwrite
```

如果你想更稳妥，推荐实际执行顺序是四步：

```bash
uv run examples/franka/convert_vla4desk_data_to_lerobot.py \
  --source_dir /media/czl/sata/franka_my_code/vla4desk/collected \
  --repo_id caozilin/my_franka_dataset

uv run scripts/compute_norm_stats.py --config-name pi05_franka_lora

uv run examples/franka/inspect_training_inputs.py \
  --config-name pi05_franka_lora \
  --num-samples 16 \
  --output-dir ./debug/inspect_pi05_franka_lora

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_franka_lora --exp-name=franka_lora_v1 --overwrite
```
