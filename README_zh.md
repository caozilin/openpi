# openpi

openpi 是由 [Physical Intelligence 团队](https://www.physicalintelligence.company/)发布的机器人开源模型和包。

目前，此仓库包含三种类型的模型：
- [π₀ 模型](https://www.physicalintelligence.company/blog/pi0)，一个基于流（flow-based）的视觉-语言-动作模型（VLA）。
- [π₀-Fast 模型](https://www.physicalintelligence.company/research/fast)，一个基于 FAST 动作分词器的自回归 VLA。
- [π₀.₅ 模型](https://www.physicalintelligence.company/blog/pi05)，一个通过[知识隔离](https://www.physicalintelligence.company/research/knowledge_insulation)训练、具有更好开放世界泛化能力的 π₀ 升级版本。注意：在这个仓库中，我们目前只支持用于 π₀.₅ 训练和推理的流匹配头。

对于所有模型，我们提供**基础模型**检查点（已在 10k+ 小时的机器人数据上预训练），以及开箱即用或微调到您自己数据集的示例。

这是一个实验：π₀ 是为我们自己的机器人开发的，与广泛使用的平台如 [ALOHA](https://tonyzhaozh.github.io/aloha/) 和 [DROID](https://droid-dataset.github.io/) 不同，虽然我们乐观地认为研究人员和从业者将能够进行创造性的新实验，将 π₀ 适配到他们自己的平台，但我们并不期望每个这样的尝试都会成功。所有这些都意味着：π₀ 可能适合您也可能不适合，但欢迎您尝试！

## 更新

- [2025年9月] 我们在 openpi 中发布了 PyTorch 支持。
- [2025年9月] 我们发布了 π₀.₅，这是 π₀ 的升级版本，具有更好的开放世界泛化能力。
- [2025年9月]：我们添加了 DROID 训练的[改进空闲过滤器](examples/droid/README_train.md#数据过滤)。
- [2025年6月]：我们添加了使用 `openpi` 在完整 [DROID 数据集](https://droid-dataset.github.io/)上训练 VLA 的[说明](examples/droid/README_train.md)。这是用于训练 pi0-Fast-DROID 的训练流水线的近似开源实现。

## 需求

要运行此仓库中的模型，您需要一块 NVIDIA GPU，至少具有以下规格。这些估算假设使用单个 GPU，但您也可以使用多 GPU 和模型并行来通过在训练配置中配置 `fsdp_devices` 来减少每 GPU 内存需求。请注意，当前的训练脚本尚不支持多节点训练。

| 模式 | 内存需求 | 示例 GPU |
| ------------------ | --------------- | ------------------ |
| 推理 | > 8 GB | RTX 4090 |
| 微调 (LoRA) | > 22.5 GB | RTX 4090 |
| 微调 (全量) | > 70 GB | A100 (80GB) / H100 |

此仓库已在 Ubuntu 22.04 上测试，暂不支持其他操作系统。

## 安装

克隆此仓库时，请确保更新子模块：

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# 或者如果您已经克隆了仓库：
git submodule update --init --recursive
```

我们使用 [uv](https://docs.astral.sh/uv/) 来管理 Python 依赖。请参阅 [uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/) 进行设置。安装 uv 后，运行以下命令来设置环境：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

注意：`GIT_LFS_SKIP_SMUDGE=1` 是拉取 LeRobot 作为依赖项所需的。

**Docker**：作为 uv 安装的替代方案，我们提供了使用 Docker 安装 openpi 的说明。如果您遇到系统设置问题，请考虑使用 Docker 简化安装。请参阅 [Docker 设置](docs/docker.md) 了解更多信息。

## 模型检查点

### 基础模型

我们提供多个基础 VLA 模型检查点。这些检查点已在 10k+ 小时的机器人数据上预训练，可用于微调。

| 模型 | 使用场景 | 描述 | 检查点路径 |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| π₀ | 微调 | 用于微调的基础 [π₀ 模型](https://www.physicalintelligence.company/blog/pi0) | `gs://openpi-assets/checkpoints/pi0_base` |
| π₀-Fast | 微调 | 用于微调的基础自回归 [π₀-Fast 模型](https://www.physicalintelligence.company/research/fast) | `gs://openpi-assets/checkpoints/pi0_fast_base` |
| π₀.₅ | 微调 | 用于微调的基础 [π₀.₅ 模型](https://www.physicalintelligence.company/blog/pi05) | `gs://openpi-assets/checkpoints/pi05_base` |

### 微调模型

我们还为各种机器人平台和任务提供"专家"检查点。这些模型从上述基础模型微调而来，旨在直接在目标机器人上运行。这些模型可能在您的特定机器人上可能适用也可能不适用。由于这些检查点是在相对较小的数据集上微调的，这些数据集是用更广泛可用的机器人（如 ALOHA 和 DROID Franka 设置）收集的，它们可能无法泛化到您的特定设置，尽管我们发现其中一些（尤其是 DROID 检查点）在实践中泛化得相当广泛。

| 模型 | 使用场景 | 描述 | 检查点路径 |
| ------------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| π₀-Fast-Droid | 推理 | 在 [DROID 数据集](https://droid-dataset.github.io/)上微调的 π₀-Fast 模型：可以在 DROID 机器人平台上在新场景中零样本执行各种简单的桌面操作任务 | `gs://openpi-assets/checkpoints/pi0_fast_droid` |
| π₀-Droid | 微调 | 在 [DROID 数据集](https://droid-dataset.github.io/)上微调的 π₀ 模型：推理速度快于 π₀-Fast-Droid，但语言命令跟随能力可能较差 | `gs://openpi-assets/checkpoints/pi0_droid` |
| π₀-ALOHA-towel | 推理 | 在内部 [ALOHA](https://tonyzhaozh.github.io/aloha/) 数据上微调的 π₀ 模型：可以在 ALOHA 机器人平台上零样本折叠各种毛巾 | `gs://openpi-assets/checkpoints/pi0_aloha_towel` |
| π₀-ALOHA-tupperware | 推理 | 在内部 [ALOHA](https://tonyzhaozh.github.io/aloha/) 数据上微调的 π₀ 模型：可以从 tupperware 容器中取出食物 | `gs://openpi-assets/checkpoints/pi0_aloha_tupperware` |
| π₀-ALOHA-pen-uncap | 推理 | 在公开 [ALOHA](https://dit-policy.github.io/) 数据上微调的 π₀ 模型：可以打开笔盖 | `gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap` |
| π₀.₅-LIBERO | 推理 | 为 [LIBERO](https://libero-project.github.io/datasets) 基准微调的 π₀.₅ 模型：获得了最先进的性能（请参阅 [LIBERO README](examples/libero/README.md)） | `gs://openpi-assets/checkpoints/pi05_libero` |
| π₀.₅-Droid | 推理/微调 | 使用[知识隔离](https://www.physicalintelligence.company/research/knowledge_insulation)在 [DROID 数据集](https://droid-dataset.github.io/)上微调的 π₀.₅ 模型：推理速度快且语言跟随能力强 | `gs://openpi-assets/checkpoints/pi05_droid` |

默认情况下，检查点会在需要时自动从 `gs://openpi-assets` 下载并缓存到 `~/.cache/openpi`。您可以通过设置 `OPENPI_DATA_HOME` 环境变量来覆盖下载路径。

## 运行预训练模型的推理

我们的预训练模型检查点可以用几行代码运行（这里是我们的 π₀-Fast-Droid 模型）：

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# 创建一个训练好的策略
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# 在一个虚拟示例上运行推理
example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    ...
    "prompt": "pick up the fork"
}
action_chunk = policy.infer(example)["actions"]
```

您也可以在[示例笔记本](examples/inference.ipynb)中测试这个。

我们为在 [DROID](examples/droid/README.md) 和 [ALOHA](examples/aloha_real/README.md) 机器人上运行预训练检查点的推理提供了详细的分步示例。

**远程推理**：我们为**远程**运行模型推理提供了[示例和代码](docs/remote_inference.md)：模型可以在不同的服务器上运行，并通过 websocket 连接将动作流式传输到机器人。这使得在机器人外使用更强大的 GPU 并保持机器人和策略环境分离变得容易。

**在没有机器人的情况下测试推理**：我们提供了一个[脚本](examples/simple_client/README.md)用于在没有机器人的情况下测试推理。此脚本将生成随机观测并使用模型运行推理。请参阅[此处](examples/simple_client/README.md)了解更多详情。

## 在您自己的数据上微调基础模型

我们将使用 [LIBERO 数据集](https://libero-project.github.io/datasets) 作为运行示例，说明如何在自己的数据上微调基础模型 π₀.₅。我们将解释三个步骤：
1. 将您的数据转换为 LeRobot 数据集（我们用于训练）
2. 定义训练配置并运行训练
3. 启动策略服务器并运行推理

### 1. 将您的数据转换为 LeRobot 数据集

我们提供了一个最小示例脚本，用于将 LIBERO 数据转换为 LeRobot 数据集，位置在 [`examples/libero/convert_libero_data_to_lerobot.py`](examples/libero/convert_libero_data_to_lerobot.py)。您可以轻松修改它来转换您自己的数据！您可以从[这里](https://huggingface.co/datasets/openvla/modified_libero_rlds)下载原始 LIBERO 数据集，然后运行：

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

**注意：** 如果您只想在 LIBERO 上微调，您可以跳过此步骤，因为我们的 LIBERO 微调配置指向预转换的 LIBERO 数据集。此步骤只是一个示例，您可以适应您自己的数据。

### 2. 定义训练配置并运行训练

要在您自己的数据上微调基础模型，您需要为数据处理和训练定义配置。我们在下面提供了带有详细注释的 LIBERO 示例配置，您可以为自己的数据集修改它们：

- [`LiberoInputs` 和 `LiberoOutputs`](src/openpi/policies/libero_policy.py)：定义从 LIBERO 环境到模型的数据映射，反之亦然。将用于训练和推理。
- [`LeRobotLiberoDataConfig`](src/openpi/training/config.py)：定义如何处理用于训练的 LeRobot 数据集中的原始 LIBERO 数据。
- [`TrainConfig`](src/openpi/training/config.py)：定义微调超参数、数据配置和权重加载器。

我们为 LIBERO 数据上的 [π₀](src/openpi/training/config.py)、[π₀-Fast](src/openpi/training/config.py) 和 [π₀.₅](src/openpi/training/config.py) 提供了示例微调配置。

在运行训练之前，我们需要计算训练数据的归一化统计量。使用您的训练配置名称运行以下脚本：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

现在我们可以使用以下命令开始训练（`--overwrite` 标志用于在重新运行相同配置的微调时覆盖现有检查点）：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

该命令将把训练进度记录到控制台并保存检查点到 `checkpoints` 目录。您还可以在 Weights & Biases 仪表板上监控训练进度。为了最大限度地使用 GPU 内存，请在运行训练之前设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` —— 这允许 JAX 使用高达 90% 的 GPU 内存（对比默认的 75%）。

**注意：** 我们提供了从预训练中**重新加载**状态/动作归一化统计量的功能。如果您正在微调到预训练混合物中包含的机器人的新任务，这可能是有益的。有关重新加载归一化统计量的更多详细信息，请参阅 [norm_stats.md](docs/norm_stats.md) 文件。

### 3. 启动策略服务器并运行推理

训练完成后，我们可以通过启动策略服务器然后从 LIBERO 评估脚本查询它来运行推理。启动模型服务器很容易（我们使用迭代 20,000 的检查点作为此示例，根据需要修改）：

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

这将启动一个监听端口 8000 的服务器，等待观测数据被发送给它。然后我们可以运行评估脚本（或机器人运行时）来查询服务器。

对于特别运行 LIBERO 评估，我们提供了（并推荐使用）一个 Docker 化的工作流程，同时处理策略服务器和评估脚本。请参阅 [LIBERO README](examples/libero/README.md) 了解更多详情。

如果您想在自己的机器人运行时中嵌入策略服务器调用，我们有一个最小示例，介绍如何在[远程推理文档](docs/remote_inference.md)中完成此操作。

### 更多示例

我们提供了更多关于如何在 ALOHA 平台上微调和运行模型推理的示例：
- [ALOHA 模拟器](examples/aloha_sim)
- [ALOHA 真实机器人](examples/aloha_real)
- [UR5](examples/ur5)

## PyTorch 支持

openpi 现在与原始 JAX 版本一起提供 π₀ 和 π₀.₅ 模型的 PyTorch 实现！PyTorch 实现已在 LIBERO 基准上验证（推理和微调）。目前有一些功能暂不支持（未来可能会改变）：

- π₀-Fast 模型
- 混合精度训练
- FSDP（完全分片数据并行）训练
- LoRA（低秩适应）训练
- 训练期间的 EMA（指数移动平均）权重

### 设置

1. 确保您已安装所有依赖项的最新版本：`uv sync`

2. 仔细检查您已安装 transformers 4.53.2：`uv pip show transformers`

3. 应用 transformers 库补丁：
   ```bash
   cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
   ```

这会用必要的模型更改覆盖 transformers 库中的几个文件：1) 支持 AdaRMS，2) 正确控制激活精度，3) 允许 KV 缓存使用而不被更新。

**警告**：使用默认的 uv 链接模式（硬链接）时，这将永久影响 uv 缓存中的 transformers 库，意味着更改将在重新安装 transformers 后保留，甚至可能传播到使用 transformers 的其他项目。要完全撤消此操作，必须运行 `uv cache clean transformers`。

### 将 JAX 模型转换为 PyTorch

要将 JAX 模型检查点转换为 PyTorch 格式：

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config name> \
    --output_path /path/to/converted/pytorch/checkpoint
```

### 使用 PyTorch 运行推理

PyTorch 实现使用与 JAX 版本相同的 API —— 您只需更改检查点路径指向转换后的 PyTorch 模型：

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = "/path/to/converted/pytorch/checkpoint"

# 创建一个训练好的策略（自动检测 PyTorch 格式）
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# 运行推理（与 JAX 相同的 API）
action_chunk = policy.infer(example)["actions"]
```

### 使用 PyTorch 的策略服务器

策略服务器与 PyTorch 模型的工作方式相同 —— 只需指向转换后的检查点目录：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=/path/to/converted/pytorch/checkpoint
```

### 使用 PyTorch 微调

要使用 PyTorch 微调模型：

1. 将 JAX 基础模型转换为 PyTorch 格式：
   ```bash
   uv run examples/convert_jax_model_to_pytorch.py \
       --config_name <config name> \
       --checkpoint_dir /path/to/jax/base/model \
       --output_path /path/to/pytorch/base/model
   ```

2. 在配置中使用 `pytorch_weight_path` 指定转换后的 PyTorch 模型路径

3. 使用以下模式之一启动训练：

```bash
# 单 GPU 训练：
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>

# 示例：
uv run scripts/train_pytorch.py debug --exp_name pytorch_test
uv run scripts/train_pytorch.py debug --exp_name pytorch_test --resume  # 从最新检查点恢复

# 多 GPU 训练（单节点）：
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# 示例：
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume

# 多节点训练：
uv run torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
```

### 精度设置

JAX 和 PyTorch 实现的精度处理如下：

**JAX：**
1. 推理：大多数权重和计算使用 bfloat16，少数计算使用 float32 以保持稳定
2. 训练：默认为混合精度：权重和梯度使用 float32，（大多数）激活和计算使用 bfloat16。您可以通过在配置中将 `dtype` 设置为 float32 来更改为全 float32 训练。

**PyTorch：**
1. 推理：与 JAX 匹配 —— 大多数权重和计算使用 bfloat16，少数权重转换为 float32 以保持稳定
2. 训练：支持全 bfloat16（默认）或全 float32。您可以通过在配置中设置 `pytorch_training_precision` 来更改。bfloat16 使用更少内存但相比 float32 显示更高损失。混合精度暂不支持。

使用 torch.compile，JAX 和 PyTorch 之间的推理速度相当。

## 故障排除

我们将在此处收集常见问题及其解决方案。如果您遇到问题，请先在此处检查。如果您找不到解决方案，请在仓库上提交问题（请参阅[此处](CONTRIBUTING.md)的指南）。

| 问题 | 解决方案 |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `uv sync` 因依赖冲突失败 | 尝试删除虚拟环境目录（`rm -rf .venv`）并重新运行 `uv sync`。如果问题持续存在，请检查您是否安装了最新版本的 `uv`（`uv self update`）。 |
| 训练耗尽 GPU 内存 | 确保在运行训练之前设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`（或更高），以允许 JAX 使用更多 GPU 内存。您也可以使用 `--fsdp-devices <n>`，其中 `<n>` 是您的 GPU 数量，以启用[完全分片数据并行](https://engineering.fb.com/2021/07/15/open-source/fsdp/)，这会以较慢的训练速度换取更低的内存使用（减速程度取决于您的特定设置）。如果仍然内存不足，您可能需要考虑禁用 EMA。 |
| 策略服务器连接错误 | 检查服务器是否正在运行并在预期端口上监听。验证客户端和服务器之间的网络连接和防火墙设置。 |
| 训练时缺少 norm stats 错误 | 在开始训练之前，使用您的配置名称运行 `scripts/compute_norm_stats.py`。 |
| 数据集下载失败 | 检查您的互联网连接。对于 HuggingFace 数据集，请确保已登录（`huggingface-cli login`）。 |
| CUDA/GPU 错误 | 验证 NVIDIA 驱动程序已正确安装。对于 Docker，请确保安装了 nvidia-container-toolkit。检查 GPU 兼容性。您不需要在系统级别安装 CUDA 库 —— 它们将通过 uv 安装。如果遇到 CUDA 问题，您甚至可能想尝试*卸载*系统 CUDA 库，因为系统库有时会导致冲突。 |
| 运行示例时出现导入错误 | 确保您已使用 `uv sync` 安装所有依赖项。一些示例可能有列在其 README 中的其他要求。 |
| 动作维度不匹配 | 验证您的数据处理变换与机器人的预期输入/输出维度匹配。检查策略类中的动作空间定义。 |
| 训练损失发散 | 检查数据集中 `norm_stats.json` 的 `q01`、`q99` 和 `std` 值。某些很少使用的维度可能最终具有非常小的 `q01`、`q99` 或 `std` 值，导致归一化后状态和动作值巨大。您可以手动调整归一化统计量作为解决方法。 |
