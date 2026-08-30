# openpi

openpi holds open-source models and packages for robotics, published by the [Physical Intelligence team](https://www.physicalintelligence.company/).

Currently, this repo contains three types of models:
- the [π₀ model](https://www.physicalintelligence.company/blog/pi0), a flow-based vision-language-action model (VLA).
- the [π₀-FAST model](https://www.physicalintelligence.company/research/fast), an autoregressive VLA, based on the FAST action tokenizer.
- the [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05), an upgraded version of π₀ with better open-world generalization trained with [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation). Note that, in this repository, we currently only support the flow matching head for both $\pi_{0.5}$ training and inference.

For all models, we provide _base model_ checkpoints, pre-trained on 10k+ hours of robot data, and examples for using them out of the box or fine-tuning them to your own datasets.

This is an experiment: $\pi_0$ was developed for our own robots, which differ from the widely used platforms such as [ALOHA](https://tonyzhaozh.github.io/aloha/) and [DROID](https://droid-dataset.github.io/), and though we are optimistic that researchers and practitioners will be able to run creative new experiments adapting $\pi_0$ to their own platforms, we do not expect every such attempt to be successful. All this is to say: $\pi_0$ may or may not work for you, but you are welcome to try it and see!

## Updates

- [Sept 2025] We released PyTorch support in openpi.
- [Sept 2025] We released pi05, an upgraded version of pi0 with better open-world generalization.
- [Sept 2025]: We have added an [improved idle filter](examples/droid/README_train.md#data-filtering) for DROID training.
- [Jun 2025]: We have added [instructions](examples/droid/README_train.md) for using `openpi` to train VLAs on the full [DROID dataset](https://droid-dataset.github.io/). This is an approximate open-source implementation of the training pipeline used to train pi0-FAST-DROID. 


## Requirements

To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. Please also note that the current training script does not yet support multi-node training.

| Mode               | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Inference          | > 8 GB          | RTX 4090           |
| Fine-Tuning (LoRA) | > 22.5 GB       | RTX 4090           |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H100 |

The repo has been tested with Ubuntu 22.04, we do not currently support other operating systems.

## Installation

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

**Docker**: As an alternative to uv installation, we provide instructions for installing openpi using Docker. If you encounter issues with your system setup, consider using Docker to simplify installation. See [Docker Setup](docs/docker.md) for more details.




## Model Checkpoints

### Base Models
We provide multiple base VLA model checkpoints. These checkpoints have been pre-trained on 10k+ hours of robot data, and can be used for fine-tuning.

| Model        | Use Case    | Description                                                                                                 | Checkpoint Path                                |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| $\pi_0$      | Fine-Tuning | Base [π₀ model](https://www.physicalintelligence.company/blog/pi0) for fine-tuning                | `gs://openpi-assets/checkpoints/pi0_base`      |
| $\pi_0$-FAST | Fine-Tuning | Base autoregressive [π₀-FAST model](https://www.physicalintelligence.company/research/fast) for fine-tuning | `gs://openpi-assets/checkpoints/pi0_fast_base` |
| $\pi_{0.5}$    | Fine-Tuning | Base [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05) for fine-tuning    | `gs://openpi-assets/checkpoints/pi05_base`      |

### Fine-Tuned Models
We also provide "expert" checkpoints for various robot platforms and tasks. These models are fine-tuned from the base models above and intended to run directly on the target robot. These may or may not work on your particular robot. Since these checkpoints were fine-tuned on relatively small datasets collected with more widely available robots, such as ALOHA and the DROID Franka setup, they might not generalize to your particular setup, though we found some of these, especially the DROID checkpoint, to generalize quite broadly in practice.

| Model                    | Use Case    | Description                                                                                                                                                                                              | Checkpoint Path                                       |
| ------------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| $\pi_0$-FAST-DROID       | Inference   | $\pi_0$-FAST model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): can perform a wide range of simple table-top manipulation tasks 0-shot in new scenes on the DROID robot platform | `gs://openpi-assets/checkpoints/pi0_fast_droid`       |
| $\pi_0$-DROID            | Fine-Tuning | $\pi_0$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): faster inference than $\pi_0$-FAST-DROID, but may not follow language commands as well                                | `gs://openpi-assets/checkpoints/pi0_droid`            |
| $\pi_0$-ALOHA-towel      | Inference   | $\pi_0$ model fine-tuned on internal [ALOHA](https://tonyzhaozh.github.io/aloha/) data: can fold diverse towels 0-shot on ALOHA robot platforms                                                          | `gs://openpi-assets/checkpoints/pi0_aloha_towel`      |
| $\pi_0$-ALOHA-tupperware | Inference   | $\pi_0$ model fine-tuned on internal [ALOHA](https://tonyzhaozh.github.io/aloha/) data: can unpack food from a tupperware container                                                                                                             | `gs://openpi-assets/checkpoints/pi0_aloha_tupperware` |
| $\pi_0$-ALOHA-pen-uncap  | Inference   | $\pi_0$ model fine-tuned on public [ALOHA](https://dit-policy.github.io/) data: can uncap a pen                                                                                                          | `gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap`  |
| $\pi_{0.5}$-LIBERO      | Inference   | $\pi_{0.5}$ model fine-tuned for the [LIBERO](https://libero-project.github.io/datasets) benchmark: gets state-of-the-art performance (see [LIBERO README](examples/libero/README.md)) | `gs://openpi-assets/checkpoints/pi05_libero`      |
| $\pi_{0.5}$-DROID      | Inference / Fine-Tuning | $\pi_{0.5}$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/) with [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation): fast inference and good language-following | `gs://openpi-assets/checkpoints/pi05_droid`      |


By default, checkpoints are automatically downloaded from `gs://openpi-assets` and are cached in `~/.cache/openpi` when needed. You can overwrite the download path by setting the `OPENPI_DATA_HOME` environment variable.




## Running Inference for a Pre-Trained Model

Our pre-trained model checkpoints can be run with a few lines of code (here our $\pi_0$-FAST-DROID model):
```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    ...
    "prompt": "pick up the fork"
}
action_chunk = policy.infer(example)["actions"]
```
You can also test this out in the [example notebook](examples/inference.ipynb).

We provide detailed step-by-step examples for running inference of our pre-trained checkpoints on [DROID](examples/droid/README.md) and [ALOHA](examples/aloha_real/README.md) robots.

**Remote Inference**: We provide [examples and code](docs/remote_inference.md) for running inference of our models **remotely**: the model can run on a different server and stream actions to the robot via a websocket connection. This makes it easy to use more powerful GPUs off-robot and keep robot and policy environments separate.

**Test inference without a robot**: We provide a [script](examples/simple_client/README.md) for testing inference without a robot. This script will generate a random observation and run inference with the model. See [here](examples/simple_client/README.md) for more details.





## Fine-Tuning Base Models on Your Own Data

We will fine-tune the $\pi_{0.5}$ model on the [LIBERO dataset](https://libero-project.github.io/datasets) as a running example for how to fine-tune a base model on your own data. We will explain three steps:
1. Convert your data to a LeRobot dataset (which we use for training)
2. Defining training configs and running training
3. Spinning up a policy server and running inference

### 1. Convert your data to a LeRobot dataset

We provide a minimal example script for converting LIBERO data to a LeRobot dataset in [`examples/libero/convert_libero_data_to_lerobot.py`](examples/libero/convert_libero_data_to_lerobot.py). You can easily modify it to convert your own data! You can download the raw LIBERO dataset from [here](https://huggingface.co/datasets/openvla/modified_libero_rlds), and run the script with:

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

**Note:** If you just want to fine-tune on LIBERO, you can skip this step, because our LIBERO fine-tuning configs point to a pre-converted LIBERO dataset. This step is merely an example that you can adapt to your own data.

### 2. Defining training configs and running training

To fine-tune a base model on your own data, you need to define configs for data processing and training. We provide example configs with detailed comments for LIBERO below, which you can modify for your own dataset:

- [`LiberoInputs` and `LiberoOutputs`](src/openpi/policies/libero_policy.py): Defines the data mapping from the LIBERO environment to the model and vice versa. Will be used for both, training and inference.
- [`LeRobotLiberoDataConfig`](src/openpi/training/config.py): Defines how to process raw LIBERO data from LeRobot dataset for training.
- [`TrainConfig`](src/openpi/training/config.py): Defines fine-tuning hyperparameters, data config, and weight loader.

We provide example fine-tuning configs for [π₀](src/openpi/training/config.py), [π₀-FAST](src/openpi/training/config.py), and [π₀.₅](src/openpi/training/config.py) on LIBERO data.

Before we can run training, we need to compute the normalization statistics for the training data. Run the script below with the name of your training config:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

Now we can kick off training with the following command (the `--overwrite` flag is used to overwrite existing checkpoints if you rerun fine-tuning with the same config):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

The command will log training progress to the console and save checkpoints to the `checkpoints` directory. You can also monitor training progress on the Weights & Biases dashboard. For maximally using the GPU memory, set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` before running training -- this enables JAX to use up to 90% of the GPU memory (vs. the default of 75%).

**Note:** We provide functionality for *reloading* normalization statistics for state / action normalization from pre-training. This can be beneficial if you are fine-tuning to a new task on a robot that was part of our pre-training mixture. For more details on how to reload normalization statistics, see the [norm_stats.md](docs/norm_stats.md) file.

### 3. Spinning up a policy server and running inference

Once training is complete, we can run inference by spinning up a policy server and then querying it from a LIBERO evaluation script. Launching a model server is easy (we use the checkpoint for iteration 20,000 for this example, modify as needed):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

This will spin up a server that listens on port 8000 and waits for observations to be sent to it. We can then run an evaluation script (or robot runtime) that queries the server.

For running the LIBERO eval in particular, we provide (and recommend using) a Dockerized workflow that handles both the policy server and the evaluation script together. See the [LIBERO README](examples/libero/README.md) for more details.

If you want to embed a policy server call in your own robot runtime, we have a minimal example of how to do so in the [remote inference docs](docs/remote_inference.md).



### More Examples

We provide more examples for how to fine-tune and run inference with our models on the ALOHA platform in the following READMEs:
- [ALOHA Simulator](examples/aloha_sim)
- [ALOHA Real](examples/aloha_real)
- [UR5](examples/ur5)

## PyTorch Support

openpi now provides PyTorch implementations of π₀ and π₀.₅ models alongside the original JAX versions! The PyTorch implementation has been validated on the LIBERO benchmark (both inference and finetuning). A few features are currently not supported (this may change in the future):

- The π₀-FAST model
- Mixed precision training
- FSDP (fully-sharded data parallelism) training
- LoRA (low-rank adaptation) training
- EMA (exponential moving average) weights during training

### Setup
1. Make sure that you have the latest version of all dependencies installed: `uv sync`

2. Double check that you have transformers 4.53.2 installed: `uv pip show transformers`

3. Apply the transformers library patches:
   ```bash
   cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
   ```

This overwrites several files in the transformers library with necessary model changes: 1) supporting AdaRMS, 2) correctly controlling the precision of activations, and 3) allowing the KV cache to be used without being updated.

**WARNING**: With the default uv link mode (hardlink), this will permanently affect the transformers library in your uv cache, meaning the changes will survive reinstallations of transformers and could even propagate to other projects that use transformers. To fully undo this operation, you must run `uv cache clean transformers`.

### Converting JAX Models to PyTorch

To convert a JAX model checkpoint to PyTorch format:

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config name> \
    --output_path /path/to/converted/pytorch/checkpoint
```

### Running Inference with PyTorch

The PyTorch implementation uses the same API as the JAX version - you only need to change the checkpoint path to point to the converted PyTorch model:

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = "/path/to/converted/pytorch/checkpoint"

# Create a trained policy (automatically detects PyTorch format)
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference (same API as JAX)
action_chunk = policy.infer(example)["actions"]
```

### Policy Server with PyTorch

The policy server works identically with PyTorch models - just point to the converted checkpoint directory:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=/path/to/converted/pytorch/checkpoint
```

### Finetuning with PyTorch

To finetune a model in PyTorch:

1. Convert the JAX base model to PyTorch format:
   ```bash
   uv run examples/convert_jax_model_to_pytorch.py \
       --config_name <config name> \
       --checkpoint_dir /path/to/jax/base/model \
       --output_path /path/to/pytorch/base/model
   ```

2. Specify the converted PyTorch model path in your config using `pytorch_weight_path`

3. Launch training using one of these modes:

```bash
# Single GPU training:
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>

# Example:
uv run scripts/train_pytorch.py debug --exp_name pytorch_test
uv run scripts/train_pytorch.py debug --exp_name pytorch_test --resume  # Resume from latest checkpoint

# Multi-GPU training (single node):
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Example:
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume

# Multi-Node Training:
uv run torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
```

### Precision Settings

JAX and PyTorch implementations handle precision as follows:

**JAX:**
1. Inference: most weights and computations in bfloat16, with a few computations in float32 for stability
2. Training: defaults to mixed precision: weights and gradients in float32, (most) activations and computations in bfloat16. You can change to full float32 training by setting `dtype` to float32 in the config.

**PyTorch:**
1. Inference: matches JAX -- most weights and computations in bfloat16, with a few weights converted to float32 for stability
2. Training: supports either full bfloat16 (default) or full float32. You can change it by setting `pytorch_training_precision` in the config. bfloat16 uses less memory but exhibits higher losses compared to float32. Mixed precision is not yet supported.

With torch.compile, inference speed is comparable between JAX and PyTorch.

## Bell Tolerance20：data → share → Arrow → 训练

本节记录 `bell_tolerance20` 数据集在 GPU 容器上的完整离线训练流程。路径约定如下：

```text
本地原始数据：
/media/czl/sata/franka_my_code/franka_mujoco/datasets/bell_tolerance20

服务器原始数据（data）：
/root/gpufree-data/bell_tolerance20

转换后的 LeRobot 数据（share）：
/root/gpufree-share/datasets/caozilin/franka_mujoco_bell_tolerance20

训练用 Arrow 缓存（data）：
/root/gpufree-data/franka_mujoco_bell_tolerance20_arrow

训练用轻量 metadata（data）：
/root/gpufree-data/franka_mujoco_bell_tolerance20_metadata

归一化统计（share）：
/root/gpufree-share/openpi-assets/pi05_franka_mujoco_joint19_state/caozilin/franka_mujoco_bell_tolerance20/norm_stats.json

训练 checkpoint（share）：
/root/gpufree-share/checkpoints/pi05_franka_mujoco_joint19_state/franka_mujoco_bell_tolerance20_joint19_state_lora
```

### 1. 连接服务器并更新代码

服务器入口：

```bash
ssh -p 31283 root@183.147.142.40
```

如果服务器访问 GitHub 需要使用本机 Clash/Mihomo，本机代理监听在 `127.0.0.1:7897`。在本机单独打开一个终端，并始终保持下面的反向隧道运行：

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -R 127.0.0.1:17890:127.0.0.1:7897 \
  -p 31283 \
  root@183.147.142.40
```

在服务器设置 Git HTTP 代理并拉取代码：

```bash
git config --global http.proxy http://127.0.0.1:17890
git config --global https.proxy http://127.0.0.1:17890
git config --global http.version HTTP/1.1

cd /root/openpi
git remote set-url origin https://github.com/caozilin/openpi.git
git pull
```

### 2. 从本机上传原始数据到 data

在本机执行。源路径末尾的 `/` 表示上传目录内容，而不是额外嵌套一层同名目录：

```bash
rsync -avP --partial --info=progress2 \
  -e 'ssh -p 31283' \
  /media/czl/sata/franka_my_code/franka_mujoco/datasets/bell_tolerance20/ \
  root@183.147.142.40:/root/gpufree-data/bell_tolerance20/
```

### 3. 定义服务器路径并创建目录

以下命令均在服务器执行：

```bash
cd /root/openpi

RAW_ROOT=/root/gpufree-data/bell_tolerance20
REPO_ID=caozilin/franka_mujoco_bell_tolerance20
LEROBOT_HOME=/root/gpufree-share/datasets
LEROBOT_ROOT=/root/gpufree-share/datasets/caozilin/franka_mujoco_bell_tolerance20
ARROW_ROOT=/root/gpufree-data/franka_mujoco_bell_tolerance20_arrow
META_ROOT=/root/gpufree-data/franka_mujoco_bell_tolerance20_metadata
ASSETS_ROOT=/root/gpufree-share/openpi-assets/pi05_franka_mujoco_joint19_state
TMP_ROOT=/root/gpufree-data/openpi_tmp

mkdir -p \
  "$LEROBOT_HOME" \
  "$ARROW_ROOT" \
  "$META_ROOT/meta" \
  "$ASSETS_ROOT/caozilin/franka_mujoco_bell_tolerance20" \
  "$TMP_ROOT" \
  /root/gpufree-share/checkpoints
```

### 4. 将原始数据转换到 share

转换器会自动读取原始数据根目录中的 `tolerance_summary.csv`。首次转换执行：

```bash
HF_LEROBOT_HOME="$LEROBOT_HOME" \
TMPDIR="$TMP_ROOT" \
uv run --no-sync python \
  examples/franka_mujoco/convert_franka_mujoco_data_to_lerobot.py \
  --raw-dir "$RAW_ROOT" \
  --repo-id "$REPO_ID" \
  --workers 12 \
  --image-writer-processes 0 \
  --image-writer-threads 6 \
  --progress-interval-seconds 10
```

若转换中断且输出目录已经存在，在相同命令末尾添加 `--resume`。只有明确需要删除并重建现有转换结果时才使用 `--overwrite`。

### 5. 计算 Bell Tolerance20 的归一化统计

必须针对当前数据集重新计算 state 和 joint19 action 的统计量，不能直接沿用 v3 全量数据集的统计：

```bash
cd /root/openpi

uv run --no-sync scripts/compute_franka_mujoco_norm_stats.py \
  --dataset-root "$LEROBOT_ROOT" \
  --joint-output-dir \
    "$ASSETS_ROOT/caozilin/franka_mujoco_bell_tolerance20" \
  --pi05-output-dir \
    /root/gpufree-share/openpi-assets/pi05_franka_mujoco_state/caozilin/franka_mujoco_bell_tolerance20
```

训练配置通过 `--data.assets.assets-dir="$ASSETS_ROOT"` 读取统计，并根据 `REPO_ID` 自动追加 `caozilin/franka_mujoco_bell_tolerance20`。

### 6. 将 metadata 和 Arrow 缓存放到 data

训练样本使用 Arrow 缓存，但 LeRobot 仍需要 metadata 提供 FPS、episode 边界和任务文本。只复制 `meta/`，无需复制 share 中的 parquet：

```bash
cp -a "$LEROBOT_ROOT/meta/." "$META_ROOT/meta/"
```

从 share 中的 LeRobot parquet 生成 data 中的 Arrow 缓存：

```bash
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
HF_DATASETS_CACHE="$ARROW_ROOT" \
HF_LEROBOT_HOME="$LEROBOT_HOME" \
TMPDIR="$TMP_ROOT" \
uv run --no-sync python -c \
"from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; d=LeRobotDataset('$REPO_ID', root='$LEROBOT_ROOT'); print(f'Arrow cache generated: {len(d)} frames')"
```

完成后，训练只从 `ARROW_ROOT` 读取样本；`META_ROOT` 仅提供轻量元数据，转换后的完整 LeRobot 数据继续保存在 share。

### 7. π₀.₅ base 在 share 中的位置

`pi05_franka_mujoco_joint19_state` 配置的底模地址是：

```text
gs://openpi-assets/checkpoints/pi05_base/params
```

训练命令设置：

```bash
OPENPI_DATA_HOME=/root/gpufree-share/cache/openpi
```

因此 OpenPI 将上述 GCS 地址映射到下面的本地 share 缓存目录：

```text
/root/gpufree-share/cache/openpi/openpi-assets/checkpoints/pi05_base/params
```

离线训练前该目录必须已经存在。`HF_HUB_OFFLINE=1` 和 `HF_DATASETS_OFFLINE=1` 不会自动下载缺失的 π₀.₅ base。

### 8. 使用 data 中的 Arrow 缓存训练

```bash
cd /root/openpi

HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
HF_DATASETS_CACHE=/root/gpufree-data/franka_mujoco_bell_tolerance20_arrow \
OPENPI_DATA_HOME=/root/gpufree-share/cache/openpi \
TMPDIR=/root/gpufree-data/openpi_tmp \
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run --no-sync scripts/train.py pi05_franka_mujoco_joint19_state \
  --exp-name=franka_mujoco_bell_tolerance20_joint19_state_lora \
  --checkpoint-base-dir=/root/gpufree-share/checkpoints \
  --data.repo-id=caozilin/franka_mujoco_bell_tolerance20 \
  --data.tolerance-trajectory-action-loss-weight=4.0 \
  --data.assets.assets-dir=/root/gpufree-share/openpi-assets/pi05_franka_mujoco_joint19_state \
  --data.lerobot-dataset-root=/root/gpufree-data/franka_mujoco_bell_tolerance20_metadata \
  --data.hf-datasets-cache-dir=/root/gpufree-data/franka_mujoco_bell_tolerance20_arrow \
  --data.lerobot-arrow-cache-dir=/root/gpufree-data/franka_mujoco_bell_tolerance20_arrow
```

该配置使用 7 维末端 state、16 步 action chunk 和 19 维联合目标。CLI 参数 `--data.tolerance-trajectory-action-loss-weight` 默认为 `1.0`，因此不传参数时所有轨迹都保持原始 action 权重，而且兼容不含 `trajectory_is_tolerance` 列的旧 LeRobot 数据。只有将该参数设为非 `1.0` 时，数据集才必须提供这个标签。上面的 Bell 命令显式设置为 `4.0`：nominal 轨迹的 loss 为 `1.0 × action + 0.5 × target rotation + 0.5 × tolerance`；tolerance 轨迹只把物理 action 项提高到 `4.0`，另外两项仍为 `0.5`。

### 9. 使用 tar + pv 拉取 10000 步 checkpoint

以下命令在本机执行。远端先用 `tar` 将整个 checkpoint 合并为单一数据流，通过 SSH 传输；本机用 `pv` 显示实时速度、已传输大小和进度，再直接解包到对应实验目录。`train_state/` 会在远端打包阶段被排除，从而避免传输优化器状态。

```bash
REMOTE_PARENT=/root/gpufree-share/checkpoints/pi05_franka_mujoco_joint19_state/franka_mujoco_bell_tolerance20_joint19_state_lora
LOCAL_PARENT=/media/czl/sata/franka_my_code/openpi/checkpoints/pi05_franka_mujoco_joint19_state/franka_mujoco_bell_tolerance20_joint19_state_lora

mkdir -p "$LOCAL_PARENT"

ssh -p 31283 root@183.147.142.40 \
  "tar -C '$REMOTE_PARENT' --exclude='10000/train_state' -cf - 10000" \
  | pv \
  | tar -C "$LOCAL_PARENT" -xf -
```

下载完成后的目录为：

```text
/media/czl/sata/franka_my_code/openpi/checkpoints/pi05_franka_mujoco_joint19_state/franka_mujoco_bell_tolerance20_joint19_state_lora/10000
```

本地 checkpoint 保留以下推理文件：

```text
10000/params/
10000/assets/
10000/_CHECKPOINT_METADATA
```

不会下载：

```text
10000/train_state/
```

该精简 checkpoint 可以用于推理，但不能用于恢复训练。如果本机已经存在旧的 `10000/train_state/`，本次解包不会主动删除它；应使用一个不含旧训练状态的目标目录。

### 10. 部署 Bell Tolerance20 的 10000 步 checkpoint

训练时通过 `--data.repo-id=caozilin/franka_mujoco_bell_tolerance20` 覆盖了通用配置，因此 checkpoint 中的归一化统计位于：

```text
10000/assets/caozilin/franka_mujoco_bell_tolerance20/norm_stats.json
```

现有 `pi05_franka_mujoco_joint19_state` 配置默认查找 `assets/caozilin/franka_mujoco/`。部署前将 Bell 统计文件复制到该配置预期的位置即可，无需新增训练配置：

```bash
cd /media/czl/sata/franka_my_code/openpi

CKPT_ROOT=/media/czl/sata/franka_my_code/openpi/checkpoints/pi05_franka_mujoco_joint19_state/franka_mujoco_bell_tolerance20_joint19_state_lora
CKPT_DIR="$CKPT_ROOT/10000"

mkdir -p "$CKPT_DIR/assets/caozilin/franka_mujoco"

cp \
  "$CKPT_DIR/assets/caozilin/franka_mujoco_bell_tolerance20/norm_stats.json" \
  "$CKPT_DIR/assets/caozilin/franka_mujoco/norm_stats.json"

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
uv run --no-sync scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_franka_mujoco_joint19_state \
  --policy.dir="$CKPT_DIR"
```

## Troubleshooting

We will collect common issues and their solutions here. If you encounter an issue, please check here first. If you can't find a solution, please file an issue on the repo (see [here](CONTRIBUTING.md) for guidelines).

| Issue                                     | Resolution                                                                                                                                                                                   |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `uv sync` fails with dependency conflicts | Try removing the virtual environment directory (`rm -rf .venv`) and running `uv sync` again. If issues persist, check that you have the latest version of `uv` installed (`uv self update`). |
| Training runs out of GPU memory           | Make sure you set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (or higher) before running training to allow JAX to use more GPU memory. You can also use `--fsdp-devices <n>` where `<n>` is your number of GPUs, to enable [fully-sharded data parallelism](https://engineering.fb.com/2021/07/15/open-source/fsdp/), which reduces memory usage in exchange for slower training (the amount of slowdown depends on your particular setup). If you are still running out of memory, you may want to consider disabling EMA.        |
| Policy server connection errors           | Check that the server is running and listening on the expected port. Verify network connectivity and firewall settings between client and server.                                            |
| Missing norm stats error when training    | Run `scripts/compute_norm_stats.py` with your config name before starting training.                                                                                                          |
| Dataset download fails                    | Check your internet connection. For HuggingFace datasets, ensure you're logged in (`huggingface-cli login`).                                                                                 |
| CUDA/GPU errors                           | Verify NVIDIA drivers are installed correctly. For Docker, ensure nvidia-container-toolkit is installed. Check GPU compatibility. You do NOT need CUDA libraries installed at a system level --- they will be installed via uv. You may even want to try *uninstalling* system CUDA libraries if you run into CUDA issues, since system libraries can sometimes cause conflicts. |
| Import errors when running examples       | Make sure you've installed all dependencies with `uv sync`. Some examples may have additional requirements listed in their READMEs.                    |
| Action dimensions mismatch                | Verify your data processing transforms match the expected input/output dimensions of your robot. Check the action space definitions in your policy classes.                                  |
| Diverging training loss                            | Check the `q01`, `q99`, and `std` values in `norm_stats.json` for your dataset. Certain dimensions that are rarely used can end up with very small `q01`, `q99`, or `std` values, leading to huge states and actions after normalization. You can manually adjust the norm stats as a workaround. |
