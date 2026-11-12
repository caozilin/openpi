# 原始 π0.5 架构、state、7 维动作 LoRA 基线

配置名：`pi05_franka_mujoco_state_action7_lora`，定义于 `src/openpi/training/config.py`。
用户已确认沿用 LoRA。用户于 2026-10-28 确认启动下述 7 卡训练。

## 已确定的模型与监督方式

- 从官方 `gs://openpi-assets/checkpoints/pi05_base/params` 初始化；保留 32 维输入、输出投影，严格检查权重形状，不重新初始化动作头。
- 保留 π0.5 网络、adaRMSNorm 时间条件、flow matching 加噪和采样流程。
- 使用 `gemma_2b_lora` / `gemma_300m_lora`，沿用官方 `get_freeze_filter()`：两套语言/动作专家主干冻结、LoRA 可训练；输入输出投影、视觉编码器等不在该冻结范围内。LoRA 不代表所有其他参数都冻结。
- `discrete_state_input=True`：7 维 state 先归一化，再按官方 π0.5 路径离散化并写入语言 token；随后补零到模型接口的 32 维。不是新增连续 state 分支。
- `max_token_len=200`，采用官方 π0.5 默认容量，避免沿用原先 48/64 token 设置导致带 state 的提示被截断。
- action horizon 暂保留当前 Franka 数据处理使用的 16，待讨论；官方 π0.5 类默认值是 50，LIBERO 示例使用 10。
- 数据只读取 `actions` 的 7 维目标，归一化后补 25 个零；不拼接、不预测容差，不读取容差轨迹标签，不使用 4 倍权重。
- loss 为前 7 维 flow matching 速度误差的均值，再对 batch / horizon 求均值。后 25 维不贡献直接 loss，仍参与原始 32 维加噪和采样计算。
- 推理反归一化后仅返回前 7 维动作。

与官方相比，明确改变的是有效维度 loss 和本项目的数据映射。模型主体、state 编码、预训练加载、LoRA 冻结规则、优化器实现及多卡分片/同步沿用原有实现。不能称为官方完整实验配置的复刻：官方 JAX loss 平均全部 32 维，不同官方示例的 horizon、state 开关、全量/LoRA 微调及超参数也不相同。

## 数据与归一化

服务器数据集：`caozilin/franka_100_nominal`。
路径：`/mnt/SharedData/czl/lerobot/caozilin/franka_100_nominal`。
转换流水线：`/mnt/SharedData/czl/runs/franka_100_nominal_20261028`。
原始计划为 2500 条 nominal 轨迹、421102 帧；必须等转换和独立审计完成再计算统计或训练。
LeRobot 文件可以包含额外容差字段，本配置不会将它们用作模型目标。

新数据必须重新计算 **7 维 state + 7 维 actions** 的统计，使用独立目录，不能复用旧 mix20 或 joint13 的统计。
转换与审计完成后可执行：

```bash
cd /home/czl/openpi
HF_ENDPOINT=https://hf-mirror.com uv run --locked scripts/compute_franka_mujoco_norm_stats.py \
  --dataset-root /mnt/SharedData/czl/lerobot/caozilin/franka_100_nominal \
  --no-joint-task-tolerance \
  --action-horizon 16 \
  --pi05-output-dir /mnt/SharedData/czl/assets/pi05_franka_mujoco_state_action7/caozilin/franka_100_nominal
```

归一化统计的 batch 是扫描分块大小，不要求等于训练 batch；训练与统计的 action horizon 应一致。π0.5 使用官方分位数归一化，state 也需要统计。

## 已确定的超参数

7 张 3090，纯数据并行（fsdp_devices=1），每卡 batch 12，全局 batch 84，总计 15000 步。
学习率 warmup 1000 步到 5e-5，再余弦衰减到第 15000 步的 5e-6；decay_steps 包括 warmup。
AdamW 保持原值，EMA 关闭，num_workers=8，loss 每 10 步记录。
每 1500 步保存并保留，含最终 15000 步，共 10 个 checkpoint。
SwanLab 项目 openpi-tasktol。实验名在启动时指定。

训练命令入口如下（填写新的实验名；先完成归一化和显存短测）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 HF_ENDPOINT=https://hf-mirror.com \
uv run --locked scripts/train.py pi05_franka_mujoco_state_action7_lora \
  --exp-name <实验名> \
  --data.lerobot-dataset-root /mnt/SharedData/czl/lerobot/caozilin/franka_100_nominal \
  --data.hf-datasets-cache-dir /mnt/SharedData/czl/hf_datasets/caozilin/franka_100_nominal \
  --data.assets.assets-dir /mnt/SharedData/czl/assets/pi05_franka_mujoco_state_action7 \
  --checkpoint-base-dir /mnt/SharedData/czl/checkpoints
```

开启 state 和扩大 token 容量后，显存与吞吐需重新检查；旧的无 state、48 token 的 batch 12 结果不能直接视为本配置保证。

## 本次运行

实验名：`Pi05_nominal100_state_action7_LoRA_7gpu_bs84_15k_lr5e-5_20261028`。
运行日志：`/mnt/SharedData/czl/runs/pi05_state_action7_bs84_15k_20261028/train.log`。
启动命令与源码快照保存在同一目录的 `training_command.json` 和 `sources/`。
Checkpoint：`/mnt/SharedData/czl/checkpoints/pi05_franka_mujoco_state_action7_lora/Pi05_nominal100_state_action7_LoRA_7gpu_bs84_15k_lr5e-5_20261028`。
归一化 SHA-256：`e1e3c4e7630f6ee524d99ae99cb0dfe8f61306d83c3feebbfa564edc0c5b7a07`。
单卡合成输入短测：真实预训练权重，batch 12，32 维接口，200 token，4 个训练步通过；JAX allocator 峰值约 17.54 GiB，编译后约 3.55 秒/步。正式 7 卡速度以训练日志为准。
