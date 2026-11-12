# 重构 LIBERO：原始 π0.5，6 卡 × batch 12

配置名：`pi05_libero_reconstructed_nostate_6gpu_bs72`。

|设置|值|
|---|---|
|初始化|`gs://openpi-assets/checkpoints/pi05_base/params`，LoRA 微调|
|模型|原始 π0.5，Gemma 2B + Gemma 300M expert 均使用 LoRA，32 维模型接口|
|state 输入|关闭：`discrete_state_input=False`，π0.5 无连续 state 分支|
|文本长度|48 tokens；已核对重构 LIBERO 最长 21、测试数据最长 32，无指令截断|
|数据集|`caozilin/libero_reconstructed_full_20261101`，1875 条、312374 帧、40 任务|
|batch|6 卡，每卡 12，全局 72；`fsdp_devices=2`，三组各两卡分片|
|训练步数|10000|
|保存 / 记录|每 2000 步保存并保留 checkpoint，每 10 步记录指标|
|学习率|前 333 步 warmup 至 5e-5，然后余弦衰减至第 10000 步的 5e-6；warmup 比例约 1/30|
|action horizon|10，沿用现有 `pi05_libero`|
|优化器 / EMA|AdamW，梯度裁剪 1.0；按官方 LoRA 示例关闭 EMA|
|SwanLab|开启 online，每 10 步记录；服务器日志保存在本次运行目录的 `swanlog/`|

使用标准 LIBERO 数据变换和原任务语言。动作是重构后的 7 维物理增量，
不再转 delta、不套 OSC 缩放；按原始 π0.5 方式补零到 32 维。
数据文件仍保留 8 维 state 用于兼容和审计，训练时不会送入语言 token 或连续 state 分支。

转换及全部像素、动作校验完成后，先计算本数据集专属的归一化统计，再启动训练：

```bash
bash examples/libero/train_reconstructed_pi05_6gpu_bs72.sh --compute-norm-stats
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash examples/libero/train_reconstructed_pi05_6gpu_bs72.sh
```

脚本使用仓库 `.venv/bin/python`，可通过 `OPENPI_PYTHON` 覆盖。
默认在服务器使用 `/mnt/SharedData/czl/lerobot`，本地使用仓库 `data/lerobot`；
可显式设置 `HF_LEROBOT_HOME`。Arrow 缓存默认位于同盘的 `.cache/hf_datasets`，
也可用 `HF_DATASETS_CACHE` 覆盖。脚本会检查完整转换审计，避免训练读取转换中的数据集。
GPU 内存按需分配（关闭 XLA 预分配），与测速设置一致；Arrow 使用系统内存池。

归一化统计保存到
`assets/pi05_libero_reconstructed_nostate_6gpu_bs72/caozilin/libero_reconstructed_full_20261101/norm_stats.json`，
不复用原始 OSC LIBERO 的统计。默认 checkpoint 写入
`checkpoints/pi05_libero_reconstructed_nostate_6gpu_bs72/Pi05_libero_reconstructed_nostate_6gpu_bs72_10k`。
训练命令后可追加 `--checkpoint-base-dir /mnt/SharedData/czl/checkpoints`、`--exp-name ...` 等参数。

测试已按用户要求停止，不再重试。正式 10000 步训练使用重构 LIBERO，
在完整转换审计、上传及远程校验完成后开始。

## 已停止的服务器 100 步性能测试（历史）

此前性能测试使用 `benchmark_pi05_lora.py` 调用服务器现有的真实 JAX 训练器，batch 为 96。
这项测试已撤销；当时只计划执行前 100 步，
每 10 步在本地文件记录指标，SwanLab 关闭，不生成测试 checkpoint。
统计首 10 步（包含编译）与后 90 步的速度，并每秒采样 6 张 GPU 的显存。

测试复用服务器已有 `caozilin/franka_100_nominal` 及其归一化统计：
该数据为 224 像素、10Hz、7 维 state；重构 LIBERO 为 256 像素、20Hz、8 维 state。
模型端均为 224 像素、7 维动作补零至 32 维、state 关闭，
此次实测用于模型显存和训练速度参考，数据读取耗时会受数据集差异影响。

首次使用默认 200 tokens 的测试在第一步 OOM。随后用同一官方 tokenizer
核对全部任务指令后，正式配置和重试配置均缩短文本补齐至 48 tokens，保留完整有效指令。
48 tokens 的纯数据并行重试仍在第一步 OOM；继续保持 batch 和模型不变，
第三次测试启用两卡 FSDP 分片，随后被用户主动停止。正式配置保留 FSDP2，
按最新指令改为每卡 12、全局 72；不再启动 100 步测试。
