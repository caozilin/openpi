> **历史文档，原生13维训练入口已停用。** 2026-10-28已统一恢复原生32维；本文旧启动/恢复命令不再适用。当前配置与迁移说明见 `JOINT32_MIGRATION_ZH.md`。旧13维checkpoint保留归档，不可用新32维配置直接恢复。

# TaskTol mix20 / 4x action loss / 7 GPUs / 9k steps

完整交接流程（数据修改、转换、归一化、Arrow、后台训练和恢复）见
[七卡训练交接文档](HANDOFF_TASKTOL_7GPU_9K_ZH.md)。

配置名：`pi05_tasktol_mix20_4xloss_7gpu_9k`。
SwanLab 项目：`openpi-tasktol`。
默认实验名：`TaskTol_mix20_4xloss_7gpu_bs84_9k`。

| 参数 | 值 |
|---|---|
| GPU | 0,1,2,3,4,5,6，数据并行，`fsdp_devices=1` |
| 全局 batch | 84，每卡 12 |
| 总步数 | 9,000 |
| 模型 | pi0.5 joint13 LoRA，horizon 16，文本长度 48 |
| 容差轨迹动作 loss 权重 | 4.0；普通轨迹为 1.0 |
| 联合 loss | 加权动作 MSE + 0.5 × 旋转容差 MSE |
| 学习率 | warmup 300 步，峰值 5e-5，余弦衰减至第 9,000 步的 5e-6 |
| 记录间隔 | 每 10 步记录这 10 步的平均 loss 和其他指标 |
| 保存 / 保留周期 | 都为 1,500 步，保留 1500、3000、4500、6000、7500、9000 |
| DataLoader workers | 8 |
| 验证 batch 系数 | 1.0；验证集需另行指定，默认未启用 |

`mix20` 应是已经准备好的 80% nominal / 20% tolerance 混合数据集。
本配置不在训练中自动混合数据，也不将所有样本的 loss 都乘四。
运行前需要显式提供该 LeRobot 数据集的 `repo_id`，避免误用基础数据集。

在仓库根目录启动，替换下面的 repo_id 和路径：

```bash
bash examples/franka_mujoco/train_tasktol_mix20_4xloss_7gpu_9k.sh \
  --data.repo-id caozilin/YOUR_MIX20_DATASET \
  --data.lerobot-dataset-root /absolute/path/to/lerobot/YOUR_MIX20_DATASET
```

如果数据集由 Hugging Face 下载或位于 LeRobot 默认缓存，可省略
`--data.lerobot-dataset-root`。启动前需完成 SwanLab 登录，以及该数据集的
13 维联合归一化统计。统计文件沿用 joint13 的目录布局：

```text
assets/pi05_franka_mujoco_joint13/<repo_id>/norm_stats.json
```

若统计文件位于其他目录，追加 `--data.assets.assets-dir /path/to/assets`。
该目录下面仍应包含 `<repo_id>/norm_stats.json`。

Checkpoint 输出目录：

```text
checkpoints/pi05_tasktol_mix20_4xloss_7gpu_9k/TaskTol_mix20_4xloss_7gpu_bs84_9k/
```

需要断点续训时，在相同命令末尾追加 `--resume`。
更换实验名使用 `--exp-name`；启动脚本默认不会覆盖已有实验。
