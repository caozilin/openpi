# Franka联合训练统一使用原生32维接口

2026-10-28：按用户要求删除原生13/15维动作投影及跳过形状不匹配后重新初始化的选择。此次不启动训练，不决定新的batch、步数或学习率。

## 当前配置

| 配置 | 模型接口 | 有效监督 |
|---|---|---|
| `pi05_franka_mujoco_joint32` | 32维 | 动作7＋旋转容差6 |
| `pi05_franka_mujoco_joint32_state` | 32维 | 同上，另启用离散state条件 |
| `pi05_tasktol_mix20_4xloss_7gpu_9k_joint32` | 32维 | 同上，保留原mix20模板的4倍容差轨迹动作权重 |

最后一项仍保留历史模板的batch84/9k数值，仅更换架构和配置、checkpoint目录身份，不表示这些超参数已重新验证或决定用于下一次训练。

旧的`pi05_franka_mujoco_joint13`、`pi05_franka_mujoco_joint13_state`和原mix20配置名已移除。新mix20配置及默认实验名包含`joint32`，避免误入已保存13维checkpoint的旧目录。旧15k续训脚本已移除，已保存的13维权重和历史审计不删除。

Franka数据配置会拒绝非32维的模型覆盖。`CheckpointWeightLoader`中`skip_mismatched_shapes`和`use_reference_for_mismatched`逻辑已删除；不匹配权重会在训练加载校验时报错，不能静默丢弃动作投影并重新初始化。新增LoRA参数仍按正常LoRA流程初始化。

## 数据和归一化

数据集仍存7维物理动作、6维旋转容差和7维state。联合监督拼成13维，先使用13维动作统计归一化，再补19个零至32维。state归一化后按模型接口补到32维。

已有`assets/pi05_franka_mujoco_joint13/<repo_id>/norm_stats.json`保存的是**13维有效数据统计**，继续使用；此目录名不再代表原生13维模型。无需为了补零重新计算32维统计，也不改变现有数据集。推理先反归一化，再取前7维动作和第7–13维容差，丢弃其余19维。

## 官方loss核查

核对官方main的JAX实现：

- `src/openpi/transforms.py:PadStatesAndActions`补零至模型维度。
- `src/openpi/models/pi0.py:compute_loss`返回32维速度预测平方误差的均值，没有动作维度padding mask。
- `scripts/train.py`再对batch和时间取均值。
- 推理输出转换只返回机器人需要的动作维度，例如LIBERO前7维。

来源：

- [官方Pi0/Pi05损失](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)
- [官方padding](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/transforms.py)
- [官方输出截取](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/libero_policy.py)

因此，官方当前JAX代码是**32维都计入loss**，补零维度也接受训练。原始动作在这些维度为零；flow matching的速度目标仍为噪声减动作，所以补零维度的速度目标等于对应噪声，而不是恒为零。希望最终生成的动作补零维度接近零，不能把这理解为每个时刻的速度输出必须为零。

这里记录的是官方当前代码行为；本次没有找到官方专门比较padding mask与全维loss优劣的明确实验建议。

## 本次保留的项目loss

本次只恢复模型接口和权重加载，保留原有分组目标：

```text
mean(w × mean(error[0:7]) + 0.5 × mean(error[7:13]))
w = 1（nominal），4（mix20配置的tolerance轨迹）
```

第13–32维目前不在项目分组loss中。该策略与官方全32维均值不同，与旧成功joint19实验“保留32维接口、只监督有效组别”的方式一致。是否追加padding监督或改为官方全维均值，留作独立的训练目标决定；本次没有同时改变loss尺度或辅助任务权重。

## 验证范围

验证包括32维配置及CLI覆盖限制、完整载入预训练动作投影、拒绝旧13维权重、13维归一化→32维padding→有效输出往返、现有分组loss与全维均值的差异，以及训练/推理相关回归。GPU训练保持停止。
