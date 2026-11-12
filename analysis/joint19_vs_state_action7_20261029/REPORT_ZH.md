# 旧 joint19 成功实验与当前 state/action7 模型的差异

核对日期：2026-10-29。此次只读训练证据并生成对照文件，没有改动训练配置或启动训练。

旧实验为 `franka_mujoco_v2_joint19_lora`，W&B run `ucflosn1`，配置名 `pi05_franka_mujoco_joint19`。配置训练上限50000，学习率在30000步衰减结束，实际最后日志30100步，运行状态killed。
当前指已完成的 `Pi05_nominal100_state_action7_LoRA_7gpu_bs84_15k_lr5e-5_20261028`；不把后来batch24的几百步短测当作当前完整模型。

## 主要结论

两者都是原始32维π0.5接口，当前模型没有原生13维实验的动作投影形状不匹配或重置问题。主要变化是：19维联合监督改为7维动作监督、关闭state改为开启、数据集变化、batch24约30k次更新改为batch84的15k次更新，以及按更新步数压缩学习率衰减。

## 模型、输入与监督

| 项目 | 旧 joint19 | 当前 state/action7 |
|---|---|---|
| 模型 | π0.5 + gemma_2b_lora / gemma_300m_lora | 相同 |
| 模型动作接口 | 32维 | 32维 |
| 有效监督维度 | 19维 | 7维 |
| 前7维 | 物理动作，权重1 | 相同 |
| 第7–12维 | 阶段目标姿态，6D旋转表示，权重0.5 | 无辅助目标，补零且不计loss |
| 第13–18维 | 6维旋转容差，权重0.5 | 无辅助目标，补零且不计loss |
| 无监督补零部分 | 第19–31维 | 第7–31维 |
| state是否进入网络 | 关闭：discrete_state_input=false | 开启：true |
| state数据 | 7维，归一化后补齐模型接口；本身不表示网络已使用state | 7维先归一化、离散化，写入官方π0.5文本token，再补齐state数组接口 |
| 文本容量 | 48 token | 200 token |
| action_horizon | 16 | 16 |
| 相机映射 | 外部相机+腕部相机；右腕占位被mask | 相同 |
| 初始化配置来源 | 官方pi05_base，32维形状兼容 | 官方pi05_base，运行日志确认32维投影完整加载 |
| 恢复训练 | resume=true，确切恢复来源未在导出中确认 | resume=false，从base开始 |
| LoRA冻结规则 | 冻结llm中非LoRA权重 | 相同 |
| 视觉、动作投影等 | 不在llm冻结范围内，可训练 | 相同 |
| EMA | None | None |
| bf16模型计算配置 | bfloat16 | 相同 |

19和7是有效监督维数，不是网络层数。两版Gemma模型变体相同，没有因此把Transformer改成19层或7层。开启state也没有新增连续state投影分支；使用的是π0.5原有离散token路径。

旧目标：

```text
L_old = mean(L_action[0:7] + 0.5 * L_stage_target_pose[7:13] + 0.5 * L_rotation_tolerance[13:19])
```

当前目标：

```text
L_current = mean(L_action[0:7])
```

每个分组先对自己的维度取均值；前7维action项的分母没有从19/32改为7，两版本来都是7维均值。旧版总loss含辅助项，不能直接与当前总loss比较。当前没有此前mix20实验的容差轨迹4倍动作权重。

阶段目标姿态是6D旋转表示，不是6维末端位置姿态，也不是额外6个控制自由度。旧模型不仅学动作，还同时学阶段目标朝向和容差描述；辅助监督可能改变学到的表示，但当前没有控制变量实验能确定它对旧成功率的贡献。

## 数据和归一化

| 项目 | 旧 joint19 | 当前 state/action7 |
|---|---|---|
| repo_id | caozilin/franka_mujoco | caozilin/franka_100_nominal |
| 实际数据快照 | 此次证据未包含原始快照 | 已审计2500条nominal轨迹、421102帧，25个来源任务 |
| 轨迹是否含扰动/容差执行 | 不能由repo_id或joint19名推断；旧快照未确认 | 100% nominal，非此前80/20混合数据 |
| 辅助标注 | stage_target_pose + rotation_tolerance | 文件可保留额外标签，但训练不读取作目标 |
| 数据协议 | 对应历史转换器存在阶段标注及容差profile路径；旧运行确切源码提交未取得 | schema8.0/release5.0 |
| 归一化方式 | π0.5工厂使用quantile | 同样quantile |
| 统计维度 | state7，actions19，再补齐32 | state7，actions7，再补齐32 |
| 统计具体值 | 未取得此旧运行的norm_stats | 按新数据全量重新计算 |
| 资产位置 | assets_base_dir=/root/gpufree-share/openpi-assets，factory默认按配置名/repo_id解析 | /mnt/SharedData/czl/assets/pi05_franka_mujoco_state_action7/caozilin/franka_100_nominal |

相同的归一化方法不表示统计值相同，也不能保证两套数据的动作分布、任务难度、标注或控制时序完全一致。历史源码用于解释19维监督语义，不能当作旧运行确切源码哈希的证明。

## 优化与日程

| 参数 | 旧 joint19 | 当前完整模型 |
|---|---|---|
| GPU | 你此前描述为4090单卡；W&B导出未直接记录卡数 | 7张3090 |
| 全局batch | 24 | 84，每卡12 |
| 实际更新次数 | 30100，通常概括为30k | 15000 |
| 配置训练上限 | 50000 | 15000 |
| warmup | 1000步 | 1000步 |
| peak_lr | 5e-5 | 相同 |
| decay_steps | 30000，包含warmup | 15000，包含warmup |
| decay_lr | 5e-6 | 相同 |
| AdamW b1/b2 | 0.9 / 0.95 | 相同 |
| eps | 1e-8 | 相同 |
| weight_decay | 1e-10 | 相同 |
| clip_gradient_norm | 1.0 | 相同 |
| seed | 42 | 相同；不同输入、batch和监督不保证相同训练轨迹 |
| fsdp_devices | 1 | 1 |
| worker数 | 2 | 8 |
| loss记录间隔 | 100步 | 10步 |
| 保存间隔 | 1000步 | 1500步 |
| 长期保留间隔 | 5000步 | 1500步 |
| 追踪 | W&B，project=openpi | SwanLab，project=openpi-tasktol |
| 验证集 | 导出没有独立验证配置证据 | 两个validation_repo_id均未设置 |

30k×24=720000个样本；15k×84=1260000个样本，后者为前者1.75倍。但更新次数仅一半。若按实际30100步计算，旧样本量722400，比例约1.744。当前数据对应约2.99次frame-anchor遍历；旧数据帧数未取得，无法计算旧epoch数。

按更新步数，当前余弦衰减周期压缩约一半；按样本量则不能这么理解。达到warmup峰值所需样本，旧约24000，当前84000。

| 更新步数 | 旧LR | 当前LR |
|---|---:|---:|
| 1000 | 5.00e-5 | 5.00e-5 |
| 7500 | 4.46e-5 | 3.00e-5 |
| 15000 | 2.87e-5 | 5.00e-6 |
| 30000 | 5.00e-6 | 当前已结束 |

## 日志对照及解释边界

| 窗口 | 动作loss均值 | 总loss均值 |
|---|---:|---:|
| 旧14500–15000，6点 | 0.02127 | 0.03795 |
| 旧29600–30100，6点 | 0.01114 | 0.02085 |
| 当前14500–15000，51点 | 0.00905 | 0.00905 |

当前最终15000步单条动作loss=0.0095；旧30100步单条动作loss=0.01127。不能据此断定当前闭环更好：数据、统计、辅助目标、更新日程和日志平滑窗口不同，没有同一验证集或同一推理协议下的对照评测。旧版效果特别好的判断来自你的实际执行经验，不是通过跨运行比较总loss得出的。

## 最值得优先对齐的内容

1. 旧数据与当前数据的实际任务、动作分布及控制/推理协议；repo_id已经不同，旧快照未确认。
2. 旧版两项辅助监督是否对动作学习有益。当前数据协议没有旧stage_target_pose的完整来源，不能简单在配置中填6个零当作复现。
3. batch24和约30k次更新，以及对应30k步学习率日程。不能仅按总样本量换算出等效训练程度。
4. state关闭改为开启。开启state通常提供额外条件，但不能从一次实验推断它必然改善或损害当前任务。

如果要复现旧模型，单独把batch改24、步数改30k，还没有恢复旧数据和19维联合监督。本次没有足够证据把任一差异指定为旧成功率高的唯一原因。

## 证据文件

- 原始旧记录：`data/wandb_exports/openpi_2026-10-26/runs/franka_mujoco_v2_joint19_lora__ucflosn1/metadata.json`、`history.csv`。
- 当前运行命令及源码哈希：`analysis/pi05_state_action7_15k_20261028/training_command.json`。
- 当前完整日志：`analysis/pi05_bs24_vs_bs84_20261029/baseline_bs84_train.log`。
- 当前实际参数（应用命令行路径和实验名覆盖）：本目录`current_actual_config.json`。
- 逐字段差异、同值字段和loss窗口统计：本目录`comparison.json`；重算脚本`compare.py`。
- 此前`analysis/joint19_vs_tasktol9k_20261028/COMPARISON_ZH.md`比较的是原生13维mix20版本，不能当作本次32维state/action7版本的差异表。
