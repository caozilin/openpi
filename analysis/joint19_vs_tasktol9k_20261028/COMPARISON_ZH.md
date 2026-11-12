# 旧 joint19 与新 TaskTol 9k 实验对比

核对日期：2026-10-28。只读取训练记录和代码；本次未修改模型、训练配置或推理逻辑。

## 结论

**这次不是单纯把旧实验改为 7 卡。模型主干深度相同，但动作接口由 32 维变成原生 13 维，三个动作投影参数重新初始化；旧版的阶段目标姿态辅助监督也被删除。数据集、样本权重、batch 和优化更新次数同时改变。**

13 和 32 均指 `action_dim`，不是网络层数。旧版 joint19 是 19 维有效监督补到 32 维，新版是 13 维有效监督直接使用 13 维接口。

此前“保持模型”的说明对照的是当前分支的 joint13 基础配置，没有对照这次指出的历史 joint19 实验。相对历史实验，确实存在下面列出的结构和训练目标变化。

## 证据来源及边界

- 旧实验：`data/wandb_exports/openpi_2026-10-26/runs/franka_mujoco_v2_joint19_lora__ucflosn1/metadata.json`，以及同目录 `history.csv`；W&B run `ucflosn1`。
- 新实验：服务器 `/mnt/SharedData/czl/swanlog/run-20261028_001729-iiaue9yel2f2z00b5nozc/files/config.yaml`，以及 `/mnt/SharedData/czl/runs/TaskTol_mix20_4xloss_7gpu_bs84_9k_potpost001_dsp16_20261027/` 下的训练日志和审计文件。
- 新实验的实际配置、初始化日志、指标、数据审计已保存为本目录 `new_run_evidence.json`；旧记录保存在 `old_run_evidence.json`；逐字段差异与 loss 窗口均值在 `comparison.json`。
- 历史代码用 `git show 96ecfbc^:<path>` 交叉核对。旧 W&B 导出没有保存确切代码提交和旧数据快照，因此历史代码用于解释当时接口，不能当作旧运行代码哈希的证明。旧 checkpoint 尚未直接取得，本次也没有加载完整权重数组。

## 模型、输入和监督

| 项目 | 旧：franka_mujoco_v2_joint19_lora | 新：TaskTol mix20 9k |
|---|---|---|
| 配置名 | pi05_franka_mujoco_joint19 | pi05_tasktol_mix20_4xloss_7gpu_9k |
| 模型类型 / 精度 | pi0.5 / bfloat16 | 相同 |
| 视觉语言主干 | gemma_2b_lora，宽 2048，18 层 | 相同 |
| 动作专家 | gemma_300m_lora，宽 1024，18 层 | 相同 |
| LoRA rank / alpha | 主干 16/16，动作专家 32/32 | 相同 |
| 视觉编码器 | SigLIP So400m/14 | 相同；新 checkpoint 的扫描层轴为 27 |
| action_dim | 32 | 13 |
| 有效监督布局 | action[0:7] + stage_target_pose[7:13] + rotation_tolerance[13:19]；余下 13 维不计组别 loss | action[0:7] + rotation_tolerance[7:13] |
| stage_target_pose | 6D 旋转表示，表示阶段目标姿态；不是额外 6 个机械臂动作 | 删除 |
| 一次预测长度 | 16 步 | 16 步 |
| 文本长度上限 | 48 tokens | 48 tokens |
| discrete_state_input | false | false |
| 输入图像 | 外部相机 + 腕部相机；右腕占位图被 mask | 相同 |
| state 数据 | 7 维，随后 padding 到模型维度 | 7 维，随后 padding 到模型维度 |
| state 是否进入网络 | pi05=true 且 discrete_state_input=false：未作为状态 token 输入，也无连续 state_proj 分支 | 相同，并非这次才取消状态输入 |
| freeze_filter | 冻结 llm 下非 LoRA 参数 | 相同 |
| 视觉塔、动作投影等 | 不在上述 llm 冻结范围内，可训练 | 相同；不能把此配置理解为只更新 LoRA 参数 |
| EMA | None | None |

主干和动作专家各 18 层是两条专家路径的配置，不应相加后当成此处讨论的 32 或 13。

### 已确认的预训练权重丢失

两版配置的预训练来源都写为 `gs://openpi-assets/checkpoints/pi05_base/params`。旧版 32 维与其动作投影形状兼容；新版启用 `skip_mismatched_shapes=True`。新训练日志明确记录：

```text
Initializing action_in_proj/kernel instead of loading checkpoint shape (32, 1024) into (13, 1024)
Initializing action_out_proj/bias instead of loading checkpoint shape (32,) into (13,)
Initializing action_out_proj/kernel instead of loading checkpoint shape (1024, 32) into (1024, 13)
```

因此新模型的动作输入映射和输出映射重新随机初始化。输入 bias 的形状仍为 1024，不因这次维度变化被跳过。重新初始化的这三个张量共 26,637 个参数；这并非整个模型重新初始化，但它们直接连接动作空间和动作专家，是需要优先验证的差异。

旧记录 `resume=true`，新记录 `resume=false`。旧配置本身不能证明其实际恢复了哪一个 checkpoint；旧 history 从 step 0 开始有记录。因此这里只能确认旧接口具备完整加载 base 投影的形状条件，不能进一步断言旧运行的所有初始化来源。

## Loss 的实际变化

旧目标：

```text
L_old = mean(L_action + 0.5 × L_stage_target_pose + 0.5 × L_rotation_tolerance)
```

新目标：

```text
L_new = mean(w × L_action + 0.5 × L_rotation_tolerance)
w = 1（nominal 轨迹），4（tolerance 轨迹）
```

这里的组别 loss 是 flow matching 速度预测误差，按对应维度取均值；不是直接以物理单位计算的闭环轨迹误差。

**4 倍只乘容差轨迹上的前 7 维物理动作 loss，后 6 维容差预测 loss 仍乘 0.5。** 轨迹类别是训练权重信息，不作为网络输入。因此网络并没有额外获得“现在要求 nominal 还是 tolerance 行为”的类别条件。

20% 是新数据集的轨迹比例，不一定等于 frame 采样比例。若容差帧也恰占 20%，在相同误差量级的假设下，容差帧贡献的物理动作 loss 权重约占 50%：`0.2×4 / (0.8×1+0.2×4)`。这说明目标分布明显偏向容差动作；是否造成动作模糊或偏移，仍需同观测条件下的动作分布和消融实验验证。

## 训练日程与运行参数

| 项目 | 旧实验 | 新实验 |
|---|---|---|
| 全局 batch | 24 | 84 = 7×12 |
| GPU 数量 | 旧导出未记录，不能仅凭 batch 推断 | 完成审计确认 GPU 0–6 |
| fsdp_devices | 1 | 1，同机数据并行 |
| 配置总步数 | 50,000 | 9,000 |
| 实际日志末步 | 30,100，run 状态 killed | 9,000，正常结束 |
| warmup | 1,000 | 300 |
| cosine decay_steps | 30,000 | 9,000 |
| peak_lr / decay_lr | 5e-5 / 5e-6 | 相同 |
| AdamW | b1=.9，b2=.95，eps=1e-8，weight_decay=1e-10 | 相同 |
| 梯度裁剪 | 全局 norm 1.0 | 相同 |
| seed | 42 | 42；张量形状不同，不代表随机初始化和噪声数值相同 |
| num_workers | 2 | 8 |
| log_interval | 100 | 10 |
| save_interval | 1,000 | 1,500 |
| keep_period | 5,000 | 1,500 |
| resume / overwrite | true / false | false / false |
| 日志平台 | W&B，project=openpi | SwanLab，project=openpi-tasktol，online |
| 验证集 | 导出无验证字段 | 新增验证配置字段，但两个 validation_repo_id 都为 null，未配置独立验证集 |

以用户此前说的 30k 为参照，旧训练采样量约 `30,000×24=720,000`，新训练约 `9,000×84=756,000`，多 5%。但优化器更新从 30,000 次降至 9,000 次，仅为 30%；AdamW 状态更新、学习率与 batch 噪声也随之变化，不能据此保证同样收敛。9k 是样本预算近似，不是收敛步数的确定换算。

新数据有 421,029 帧；新训练只相当于约 1.80 次 frame-anchor 数据遍历。每个 anchor 取 16 步窗口，窗口会重叠，不能当成 16 倍独立样本。旧数据帧数尚未取得，无法计算其真实 epoch 数。

## 数据、归一化和推理接口

| 项目 | 旧实验 | 新实验 |
|---|---|---|
| dataset repo_id | caozilin/franka_mujoco | caozilin/franka_nominal80_tolerance20_potpost001_dsp16_20261027 |
| 轨迹数量 / 混合 | 历史导出未提供，比例未知 | 2,500 条：2,000 nominal + 500 tolerance |
| 帧数 | 未取得旧快照 | 421,029 |
| TaskTol 协议 | 历史转换器使用 stage_annotations / stage_target_pose | schema 8.0、release 5.0；独立 tolerance_labels.json |
| 容差语义 | 旧转换器按阶段目标姿态和 profile 生成；确切旧数据标签尚未取得 | 6 个正负方向界限、弧度、realtime_nominal 参考；Pre/Post 标注，grasp/release 为零 |
| sequence keys | actions、stage_target_pose、rotation_tolerance | actions、rotation_tolerance |
| 归一化模式 | pi05 工厂设置 quantile | 同样 quantile；配置中 base_config.use_quantile_norm=false 是工厂处理前的值，不能据此误判 |
| 归一化统计维度 | 有效 actions 为 19 维，随后 padding 到 32；旧统计文件未取得 | actions 13 维、state 7 维，基于全部 421,029 帧重新计算 |
| 训练读取位置 | 配置中本地 root/cache 为 null | 显式指定 /mnt/SharedData/czl 下的 LeRobot 和 HF cache |
| Arrow 专用覆盖 | null | null，使用本次指定的 HF datasets cache |
| 资产位置 | /root/gpufree-share/openpi-assets 下按 dataset id 查找 | /mnt/SharedData/czl/assets/pi05_franka_mujoco_joint13 下按新 dataset id 查找 |
| checkpoint 位置 | /root/gpufree-share/checkpoints | /mnt/SharedData/czl/checkpoints |
| 推理返回 | actions[16,7]、stage_target_pose[16,6]、rotation_tolerance[16,6] | actions[16,7]、rotation_tolerance[16,6] |

新数据审计的 25 个来源任务与 LeRobot info 的 total_tasks=75 是两种记录口径；不能把它直接当作和旧模型相同的任务集合。数据内容和标签语义变化均不是只改网络维度能够概括的。

新训练日志明确加载新 dataset 对应的统计文件。完成审计确认所有 checkpoint 的归一化资产一致。目前服务器 `/mnt/SharedData/czl/runs/openpi_9k_unseen3_dr_policy_20261028/server.log` 也确认使用相应 9000 checkpoint、新 dataset asset id，统计文件 SHA256 为 `ceefe7d23f307025d6ab06711faeecb6c2a815029f91c814ad0ad5e06c8dc21b`，与训练审计一致。因此没有发现这条服务明显加载错归一化文件；这不能替代对实际评测客户端、执行动作长度、相机、prompt、控制坐标和扰动设置的核对。

## 训练 loss 提供的线索

| 日志窗口均值 | 旧 29,600–30,100（6 个记录点） | 新 8,500–9,000（51 个记录点） |
|---|---:|---:|
| loss/action，未乘轨迹权重 | 0.01114 | 0.02635 |
| loss/rotation_tolerance | 0.01279 | 0.07719 |
| loss/stage_target_pose | 0.00664 | 无 |
| grad_norm | 0.3155 | 0.9462 |
| total loss | 0.02085 | 0.08028 |

新 run 的动作 loss 从 step1000 的 0.1407 降到 step9000 的 0.0267，容差 loss 从 0.3864 降到 0.0741，训练没有出现明显 NaN/发散，仍呈下降。旧 run 在 8,500–9,000 的动作 loss 约 0.02952，随后至 30k 附近约 0.0112，也说明不能忽略更新次数。

两版数据、归一化统计和 loss 目标不同，上表不是统一验证集测试，不能把这些比值解释为真实动作误差或成功率恶化的倍数。当前没有证据确定某一项是唯一原因。

## 建议的排查顺序

1. 先固定同一评测任务、相机、prompt、控制约定、执行 chunk、采样设置和扰动种子，重测旧/新 checkpoint，确认比较口径。当前服务的配置与 norm 资产身份已核对通过。
2. 优先做动作投影兼容性的对照：保持当前数据、13 维监督与训练目标，使用 32 维模型接口，13 维目标 padding 到 32，并调整输出解包以只取有效维度，让 base 动作投影可以完整加载。这需要配套修改接口，不能只把 action_dim 改成 32。
3. 在同一模型与数据上比较 tolerance 动作权重 1 和 4。先建立 nominal 行为基线，再判断强调容差轨迹是否有利。
4. 保留相同 batch 与结构，延长训练并根据统一验证集/闭环成功率选 checkpoint；不能保证补到 30k 就一定恢复，也不应继续把 9k 当作 30k 的等效收敛承诺。
5. 若要完整复现旧 joint19，则需取得旧数据与规范，恢复 stage_target_pose 的真实监督，并恢复其输出解包/容差参考定义。当前 schema8 没有该字段，不能用随意填充标签代替复现。

优先级依据是：动作投影重新初始化和辅助监督删除已经确认；它们对性能的实际贡献，以及数据混合和训练预算分别占多少，必须用上述受控比较才能区分。
