# 原生13维接口、投影重新初始化与后续checkpoint溯源

核对日期：2026-10-28。只读取历史和训练产物，没有改训练配置或启动正式训练。

## 结论

1. **首次引入“缩小原生接口＋跳过不匹配投影权重”的提交是2026-09-12的 `bf0556c00318d03b1e1ab23a3b53f3a7ea193d1d`，当时是32→15维。**
2. **首次在当前可见Git历史中提交原生13维配置的是2026-10-27的 `5e0f53646038836babdd36675fcd1bba845e9568`，由15→13维，并沿用上述加载机制。**
3. 当前可访问证据中，确认的13维训练只有这次TaskTol mix20实验及其同一实验的续训。已保存6个checkpoint，最后为9000步；续训日志到10490步后由用户终止，没有新增10500或15000步checkpoint。
4. 旧W&B导出的45个实验配置全部为32维；本地旧checkpoint的动作投影也全部为32维。未发现joint15训练checkpoint或另一组独立joint13训练checkpoint。

这些日期是最早已提交证据，不代表能够确定更早未提交编辑的发生时间。未访问旧gpufree服务器，不能据此排除那里未导出的训练产物。

## Git源头

### 2026-09-12 14:58:40 +08:00：缩小接口和跳过机制同时引入

- 提交：`bf0556c00318d03b1e1ab23a3b53f3a7ea193d1d`
- Git作者：`czl`
- 标题：`feat: replace joint19 with joint15 tolerance descriptor`
- 父提交：`37054c12d84feecd4423794e43d7a3a49d82fa9e`

该提交同时做了以下修改：

- `pi05_franka_mujoco_joint19`改为`pi05_franka_mujoco_joint15`，明确设置`action_dim=15`；state版本也同样修改。
- 有效监督由动作7＋阶段目标姿态6＋容差6，改为动作7＋容差参考框架编码2＋容差6。
- 两个joint15配置启用`skip_mismatched_shapes=True`。
- `CheckpointWeightLoader`首次新增该开关；`_merge_params`遇到不匹配形状时返回参考初始化，而不载入checkpoint中的该张量。
- 输出解包由允许至少19维改为严格15维。

因此，用该joint15配置从32维pi05_base初始化，就会触发动作输入kernel、动作输出kernel和输出bias的形状不匹配处理。是否实际跑过该配置，不能由代码提交本身证明。

代码快照：

- [15维配置](/media/czl/sata/franka_my_code/openpi/analysis/joint19_vs_tasktol9k_20261028/history_origin/bf0556c/src/openpi/training/config.py:884)
- [首次跳过机制](/media/czl/sata/franka_my_code/openpi/analysis/joint19_vs_tasktol9k_20261028/history_origin/bf0556c/src/openpi/training/weight_loaders.py:49)
- 完整相关diff：`history_origin/bf0556c.diff`

### 2026-10-27 10:20:08 +08:00：15→13维

- 提交：`5e0f53646038836babdd36675fcd1bba845e9568`
- Git作者：`czl`
- 标题：`feat: migrate tracking to SwanLab and add joint13 LIBERO workflows`
- 父提交就是上述`bf0556c`。

该提交把joint15及state配置改成joint13及state，明确设置`action_dim=13`，去掉2维`tolerance_frame_code`，仅保留动作7＋容差6，并沿用`skip_mismatched_shapes=True`。同时更新数据协议、归一化与输出接口，切换SwanLab。

代码快照：[13维配置](/media/czl/sata/franka_my_code/openpi/analysis/joint19_vs_tasktol9k_20261028/history_origin/5e0f536/src/openpi/training/config.py:948)，相关diff：`history_origin/5e0f536.diff`。

本次七卡9k配置是工作区中从`pi05_franka_mujoco_joint13`派生的配置，因此继承了原生13维和跳过不匹配形状的行为。它没有在派生时才首次引入这两项。

## 实际训练及checkpoint证据

### 旧记录

W&B导出：`data/wandb_exports/openpi_2026-10-26/`，manifest记载2026-10-26完整导出`1848166725-hit/openpi`项目45个run，无导出失败。45个配置的`model.action_dim`全部为32，最新run创建于2026-08-31；没有找到joint15/joint13运行记录。

旧joint19v2、joint19v3及其state实验都使用32维模型接口。这里不能根据实验名字中出现19推断模型接口就是19维。

### 本地checkpoint

直接读取17个`params/_METADATA`，没有恢复大权重数组：

| 本地实验 | 已保存步数 | action_out_proj/kernel |
|---|---|---|
| franka_lora_v1 | 1000、2000、3000、5000 | [1024,32] |
| franka_lora_v2 | 5000、10000、15000、20000 | [1024,32] |
| pi05_libero_lora | 5000、10000、20000、29999 | [1024,32] |
| pi05_libero_lora_discrete | 5000、10000、20000、29999 | [1024,32] |
| 本次TaskTol mix20 | 9000，本地推理副本 | [1024,13] |

### 当前服务器

服务器`/mnt/SharedData/czl/checkpoints`只有本次TaskTol配置的实验目录；`/home/czl/openpi/checkpoints`不存在。

实验名：`TaskTol_mix20_4xloss_7gpu_bs84_9k_potpost001_dsp16_20261027`。

| 会话 | 实际运行 | 初始化 | 新增保存点 |
|---|---|---|---|
| 2026-10-28首次9k训练 | 正常完成9000步 | 日志明确记录三个动作投影参数32→13重新初始化 | 1500、3000、4500、6000、7500、9000 |
| 当天续训到15k的会话 | 最后日志10490步，用户终止，退出码130 | 从9000步完整恢复；没有重新执行base投影初始化 | 无；10500保存点尚未产生 |

上述6个服务器checkpoint逐个检查，输出投影形状全部为[1024,13]，并且都包含完整train_state。两个SwanLab日志目录对应同一run id：`iiaue9yel2f2z00b5nozc`，不是两组独立从头训练的模型。

目录`/mnt/SharedData/czl/runs/franka_100_nominal_20261028`目前只有数据转换和审计文件，本次盘点没有找到其训练日志或模型checkpoint。

## 可复查文件

- `history_origin_evidence.json`：提交信息、45个W&B运行概要、17个本地checkpoint形状。
- `history_server_evidence.json`：服务器6个checkpoint形状、两次训练会话及SwanLab配置来源。
- `history_origin/`：两个关键提交的配置、权重加载器、数据接口源码快照和diff。
- `new_run_evidence.json`：本次训练的三个投影重新初始化日志及完成审计。

在本次可见记录中，尚无另一组训练实验能够支持“原生13维且投影重新初始化已经验证过效果良好”这一说法；这次9k实验才是目前确认存在完整checkpoint的该接口训练。
