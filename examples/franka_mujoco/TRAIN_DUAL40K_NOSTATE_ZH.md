# 纯 nominal 与 joint13 混合数据：各三卡、batch36、40k步

用户于2026-10-29确认同时从官方pi05_base重新训练这两组，均采用LoRA。

| 项目 | nominal基线 | TaskTol混合组 |
|---|---|---|
| 配置 | pi05_franka_nominal100_action7_nostate_3gpu_bs36_40k | pi05_tasktol_mix20_joint32_nostate_4xloss_3gpu_bs36_40k |
| 实验名 | Pi05_nominal100_action7_NoState_LoRA_3gpu_bs36_40k_20261029 | TaskTol_mix20_joint13_4xActionLoss_NoState_LoRA_3gpu_bs36_40k_20261029 |
| GPU | 0,1,2（NUMA0） | 4,5,6（NUMA1） |
| 数据ID | caozilin/franka_100_nominal | caozilin/franka_nominal80_tolerance20_potpost001_dsp16_20261027 |
| 数据组成 | 2500 nominal轨迹，421102帧 | 2000 nominal+500 tolerance轨迹，421029帧 |
| 训练目标 | 前7维动作 | 前7维动作+6维旋转容差 |
| 轨迹动作权重 | 全部1 | nominal=1，tolerance=4 |
| 旋转容差loss权重 | 无 | 0.5，不乘轨迹4倍权重 |
| 归一化统计 | state7/actions7 | state7/actions13 |

两组共同设置：原始32维π0.5输入输出投影，不重新初始化动作头；有效目标归一化后补零到32。
`joint13`指13维有效监督，不是原生13维架构。两组均为state关闭（discrete_state_input=False），无连续state分支；文本容量48，action_horizon16。
`gemma_2b_lora`、`gemma_300m_lora`，使用原有get_freeze_filter；视觉编码器及动作输入输出投影不在llm冻结范围内，仍可训练。

每组3张3090、每卡12、全局batch36，fsdp_devices=1，无梯度累积。
总计40000次更新，warmup1000到5e-5，余弦衰减至总第40000步的5e-6；decay_steps包含warmup。
AdamW b1=.9,b2=.95,eps=1e-8,weight_decay=1e-10,clip_gradient_norm=1；EMA关闭；seed42。
num_workers=8，loss每10步记录，5000/10000/15000/20000/25000/30000/35000/40000保存并全部保留，共每组8个checkpoint。
SwanLab项目openpi-tasktol，online模式。训练使用`.venv/bin/python`，无需更换依赖版本。

## 资产与缓存

两组分别沿用已按对应数据完整计算过的统计；关闭state不改变数值归一化定义。horizon16也与统计计算时保持一致。
nominal资产目录：`/mnt/SharedData/czl/assets/pi05_franka_mujoco_state_action7/caozilin/franka_100_nominal`，目录名含state仅是历史资产名称，不会开启state。
混合资产目录：`/mnt/SharedData/czl/assets/pi05_franka_mujoco_joint13/caozilin/franka_nominal80_tolerance20_potpost001_dsp16_20261027`，joint13同样只是有效统计维数。
数据根：`/mnt/SharedData/czl/lerobot/<repo_id>`。
各自HF/Arrow缓存：`/mnt/SharedData/czl/hf_datasets/<repo_id>`。
checkpoint根：`/mnt/SharedData/czl/checkpoints/<配置名>/<实验名>/<step>`，完整保存模型参数、优化器和归一化资产。
预计两组16个完整checkpoint约150GB；启动检查时共享盘剩余约462GB。不删除或覆盖旧checkpoint。

## 启动与核对

服务器运行父目录：`/mnt/SharedData/czl/runs/dual40k_nostate_bs36_20261029`。
两个子目录`nominal/`、`joint/`各有`train.log`、`events.jsonl`、`status.json`、`resources.jsonl`、`actual_config.json`、`training_command.json`、`sources/`。
外部监督器`analysis/dual40k_nostate_bs36_20261029/run_group.py`直接调用原始train.py，没有短跑计时器，不会在15分钟后停止。
训练进程和监督器均脱离SSH会话，本地关机或关闭Codex不会终止服务器训练。
退出后写入`exit_status.json`；returncode=0才表示正常完成。

启动前检查配置、两套统计SHA、目标维度、真实样本权重、全部75条指令长度；nominal只读动作，joint样本拼接13维目标后补齐32。
启动后核对真实batch=(36,16,32)、state token关闭、预训练投影形状(32,1024)/(1024,32)、loss分项关系和GPU占用。

## 比较的含义

两组同时改变数据组成、辅助容差监督和tolerance轨迹动作权重，因此比较的是两套完整训练方案，不能将结果差异单独归因为辅助监督。
nominal总loss仅动作项；joint总loss是加权动作项+0.5容差项，不能直接比较总loss。
优先对齐固定执行后端、预测16执行10、相同任务/seed/随机化，再比较成功率、失败类型和共同的未见seed评测。
本次未额外启用独立验证数据集或改写推理配置；后续评测应与本次NoState训练保持一致。

## 实际启动记录

北京时间2026-10-29 22:27两组同时启动。
nominal SwanLab：https://swanlab.cn/@caozilin/openpi-tasktol/runs/xhq6qml1hepi5c3giuq2j 。
joint SwanLab：https://swanlab.cn/@caozilin/openpi-tasktol/runs/sph38xasw78dvvq8rq0go 。
数据预检查通过：全部75条实际指令最长32 token；nominal样本仅1倍权重，joint样本同时确认1/4倍权重。
本地启动参数快照：`analysis/dual40k_nostate_bs36_20261029`。
