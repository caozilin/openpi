# 两组40k训练已启动

2026-10-29 22:27，北京时间。两组同时从官方pi05_base开始，LoRA；不是恢复15k或旧13维checkpoint。

| 组别 | GPU | 全局batch | 目标步数 | 首批已确认步数 | loss |
|---|---|---:|---:|---:|---:|
| nominal | 0,1,2 | 36 | 40000 | 20 | 0.3008 |
| joint | 4,5,6 | 36 | 40000 | 20 | 0.9957 |

共同参数：每卡12、NoState、32维预训练接口、max_token_len48、horizon16；warmup1000/peak5e-5/cosine40000/end5e-6；每5000步保存并保留，loss每10步记录。
joint使用13维有效监督，旋转容差项权重0.5，只有tolerance轨迹的动作项乘4；nominal仅前7维动作。
43项相关回归测试通过；真实数据目标、样本权重、统计SHA、全部75条指令长度均已检查；最长32 token。
运行日志确认两组真实batch=(36,16,32)，投影(32,1024)/(1024,32)，state关闭，官方base权重加载，loss有限且分项正确。
监督器父进程PID=1，无TTY；训练不依赖本地电脑或SSH，长期运行没有短跑时间限制。
共享盘启动前剩余约462GB，预估两组16个完整checkpoint约150GB。GPU3留空。
两组改变数据组成、辅助监督及轨迹权重，比较完整方案效果，不能当成单变量消融。评测须使用对应NoState配置，不能直接复用上一模型的state开启设置。

nominal SwanLab：https://swanlab.cn/@caozilin/openpi-tasktol/runs/xhq6qml1hepi5c3giuq2j
checkpoint：/mnt/SharedData/czl/checkpoints/pi05_franka_nominal100_action7_nostate_3gpu_bs36_40k/Pi05_nominal100_action7_NoState_LoRA_3gpu_bs36_40k_20261029

joint SwanLab：https://swanlab.cn/@caozilin/openpi-tasktol/runs/sph38xasw78dvvq8rq0go
checkpoint：/mnt/SharedData/czl/checkpoints/pi05_tasktol_mix20_joint32_nostate_4xloss_3gpu_bs36_40k/TaskTol_mix20_joint13_4xActionLoss_NoState_LoRA_3gpu_bs36_40k_20261029

服务器父目录：/mnt/SharedData/czl/runs/dual40k_nostate_bs36_20261029；子目录nominal/joint有train.log、status.json、events.jsonl和资源记录。
本地文件为启动快照，实时进度以服务器和SwanLab为准。完整配置和说明见 examples/franka_mujoco/TRAIN_DUAL40K_NOSTATE_ZH.md。
