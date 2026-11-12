# π0.5 state + action7 训练启动记录

已授权启动正式训练，启动日期 2026-10-28，时区 Asia/Shanghai。
实验：Pi05_nominal100_state_action7_LoRA_7gpu_bs84_15k_lr5e-5_20261028。
SwanLab：https://swanlab.cn/@caozilin/openpi-tasktol/runs/bxuk75iphdqejab69f03x。

- 7 张 RTX 3090；全局 batch 84，每卡 12；总步数 15000。
- 原始 32 维模型接口，严格加载官方 pi05_base，LoRA 微调。
- 启用官方离散 state 输入；max_token_len 200；action_horizon 16。
- 仅前 7 维动作参与 loss，不拼接容差，不使用轨迹额外权重。
- warmup 1000 到 5e-5，余弦衰减至总第 15000 步的 5e-6。
- 每 1500 步保存并保留，loss 每 10 步记录。
- 数据转换、独立数值审计和图像抽查均通过；重新计算本数据的 state7/actions7 统计。
- 单卡合成输入短测通过，真实预训练初始化和 train_step，峰值 17.54 GiB；短测更新没有保存或用于正式训练。
- 正式首条记录 Step 10：loss=0.3808，grad_norm=0.8777；原始 32 维投影和真实 batch 形状已从运行日志核对。

服务器运行目录：/mnt/SharedData/czl/runs/pi05_state_action7_bs84_15k_20261028。
日志：上述目录/train.log；退出状态完成或失败时写入 exit_status.json。
本目录 JSON 为启动时快照，后续进度以服务器和 SwanLab 为准。
