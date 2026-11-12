# 2026-10-28：从 9000 步续训至 15000 步

**当前状态：已按用户要求终止。** 后台任务退出码130，七张GPU的利用率恢复为0%，显存恢复为约44–59MiB。已有完整checkpoint为1500、3000、4500、6000、7500、9000；没有完成15000步。下文是已终止实验的启动配置记录，不代表当前仍在训练。

用户授权：检查服务器占用，根据 batch 差异选择学习率，继续训练至 15000 步。

启动验证：恢复的 step、Adam count、schedule count 均为9000；已经实际推进到9010步以上，首个10步汇总 loss=0.0776，七卡利用率约99–100%。核验记录见 `resume_startup_verified.json`。这里只确认续训已启动，15000步尚未完成。

## 配置

- 继续原 joint13 模型、同一 mix20 数据集、同一归一化参数与 AdamW；不重建动作投影。
- GPU 0–6，共七张 3090；每卡 batch12，全局 batch84。
- 从服务器 9000 步的 `params` 和 `train_state` 恢复；核验训练 step、Adam count、学习率 schedule count 均为 9000。
- 9000步学习率 `5e-6`；500步内线性升至 `5e-5`；9500–15000步保持 `5e-5`。
- 选择依据：batch84 是旧 batch24 的 3.5 倍，但模型目标、4倍容差轨迹动作权重也有变化，因此不按 batch 比例机械放大到 `1.75e-4`。采用本次原训练已经使用过的峰值 `5e-5`，通过平滑回升避免突变。此为续训实验选择，并非已验证的最优学习率。
- 训练步数上限15000；每1500步保存并保留 checkpoint，每10步汇总记录 loss 和学习率。
- 继续原 SwanLab run，项目 `openpi-tasktol`；checkpoint 路径、实验名仍含9k，以保留原位置和记录身份；实际新日程在 `resume_plan.json` 中保存。
- 当前数据加载器不保存迭代位置，续训会重新建立 shuffled iterator；模型参数与优化器状态则完整恢复。

## 文件与运行位置

启动脚本：`examples/franka_mujoco/resume_tasktol_mix20_to15k.py`，本地与服务器均有副本。

服务器代码目录：`/home/czl/openpi`。

服务器续训日志目录：

```text
/mnt/SharedData/czl/runs/TaskTol_mix20_4xloss_7gpu_bs84_resume15k_20261028
```

其中 `resume_plan.json` 保存实际配置与学习率检查点，`restore_verified.json` 保存恢复状态核验，`train.log` 为训练日志，`exit_status.json` 由后台 supervisor 在训练退出时生成。

原训练控制台统一保留四位小数，因此 `learning_rate=0.0000` 是显示舍入，不代表学习率为零；SwanLab中的数值保留实际精度。

继续写入原 checkpoint 目录：

```text
/mnt/SharedData/czl/checkpoints/pi05_tasktol_mix20_4xloss_7gpu_9k/TaskTol_mix20_4xloss_7gpu_bs84_9k_potpost001_dsp16_20261027
```

新保存步数预期为10500、12000、13500、15000，原9000步保留。

## 运行方式

已在服务器用独立会话的后台 supervisor 启动，不依赖本地 SSH 会话保持连接。重复运行前先检查该日志目录的 `launch.json`、`process.json`、`exit_status.json`，避免启动两个训练任务。

服务器资源检查时七卡利用率均为0%，显存占用约44–59MiB；有其他登录会话和CPU进程，因此这里只确认GPU空闲。共享盘当时剩余约609.5GiB，新四个checkpoint预计约36GiB，有充足余量。
