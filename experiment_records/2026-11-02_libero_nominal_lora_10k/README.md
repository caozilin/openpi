# LIBERO名义数据：6卡LoRA训练10000步历史归档

状态：原训练2026-11-02完成并通过独立审计；2026-11-09只读核对历史耗时，未启动或恢复任何训练。

## 配置、时间与结果

1875条成功重构演示、312374帧、40任务、20Hz；pi05_base官方LoRA、nostate、48tokens，6张RTX3090，GPU0/1/2/3/4/6，各卡batch12、全局72，FSDP2；10000步、warmup333、5e-5余弦到5e-6，每2000步保存。最终loss0.0248、grad_norm0.0765，退出码0；2000/4000/6000/8000/10000五个checkpoint完整恢复审计通过。

北京时间2026-11-02 **00:37:58启动正式训练进程，19:09:35训练结束**，墙钟用时 **18小时31分37秒（18.527小时）**。除以10000步为6.6697秒/步；该口径包括进程启动、参数加载、JAX编译和定期checkpoint保存，不是纯GPU单步时间，不包含前面的上传和CPU归一化。CPU归一化此前已完成；原归一化spawn导入错误和SwanLab环境别名启动错误保留在历史run，不并入本次正常训练进程耗时。

完成后五checkpoint全CPU恢复审计于19:15:59完成，最终训练独立审计于19:22:43完成，均不计入18小时31分37秒。

## 本地与远端源地址

- 本地运行交接及启动/结束时间证据：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261101_full_v1/SERVER_HANDOFF.md`，其中“最新正式训练启动（2026-11-02 00:39）”记录PID1731826/1731828于00:37:58启动，“训练本身已于2026-11-02 19:09:35完整结束”记录完成。
- 配置：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261101_full_v1/training_request.json`；完成凭据：同目录`training_completion.json`；独立审计`training_independent_final_audit.json`；报告`TRAINING_FINAL_REPORT.md`。本次计算与源SHA清单：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-02_libero_nominal_lora_10k/timing_verification.json`。
- 本地原数据：`/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_full_20261101`；原始物理轨迹/视频来源见`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261101_full_v1`的重构来源索引与原记录，未复制或修改。
- 远端训练run：`czl@100.73.28.99:/mnt/SharedData/czl/runs/pi05_libero_reconstructed_lora_6gpu_bs72_10k_20261101`；其`train.log`、`training_process.json`、`training_state.json`、`normalization.log`及attempts保留。远端模型：`czl@100.73.28.99:/mnt/SharedData/czl/checkpoints/pi05_libero_reconstructed_nostate_6gpu_bs72/Pi05_libero_reconstructed_NoState_LoRA_6gpu_bs72_10k_20261101`。远端路径不能视作本地文件，本次未连接服务器。
- SwanLab历史实验：[最终10000步训练](https://swanlab.cn/@caozilin/openpi/runs/95dgsykkqsun5pvkuts3t)，原独立审计已记录FINISHED与1000行真实指标。此后闭环评测的次数和成功率另见400/2000回合记录，不属于训练耗时。
