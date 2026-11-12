# joint13 successmix20训练最终结果

全部训练和独立审计已完成；最终核验2026-11-10T16:25:01.146835+08:00。

- 1875条/312374帧/40任务，1574名义观测＋301容差观测，原始名义actions7逐帧不变；全部13维标签齐全，来源标识完整。
- 六卡0/1/2/3/4/6，batch72=6×12，10000步，warmup333，5e-5余弦到5e-6，官方pi05_base＋LoRA，nostate/horizon10/48tokens/原32维接口监督13维。
- 容差来源动作损失4倍、名义1倍、容差组0.5；SwanLab在线真实FINISHED，1000行×10组指标完整匹配。
- 最终loss0.1988，action0.1219，rotation_tolerance0.0710，weighted_action0.1633，weighted_tolerance0.0355。
- 五checkpoint全部params＋train_state CPU恢复，有限、实际step正确、归一化SHA一致、五份参数确有变化。
- 远端1890文件全SHA再次复核；原仓库、overlay、请求保护通过；所有原训练/监督/审计PID退出，锁释放。

本地数据：`/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109`。

服务器数据：`czl@100.73.28.99:/mnt/SharedData/czl/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109`。

最终checkpoint：`czl@100.73.28.99:/mnt/SharedData/czl/checkpoints/pi05_libero_joint13_successmix20_nostate_6gpu_bs72/Pi05_libero_joint13_successmix20_NoState_LoRA_6gpu_bs72_10k_20261109/10000`。

SwanLab：[训练记录](https://swanlab.cn/@caozilin/openpi/runs/kl28cdu6bv1mywr8n5wv0)。

本地原始证据：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/final_snapshot`，含training_completion.json、training_independent_final_audit.json、checkpoint_final_audit.json、逐checkpoint恢复凭据、swanlab_cloud_final_audit.json、swanlab_cloud_verified_metrics.csv、完整train.log和实际配置。下载SHA与实际PID/锁检查：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/final_evidence_download_verification.json`。

历史纯名义方案已被用户明确替换为本混合方案；20%以每任务成功候选池为分母，总体301/1875=16.05%。损失与旧32维均值不可直接比较；未使用本训练loss声称任务成功率提升。新的400回合评测单独记录。
