# LIBERO joint13：成功候选池20%容差观测替换训练

状态：**10000步训练与完整独立审计均已完成**。最终证据核验时间2026-11-10T16:25:01.146835+08:00。此实验延续用户原任务；用户后续明确将“全名义”改为每任务成功容差候选池随机20%替换，并要求容差来源动作损失4倍及SwanLab在线。

## 当前证据

| 要求 | 当前结果 | 本地证据 |
|---|---|---|
| 新数据与本地全量独立审计 | 1875条/312374帧/40任务，1574名义＋301容差来源；缺标0 | [dataset_independent_audit.json](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/dataset_independent_audit.json) |
| 远端逐文件SHA与实际LeRobot读取 | 1890文件/38,388,719,043字节全部通过 | [lerobot_upload_verification.json](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/lerobot_upload_verification.json) |
| 专属state8/actions13归一化 | 4338batch真实exit0，全部有限 | [normalization_completion.json](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/normalization_completion.json) |
| 4倍动作损失部署 | 40回归＋真实样本CPU/Observation/JIT预检通过 | [training_preflight_verification.json](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/four_x_protocol_update/training_preflight_verification.json) |
| 真实训练与SwanLab | Step10000；1000条指标全部有限、4x恒等式和云端10组指标逐值核验通过 | [training_prefix100_verification.json](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/training_prefix100_verification.json) |
| 10000步、五checkpoint及最终独立审计 | **全部通过**，原训练/监督/审计进程退出、锁释放 | [最终证据](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/final_snapshot/training_independent_final_audit.json) |

最终日志：`Step 10000: batch/action_loss_weight_mean=1.4208, batch/reweighted_action_fraction=0.1403, grad_norm=0.7136, loss=0.1988, loss/action=0.1219, loss/action_nominal_contribution=0.1081, loss/action_reweighted_contribution=0.0138, loss/rotation_tolerance=0.0710, loss/total=0.1988, loss_weighted/action=0.1633, loss_weighted/rotation_tolerance=0.0355, param_norm=1804.9407`。训练损失仅说明本训练目标的数值状态，不能替代最终checkpoint审计或任务成功率测评。

## 数据与替换规则

逐任务仅从已审计、完整执行且最终原生成功谓词为真的容差回放中抽样。1503成功候选，按任务N取round(N×20%)，随机种子42及固定SHA派生子种子，合计301条。最终容差来源占1875的16.05%，分母是成功候选池，不能写成1875的20%。完整抽样在[selection.json](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/selection.json)，组成在[dataset_composition.json](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/dataset_composition.json)。

| 套件 | 全部 | 名义观测 | 容差观测 |
|---|---:|---:|---:|
| Spatial | 465 | 390 | 75 |
| Object | 493 | 408 | 85 |
| Goal | 458 | 374 | 84 |
| Long | 459 | 402 | 57 |
| 合计 | 1875 | 1574 | 301 |

301入选来源替换为同源容差物理回放pre_action state8与双相机256视频解码观测，trajectory_is_tolerance=1；1574条保留原名义观测、flag=0。**所有1875条原名义物理actions7保持逐帧不变**；时间戳、语言、task/episode/frame/demo身份不变。容差标签沿用原审计1851＋补24，单位rad，顺序rx-/rx+/ry-/ry+/rz-/rz+，使用搜索边界而非实际释放量。用户已确认标签未修改。

原MP4是有损H.264，转换严格匹配其解码像素，不宣称与原渲染无损一致；图像此前已旋转180度一次，转换不重复旋转。全量审计覆盖全部动作/标签/身份、1574原观测与301双视频所有帧/state，并完成40任务×2来源类型的80个实际LeRobot十步读取。301条共47564帧，总帧数仍312374，20Hz。

## 正式训练协议

- 原始pi05_base，官方Gemma2B与300M expert LoRA配置；nostate、48tokens、horizon10，原32维输入/输出投影，前13维作为动作7＋容差6监督。未随机新建13维头；新增官方LoRA照常初始化，两次seed42。原32维投影从预训练checkpoint加载。
- GPU0/1/2/3/4/6，batch72=6×12，FSDP2，8数据worker，seed42，10000步。
- warmup333，5e-5 cosine→5e-6；AdamW b1=.9、b2=.95、eps1e-8、weight_decay1e-10、clip1；无EMA。
- log10；save2000/keep2000，保留2000/4000/6000/8000/10000五份。
- 对batch和horizon求均值：`loss = mean(w_i * mean(MSE动作7) + 0.5 * mean(MSE容差6))`。名义来源w=1，容差来源w=4；不按权重总和重新归一化，剩余19维填充不计损失。
- SwanLab开启在线，云端实验[kl28cdu6bv1mywr8n5wv0](https://swanlab.cn/@caozilin/openpi/runs/kl28cdu6bv1mywr8n5wv0)，cfg online/SDK环境cloud；记录原始/加权动作与容差损失、总损失、来源贡献和批次权重。

冻结请求SHA：`4c9a7fc7d88dc7c1c07df20c7276a264ed6f9df052cca151318370c8a7268fb0`。数据manifest SHA：`c83584914ad3eb169dcee7bb489990f562a2db4175d8d2ebd121f1002aca5d8d`。归一化SHA：`e26cd8d5baade92e2f081435cb497df85f569439fa1e12e4f93fa7b66d728e42`。请求和实际配置在[training_request.json](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/training_request.json)、[actual_config.json](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/four_x_startup_snapshot/actual_config.json)。

旧训练loss0.0958是对32维求均值，本次raw action0.4343对7维求均值；换算当前动作7贡献为32维分母约0.0950。动作前7维mean/std/q01/q99逐值完全相同。动作误差定义未变，但均值系数进入反传，加上容差监督、4x权重和观测替换，整体训练不等价；不能把不同口径曲线直接用于性能排名。[只读诊断](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/loss_scale_comparison.json)。两次首条Step10均为前10步平均。

## 本地与服务器原始地址

- 正式本地LeRobot：`/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109`。请求replacement_request.json、转换manifest、meta独立审计和1875个Parquet均在此目录。
- 原名义数据（保护）：`/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_full_20261101`。
- 容差物理回放（保护）：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1`；successful_sources.json、independent_final_audit.json与episodes下逐源双MP4/state/trajectory/report。
- 搜索母集（保护）：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107`；补24：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_missing24_baseline_observation_20261109`。
- 服务器数据：`czl@100.73.28.99:/mnt/SharedData/czl/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109`。
- 服务器run：`czl@100.73.28.99:/mnt/SharedData/czl/runs/pi05_libero_joint13_successmix20_lora_6gpu_bs72_10k_20261109`；training_request/actual_config/training_process/state、training.lock、train.log、pipeline.log、swanlog、source_overlay及完成监督/审计日志均在此目录。
- checkpoint父目录：`czl@100.73.28.99:/mnt/SharedData/czl/checkpoints/pi05_libero_joint13_successmix20_nostate_6gpu_bs72/Pi05_libero_joint13_successmix20_NoState_LoRA_6gpu_bs72_10k_20261109`。五份完整checkpoint及最终训练审计：**均已生成并通过CPU全量恢复检查**。
- 最终证据（已生成并下载本地final_snapshot）：上述run/training_completion.json、checkpoint_final_audit.json、swanlab_cloud_final_audit.json、training_independent_final_audit.json、task_completion.json。

完整机器可读地址：[sources.json](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/sources.json)。当前恢复与监督入口：[HANDOFF.md](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/HANDOFF.md)。

## 阶段变更与历史证据

1. 用户授权删除从未训练的旧纯名义joint13转换副本及部分上传，已完成。保护原名义数据、原始HDF5、容差回放与标签。[本地退役](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/nominal_conversion_retirement.json)、[远端删除核验](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/remote_nominal_partial_upload_deletion_verification.json)；小型旧元数据保留。
2. 原rsync真实活跃但较慢，证明三源Parquet逐字节复现后切换紧凑传输。240,203,401字节/2479紧凑文件，服务器802已有SHA正确文件复用、1073重建，最终1875 Parquet均与原冻结SHA一致；错误/原文件回退0。早期1个不完整文件移独立partial_files保存，未改数据manifest。[传输小结](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/exact_transport_reconstruction_summary.json)。
3. 正式归一化旧总控3312670/norm3312672完整4338batch正常退出。用户随后新增4x请求，曾仅STOP旧父以阻止旧配置启动GPU，统计继续。完成后定向退出旧父/旧结束监督3210109，保留原请求/源码/receipt；旧GPU优化步数0。新4x总控3339995、训练3339998、最终监督3339996于20:37启动，统计目标与输入未变，原receipt SHA保留。[版本切换证据](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/four_x_protocol_update/transition_completion.json)、[统计复用证据](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/four_x_protocol_update/normalization_reuse_verification.json)。
4. 本地两次早期监督启动兼容问题均在attempts保留，未启动额外训练或重测样本。曾有观察超时/审批额度问题，后续真实核验恢复，不能当训练失败。[历史全文](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/README_HISTORY_20261109_2110.md)保留整理前所有文字；运行交接历史另在HANDOFF_HISTORY_20261109_2110.md。

## 最终验收

真实训练10000步/exit0；2000、4000、6000、8000、10000五份checkpoint的params及train_state全部CPU恢复，全部有限、实际step精确、归一化SHA一致、32维原投影保留，五份参数摘要不同。10..10000共1000条日志完整且有限，全部4x分组恒等式通过；SwanLab FINISHED，1000行×10指标与日志逐值匹配。全1890数据文件SHA再次复核，原服务器仓库和训练overlay未变。原训练总控3339995、训练3339998、结束监督3339996、审计3906177均已退出，三把锁均释放。

最终总损失0.1988、动作原始损失0.1219、容差原始损失0.0710；加权动作0.1633、加权容差0.0355。训练完成于2026-11-10 16:11:00（北京时间），从11月9日20:37:36启动约19小时33分24秒。训练完成不等价于闭环任务成功率。

[RESULTS.md](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/RESULTS.md)、[最终全量审计](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/final_snapshot/training_independent_final_audit.json)、[取回及进程/锁核验](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/final_evidence_download_verification.json)。用户后续已授权仅对最终10000步做预测容差400回合，见[新评测记录](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_predicted_tolerance_400/README.md)。这是另一次闭环评测，未重训。

## 长期服务器文档

用户要求的SSH/密钥路径与指纹、项目及大文件路径、7×3090硬件和多人共用规则已加入本地[AGENTS必读入口](/media/czl/sata/franka_my_code/openpi/AGENTS.md)与[TRAINING_SERVER.md](/media/czl/sata/franka_my_code/openpi/docs/TRAINING_SERVER.md)。用户额外明确要求保存口令，已仅写本机Git忽略的私有凭据文件（目录700/文件600），未将口令或私钥内容写入本实验记录或公共说明；未提交/推送/上传，未改远端仓库或authorized_keys。公钥登录当前未获远端接受，不能误称已配置密钥登录。凭据保护验证server_documentation_verification.json、硬件/主机公开指纹验证server_inventory_verification_20261109.json位于本实验根；当前训练保持运行。
