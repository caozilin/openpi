21:06本机关机依赖核查通过：训练3339995/3339998、最终监督3339996无终端，两个总控PPID1且独立SID，stdout/stderr仅服务器日志；数据挂载为服务器/dev/sdc1 ext4，不依赖本机/SSH。服务器任务继续，本地关机只中断Codex实时观察；local_shutdown_readiness.json有凭据。用户只是询问关机影响，尚未要求暂停goal或停止任何进程。

20:46实测：新版已真正完成至少10步；六张GPU0/1/2/3/4/6全部真实计算，trainer3339998及8worker、总控3339995、最终监督3339996均存活。Step10 loss1.0650/grad_norm2.5589，raw action0.4343、weighted action0.6414、raw tolerance0.8473、weighted tolerance0.4236；批次动作平均权重1.4667，对应容差来源比例0.1556。SwanLab云端相同步10的全部10指标已读取，数值匹配本地，真实4x贡献恒等式通过。证据four_x_protocol_update/first_real_metrics_and_cloud_verification.json与training_live_verified_20261109_2046.json。此仅首10步验证，不能替代完整10000/五checkpoint/1000云端行/最终独立审计，继续复用当前实例，不重启。

## 最新状态：4倍权重正式训练已启动（20:40）

本地与远端数据完整审计已通过；4338batch归一化真实exit0，state8/actions13全部有限，统计SHA `e26cd8d5baade92e2f081435cb497df85f569439fa1e12e4f93fa7b66d728e42`。用户训练前追加4倍容差动作损失，已完成版本化部署与真实LeRobot/归一化/Observation/JIT预检；原统计输入未变、原receipt SHA未改，不重算统计。旧总控3312670与旧结束监督3210109已定向退出，旧请求/源码完整保存至本实验four_x_protocol_update/before_4x，原GPU优化步数0。

新唯一训练总控3339995、训练子3339998、最终监督3339996真实活跃。请求SHA `4c9a7fc7d88dc7c1c07df20c7276a264ed6f9df052cca151318370c8a7268fb0`，实际数据变换tolerance_trajectory_action_loss_weight=4.0；名义动作7维×1、容差观测来源动作7维×4、容差6维组权重0.5。其他batch72/6卡0,1,2,3,4,6/FSDP2/10000steps/warmup333/原LoRA/48tokens/nostate/horizon10不变。原“无4倍”和“统计运行中/父T”段落均为历史，不能据此CONT旧总控或重复启动。

SwanLab新在线实验已通过OpenApi真实查询，云端RUNNING：[kl28cdu6bv1mywr8n5wv0](https://swanlab.cn/@caozilin/openpi/runs/kl28cdu6bv1mywr8n5wv0)。当前处于训练worker/模型加载阶段，首个真实Step指标及云端loss记录仍待产生；不能把上线/启动等同于10000完成。后续继续检查实际8worker、6卡、前10步4x损失分解与云端同步，最终须五checkpoint完整CPU恢复和全1000条云端逐行审计。

本地证据根 `/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/four_x_protocol_update`：transition_completion、original_normalization_exit、prior_protocol_retirement、normalization_reuse_verification、training_preflight_verification、startup_verification、swanlab_online_startup_verification及unit_test_verification（34+6=40回归）。当前源码/实际配置/进程快照在本实验four_x_startup_snapshot。远端run仍为 `czl@100.73.28.99:/mnt/SharedData/czl/runs/pi05_libero_joint13_successmix20_lora_6gpu_bs72_10k_20261109`，禁止改冻结源码/请求，不动服务器仓库或其他任务。

---

> 最新用户更改（20:20）：容差轨迹动作7维损失4倍、名义1倍、容差6维组权重0.5不变；SwanLab同时记录原始/加权及分组贡献。旧“无额外4倍”描述已被覆盖。训练父3312670已确认后SIGSTOP以阻止旧协议启动GPU，归一化3312672及8workers继续，不重算。准备版本化更新，完整保留旧请求/源码及归一化凭据；不能误判父进程T为故障或直接CONT旧父。后续见four_x_protocol_update/。

# joint13成功候选池20%混合数据：当前交接

最新用户方案覆盖旧“全名义”目标：1875条中按每任务成功容差回放候选池round20%抽样，1503成功候选→301替换＋1574名义，312374帧、原名义actions7不变。用户确认标签未改，沿用审计1851＋补24；user已授权删除旧全名义joint13转换副本和服务器半上传，已完成，仅小型证据保留。原始名义/物理容差回放/标签均保护。

## 当前阶段

本地与远端数据已完整审计通过。正式服务器总控3312670已唯一启动，归一化3312672及8worker真实活动；GPU训练尚未开始，完整10000步及最终审计仍待完成。

旧普通rsync真实活跃但网络较慢；已在独立目录证明名义0、容差2、Long1420（97/89/307帧）重建Parquet与本地冻结SHA逐字节一致，主动切换到紧凑传输。旧上传不是误判失败或断点重测，保护全部已提交文件。原数据manifest和训练代码/超参未改。每个重建文件必须原SHA相等才原子提交；不相等者传原始Parquet回退，最后仍用原独立远端验证入口全量验收。

- 上传136691/136694及训练调度82814均已正常完成并真实退出；upload_state=complete，training_dispatch_state=launched。不要重复启动。新唯一服务器训练总控3312670，归一化3312672，实际8worker见training_startup_verification.json与server_startup_snapshot。
- 原训练调度82814已完成唯一启动并退出，冻结请求SHA9077cac73ceae17e0e9226af65d5221cd1b0089b6a681ba978bee22a6bcb9917；实际训练父子活动以服务器training_process/state和真实/proc为准。
- 原上传82812/120234/120235和远端3219491/3219492已定向退出；803个已有Parquet保留。停止凭据`transport_transition/local_stop_verification.json`及`before_stop.json`。SSH主连接未停止，其他任务/viewer未动。
- 紧凑包 **240,203,401字节/2479文件**（数值IPC＋原MP4＋schema），源 `/tmp/libero_successmix20_exact_transport_20261109`；MP4是原文件软链，rsync-L传实际字节，不再次有损编码。
- `exact_transport_preparation.json`和`transport_transition/plan.json`冻结传输说明。原manifest SHA **c83584914ad3eb169dcee7bb489990f562a2db4175d8d2ebd121f1002aca5d8d**；服务器重建脚本SHA **1501fdec7479db6ccab9de96ba30aafe060c7f556881959eee9d81e3b91c5dd1**。
- 本地源Parquet不改；本次还须远端完整SHA、原动作/标识、实际读取、归一化、10000训练及完整独立审计。

## 路径与观察

仓库 `/media/czl/sata/franka_my_code/openpi`，记录根为本文件所在目录，先读AGENTS.md和experiment_records索引。
新数据 `/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109`；原名义RAW `/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_full_20261101`（保护）。
容差回放 `/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1`；标签在franka_mujoco/analysis_results的full1875_20261107与missing24_20261109，完整路径见README/sources.json。

只读观察：`python3 /media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/observe_progress.py`（require_escalated主机/proc）。它检查真实父子进程、紧凑输入字节、原Parquet数量、服务器重建/验证/归一化/训练及最终监督状态。SSH观测超时不是任务终止；同一实例活跃应复用，不能凭慢或旧状态重启。
旧本地组装82799/审计83257已完成退出；stage47516也退出。原两次启动兼容问题在attempts，未重测数据。曾有一次只读SSH观测超时后恢复，不是上传失败。

## 服务器

czl@100.73.28.99，专用socket `/tmp/libero_joint13_20261109.sock`。凭据仅交互，不写文件日志。Python `/home/czl/openpi/.venv/bin/python`，不装依赖、不修改服务器仓库。
新run `/mnt/SharedData/czl/runs/pi05_libero_joint13_successmix20_lora_6gpu_bs72_10k_20261109`；新数据 `/mnt/SharedData/czl/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109`。
紧凑输入放run/exact_transport_bundle；重建入口run/reconstruct_joint13_exact_transport_server.py（本地源在examples/libero同名）；CPU六并行，无物理仿真/推理/新rollout。`exact_transport_reconstruction_state/process/completion.json`记录真实进度和逐SHA；不匹配的早期上传部分移run/transport_transition/partial_files保留。
远端验证仍为原`verify_successmix20_joint13_upload_server.py`，完成文件lerobot_upload_verification.json需取回本记录并与原manifest一致。
服务器完成监督 **3210109** `finish_successmix20_joint13_training_server.py`保持活跃等待本训练；不得重复启动。

## 训练与最终验收

配置pi05_libero_joint13_successmix20_nostate_6gpu_bs72；实验Pi05_libero_joint13_successmix20_NoState_LoRA_6gpu_bs72_10k_20261109。原pi05_base＋官方Gemma2B/300Mexpert LoRA、nostate48tokens、horizon10，原32维接口/有效13维（动作7＋signed方向容差6），组损失1/0.5，无额外4倍动作权重。
GPU0,1,2,3,4,6，batch72=6×12，FSDP2，8worker，seed42，10000步，warmup333，5e-5 cosine→5e-6，AdamW(.9,.95,eps1e-8,wd1e-10,clip1)，无EMA，log10/save2000/keep2000。新state8/actions13专属统计必须完整计算，不复用旧动作7统计。SwanLab cfg online/SDK cloud，URL待生成。
checkpoint父目录 `/mnt/SharedData/czl/checkpoints/pi05_libero_joint13_successmix20_nostate_6gpu_bs72/Pi05_libero_joint13_successmix20_NoState_LoRA_6gpu_bs72_10k_20261109`。
只有10000步真实退出、1000条有限日志、五份2000/4000/6000/8000/10000参数＋train_state CPU完整恢复/精确step/有限/normSHA，云端FINISHED真实1000行、服务器原仓库保护均通过，才能完成goal。完成监督会调用独立audit_successmix20_joint13_training_server.py并写task_completion，仍需真实核验全部条件后更新README/RESULTS/索引并最终报告路径、SwanLab和数据范围。不要启动额外训练/评测、关机、归档或停止viewer。

最新19:38：紧凑包2479文件/240203401字节全部到达；服务器重建3267870已真实退出，全1875 Parquet原SHA通过（802复用＋1073重建，0错误/0原文件回退）。803原路径中1条早期不完整文件已移入独立partial_files证据保留。当前local upload_state=remote_verifying，原全量verifier在服务器实际工作；训练与归一化仍待验证。

最新19:54：数据远端独立审计全部通过，归一化计划4338batches，正在用新混合数据计算state8/actions13，不能复用旧7维统计。总控3312670、统计3312672、workers3314577/3314638/3314684/3314774/3314836/3314886/3315007/3315073都已实际核对，阶段computing_normalization。该阶段会持续一段时间，日志/CPU/IO仍推进不能误判为死。SwanLab新ID和GPU训练steps尚未产生，只有完整10000＋独立最终审计通过才能完成goal。服务器结束监督3210109保持工作，后续自动捕获真实6GPU/8训练worker并在结束后完整CPU恢复五checkpoint＋云端审计。

20:14真实核查：审批服务此前额度问题导致上轮最后一次观察未执行，本轮require_escalated恢复成功；不是训练失败。归一化2351/4338(约54%)，总控3312670/norm3312672及8worker全部真实活跃且CPU/IO推进，最终监督3210109活跃，GPU0/1/2/3/4/6仍空闲待训练。服务器free约64.61GiB，缓存已完整建立。尚未产生归一化完成或训练step证据，不重启。观测证据normalization_observation_20261109_2014.json。


20:27：唯一4x切换监督3334760已在服务器实际运行，stage=waiting_for_original_normalization。仅原总控3312670有意SIGSTOP，统计3312672继续、当前无GPU训练step。监督将确认原4338batch日志/统计维度有限/真实exit0后，写原统计凭据，定向结束旧总控与旧结束监督3210109，完整保存旧请求/源码到run/four_x_protocol_update/before_4x，部署新4x源码并做CPU真实LeRobot/归一化/Observation/JIT损失预检，通过才唯一启动训练及新最终审计监督。若transition_state=failed应查看原因；不能CONT旧父或重复启动。归一化不因损失权重变更重算，保留原receiptSHA，并独立证明旧输入与新输入去掉仅用于loss的weight字段逐项一致。

新版参数：仅301容差观测来源的动作7维loss×4；1574名义×1；容差标签6维组权重0.5不变。总loss=mean(w_i*action_mse7+0.5*tolerance_mse6)，不按权重和重新归一化。SwanLab在线记录loss/action、loss/rotation_tolerance、loss_weighted/action、loss_weighted/rotation_tolerance、loss/total、loss/action_nominal_contribution、loss/action_reweighted_contribution、batch/action_loss_weight_mean、batch/reweighted_action_fraction；每10步记录，最终云端逐行复核4x贡献恒等式。34测试通过含JIT真实梯度4倍与容差不变/无填充梯度。用户未修改数据、标签、模型或其余训练超参数。

新证据本地绝对根：/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/four_x_protocol_update；实际远端四倍配置/新进程/预检待统计结束后生成，此前training_request.json与startup_verification仍代表旧归一化阶段，不能把它们说成新GPU已启动。远端状态/run/four_x_protocol_update/transition_state.json（完整run路径见服务器章节），完成后应取回normalization_reuse_verification.json、training_preflight_verification.json、transition_completion.json及新的training_request/actual_config/launch。
