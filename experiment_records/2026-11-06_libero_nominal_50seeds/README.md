# LIBERO 正式无容差评测：40任务 × 50 seed

更新时间：2026-11-06T16:27:57.735658+08:00（Asia/Shanghai）。
状态：**已完成并通过独立审计**。

## 目的与样本来源

将最终10000步模型的正式无容差评测从每任务10次扩展到50次，得到40任务×seed0–49的2000个唯一组合。保留正式旧400条（380成功），追加seed10–49的1600条（1523成功）。旧400包含重新完整执行的seed0–5与首次执行seed6–9，未混入此前只重跑失败回合的结果。所有已提交成功与失败均保留，没有按结果替换。

## 模型、控制与判定协议

- 模型：重构LIBERO数据微调的pi05_base＋官方LoRA，最终10000步checkpoint；nostate，语言48tokens，预测10步，每执行5步重新推理。训练数据1875条成功演示、312374帧、40任务；训练/数据审计见源地址。
- 计算：本地6个并行MuJoCo仿真客户端，服务器GPU0/1/2/3/4/6仅推理；官方初态索引与seed相同，噪声按seed/case/chunk确定。
- 控制：冻结shared关节伺服＋默认IPOPT，无容差mask000/boundsNone，20Hz控制、500Hz物理、每帧25物理步；动作平移米、世界旋转向量弧度，不套OSC缩放；夹爪deadband0.5。
- 求解失败：失败动作不更新名义参考，保持上一接受参考；连续20帧（1秒）失败才终止，成功清零。
- 原任务上限：Spatial220、Object280、Goal300、Long520步，settle10。
- 成功率：首次原生LIBERO成功谓词为准。新增1600首次成功后继续1秒20步；保留400首次成功即结束。尾段不同，不能将两者最终稳定成功率视为同一指标。

## 总体与套件结果

合计 **1903/2000，95.15%**；失败97条，其中连续求解失败终止20条、步数耗尽77条。以下是记录的终止标签，未据此断言物理不可达或失败因果。

| 套件 | 成功/总数 | 成功率 | 连续求解失败终止 | 步数耗尽 |
|---|---:|---:|---:|---:|
| Spatial | 490/500 | 98.00% | 3 | 7 |
| Object | 491/500 | 98.20% | 0 | 9 |
| Goal | 463/500 | 92.60% | 5 | 32 |
| Long | 459/500 | 91.80% | 12 | 29 |

## 各套件关键失败任务

task_id从0开始；按各套件失败数排序。

| 套件/task | 任务 | 成功/50 | 连续求解失败终止 | 步数耗尽 |
|---|---|---:|---:|---:|
| Spatial/04 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 44/50 | 2 | 4 |
| Spatial/09 | pick up the black bowl on the wooden cabinet and place it on the plate | 48/50 | 0 | 2 |
| Spatial/06 | pick up the black bowl next to the cookie box and place it on the plate | 49/50 | 1 | 0 |
| Object/00 | pick up the alphabet soup and place it in the basket | 48/50 | 0 | 2 |
| Object/06 | pick up the butter and place it in the basket | 48/50 | 0 | 2 |
| Object/02 | pick up the salad dressing and place it in the basket | 49/50 | 0 | 1 |
| Goal/05 | push the plate to the front of the stove | 37/50 | 1 | 12 |
| Goal/03 | open the top drawer and put the bowl inside | 39/50 | 2 | 9 |
| Goal/09 | put the wine bottle on the rack | 45/50 | 0 | 5 |
| Long/08 | put both moka pots on the stove | 36/50 | 3 | 11 |
| Long/09 | put the yellow and white mug in the microwave and close it | 42/50 | 2 | 6 |
| Long/05 | pick up the book and place it in the back compartment of the caddy | 43/50 | 4 | 3 |

## 每任务结果

| 套件 | task_id（0起始） | 成功/50 |
|---|---:|---:|
| Spatial | 00 | 50/50 |
| Spatial | 01 | 50/50 |
| Spatial | 02 | 50/50 |
| Spatial | 03 | 50/50 |
| Spatial | 04 | 44/50 |
| Spatial | 05 | 50/50 |
| Spatial | 06 | 49/50 |
| Spatial | 07 | 50/50 |
| Spatial | 08 | 49/50 |
| Spatial | 09 | 48/50 |
| Object | 00 | 48/50 |
| Object | 01 | 50/50 |
| Object | 02 | 49/50 |
| Object | 03 | 50/50 |
| Object | 04 | 49/50 |
| Object | 05 | 49/50 |
| Object | 06 | 48/50 |
| Object | 07 | 49/50 |
| Object | 08 | 49/50 |
| Object | 09 | 50/50 |
| Goal | 00 | 46/50 |
| Goal | 01 | 50/50 |
| Goal | 02 | 47/50 |
| Goal | 03 | 39/50 |
| Goal | 04 | 50/50 |
| Goal | 05 | 37/50 |
| Goal | 06 | 49/50 |
| Goal | 07 | 50/50 |
| Goal | 08 | 50/50 |
| Goal | 09 | 45/50 |
| Long | 00 | 48/50 |
| Long | 01 | 48/50 |
| Long | 02 | 49/50 |
| Long | 03 | 48/50 |
| Long | 04 | 47/50 |
| Long | 05 | 43/50 |
| Long | 06 | 48/50 |
| Long | 07 | 50/50 |
| Long | 08 | 36/50 |
| Long | 09 | 42/50 |

## 审计与中断历史

- 回合执行完成：新增1600，六客户端exit0，活动基础设施错误0。
- 11月5日3次600秒推理通信超时，保留全部762提交回合，只归档未提交目录后续跑；不计作任务失败。
- 11月6日再次发现本地进程消失，最后日志11:32、本机15:33启动；确切中断原因未证实。保留1503提交回合SHA，归档未提交目录，续跑最后97条。
- 首次跟随审计最终进程退出检查存在竞态。确认主控/客户端退出后，使用同一冻结审计源码完成全1600只读复核，16:26:53通过；16:26:59写出task_completion complete。未重跑计分回合。
- 最终独立审计与task_completion均complete；另以本记录的`verification.json`核对2000唯一组合、40任务各50次、CSV2000行、全8000份artifact SHA、冻结源码及来源清单一致。
- 本次服务器六副本及六转发已定向停止，专用SSH控制连接关闭，用户viewer未干预。

## 本地源数据地址

完整机器可读索引：[sources.json](sources.json)；汇总复核证据：[verification.json](verification.json)。下面均为本地绝对路径：

- **new_run**：[pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105)
- **request**：[evaluation_request.json](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105/evaluation_request.json)
- **old400_provenance**：[preserved_formal400.json](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105/preserved_formal400.json)
- **old400_manifest**：[combined_formal_400.json](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_formal_seed0to5_20261104/combined_formal_400.json)
- **formal_seed0to5_run**：[pi05_reconstructed_10000_nominal_hold20_formal_seed0to5_20261104](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_formal_seed0to5_20261104)
- **first_pass_seed6to9_run**：[pi05_reconstructed_10000_nominal_hold20_40tasks_10seeds_extension_20261104](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_40tasks_10seeds_extension_20261104)
- **official_initial_states**：[official_initial_states.npz](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105/official_initial_states.npz)
- **official_tasks**：[official_tasks.json](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105/official_tasks.json)
- **new_summary**：[summary.json](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105/summary.json)
- **new_completion**：[evaluation_completion.json](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105/evaluation_completion.json)
- **final_audit**：[evaluation_independent_final_audit.json](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105/evaluation_independent_final_audit.json)
- **final_audit_state**：[independent_audit_state.json](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105/independent_audit_state.json)
- **combined_json**：[combined_formal_2000.json](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105/combined_formal_2000.json)
- **combined_csv**：[combined_formal_2000.csv](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105/combined_formal_2000.csv)
- **run_report**：[NOMINAL_50SEED_REPORT.md](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105/NOMINAL_50SEED_REPORT.md)
- **task_completion**：[task_completion.json](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105/task_completion.json)
- **audit_log**：[independent_audit.log](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105/independent_audit.log)
- **evaluation_log**：[evaluation_pipeline.log](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105/evaluation_pipeline.log)
- **recovery_log**：[final_audit_recovery.log](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105/final_audit_recovery.log)
- **resume_history**：[RESUME_HISTORY.md](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105/RESUME_HISTORY.md)
- **first_resume_receipt**：[resume_after_transport_failure.json](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105/resume_after_transport_failure.json)
- **second_resume_receipt**：[resume_after_host_restart.json](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105/resume_after_host_restart.json)
- **attempts**：[attempts](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105/attempts)
- **server_shutdown**：[inference_shutdown_verification.json](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105/inference_shutdown_verification.json)
- **tunnel_cleanup**：[local_cleanup_verification.json](/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105/local_cleanup_verification.json)
- **training_audit**：[training_independent_final_audit.json](/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261101_full_v1/training_independent_final_audit.json)
- **dataset**：[libero_reconstructed_full_20261101](/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_full_20261101)
- **handoff**：[SERVER_HANDOFF.md](/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261101_full_v1/SERVER_HANDOFF.md)
- **eval_manager_source**：[eval_nominal50.py](/media/czl/sata/franka_my_code/openpi/examples/libero/eval_nominal50.py)
- **shared_eval_worker**：[eval_reconstructed_libero_local.py](/media/czl/sata/franka_my_code/openpi/examples/libero/eval_reconstructed_libero_local.py)
- **auditor_source**：[audit_formal_nominal.py](/media/czl/sata/franka_my_code/openpi/examples/libero/audit_formal_nominal.py)
- **final_audit_recovery_source**：[final_audit_recovery.py](/media/czl/sata/franka_my_code/openpi/analysis/libero_nominal50_20261105/final_audit_recovery.py)
- **frozen_runtime**：[frozen_shared_runtime](/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261025_success10_v1/frozen_shared_runtime)

逐回合文件位于各来源run的 `episodes/<suite>/task_XX/seed_N/`，包括 `result.json`、`trajectory.json`、`trajectory.npz`、`rollout.mp4`。每个正式汇总行的`source_result`指向真实来源，不能假设2000条都在新run内。

## 远端与在线记录（不是本地路径）

- 推理/训练服务器：`czl@100.73.28.99`。
- 最终模型：`czl@100.73.28.99:/mnt/SharedData/czl/checkpoints/pi05_libero_reconstructed_nostate_6gpu_bs72/Pi05_libero_reconstructed_NoState_LoRA_6gpu_bs72_10k_20261101/10000`。
- 推理run：`czl@100.73.28.99:/mnt/SharedData/czl/runs/libero_eval_pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105`。
- 训练run：`czl@100.73.28.99:/mnt/SharedData/czl/runs/pi05_libero_reconstructed_lora_6gpu_bs72_10k_20261101`。
- SwanLab：[最终10000步训练实验](https://swanlab.cn/@caozilin/openpi/runs/95dgsykkqsun5pvkuts3t)。
