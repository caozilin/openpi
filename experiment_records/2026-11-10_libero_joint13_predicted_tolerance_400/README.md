# joint13最终10000步模型：预测容差400回合

状态：**400回合已全部提交，366成功（91.50%），六客户端exit0并真实退出；完整独立审计进行中，尚未宣布全部完成**。用户2026-11-10明确授权：只测评最终10000步checkpoint，40任务各官方初态seed0–9，共400回合，服务器六卡推理、本地六客户端仿真。后续以run/pipeline_state.json、summary.json和真实PID为准。

使用新模型输出动作7＋六方向容差（rx-/rx+/ry-/ry+/rz-/rz+，rad），不使用旧任务固定容差表。对模型容差数值裁剪到训练标签支持的0–30°，保存原始预测、裁剪值与实际施加值。保留共享因果手指阶段分类、ENG/DIS严格、heading退化保护；阶段保护是既有控制协议，非额外任务语义mask。每次预测10步，执行前5步对应动作和容差，再闭环重规划。

采用当前共享MuJoCo后端快照，IPOPT默认20迭代、既有delta_q_squared＋EMA10、共享关节伺服、20Hz/500Hz、每动作25物理步、nostate、48tokens、夹爪deadband0.5；10步settle，Spatial220/Object280/Goal300/Long520上限。求解失败保持上一接受名义/关节/夹爪参考，连续20失败帧终止；首次原生成功后继续20步（1秒），首成功与最终谓词分开记录。该运行时较历史20261025 benchmark快照更新，不能将与旧模型的差异全部归因于模型。

原训练数据1875条（1574名义＋301容差观测，所有名义actions7不变），joint13，容差来源动作损失4倍、容差6维组权重0.5。训练目标是10000步；其他2000/4000/6000/8000仅作为原训练完整性检查，不创建额外rollout。

## 验证与结果范围

- 29项预测输出配对/单位/裁剪/阶段保护及失败保持、成功续跑测试通过。
- 冻结400唯一组合，六客户端分配67/67/67/67/66/66，官方初态和任务文件与原0–9评测逐字节一致。
- 请求冻结479个来源/运行时文件，未修改服务器仓库和原训练overlay；未覆盖任何历史结果。
- 本地可用约5 GiB，参考既有400回合0.56 GiB，按0.8 GiB预估本次输出，原数据不清理。
- 只有400回合全部提交、6客户端exit0、基础设施错误0，以及逐帧动作/容差/状态/25物理步/保持规则、原始初态、原生首/末成功谓词、全视频解码和六服务身份全部独立审计通过，才宣布完成。部分回合成功率见下方最新监测，完整结果及最终审计待完成。

## 本地源数据地址

- 原始run：`/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_predtol_hold20_40tasks_10seeds_20261110`。
- 冻结请求：上述run/evaluation_request.json；官方输入official_initial_states.npz、official_tasks.json；共享快照frozen_shared_runtime/tasktol。
- 结果、视频、轨迹：上述run/episodes/<suite>/task_XX/seed_N/{result.json,trajectory.json,trajectory.npz,rollout.mp4}（待生成）。
- 汇总、日志：上述run/summary.json、evaluation_pipeline.log、client_0..5.log（待生成）。
- 首批审计：上述run/initial_client_integration_audit.json；完整审计evaluation_independent_final_audit.json、independent_audit.log（待生成）。
- 最终报告：上述run/PREDICTED_TOLERANCE_400_REPORT.md、predicted_tolerance400_report.json/csv和task_completion.json（待生成）。
- 准备证据：本实验preparation_verification.json、evaluation_request.json。
- 本地监督：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_predicted_tolerance_400/pipeline.py`，pipeline.log及run/pipeline_state.json。
- 服务器推理run：`czl@100.73.28.99:/mnt/SharedData/czl/runs/libero_eval_joint13_successmix20_predtol_400_20261110`（已启动）。
- 仅用checkpoint：`czl@100.73.28.99:/mnt/SharedData/czl/checkpoints/pi05_libero_joint13_successmix20_nostate_6gpu_bs72/Pi05_libero_joint13_successmix20_NoState_LoRA_6gpu_bs72_10k_20261109/10000`。
- 训练审计：`czl@100.73.28.99:/mnt/SharedData/czl/runs/pi05_libero_joint13_successmix20_lora_6gpu_bs72_10k_20261109/training_independent_final_audit.json`。

仅在400回合全部完成审计后定向停止本次六推理副本和专用隧道，保留所有视频/轨迹和用户viewer。基础设施错误单列，不计作任务失败或按成败选重跑。


## 启动与审计修正历史

初次沙箱监督启动未持久化，主机核验无实例且锁释放、零评测开始后，才在主机启动唯一实例（attempts/0001_sandbox_launch）。首批6回合全部成功后，旧固定容差审计误将瞬时预测边界当作有效硬边界，忽略共享stage-local limits EMA和边界收缩恢复包络，审计触发后本地客户端已全部退出。未修改物理控制、模型或冻结请求；改用独立只读复算canonical configure_stage/prepare_cycle，核验EMA后的声明边界、实际恢复硬盒与FK姿态坐标。6条原始回合全部通过，24项原artifact SHA未变，未重跑。旧审计源码仍按原SHA保留，新审计独立绑定于audit_implementation_amendment.json。未提交部分归档attempts/0002_initial_audit_bounds_ema/uncommitted，再同请求继续其余394条；服务器六副本及隧道复用未重启。

startup_verification.json、initial_client_integration_audit.json及audit_implementation_amendment.json是当前证据；首6通过不代表完整400。模型预测经0–30°裁剪后进入共享stage-local容差EMA及既有恢复规则，因此当帧实际硬盒并非简单等于未经滤波的模型输出；完整审计按共享实现复算，未通过放宽阈值绕过。

每15分钟监测与成功率汇报已开启，本聊天heartbeat libero-400-15；以独立已提交result计数作为分母，报告总体和四suite、增量及错误，最终完整审计后关闭。起始统计：25/27（92.59%），完整400仍待完成；monitor_last_report.json保留上次检查值。


## 可视化只读代码检查

按用户要求只检查代码，未启动viewer或HTTP测试，未更改评测。Benchmark模式会自动发现本run已提交result，支持任务/seed双列表、视频、3秒刷新与已接受rotation_release_rad；只显示已提交回合，不直播未提交视频。新模型预测模式尚未完整适配：live.js标题回退为固定±30°，API未透传predicted_bounds_raw_rad，边界阴影仍使用原始裁剪bounds_rad而非共享EMA与收缩恢复后的有效硬盒，不能当作本帧实际硬约束。证据viewer_code_review.json含全部本地代码绝对路径。


## 2026-11-10 keepalive中断与同参数续跑

完成66条时57成功、9step_limit；另6条未提交Long回合收到server1011 keepalive ping timeout，非任务失败。原本地父子全部退出，六远端副本始终存活。66条全部264项artifact SHA保护，旧状态/日志与未提交目录归档attempts/0003_server_keepalive_ping_timeout；只新增本地同packet/同seed/同模型metadata重连wrapper，3次上限和600秒原总deadline，6测试通过。物理入口479源、冻结请求SHA、模型/控制协议未变；66条成败不重测。17:02监督30768/总控30823/六worker30824..30829恢复，远端六服务和隧道复用。新传输SHA与历史证据见transport_recovery_amendment.json，恢复主机核验transport_recovery_startup_verification.json。

本次监测：57/66成功，当前活动基础设施错误0、历史未提交中断6，完整400仍待完成。

## 最新定时监测

2026-11-10T17:56:50.861738+08:00；400回合执行完成，完整独立审计进行中。

| 套件 | 成功/已完成 | 成功率 | 进度 |
|---|---:|---:|---:|
| Spatial | 92/100 | 92.00% | 100/100 |
| Object | 95/100 | 95.00% | 100/100 |
| Goal | 92/100 | 92.00% | 100/100 |
| Long | 87/100 | 87.00% | 100/100 |

总体366/400=91.50%，较上次新增93条、成功84条；终止类型{'task_success': 366, 'step_limit': 29, 'ik_failure_streak': 5}。活动基础设施错误0、历史6次未提交中断保留。全部400唯一组合已核对；六客户端exit0、原PID均已退出，评测锁释放；审计72/400，以真实审计PID/日志为准。原66条成败SHA及479冻结来源/请求SHA未变。审计、最终报告及定向服务器清理尚未完成，不提前删除监督或停止推理副本。

证据：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_predicted_tolerance_400/monitor_last_report.json`。


## 动作块容差预测稳定性只读分析

2026-11-10T17:33:46.216066+08:00快照238条已提交成败回合；预测10/执行5，仅有前5帧记录。APP/EXEC的6378完整块、5695相邻块对：原始块内极差均值0.424°，相邻块均值绝对差均值2.286°/P95=11.652°；93.01%块六方向各自极差均≤1°。ry±跨块变化最大；不能推断未保存的后5帧或16步。详见`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_predicted_tolerance_400/prediction_stability/README.md`，明细和输入SHA均保存在同目录，原始结果未改。

## 累计成功率回升的只读核对

17:26到17:41，累计157/178→282/307；新增125/129=96.90%，因此提高整体比例。新增Spatial26/27、Object37/37、Goal37/38、Long25/27。以完整40任务的seed比较：seed0为31/40（77.5%），seed1/2各36/40（90%），seed3/4/5各38/40（95%）。说明后续已执行任务/seed组合成功率更高；不表示评测过程中模型在学习。不能只归因摆放：seed同时绑定reset随机性与确定性推理噪声。479冻结来源SHA未变，容差EMA/物理/终止协议未变；历史6次未提交基础设施中断不计任务失败。并行任务按seed/任务顺序推进，部分完成样本不是随机均匀子集；最终需完整400及独立审计。证据和307条result SHA：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_predicted_tolerance_400/success_rate_rebound_diagnosis.json`。

## 历史seed趋势对照

历史正式无容差400：seed0–4为191/200=95.5%，seed5–9为189/200=94.5%；原模型预设容差400为184/200=92%→186/200=93%，没有同样大幅回升。按本次前178/后129完全相同组合，历史两组提高约2个百分点，本次提高8.70个百分点；seed0历史两组90%，本次77.5%。精确计数、初态同一性和原始result SHA见`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_predicted_tolerance_400/historical_seed_trend/README.md`及summary/sources.json。未新增推理或rollout。


最终：已完成并通过独立审计，366/400。详见RESULTS.md；六副本与隧道已定向结束，viewer保留。
