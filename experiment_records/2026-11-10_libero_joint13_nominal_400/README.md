# joint13同模型无容差对照：40任务×seed0–9

状态：全部400回合与完整独立审计通过，368/400成功（92%）；本地/远端任务进程退出、全部锁及专用隧道释放。2026-11-10用户明确授权：使用刚完成预测容差评测的同最终10000步joint13 successmix20模型、同40任务、同官方seed0–9，关闭容差重新评测400回合。

## 唯一干预与固定协议

客户端使用原生`session.step(action)`默认`bounds_rad=None`，进入canonical严格无容差分支、mask000；不使用零宽容差盒替代无容差。模型仍输出相同动作7和容差6，动作采用原解码/夹爪deadband0.5，容差仅保存诊断、完全不进入控制器。无容差路径不使用容差值EMA；优化器delta_q_squared+EMA10目标仍保持原样。

其余与本次已审计预测容差组逐项相同：冻结shared运行时、IPOPT20/现成servo、20Hz500Hz/25physics、nostate/48tokens、预测10执行5、settle10、上限Spatial220/Object280/Goal300/Long520、失败不推进名义参考/保持上次接受参考/连续20帧终止、首次原生成功后再20步且末成功另列。相同seed/case/chunk确定性噪声。官方初态NPZ、任务文件及runtime源码逐字节相同，400组合/6worker分配相同。

本组全新400，不复用旧模型名义结果，不覆盖旧成败或按结果选重测。服务器GPU0/1/2/3/4/6只推理，本地六客户端物理仿真。旧预测容差组400完整审计通过366成功；旧六副本已定向退出。新run独立身份/SSH/源SHA，不改旧冻结入口、训练checkpoint或共享环境。33项适配/有界重连/失败保持/成功续跑测试通过。

## 验证与完成条件

先审计首6真实提交：同初态/首prestate、首chunk动作7及容差6差异记录（不要求跨副本数值完全相同），全部控制帧boundsNone/mask000、独立纯FK严格姿态/位置约束与参考累计/失败回滚、25physics、原生首末谓词及双视角视频全解码。首6通过后继续全部400。完成需400唯一组合、6clientexit0、活动infra0、完整独立审计、task_completion/pipeline_state complete、真实进程退出/锁释放、仅本六推理和隧道定向收尾。最终逐任务逐seed与预测容差组配对，列改善/退化/共同成败。只读审计不再执行求解/物理/推理。

有界传输沿用原已验证wrapper：同packet/seed/metadata重连最多3次，原600秒总deadline；若基础设施错误保留日志和全部成败，不自动盲重试。GPU5/他人桌面、用户viewer、其他SSH和旧数据不动。不训练、不评测其他checkpoint。

## 本地源数据绝对路径

- 原始run：`/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_nominal_hold20_40tasks_10seeds_20261110`。
- 请求：`/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_nominal_hold20_40tasks_10seeds_20261110/evaluation_request.json`（已冻结，SHA见preparation_verification.json）。
- 官方初态/任务/运行时：`/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_nominal_hold20_40tasks_10seeds_20261110/official_initial_states.npz`、`/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_nominal_hold20_40tasks_10seeds_20261110/official_tasks.json`、`/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_nominal_hold20_40tasks_10seeds_20261110/frozen_shared_runtime`。
- 客户端：`/media/czl/sata/franka_my_code/openpi/examples/libero/eval_joint13_nominal_libero_local.py`；适配：`/media/czl/sata/franka_my_code/openpi/examples/libero/nominal_joint13_policy.py`。
- 监督/包装/审计：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_nominal_400/pipeline.py`、`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_nominal_400/run_client.py`、`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_nominal_400/audit_evaluation.py`、`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_nominal_400/audit_nominal_reference.py`。
- 逐回合（待生成）：`/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_nominal_hold20_40tasks_10seeds_20261110/episodes/<suite>/task_XX/seed_N/{result.json,trajectory.json,trajectory.npz,rollout.mp4}`。
- 日志/进程（待生成）：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_nominal_400/pipeline.log`、`/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_nominal_hold20_40tasks_10seeds_20261110/evaluation_pipeline.log`、`/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_nominal_hold20_40tasks_10seeds_20261110/client_0..5.log`和对应process/launch.json。
- 初审/完整审计（待生成）：`/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_nominal_hold20_40tasks_10seeds_20261110/initial_client_integration_audit.json`、`/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_nominal_hold20_40tasks_10seeds_20261110/evaluation_independent_final_audit.json`。
- 最终报告/配对（待生成）：`/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_nominal_hold20_40tasks_10seeds_20261110/NOMINAL_400_REPORT.md`、`/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_nominal_hold20_40tasks_10seeds_20261110/nominal400_report.json`、`/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_nominal_hold20_40tasks_10seeds_20261110/nominal400_results.csv`、`/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_nominal_hold20_40tasks_10seeds_20261110/paired_nominal_vs_predicted_tolerance.json`。
- 预测容差配对组：`/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_predtol_hold20_40tasks_10seeds_20261110`，400完整独立审计已通过。
- 准备/代码差异：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_nominal_400/preparation_verification.json`、client_changes.patch、audit_changes.patch、pipeline_changes.patch。

## 远端（不是本地路径）

- 推理run：`czl@100.73.28.99:/mnt/SharedData/czl/runs/libero_eval_joint13_successmix20_nominal_400_20261110`。
- checkpoint：`czl@100.73.28.99:/mnt/SharedData/czl/checkpoints/pi05_libero_joint13_successmix20_nostate_6gpu_bs72/Pi05_libero_joint13_successmix20_NoState_LoRA_6gpu_bs72_10k_20261109/10000`。
- SSH专用socket：`/tmp/libero_joint13_nominal400_20261110.sock`，端口18760..18765 loopback六转发。凭据按AGENTS/TRAINING_SERVER仅内存交互。

延续每15分钟四套件汇报；使用同一现有heartbeat切换监督本组，不创建重复监测。用户未要求启动viewer，本次不启动。


启动核验：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_nominal_400/startup_verification.json`，八个真实本地进程、六实际WebSocket身份与原容差组模型参数/norm digest一致；490源码SHA未变。

## 首块数值检查中断与保留续跑

18:16首6已提交且成功，但初审额外要求两次运行首5输出逐项1e-6一致，检查失败导致本地父子正常退出（已主机核实）并释放锁；不是任务失败。全部6条24项artifact SHA保护，原始审计/监督/请求/日志及未提交目录保存在attempts/0001_first_chunk_numeric_comparison。没有重测已提交成败。

同一固定观测packet字节、同噪声种子对六副本各重复2次诊断：各副本内部重复差0，但副本间actions最大差0.002929103、容差最大差0.005176160rad；同params/norm身份已核对。该packet来自一帧已存有损视频，只用于比较相同packet的副本数值差，不能证明旧运行原始输入字节相同，也不能将跨运行全部差异唯一归因GPU。首prestate一致不保证像素字节完全一致；历史请求没有完整原始packet。因此撤销“同模型/seed必保证1e-6逐输出一致”的额外审计条件，保留首chunk差值作为诊断，不放宽任何物理/FK/求解/原生成功门限，不改seed/模型/runtime。相同模型seed不声称比特级相同动作，后续比较按任务/seed配对。

原冻结auditor/pipeline保留不改，新增audit_evaluation_v2.py/pipeline_v2.py独立补充清单numeric_comparison_audit_amendment.json绑定SHA。原请求ccb34c973fca0edc6ac08e1f323e195702bc9a8ec763f38eee37ee75f23c1575和490来源不变。修正初审18:22:50通过原6全部：同官方初态/首prestate1e-12、无容差boundsNone/mask000、严格参考累计/FK、25physics、失败保持/原生首末谓词/视频完整解码。证据run/initial_client_integration_audit.json和initial_audit_v2.log。

确认原本地进程/只读初审退出后，18:23使用同冻结客户端和有界传输续跑剩余394；新监督pipeline_v2.py PID71157/start_ticks1258864，真实总控/6客户端见evaluation_state及主机/proc。远端原3979754和3979755..3979760全部复用未重启。完整400审计待完成；不要恢复旧66337或旧原始audit_evaluation.py。原independent_audit_state.failed是旧额外数值比较残留，历史已归档，当前全审计待全部400。

## 最新定时监测

2026-11-10T20:17:13.891065+08:00；400执行完成，最终只读v3审计253/400。

| 套件 | 成功/已完成 | 成功率 | 进度 |
|---|---:|---:|---:|
| Spatial | 96/100 | 96% | 100/100 |
| Object | 97/100 | 97% | 100/100 |
| Goal | 88/100 | 88% | 100/100 |
| Long | 87/100 | 87% | 100/100 |

368/400=92%，相对上次回合结果新增0；22步数上限/10连续20失败终止，活动infra0。两项历史审计实现差异已保留，不改物理/模型/原始结果；现v3按冻结IPOPT逐分量最大残差而非额外L2判据。全部1600artifact、490冻结来源及v3补充SHA不变。finisher/审计真实活跃时不重复启动，完成文件/配对报告/定向清理仍需核验。证据本实验monitor_last_report.json。


最终400审计通过，368/400成功，逐seed配对报告已生成，六推理副本及专用隧道定向退出；历史2项审计实现错误保留，未重跑已提交成败。详见RESULTS.md及两份audit_amendment。

最终交付核验：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_nominal_400/final_delivery_verification.json`，400×4文件SHA与完整审计一致；报告RESULTS.md、40任务表NOMINAL_400_REPORT.md、逐seed配对run/paired_nominal_vs_predicted_tolerance.json。历史两项审计实现修正完整保留；所有400物理结果不变。
