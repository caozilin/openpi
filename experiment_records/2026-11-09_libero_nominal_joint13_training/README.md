# 名义LIBERO：补标24条并训练动作7+容差6

状态：旧纯名义训练方案已被用户确认的新混合方案替代，旧训练从未启动。用户随后确认未修改标注，并授权删除旧全名义joint13转换副本及其服务器部分上传；本地旧1875数据文件与远端572上传文件已退役清理，小型审计/请求记录保留。最新工作见 `/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/README.md`。以下保留历史执行、暂停及补标证据，不作为恢复旧训练的授权。

## 目标和协议

保留之前训练的1875条成功名义轨迹、312374帧、40任务、20Hz、原双相机PNG及state8/actions7，新增搜索六方向容差（rad）和trajectory_is_tolerance=0。此次全部为名义轨迹，不能混入新容差物理回放。原1851条已审计标签保留；24条原baseline_unavailable单独尝试补标，成功与失败均保留。

补标使用原cache85冻结执行源码的隔离副本，扩展共享FK后端的基线观测模式：零容差求解拒绝时保持上一接受关节参考、名义目标仍按原动作推进；自碰撞如实记录，继续记录完整逐帧净空。基线只作为几何参考，不要求运动学可行；非有限值或缺失帧仍不能构造参考。候选仍使用正常首失败停止的后端，完整可行、自碰撞通过、逐帧净空余量≥−0.1mm才可作为标签。保持逐帧min(5mm,c0)、原语义方向和5/15/30°上限、IPOPT20迭代/EMA10、每块结构化初始化+16追加；不能将失败基线复用为成功的零点候选。此为补标协议差异，不混写为原完全同协议结果。

训练拟沿用原6卡0,1,2,3,4,6；pi05_base官方LoRA、nostate、48tokens、horizon10、全局batch72、FSDP2、10000步、warmup333、5e-5余弦至5e-6、每2000保存。有效动作13维经原32维接口补零；分组损失动作1、容差0.5。必须重新计算13维归一化，开启SwanLab在线，并完整审计后才宣布完成。最终训练样本范围取决于本次补标真实结果。

## 本地源数据地址

- 原名义LeRobot：`/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_full_20261101`。
- 原标签及审计：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107`。
- 24条诊断：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/BASELINE_UNAVAILABLE_DIAGNOSIS_20261109.md`。
- 补标请求/结果/日志/审计根（准备中）：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_missing24_baseline_observation_20261109`。
- 原六卡训练请求：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261101_full_v1/training_request.json`。
- 新训练LeRobot、归一化、远端run、checkpoint、SwanLab链接与最终训练审计：待生成。

原1851标签、24失败证据、原始物理轨迹及冻结请求均保留。没有启动额外物理回放或VLA评测。

## 补标完成与数据入口验证

24/24均找到有效候选，660份原始trial通过独立审计，活动错误0。原5578份受保护状态/标签/基线文件SHA未变，补标总控及24搜索/审计进程已退出、锁释放。原1851+新24共1875条有效标签，全部保留动作训练和容差监督。Spatial10/10、Goal10/10、Long4/4补标成功。基线观测允许失败不意味着候选允许失败，零点候选独立正常回放，未复用不合格基线作为合格观测。

15项基线/候选验收测试、14项新LIBEROjoint13数据/变换测试通过；实际LeRobot读取episode0首、中、末帧及10步动作/容差序列通过，原PNG字节和actions7不变，state8和全部名义标识正确。实际读取证据：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_nominal_joint13_training/real_reader_smoke.json`。

补标完整审计：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_missing24_baseline_observation_20261109/annotation_independent_final_audit.json`；原结果保护：同根`original_results_protection.json`、`original_results_verification.json`；搜索及审计日志：同根`pipeline.log`、`independent_audit.log`和各demo/search.log。新数据：`/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_nominal_joint13_20261109`；转换请求、逐episode收据、状态和日志均在该根目录。独立新数据审计：待生成。

服务器2026-11-09只读核查：6张目标3090利用率均0%；GPU5桌面保留。SharedData剩余约137GiB，旧单checkpoint约8.91GiB，旧数据Arrow缓存约36GiB；后续必须按新数据+缓存+5份checkpoint及余量测算磁盘，未清理任何旧数据。

转换实现历史：首次主机运行在第1个Parquet提交前被严格元数据往返检查拦截，原因是Arrow新列表子字段默认名item经Parquet转为element。已保留原脚本/日志/启动凭据，原进程退出后将新增字段显式定义为element；6项转换回归及真实PNG/state/action/label严格往返通过后按同一数据请求恢复。没有已提交结果被覆盖，没有放宽值或元数据相等检查。沙箱首次启动未形成持久主机进程的凭据亦单列保留。

新增实现：`/media/czl/sata/franka_my_code/openpi/examples/libero/enrich_nominal_joint13_dataset.py`、`audit_nominal_joint13_dataset.py`、`train_nominal_joint13_pi05_server.py`、`verify_joint13_upload_server.py`；数据策略`src/openpi/policies/libero_joint_policy.py`及专用20Hz配置`src/openpi/training/libero_joint.py`。补标24逐条搜索边界：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_nominal_joint13_training/supplement24_results.json`。

远端准备根：`czl@100.73.28.99:/mnt/SharedData/czl/runs/pi05_libero_nominal_joint13_lora_6gpu_bs72_10k_20261109`；仅部署本次source_overlay并完成配置构造检查。服务器原仓库Python文件保护清单repository_source_before.json、overlay_manifest.json及overlay_preparation.json在该远端run。当前远端training_request.json明确为待数据验收的草案，不是已启动训练。SSH专用连接已建立，不记录凭据。

转换后的监督将自动全量审计；上传监督等待该独立审计通过才上传，并用旧服务器名义副本作rsync复制差分基底（新文件独立，非硬链接），随后远端逐SHA、全1875原列相等及实际LeRobot读取验证。上传根为新数据路径，不修改旧数据。启动凭据local_pipeline_launch.json和upload_launch.json、状态和日志均在本记录；训练仍未启动。

当前持久监督与精确训练协议见[运行交接](HANDOFF.md)。上传和训练均受前序本地/远端审计门槛保护，不能把等待状态误判为训练运行。15项joint13/转换回归与15项基线验收回归通过。

全量数据审计第二次完成：1875条原始列/双相机PNG字节完全一致，全部标签等于审计搜索输出；40任务各首/中/末帧与10步动作/容差序列实际LeRobot读取通过，缺标0，名义标识全部0。审计证据：`/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_nominal_joint13_20261109/meta/joint13_independent_audit.json`。首轮读取审计触发LeRobotv2.1非连续episode子集偏移索引越界，数据本身未变；仅在两个只读审计入口的子类映射原始episode编号至子集偏移，不修改第三方库或正式全量训练读取。两项真实审计类索引回归通过，失败证据在本记录`attempts/0001_lerobot_selected_episode_index`；原监督全部退出后重新完整审计，未重生成或改写数据。

当前主机上传26533、训练调度26535；它们会在前序校验通过后继续。本地监督26531/审计26532已退出。最新上传进程已实际读到episode1772并完成约35GB读取，处于计算SHA清单，不能把旧waiting状态当作当前已开始网络上传。

## 用户暂停

本地上传26533和训练调度26535已核对真实命令后停止；服务器本次结束审计监督3170795已停止，training_launch.json不存在、训练从未启动。原数据、标签、新数据和部分上传文件全部保留，不继续归一化/训练。暂停凭据：本记录`user_pause.json`、`pause_local_verification.json`；远端同run的`pause_verification.json`和`pause_transfer_verification.json`。恢复须由用户明确授权，先读取调整后的标签，不能直接复用本轮旧标签数据或旧上传清单启动训练。

## 转换副本退役

旧本地数据路径已无Parquet；审计、请求、逐episode收据的归档地址为 `/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/retired_nominal_joint13_conversion_metadata`。删除证据为同记录 `nominal_conversion_retirement.json`。原始名义数据、补标24与原1851标签及容差物理回放均保留。
