# 原模型闭环rollout：外部语义预设±30°，400次

状态：历史2026-11-04正式400容差回合及400无容差对照均通过原独立审计；2026-11-09补充只读归档和保存轨迹的运动统计复算。没有新rollout、训练、推理、FK或修改旧请求/数据。

## 运行范围与成功率

最终10000步pi05_base官方LoRA重构LIBERO模型，nostate、预测10步/执行5步后重推理；服务器6副本只推理，本地6客户端物理仿真。40任务×官方seed0–9，容差组完整新跑400次；外部使用2026-10-31用户确认的任务/子任务/APP/ENG/EXEC/DIS语义mask，允许轴双向30°，ENG/DIS严格，保留heading退化保护。阶段来自实时手指反馈，子任务来自因果仿真语义先验，非未来帧、非学习预测容差、非所有任务统一111。

默认IPOPT/共享关节伺服，20Hz控制、500Hz物理，每动作25物理步；失败保持上一接受参考，连续20帧失败终止。模型和动作时序与正式名义对照相同，后续动作各自闭环产生，非固定动作重放。成功以原生LIBERO谓词为准。

|套件|无容差|外部预设容差|差值|
|---|---:|---:|---:|
|Spatial|98/100|95/100|−3个百分点|
|Object|97/100|95/100|−2个百分点|
|Goal|92/100|89/100|−3个百分点|
|Long|93/100|91/100|−2个百分点|
|全部|380/400=95.00%|370/400=92.50%|−2.50个百分点|

同seed配对：双方成功356，无容差成功而容差失败24，反向改善14，双方失败6，净减少10成功。关键容差失败任务为Goal03开抽屉放碗5/10、Goal05推盘6/10、Long09杯入微波炉并关门6/10；逐任务与精确失败seed见原报告和formal_nominal_vs_tolerance.json。

正式无容差400由seed0–5完整新跑240次，加保留seed6–9首次一次性160次构成，按seed预先分区，无失败筛选。与容差400合计800个正式回合记录。更早的失败定向重测17条不纳入该对照；旧latest385/400和original372/400探索性基线不替代正式380/400。后续无容差扩展到2000次、1903成功是另一分母，不与这400容差组直接配成50seed比较。

## 保存轨迹的运动学复算（2026-11-09）

从每帧post_action真实关节角取20Hz有限差分，速度/加速度/jerk为7关节向量L2范数的帧均值，关节路程为相邻关节角L2距离和；不填充初始导数，不跨回合差分，每回合等权。另独立报告joint_target_rad参考序列；不能与真实物理关节轨迹混写。所有400样本含成功与失败，双方成功356另列，不能把失败停止运动或步数耗尽轨迹当成成功轨迹。

|真实物理关节轨迹指标|全400无容差→容差均值|降低|双方成功356降低|
|---|---:|---:|---:|
|速度rad/s|0.605157→0.470893|22.19%|23.16%|
|加速度rad/s²|1.964056→1.543601|21.41%|24.19%|
|jerk rad/s³|39.300158→29.352660|25.31%|28.21%|
|关节路程rad|4.534833→3.752204|17.26%|20.61%|

参考关节序列全400的速度/加速度/jerk分别降低22.70%/25.00%/25.89%；双方成功356分别降低23.40%/27.90%/29.11%。两组闭环动作和轨迹长度不同，因此这些是整套闭环方案的描述性对照，不能解释为控制相同轨迹的纯容差因果效应。平均时长全400为7.970→8.316秒，双方成功为7.297→7.550秒；运动量下降不等于任务完成更快。

本次没有重算mY/最小奇异值，不引用后续离线标注实验的四维收益冒称本次闭环结果。

## 本地绝对源地址与审计

- 容差原run：`/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_tol30_confirmedmask_hold20_40tasks_10seeds_20261104`。请求`evaluation_request.json`，原总结`CONFIRMED_MASK_10SEED_REPORT.md`、`confirmed_mask400_report.json`，原独立审计`evaluation_independent_final_audit.json`，完成凭据`task_completion.json`、`evaluation_completion.json`；逐回合`episodes/<suite>/task_XX/seed_N/{result.json,trajectory.json,trajectory.npz,rollout.mp4}`。原日志/源码快照仍在run，未改写。
- 正式名义run：`/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_formal_seed0to5_20261104`；`FORMAL_NOMINAL_400_REPORT.md`、`combined_formal_400.json`、`formal_nominal_vs_tolerance.json`、`formal_400_audit.json`，以及原`evaluation_request.json`、`evaluation_independent_final_audit.json`。seed6–9来源为`/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_40tasks_10seeds_extension_20261104`，每条精确result路径在per_pair.json绑定。
- 模型为远端`czl@100.73.28.99:/mnt/SharedData/czl/checkpoints/pi05_libero_reconstructed_nostate_6gpu_bs72/Pi05_libero_reconstructed_NoState_LoRA_6gpu_bs72_10k_20261101/10000`，这是远端地址，本次未连接服务器。
- 本次归档根：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-04_libero_preset30_rollout400`；只读入口`recall.py`，日志`recall.log`，全部统计`summary.json`，逐配对结果和来源`per_pair.json`，源文件绝对路径及SHA`sources.json`，复核`verification.json`。原始视频/状态/轨迹未复制、移动或修改。
- 原容差400审计全部每帧动作/state、25物理步、失败回滚、实时阶段/mask、容差边界、初态、原生成功及视频完整解码。首轮审计对原生缓存观测与结束几何的读取时序差异保留于run/audits/0001_native_observable_vs_end_geometry；之后使用正确同观测时序审计原已存回合，未重跑400条。
- 正式无容差全400审计链排除了旧17失败定向重测；原审计历史与恢复见原run。此次只读计数、来源SHA与已保存轨迹有限差分复算，不新增物理结论。
