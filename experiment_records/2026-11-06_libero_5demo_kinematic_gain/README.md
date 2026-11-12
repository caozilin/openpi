# 各5demo选中候选：四项标准运动学指标收益

状态：2026-11-06只读统计完成，源哈希与数值复算通过。没有新增搜索、策略推理或物理回放。这里的最佳解是有限搜索中按平均关节速度最小选出的合格候选，不是全局最优证明，也不是四目标联合最优。

## 协议与样本

沿用每任务前5条成功重构demo、手工分轴5/15/30度上限、每块16次贝叶斯追加、20Hz理想运动学回放、原完整机器人/支撑面及相对零基线净空epsilon=0.1mm协议。200条选样中199有标签，Spatial04/demo_1缺完整基线而排除。包含198条原协议标签和1条Long08/demo_4的新零容差无阶段偏置补标，不把补标混写为同协议重跑。Goal00、Goal05共10条全禁用容差样本保留在总体平均中。

对每个独立初始化的标注块读取选中候选与其配对零容差baseline；共245对。标准顺序为速度、加速度、平均Yoshikawa可操作度mY、速度能力归一化后的全6维Jacobian最小奇异值sigma_min。这里使用实际20Hz采样，并非项目其他任务的10Hz默认域。速度按各块N−1个有效差分、加速度按N−2个有效差分合并，mY按各块N个结点合并，sigma_min取各块最小值；不凭空增加独立块间的跨边界差分，不把该统计当成重新连续执行整条拼接轨迹的验证。随后每条demo等权取平均。

## 整体结果

|指标|零容差均值|选中候选均值|由整体均值计算的变化|
|---|---:|---:|---:|
|速度，rad/s|0.654881|0.528661|降低19.27%|
|加速度，rad/s²|5.435391|4.020520|降低26.03%|
|平均mY|0.079255|0.077290|降低2.48%，退化|
|每条轨迹最小sigma_min的均值|0.900581|1.102495|提高22.42%|

变化由两个整体均值相除得到，不是逐demo百分比再平均。后一算法遇到接近零的sigma_min基线会产生极大百分比，因此不把该数作为主结果；机器可读summary.json同时保留两种算法以便复核。逐demo看：速度188改善、0退化、11基本不变；加速度188改善、0退化、11基本不变；mY89改善、99退化、11基本不变；sigma_min145改善、37退化、17基本不变。

|套件|有效demo|速度降低|加速度降低|mY变化|sigma_min提高|
|---|---:|---:|---:|---:|---:|
|Spatial|49|22.83%|32.45%|+1.72%|45.77%|
|Object|50|25.82%|32.52%|−13.68%|12.33%|
|Goal|50|13.04%|17.67%|+1.28%|18.16%|
|Long|50|15.59%|18.53%|+2.36%|20.28%|

整体mY下降主要集中在Object：42/50条退化。说明速度目标下出现可操作度权衡，不能声称四项都优化。剔除新Long补标的原协议198条，变化为速度降低19.29%、加速度降低25.99%、mY降低2.53%、sigma_min提高21.98%，方向一致。剔除10条全禁用任务，189条允许释放样本对应20.26%、27.30%、−2.64%、+23.59%。未对退化样本重选、重测或删除。

## 本地源数据与验证

- 脚本：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-06_libero_5demo_kinematic_gain/analyze.py`。
- 整体、分套件、敏感性统计：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-06_libero_5demo_kinematic_gain/summary.json`。
- 199条、245对块指标与精确trial/annotation绝对地址和SHA：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-06_libero_5demo_kinematic_gain/paired_metrics.json`。
- 复算证据：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-06_libero_5demo_kinematic_gain/verification.json`。核对199标签SHA、490次配对trial读取SHA、完整回放/可解状态、20Hz/零物理步、成对结点数和原帧覆盖；1592项块合并复算及整体均值复算通过。
- 合并来源清单：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_zero_nominal_continuation_20261106/combined_200_source_manifest.json`。
- 原198条与配置、审计、候选轨迹：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_dsp16_first5pertask_20261106`，其中`annotation_request.json`、`selection.json`、`annotation_independent_final_audit.json`，日志`pipeline.log`、`annotation.log`、`independent_audit.log`。
- 补标与配置、审计、轨迹：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_zero_nominal_continuation_20261106`，其中`annotation_request.json`、`selection.json`、`annotation_independent_final_audit.json`，日志`annotation.log`、`independent_audit.log`。
- 原物理视频/轨迹保持原位：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261025_success10_v1/episodes`及`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261101_full_v1/episodes`；精确来源由prepared_input.json溯源。本次不生成物理视频。

统计验证不替代原运动学与净空独立审计；后者沿用上述正式run中的审计文件。无远端源数据、无额外运行进程。
