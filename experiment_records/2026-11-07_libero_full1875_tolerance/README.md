# 完整1875条重构轨迹：新版EMA容差值标注

状态：已完成并通过完整独立审计；后续存储默认已应用并验证。更新时间：2026-11-07T18:16:38.218249+08:00。

1875/1875已处理（100%），1851条有标签（98.72%）、24条零基线不可用（1.28%），无合格标签0、活动适配错误0。历史GP基础设施错误1条，已单独记录并断点恢复；没有覆盖成功/失败候选或挑选重测。

## 各套件状态

|套件|已处理|有标签|零基线不可用|无合格标签|活动适配错误|
|---|---:|---:|---:|---:|---:|
|Spatial|465|455|10|0|0|
|Object|493|493|0|0|0|
|Goal|458|448|10|0|0|
|Long|459|455|4|0|0|
|合计|1875|1851|24|0|0|

## 目的与冻结协议

全部1875条成功重构来源、312374帧、40任务重新标注，未混用旧199条标签。手工语义表5°/15°/30°为分轴搜索上限，0为严格轴；cache85新版EMA同轴采样，整段零容差不继承阶段偏置。共享IPOPT20迭代、20Hz、delta_q_squared+EMA权重10、结构化初始化后每块16次贝叶斯追加、6本地worker。仅机器人自碰撞及原生支撑面的离线运动学，无物理步进、任务物体、服务器推理或模型训练；Object保留floor，其他来源保留有限桌面。

每帧参考净空为min(5mm,c0)，其中c0来自本条零容差运动学回放；候选完整可解且全程相对净空余量≥−0.1mm才可接受。可保留原有负净空，本协议不等同于绝对碰撞安全。零基线不能完整求解的24条保留baseline_unavailable，不补造净空或标签，也不认定为物理不可达证明。

## 四项配对运动学结果

|指标|零容差基线均值|选中候选均值|变化|
|---|---:|---:|---:|
|平均关节速度（rad/s）↓|0.660007|0.531313|降低19.50%|
|平均关节加速度（rad/s²）↓|5.491685|4.048297|降低26.28%|
|平均可操作度 mY ↑|0.079178|0.077162|降低2.55%|
|归一化最小奇异值 ↑|0.898152|1.102212|提高22.72%|

统计分母1851条有标签且零基线完整的demo，共2275对选中候选/零基线标注块。每demo内部速度按N−1、加速度按N−2、mY按N加权合并独立块，最小奇异值取各块最小值；不虚构跨块差分。整体每demo等权，百分比由两组整体均值之比计算。逐demo百分比均值另存JSON作诊断；近零基线会放大最小奇异值的百分比，故不将其作为主要收益。

这里“选中”是已评估可接受候选中平均关节速度最小者，不是已证明的全局最优，也不是四目标同时优化。四项均为理想运动学回放指标，不能直接当作物理跟踪或闭环任务成功率。0容差严格任务仍纳入主统计；排除Goal00/05后1768条的补充统计在JSON中单列，不替换主分母。

|套件|配对demo|速度降低|加速度降低|mY变化|最小奇异值提高|
|---|---:|---:|---:|---:|---:|
|Spatial|455|22.78%|32.36%|+3.38%|46.71%|
|Object|493|25.80%|32.90%|-14.57%|12.69%|
|Goal|448|13.06%|17.05%|+0.60%|17.52%|
|Long|455|15.98%|19.39%|+3.45%|22.39%|

## 基线不可用条目

|套件/task（0起始）|零基线不可用demo|
|---|---|
|Spatial / 02|demo_37|
|Spatial / 04|demo_1, demo_20, demo_21, demo_24, demo_37, demo_42|
|Spatial / 06|demo_14|
|Spatial / 07|demo_49|
|Spatial / 09|demo_39|
|Goal / 02|demo_11, demo_17, demo_19, demo_20, demo_21, demo_23, demo_24, demo_25, demo_27, demo_45|
|Long / 05|demo_15, demo_43|
|Long / 08|demo_27, demo_36|

上述为零基线无法完整生成的终止状态，未进行额外重搜或追加迭代，不据此宣称原数据物理不可达。

## 历史数值异常、恢复与审计

原批次处理完后因1条历史GP拟合错误退出1。Long task02/demo40（源episode1555）第一块已完成，第二块在17观测（4初始化+13追加）时L-BFGS ABNORMAL，既有5次拟合重试耗尽。原进程全部退出并取得原锁后，单独冻结恢复请求；仅当原拟合失败时恢复相同模型状态/Torch RNG，将内部线搜索maxls20改100，GP上限仍100迭代。只追加缺失3查询，原62738份候选/标签/状态保护文件SHA全部未变，原失败证据保留。IPOPT、seed、追加总预算16、净空规则、动作和mask均未变。这是有记录的数值配置差异，不冒充完全相同数值设置或独立重测。

恢复完成后完整1875条既有独立审计exit0，审计报告56715份raw trial、逐帧标签、分轴上限、来源SHA、零物理步和无任务物体均通过。原总控/标注/恢复/审计全部真实退出，三把锁已释放。

剔除该数值恢复样本后1850条：速度降低19.50%、加速度降低26.29%、mY变化-2.55%、最小奇异值提高22.72%。差异不足0.01个百分点。

## 已应用的后续存储默认

全量审计、四维统计及真实进程退出核实后，已应用用户授权的后续默认存储格式：robot_xml、assets、asset_sha256按suite/task/demo/episode_index+prepared SHA绑定到已有prepared_input.json；候选保留相对路径及SHA引用，候选专有元数据和全部逐帧指标照常保存。三个生产读取入口还原相同内存结构，源缺失/错配/篡改明确报错；旧内嵌格式兼容。89项相关回归、完整tasktol/tests语法检查及6份真实候选旧/新格式往返检查通过。

当前1875批次的原始候选未迁移、删除、压缩或重写；旧标签/原始数据/用户viewer保留。新默认仅适用于之后新请求。当前生产源码因此有已记录存储差异；本批次复现/审计应使用冻结请求和source_snapshot，不能拿更新后的root源码冒称原冻结SHA。

## 本地源数据地址

- 正式原始结果根目录：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107`。
- 请求/选样/语义上限：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107/annotation_request.json`、`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107/selection.json`、`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107/authored_tolerances_deg.json`。
- 原冻结源码：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107/source_snapshot`（190份文件SHA已再次核对）；1875来源索引：`/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_full_20261101/meta/reconstruction_sources.json`。
- 每demo：上述run下`<suite>/task_XX/demo_N/prepared_input.json`、`zero_baseline.json`、`annotation_status.json`、`tolerance_labels.json`（有标签1851条）；`subtask_*/annotation.json`、`subtask_*/evaluations/*/trial.json`及`zero_baselines/`。确切路径与SHA在独立审计及配对统计清单中。本次未生成物理视频。
- 完整独立审计：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107/annotation_independent_final_audit.json`；运行终态：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107/task_completion.json`、`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107/final_summary.json`；日志：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107/annotation.log`、`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107/pipeline.log`、`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107/independent_audit.log`、`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107/recovery.log`及每demo/search.log。
- 数值恢复原始失败/保护清单/请求/回退trace/完成凭据：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107/attempts/0001_gp_fit_recovery`；先前只读诊断：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/gp_fit_failure_libero10_task02_demo40`。
- 四维来源、逐demo/块配对及统计：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/kinematic_gain_sources.json`、`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/kinematic_gain_pairs.json`、`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/kinematic_gain_summary.json`、`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/kinematic_gain_verification.json`；只读计算脚本：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/analyze_full_kinematic_gain.py`、`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/verify_full_kinematic_gain.py`。
- 存储改动前后SHA/补丁：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/pending_storage_change.json`、`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/pending_storage_change.patch`；应用凭据和后验回归：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/storage_change_applied.json`、`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/storage_change_postapply_verification.json`、`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/verify_applied_trial_io.py`。
- 最终进程/锁/源码快照核验：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/final_verification.json`；历史监测：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/monitor_history.jsonl`。先前空间快照`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/storage_snapshot.json`、重复字段分析`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/storage_redundancy.json`是运行中历史样本，不是最终run体积。
- 可视化状态：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261101_full_v1/viewer/annotation_status.json`，已导出本次完整1875条状态；本次未停止用户viewer。旧199标签保留于`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_dsp16_first5pertask_20261106`及`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_zero_nominal_continuation_20261106`。
- 无远端数据、无新模型或训练。


本次15分钟监督libero-1875已在所有工作完成后删除。完成交付凭据：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/delivery_completion.json`。

## 搜索选中与语义上限全开对照

对1851条有效基线demo，上限全开验收1238/1851（66.88%），搜索选中1851/1851（100%）；全开失败33条运动学、580条净空。双方均通过的1238条，速度0.517966→0.513589rad/s，降低0.85%。若将净空违规但运动学完整全开也纳入1818条，搜索速度反而提高2.12%；主要收益是约束满足。此为离线样本内验收，不是物理任务成功率。

完整定义、套件/任务表、配对清单和源SHA见[对照报告](BEST_VS_FULL_REPORT.md)。只读复用已提交最大点，无新增回放。

## 24条零基线不可用的逐条诊断（2026-11-09）

22条IPOPT迭代上限未通过、2条机器人自碰撞；24条候选搜索数量全部为0，并非贝叶斯搜索后全无合格候选。原名义物理录制均成功。[逐条终止位置与原始证据](BASELINE_UNAVAILABLE_DIAGNOSIS_20261109.md)。

## 2026-11-09后续补标（原批次保留）

用户要求基线不必可行、直接补标尝试。单独隔离原cache85执行源码，为原24条增加失败保持的基线几何观测模式，候选正常严格验收；24/24补标并通过独立审计（660原始trial），原5578份保护文件SHA未变。原批次1851/1875结果及本页原统计保持原分母，新训练可合并原1851+新24标签；不能把补标基线当作完整可行的运动学对照。本次补标协议、路径与后续训练状态见[新实验记录](../2026-11-09_libero_nominal_joint13_training/README.md)。
