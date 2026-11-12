# 完整1875条重构轨迹：新版EMA容差值标注

状态：已完成处理并通过独立审计；有标签1851/1875。最终结果见RESULTS.md。

已于2026-11-07 14:43启动，总控9702、标注主进程9704，6个真实搜索子进程已核实活跃；首6条零容差基线完整通过，实际结果cache85/20Hz/零物理步/名义无阶段偏置和全部冻结源码SHA已核对。启动检查见本目录startup_verification.json；该检查不代替1875条最终独立审计。

用户随后要求定时监测并汇报，已建立本聊天每15分钟监督libero-1875。每次汇报真实已提交数量、标签状态、增量与阶段，异常及时说明，全部完成并通过审计后删除监督。该监督不改变标注协议或启动第二套计算。

用户于2026-11-07授权本地完整数据集标注。样本为重构数据集全部1875条成功轨迹、312374帧、40任务；Spatial465/Object493/Goal458/Long459。原2000来源中125条重构未成功轨迹不在已发布LeRobot成功数据集中，本次不额外纳入。所有1875条用本次统一新版重新计算，旧199条标签与原数据保留原位，不混用旧缓存冒充新版。

沿用手工语义表各轴5/15/30度作为双向搜索上限，0为严格轴；保持原阶段分割规则。正式共享MuJoCo运动学后端、IPOPT20次迭代、20Hz、运动目标delta_q_squared与EMA权重10、结构化初始化后每块16次贝叶斯追加，6本地并行worker。采用当前EMA在推进名义姿态后同轴采样的cache85实现；整段零容差不继承阶段偏置。无服务器推理、无物理步进、无任务物体，完整机器人自碰撞及原生支撑面几何；Object保留原floor，其他来源保留原桌面水平范围。

逐帧零容差回放冻结c0；所需净空min(5mm,c0)，候选验收完整运动学通过且最小相对余量>=−0.1mm。原名义负净空可被保留，本规则不是绝对碰撞安全证明。零基线无法完整求解时明确记baseline_unavailable，不补造净空或标签，不提高迭代预算。无合格候选与适配错误另列。结果尚未产生，不报告成功率；处理完成不等于所有1875条都有标签。

79项相关回归通过（EMA、零容差参考、相对净空、手工上限、floor、适配器），使用现有uv环境，不安装依赖。批次请求冻结全部相关源码SHA与容差表SHA，保存源码快照；中途源码变化会显式报错，不修改冻结请求绕过。保留已完成结果；无全量活动实例时才启动新总控。最终调用既有独立audit_batch核对逐帧来源、零基线、全部候选验收、上限、最佳选择、标签和资产/源码SHA。完成后生成RESULTS.md、final_summary.json及task_completion.json；这些文件当前待生成。

## 本地源数据地址

- 正式run：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107`。
- 请求/选样/手工表：run下annotation_request.json、selection.json、authored_tolerances_deg.json；源码快照source_snapshot/；启动证据launch_verification.json、pipeline_process.json、pipeline_launch.json。
- 每条原始输入/零基线/候选与标签：run下`<suite>/task_XX/demo_N/prepared_input.json`、zero_baseline.json、subtask_*/annotation.json、evaluations下trial.json、tolerance_labels.json及annotation_status.json。尚未提交的条目待生成。
- 状态与日志：run下pipeline_state.json、annotation_process.json、annotation.log、pipeline.log及每条search.log；独立审计阶段independent_audit_state.json、independent_audit.log、annotation_independent_final_audit.json待生成。
- 监督脚本：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/pipeline.py`，只调用既有标注CLI和独立审计，不复制优化器或物理循环。
- 正式标注入口：`/media/czl/sata/franka_my_code/franka_mujoco/annotate_libero_tolerance.py`。
- 发布数据来源索引：`/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_full_20261101/meta/reconstruction_sources.json`；原来源根目录`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261025_success10_v1/episodes`及`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261101_full_v1/episodes`，其trajectory.json、states.npz、recording.hdf5、state_reference.hdf5、report.json和已有视频保留原位。精确来源在selection及prepared_input溯源。
- 旧199条标注：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_dsp16_first5pertask_20261106`及`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_zero_nominal_continuation_20261106`；不覆写。
- 可视化状态：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261101_full_v1/viewer/annotation_status.json`，由本次run导出；切换前状态备份在run/previous_viewer_annotation_status.json。原1875条demo目录不变。本次不生成物理视频。

最终结果文档与独立审计待生成；无远端数据，无新模型或训练。

## 最近定时核验

2026-11-07T18:08:14.259820+08:00：真实已提交完成状态1875/1875（100.00%），有标签1851、无合格标签0、基线不可用24；另有基础设施错误0条未计入完成。较上次1753条新增122条。原9702/9704已真实退出，主批次因历史GP错误exit1；唯一恢复监督60491已接手，当前阶段auditing_recovered_batch，原候选全量SHA快照需读取大量文件，D态IO不当作卡死。冻结源码SHA不变，原两锁由恢复监督持有；磁盘余量44.89GiB。仅补Long02/demo40剩余3查询后全量终审，当前未宣布完成或应用存储补丁。

逐轮证据：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/monitor_history.jsonl`；最近快照/原进程退出/恢复PID与IO：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/monitor_latest.json`；恢复状态/日志：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107/recovery_state.json`、`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107/recovery.log`。

## 文本结果空间核查

369条已提交时，本次run文件逻辑体积1.30GiB，媒体文件0；10541份trial.json共1143.63MiB，prepared_input共88.05MiB，最终标签23.63MiB，日志仅0.55MiB。主要是全部搜索候选及各trial重复包含robot_xml、asset路径/SHA、逐帧求解状态与净空，非视频占用。剩余磁盘约51.56GiB是整个文件系统剩余空间，不是本次已用量。当前冻结SHA与断点缓存依赖原文件，不在运行中压缩、删除或去重。快照地址：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/storage_snapshot.json`。

### 用户授权默认按源demo绑定静态元数据

用户明确不需反复保存模型/资产，默认绑定来源。已准备并测试新默认格式，但当前批次尚在运行，未修改其冻结源码或任何候选。只有本批次处理/独立审计完成、原总控和子进程全部退出后，才应用到后续新请求；不擅自暂停或重启标注，不修改现有SHA。当前数据不迁移、不删除，之后若需整理历史候选须有明确迁移协议与重新审计。

新候选仅省略domain_randomization中的robot_xml、assets路径、asset_sha256，引用同demo已有prepared_input.json/metadata；绑定键suite/task_id/demo/episode_index+prepared SHA，不使用会跨任务重名的单独seed数字。文件路径相对引用，可随完整demo目录移动；读取时校验SHA及身份。源缺失/变更/错配报错，不凭当前全局配置默默重建。候选专有execution_mode/task_interaction等元数据及所有指标、逐帧求解/净空完整保留。共享read_json还原原内存结构，老内嵌格式仍可读取。非LIBERO写入保持原样；直接json.load新文件会看到引用，需要通过统一读取入口取完整模型元数据。

修改范围为共享纯JSON trial_io、新的来源读取和label_franka_nominal_dsp.evaluate写入，以及相关测试/文档，不修改优化器或搜索算法。初版“整块元数据完全等于prepared”的真实样本检查发现候选多了execution_mode/task_interaction，已改为仅绑定三项静态字段，避免丢弃候选专有信息。最终10项测试通过（旧格式、双任务同seed、源篡改、错配、相对路径、往返完整性、动态信息保留）；6份真实普通/零基线候选往返精确相等；staged语法与git apply --check通过，当前全量冻结源码SHA未变。

本地交付地址：

- 待应用补丁：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/pending_storage_change.patch`。
- 修改前后SHA及待应用状态：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/pending_storage_change.json`。
- 完整待应用源码/测试：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/storage_change/`。
- 测试及6份真实候选SHA/往返检查：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/storage_change_verification.json`。
- 完成后带进程/审计/源码冲突保护的应用脚本：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/apply_storage_change_when_complete.py`。
- 应用凭据：本目录storage_change_applied.json，当前待生成。若应用前用户又改动相关源码，脚本拒绝覆盖，须保留并重新合并验证。

## 重复元数据实测

2026-11-07T15:24:01.530971+08:00固定文件列表快照：11109份trial，整个run逻辑体积1.368GiB。对XML、assets路径表和asset_sha256分别做内容哈希，各只有1个独特值；重复部分合计706.15MiB，占run的50.40%。这一比例仅指当前已产生的结果，后续套件可能出现其他模型/资产内容。

|重复字段|重复部分MiB|
|---|---:|
|robot_xml|331.19|
|assets|203.20|
|asset_sha256|171.75|

若更保守地将整个domain_randomization按完整相同对象共享（保留393份独特对象），重复部分688.75MiB，占全部trial的57.15%、整个run的49.15%。这与前三字段全局去重是两种方案，不能相加。引用/索引会有少量额外开销。

统计对现存JSON编码逐文件核对一致，另对40个属性的真实原始字节切片复算通过。逐帧求解、净空、边界和指标未假设重复，不删除；仅分析、不迁移或改写运行结果。

脚本：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/analyze_storage_redundancy.py`；来源/校验和/数值：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/storage_redundancy.json`。

## GP拟合基础设施异常

Long task02/demo40（源episode1555）第一标注块完成，第二块完成13/16追加查询后，在17观测点拟合净空GP时，SciPy L-BFGS线搜索ABNORMAL导致BoTorch既有5次拟合尝试耗尽。原搜索PID50275已退出，主批次继续处理其他来源；不将此错误记为运动学失败或无合格标签，不覆盖已提交候选。已保存原错误、日志、checkpoint与所有原trial SHA。

只读数值诊断：原100迭代/相同query随机种子/相同checkpoint可复现拟合错误；仅诊断将L-BFGS线搜索maxls改100（默认20），目标/净空/可解性三模型分别21/35/37迭代收敛，均未超过原100迭代上限。没有执行ask/tell、物理回放或新候选，没有修改生产源码/请求。该诊断不是已完成恢复；数值选项差异须单列，不能声称完全相同数值参数。主批次退出后需保护已有SHA并单独续完未完成的3次查询，再汇总并独立审计；当前主批次将因其历史错误返回非零，不能仅用后续summary宣告成功。

诊断目录（原始错误/日志/checkpoint、两种拟合诊断、脚本与原候选SHA）：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/gp_fit_failure_libero10_task02_demo40`。

### 异常恢复与终审衔接已准备

单条恢复监督已启动等待原批次退出，PID60491，脚本`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/recover_gp_then_audit.py`。等待期间不改主批次，不执行恢复查询；原进程/搜索全部退出并取得原两把锁后，保存全部原trial/标签/完成状态SHA与失败summary，再仅恢复Long02/demo40缺失3次追加。仅原GP拟合五重试失败后回退maxls100，恢复前模型和Torch RNG；真实checkpoint只读回归已验证默认失败→回退成功，不改变初始query、随机种子、GP100迭代/额外16/IK20/净空与mask。恢复后原候选SHA必须不变，旧adapter_error归档保留，合并真实1875状态并调用既有完整独立审计，再复用原finish生成报告与完成文件。

单独数值恢复请求在run/attempts/0001_gp_fit_recovery/recovery_request.json（执行时生成）；保留文件清单preserved_files_before.json、原错误original/及original_case/、数值回退numerical_fallback_trace.json、完成recovery_completion.json；run/recovery_process.json、recovery_state.json、recovery.log供监督核实。任何不符合保护条件会明确停止恢复，不绕过SHA或重测完整回合。待生效存储应用脚本已加入恢复监督退出检查，不能在恢复/审计未结束时应用。
