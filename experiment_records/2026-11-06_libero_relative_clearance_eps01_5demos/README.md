# 每任务前5条成功demo：相对零基线净空，epsilon=0.1mm

状态：200条已处理并通过独立审计。原名义参考、原40任务/200条demo、手工5/15/30度上限和16次贝叶斯追加冻结；完整机器人、自碰撞、原生水平支撑面，零物理步、无任务物体或策略推理。

结果：198条有标签，0条无合格标签，2条零基线不可用，适配错误0；核验5992条候选。

零基线不可用代表零容差不能完整满足运动学要求，不伪造基线；无合格标签不等于证明连续域无解。Object按来源XML保留无限floor平面，其他任务沿用有限XY桌面，未补造桌子。新相对规则允许保留原负净空，不是绝对碰撞安全保证。

|套件|有标签|无合格标签|零基线不可用|
|---|---:|---:|---:|
|libero_spatial|49|0|1|
|libero_object|50|0|0|
|libero_goal|50|0|0|
|libero_10|49|0|1|

本地源数据目录：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_dsp16_first5pertask_20261106`。其中annotation_request.json/selection.json冻结请求与来源；各suite/task_XX/demo_N含prepared_input.json、zero_baseline.json、subtask_*/annotation.json和候选trial、tolerance_labels.json；annotation_independent_final_audit.json为独立审计；pipeline.log/annotation.log/independent_audit.log为日志；final_summary.json含逐demo/子任务速度和余量。原数据由来源索引映射，未改写。

### 论文定义核对与统一方向

只读核对当前英文`/home/czl/桌面/TaskTol/TaskTol.tex`第168/178/185、424–439行，以及中文`/home/czl/桌面/TaskTol/TaskTol_zh.tex`第449–462、534–552行。论文明确将阶段相对旋转在重力对齐容差系中提取固定XYZ/RPY坐标，Specific坐标为零，Tolerant坐标按带符号界约束；离线回放与在线执行共享此问题。故统一方向应为论文的阶段相对RPY路径，零上限也通过同一坐标定义，而不是为救回单条失败把统一模式直接改成原绝对SO(3)路径。此前统一回原路径的建议在查阅论文前过早，已更正。名义动作依旧为SO(3)旋转向量增量；混合mask的有限RPY盒不能直接替换成log向量分量盒。未修改论文、代码、请求或任何结果。证据与论文SHA：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-06_libero_relative_clearance_eps01_5demos/manuscript_constraint_alignment.json`。

### 用户继续授权：名义轨迹不必全帧求解成功

当前原批次实际已完成：200条、198条有标签、2条零基线不可用、适配错误0，5992条候选独立审计通过；上述完成范围不等于200条均有标签。用户随后明确名义轨迹不要求保证求解成功，继续标注。应撤销把零基线可解性当作整条样本搜索资格的前置条件，候选标签仍须符合原完整运动学与净空验收。当前两条未搜索：Spatial04/demo_1及Long08/demo_4。保留原198条标签及全部候选，不改写既有结果。

需确认的定义：原逐帧相对净空公式依赖完整c0_t；两条零容差回放提前中断，缺少后续帧净空。已询问用户：零回放失败保持上一接受关节解并继续测量，或采用原重构保存关节目标FK。确认前不补造缺失净空、不把前缀最小值填充为全轨迹、不静默改为固定5mm、不启动重复200条标注。

诊断归因范围更正：此前SO(3)对照实际同时改变了姿态误差的表达坐标系及log/RPY参数化；受控证据隔离的是整个旋转硬约束表达组合，并未分别隔离F坐标变换与参数化的作用。保持初值、目标和历史相同恢复求解的结论不变。

### 两条未标注回合是否接近极限：已完成诊断

仅诊断，不改变原198条标签、两条原状态、失败门限、阶段参考或候选预算。复用run_counterfactual_trial和IpoptIK.fork，不新增执行循环、无物理步。源代码188文件核对未变；两条zero trial仍与原冻结SHA一致。

- Spatial04/demo_1：135帧/6.75s，141总帧；原生任务首次成功在108帧，失败发生于成功后保留的完整尾段。原候选最近关节限位仍19.33度，未贴限位；速度归一化sigma_min从此前中位约1.109降至0.003129，原保存关节解也只有0.003168，属于接近奇异的参考构型。20次迭代位置残差0.014067mm，仅超过0.01mm门限0.004067mm；21次变为0.003649mm并完整通过。阶段偏移仅0.0000125度。
- Long08/demo_4：失败为第二标注块本地200帧，即原413帧/20.65s，420总帧；不是原200帧。原生任务首次成功409帧，之后的原完整尾段仍保留。第二块从原213帧的记录实际关节状态初始化，继承相对名义姿态约0.531414度的阶段偏移。此时原保存关节目标符合原名义目标，但对当前阶段参考的严格旋转残差约0.367232度，不再是同一个姿态目标的可行见证。20次位置原始残差0.764680mm，100次仍0.677365mm。最近限位22.70度，未贴限位；候选归一化sigma_min约0.004349，此前中位约1.018。只消除交接姿态偏移后，同一原413帧5次迭代即满足目标，整个207帧块通过；该对照改变了目标，仅用于归因，不能替代正式基线。

Long固定原阶段参考、原目标及目标函数，使用12个数值初值（上一命令、原关节见证、原始候选、实际状态、8个固定随机扰动），每次最多300迭代：0/12合格；其中10个收敛到局部不可行，9个停在几乎同一位置残差0.722608mm处（另1个1.165530mm），姿态残差接近零，原见证初值所得局部点归一化sigma_min约9.95e-9。支持偏移后参考位姿触及局部运动学边界，而非仅仅20次预算不足；不提供全局不可达证明。所有多初值残差均取IPOPT原始最后迭代点，未将保持旧命令的残差混用。

补充坐标隔离：保留同一F和阶段参考，仅将RPY改为同F下SO(3)log，Spatial20次仍失败（位置0.011779mm），Long20次仍失败（0.764631mm）；故此前“SO(3)恢复”来自整套硬约束数值表达，不能把原因单独归于RPY。Spatial更像近奇异条件下对误差坐标/有限迭代敏感，Long则有明确的阶段交接偏移效应。

本地源数据地址：

- 诊断脚本：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-06_libero_relative_clearance_eps01_5demos/diagnose_baseline_boundaries.py`。
- 汇总及各组SHA：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-06_libero_relative_clearance_eps01_5demos/two_case_boundary_diagnosis/summary.json`。
- 两条逐帧原始输入/关节/奇异值/限位/候选残差：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-06_libero_relative_clearance_eps01_5demos/two_case_boundary_diagnosis/<suite>/task_XX/demo_N/<mode>.json`。
- Long12初值原始结果：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-06_libero_relative_clearance_eps01_5demos/two_case_boundary_diagnosis/libero_10/task_08/demo_4/multistart_raw.json`。
- 正式批次与来源路径由summary.json逐条列出。原标注补齐仍等待缺失c0基线的定义，不因诊断擅自改变基线或剪掉尾段。

### 用户授权修正：整段零容差不继承阶段偏置（补标运行中）

新协议仅当完整回放描述符所有有效方向上限均为零时生效：仍采用RPY/F硬约束，硬参考等于独立名义姿态；阶段和子任务切换不继承实际跟踪/上一解数值偏差。不由单帧ENG/DIS或预测边界收缩触发，原有容差释放的阶段交接保留。共享RotationReleaseState新增inherit_stage_offset，正式counterfactual后端自动按完整描述符选择；rebase_rotation_subtask在此策略下不重锚名义参考。缓存版本84，结果记录明确策略字段。未修改优化器/预算/残差门限/动作/手工上限/epsilon。

80项相关回归通过；2项既有测试因历史benchmark_optimized_trajectories.py和datasets/v1_tolerance_summary_v2.csv缺失而未运行。真实104帧非零候选在原正式MuJoCo3.10.0搜索环境中，与原缓存的全部运动学指标和逐帧净空完全一致。初次跨原生MuJoCo3.2.3环境比较出现浮点/距离差异，已改用同版本环境验证，未放宽正式验收。

真实零回放：Long08/demo_4第二块207帧在原20迭代预算下完整通过；Spatial04/demo_1仍在原135帧微小超过位置门限，不伪造完整c0。两条均在新目录补标，旧198标签、5992候选及所有原数据保留；retained_198_labels.json逐一绑定原标签SHA。新参考协议会改变零基线及后续相对净空要求，因此补标明确为新协议续项，不冒充原批次同协议重跑。未执行整200条或1875条重新标注。

本地源数据地址：

- 新补标run：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_zero_nominal_continuation_20261106`。
- 原协议衔接：run下continuation_protocol.json、retained_198_labels.json。
- 新请求/选样：run下annotation_request.json、selection.json。
- 修改前后源码：run下source_before/、source_after/。
- 验证：run下implementation_verification.json、positive_candidate_regression.json、zero_reference_regression_search_runtime/。
- 日志/进程：run下annotation.log、annotation_process.json、各demo/search.log。
- 完成审计与新结果：待生成。

补标完成并通过独立审计：新run处理2条，Long08/demo_4完整420帧生成标签（2块、40候选），Spatial04/demo_1仍为baseline_unavailable；适配错误0。旧198条标签SHA全部不变。现合并199/200有标签，1条缺完整基线；来源为旧参考协议198+新零参考策略1，不是新版全200重跑。viewer状态已合并为199有标签/1基线不可用，原始1875目录不变。新run的task_completion.json、combined_200_source_manifest.json、annotation_independent_final_audit.json已生成。
