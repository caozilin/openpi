# 重构LIBERO数据：每任务前5成功demo离线容差标注

状态：200条均已处理，批次因50条适配错误退出，未通过完整独立审计。73条生成标签，77条有限搜索未找到合格标签；50条Object因原记录无法提取桌面几何而失败，不能混作无合格解。用户2026-11-06确认：40任务各按原demo编号取前5条成功轨迹，共200条；语义表5°/15°/30°分别作为各轴搜索上限。

## 主体与协议

实现、运行和完整实验总结均在MuJoCo项目。复用现有DSP贝叶斯搜索/统一运动学后端；APP+EXEC六向联合连续搜索，结构化初始化+每子任务16次自适应查询，ENG/DIS0。完整机器人（含基座）/自碰撞/有限XY桌面高度几何，5mm净空；原20Hz名义参考/动作/手指/阶段冻结，零物理步、无任务物体或服务器推理。选完整合格观测中整段平均关节速度最小者，输出连续六向容差逐帧标签。

## 文档与本地源数据地址

- MuJoCo主总结：[README.md](/media/czl/sata/franka_my_code/franka_mujoco/experiment_records/2026-11-06_libero_reconstructed_authoredcaps_5demos/README.md)。
- 本次运行：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_reconstructed_authoredcaps_dsp16_first5pertask_20261106`。
- 200条来源清单：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_reconstructed_authoredcaps_dsp16_first5pertask_20261106/selection.json`。
- 冻结语义表：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_reconstructed_authoredcaps_dsp16_first5pertask_20261106/authored_tolerances_deg.json`。
- 配置与全体源码SHA：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_reconstructed_authoredcaps_dsp16_first5pertask_20261106/annotation_request.json`。
- 日志/进度：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_reconstructed_authoredcaps_dsp16_first5pertask_20261106/pipeline.log`、`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_reconstructed_authoredcaps_dsp16_first5pertask_20261106/annotation.log`、`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_reconstructed_authoredcaps_dsp16_first5pertask_20261106/pipeline_state.json`。
- 逐回合路径：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_reconstructed_authoredcaps_dsp16_first5pertask_20261106/<suite>/task_XX/demo_N/`，含prepared_input.json、逐子任务annotation.json/evaluations/*/trial.json、annotation_status.json；有合格解时含tolerance_labels.json。
- 原数据来源：`/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_full_20261101/meta/reconstruction_sources.json`；该索引映射原始记录，文件保持原位只读。
- 最终审计/完成（待生成）：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_reconstructed_authoredcaps_dsp16_first5pertask_20261106/annotation_independent_final_audit.json`、`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_reconstructed_authoredcaps_dsp16_first5pertask_20261106/task_completion.json`。

## 已完成验证与统计边界

34项回归测试及编译通过；所选Spatial04/demo0真实28查询的5°APP/15°EXEC边界、实际IPOPT负/正方向限制及逐帧标签通过；Long06白杯与布丁mask相同但角度不同的路由验证通过。证据`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_reconstructed_authoredcaps_dsp16_first5pertask_20261106/initial_authored_caps_integration_audit.json`，仅代表集成检查，不代表200条完成。

未找到合格标签与程序/基础设施错误分开统计；基座及手指5mm净空不满足时不排除几何或放宽约束，有限预算无合格标签不代表证明整个连续域无解。独立审计和最终结果由本次总控生成后追加。


## 2026-11-06：参考轨迹与跟踪误差的处理建议（尚未实施）

按用户描述的论文假设，继续采用目标参考的理想运动学评价。当前适配器从`state_reference.hdf5`读取positions/rotations；重构参考含执行跟踪补偿，和记录实际关节FK轨迹不同。补偿参考的运动学净空不等于实际执行净空；既有15条阶段诊断支持这一区别，不能推断全部77条无标签均为同一原因。

建议先区分几何名义参考与执行补偿参考：可从原记录实际关节FK生成独立派生的几何参考，再在该参考上做零容差基线检查和姿态容差搜索。原始数据保持不变。若采用派生几何参考训练，动作增量和标签必须与新参考在坐标系、时序和积分规则上严格一致；不能只换标注参考而沿用旧补偿动作，并称其为原动作的标签。几何修复应与容差优化分开记录。

桌面零穿透与5mm额外净空须分别报告。若论文约束是5mm就保留；若论文只有零穿透，5mm属于额外安全裕度，不能混写。实际轨迹也可能有接触/轻微穿透；实际FK参考不保证通过5mm检查，且同末端位姿的不同IK关节解仍需碰撞检查。基线不合格且严格阶段/位置固定时，姿态容差未必能修复，应显式记录输入不满足约束，或另做有界几何修复。

建议先在固定小样本上比较补偿参考/实际FK参考的逐帧净空与修正量，再选择派生数据方案；此节仅记录讨论，没有改桌面、裕度、原数据或冻结搜索请求，没有新增实验。


### 去掉桌面判定的影响边界（代码核对，未实施）

共享`tasktol/counterfactual/runner.py`先调用solve_stage_relative_target，检查求解器可行性/有限性，再对求得关节做FK、自碰撞与桌面几何查询；当前DSP继续记录桌面接触，并在`benchmark_joint_pre_post_t1_60.py:result_outcome`以kinematic_feasible_without_table和5mm净空作分层验收。因此固定同一输入与容差查询、只取消桌面验收时，IK求解本身及原始运动学量不因后验净空门限改变；自碰撞等其余检查仍应保留。重新运行贝叶斯搜索时，成功反馈、可接受集合和最佳标签可能变化，不能保证最终标签或轨迹不变。跟踪补偿也能改变目标自身的关节可达性/数值条件，不能断言其影响只限桌面。未执行任何新查询、修改冻结请求或改变约束。


### Spatial00/demo4：最小净空帧核对（2026-11-06 18:25）

相同冻结robot_xml、有限XY桌面、完整机器人碰撞几何，读取原104帧关节/手指状态，仅FK，禁止物理步。实际pre-action视频最小净空在0起始帧52，2.60s，左手指+7.6523mm；等同动作51的post-action状态。原记录joint_target与缓存离线零容差IK均在动作89目标最小，-4.3820mm，左手指，EXEC；该命令输入帧89/4.45s，动作末目标时刻4.50s。动作89同期实际pre-state净空+16.6508mm，post-state（视频帧90）+11.1442mm。实际整段始终正净空，参考IK负净空不可描述成实际穿桌；本条原joint_target与离线零容差最小值完全一致。缓存离线trial中time_s=0是零物理步时间，不能用来定位视频，已按source_frame映射。

完整逐帧数值及来源：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-06_libero_reconstructed_authoredcaps_5demos/spatial00_demo4_clearance_frames.json`；复现脚本位于同目录`diagnose_spatial00_demo4_minimum_frames.py`。原始记录：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261101_full_v1/episodes/libero_spatial/00/demo_4/trajectory.json`。未新增搜索或修改数据。


### 下压接触假设的证据边界

动作89参考末端z=907.743mm，实际post末端z=923.213mm，实际高15.470mm；同期实际减参考XYZ=[19.511,-1.185,15.470]mm。因此“目标比实际更向下”有数据支持，但偏差不纯竖直。动作85–88连续下降时实际落后参考约17–18mm；动作90参考上移3.483mm而实际仍下降0.604mm，亦支持动态跟踪滞后。碗受接触支撑、有限刚度伺服导致下压参考偏低是合理机制，当前未检查该时刻接触对/接触力，不能确定为唯一或已证实原因。这次执行为native关节伺服，不直接等同于笛卡尔阻抗控制。原始位姿证据：同目录spatial00_demo4_reference_actual_85to91.json；未新增仿真。
