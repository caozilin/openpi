# 24条无标注来源：零容差基线终止诊断

状态：2026-11-09只读逐条核对24个原始zero trial SHA、状态和搜索目录完成，没有新搜索/回放。原24条名义物理录制均为任务成功。

22条零容差IPOPT在原20迭代预算内未通过可行性检查，终止状态全部为Maximum number of iterations exceeded；2条IK已求解，但FK机器人自碰撞失败。这24条均尚未进入贝叶斯候选搜索，搜索trial和subtask annotation全部为0；不是搜索穷尽后没有可接受候选，也不是桌面净空阈值拒绝了全部候选。

流程先为所有独立标注块生成完整零容差c0，再构造逐帧min(5mm,c0)验收要求；任一块基线失败即返回baseline_unavailable。因此Long08两条虽第1块基线完成，第2块失败后整条依然未搜索。不能虚构缺失帧c0、补0标签或把状态解释为释放容差后也不可解。

|套件/task（0起始）|demo|失败源帧（0起始）|原因|位置残差mm|旋转残差deg|自碰撞深度mm|
|---|---|---:|---|---:|---:|---:|
|libero_spatial/02|demo_37|114|optimizer_infeasible|1.1898|0.1878|—|
|libero_spatial/04|demo_1|135|optimizer_infeasible|2.6350|0.6047|—|
|libero_spatial/04|demo_20|106|optimizer_infeasible|13.4283|0.4858|—|
|libero_spatial/04|demo_21|66|optimizer_infeasible|11.1323|0.8359|—|
|libero_spatial/04|demo_24|77|optimizer_infeasible|11.0664|0.1314|—|
|libero_spatial/04|demo_37|82|optimizer_infeasible|10.4834|0.3284|—|
|libero_spatial/04|demo_42|111|optimizer_infeasible|7.5885|0.7707|—|
|libero_spatial/06|demo_14|48|optimizer_infeasible|8.2608|0.8069|—|
|libero_spatial/07|demo_49|150|optimizer_infeasible|6.5745|1.1049|—|
|libero_spatial/09|demo_39|124|optimizer_infeasible|5.4752|0.7949|—|
|libero_goal/02|demo_11|85|optimizer_infeasible|9.6496|0.5393|—|
|libero_goal/02|demo_17|98|optimizer_infeasible|12.1229|1.2066|—|
|libero_goal/02|demo_19|89|optimizer_infeasible|4.5829|1.0745|—|
|libero_goal/02|demo_20|86|optimizer_infeasible|10.4883|0.5727|—|
|libero_goal/02|demo_21|72|optimizer_infeasible|3.5669|0.9624|—|
|libero_goal/02|demo_23|100|optimizer_infeasible|10.4553|1.2358|—|
|libero_goal/02|demo_24|99|optimizer_infeasible|12.9438|0.4046|—|
|libero_goal/02|demo_25|73|optimizer_infeasible|3.9946|1.4106|—|
|libero_goal/02|demo_27|75|optimizer_infeasible|5.8511|0.4625|—|
|libero_goal/02|demo_45|120|optimizer_infeasible|2.9745|0.6625|—|
|libero_10/05|demo_15|230|robot_self_collision|—|—|2.3957|
|libero_10/05|demo_43|242|robot_self_collision|—|—|3.3671|
|libero_10/08|demo_27|384|optimizer_infeasible|6.1796|0.7768|—|
|libero_10/08|demo_36|511|optimizer_infeasible|1.9803|0.9938|—|

逐条原始路径和SHA：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/baseline_unavailable_diagnosis_20261109.json`；原审计：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107/annotation_independent_final_audit.json`；原冻结流程：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107/source_snapshot/tasktol/data/libero_annotation/dsp.py`。保留全部原始数据，未修改协议或冻结请求。
