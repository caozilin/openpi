# APP Ry接近零：原搜索过程诊断

状态：2026-11-09，只读诊断和来源/逐帧净空复算完成。沿用同一47条demo与2026-11-07冻结协议，无新训练、搜索或回放。近零定义为某侧边界≤1°；每条有12个结构化初始点+16个追加点，联合搜索APP 2维与EXEC 6维。候选不是逐轴独立求最大可行值。

47条中15条至少一侧近零，其中4条双侧近零，11条仅单侧。

## 验收规则与选择过程

每帧要求 c_candidate(t) ≥ min(5mm,c_zero(t))−0.1mm，全程通过才可进入速度排名。完整可解与相对净空分别验收，不能只看整段最小净空。选中行严格按(velocity_rad_s, observation index)取最小；没有为大容差加偏好，也没有对微小速度差设置等价阈值。最终将整组候选的APP/EXEC边界直接写成标签。

## 全部近零样本

|demo|选中APP Ry−/+°|实际APP峰值−/+°|合格/查询|近零侧有>1°的合格候选|选中较该侧最佳较大候选降速|
|---|---:|---:|---:|---|---:|
|demo_5|30.0000 / 0.0000|12.1754 / 0.0000|4/28|positive: 0|—|
|demo_8|0.0467 / 0.1292|0.0467 / 0.1292|2/28|negative: 0；positive: 0|— / —|
|demo_9|30.0000 / 0.0000|12.7644 / 0.0000|28/28|positive: 20|0.01706892%|
|demo_13|0.7432 / 9.4339|0.7432 / 9.4339|7/28|negative: 0|—|
|demo_18|30.0000 / 0.0000|8.4505 / 0.0000|15/28|positive: 6|2.56574148%|
|demo_20|28.3834 / 0.0000|12.9966 / 0.0000|17/28|positive: 5|5.02888803%|
|demo_21|30.0000 / 0.0000|9.7574 / 0.0000|26/28|positive: 21|0.17203625%|
|demo_24|0.0000 / 29.5248|0.0000 / 8.8146|5/28|negative: 1|0.55860187%|
|demo_29|30.0000 / 0.0000|5.4360 / 0.0000|28/28|positive: 16|0.00006672%|
|demo_38|0.0000 / 30.0000|0.0000 / 11.6659|28/28|negative: 24|0.00000078%|
|demo_42|28.2184 / 0.0000|6.4720 / 0.0000|26/28|positive: 12|1.26811370%|
|demo_44|0.0000 / 0.0000|0.0000 / 0.0000|1/28|negative: 0；positive: 0|— / —|
|demo_45|0.0000 / 0.0000|0.0000 / 0.0000|1/28|negative: 0；positive: 0|— / —|
|demo_46|0.0000 / 26.1187|0.0000 / 19.2561|28/28|negative: 19|0.00028716%|
|demo_49|0.0000 / 0.0000|0.0000 / 0.0000|1/28|negative: 0；positive: 0|— / —|

有该侧较大合格候选的9条：demo_9, demo_18, demo_20, demo_21, demo_24, demo_29, demo_38, demo_42, demo_46。近零是现有候选中的速度选择；其中也有EXEC同时改变的联合配置，不能全部当作仅改变一个APP方向的因果比较。

未找到该侧较大合格候选的6条：demo_5, demo_8, demo_13, demo_44, demo_45, demo_49。这表示28次记录中没有找到，不代表数学上不存在。

## 双侧近零的4条：约束过滤

|demo|全开APP实际峰值−/+°|全开净空余量mm|全开仅APP净空余量mm|最终APP边界−/+°|合格数|
|---|---:|---:|---:|---:|---:|
|demo_8|2.6306 / 7.5723|-0.921108|-0.921108|0.0467 / 0.1292|2/28|
|demo_44|4.8943 / 8.3602|-7.988780|-4.539252|0.0000 / 0.0000|1/28|
|demo_45|0.1465 / 10.6941|-4.072266|-0.111062|0.0000 / 0.0000|1/28|
|demo_49|0.0000 / 14.5535|-3.358981|-0.317672|0.0000 / 0.0000|1/28|

demo44/45/49：第0个全零候选合格，另外27个均完整可解但相对净空不合格，零是原第0候选被保留。demo8：只有全零和第19查询合格，最终选择第19，APP=0.046718/0.129195°，实际峰值几乎贴住两侧上限；速度1.034235→0.969682rad/s，降低6.24%。

demo8全开在源frame36（grasp）零参考净空8.202539mm，因此要求5mm；候选3.978892mm，加0.1mm容忍后余量−0.921108mm，被淘汰。其整段最小距离1.233774mm反而高于零参考1.104836mm，但不是发生于同一帧，不能据此判为合格。

demo44仅APP全开/EXEC严格零，在frame34（pregrasp）要求5mm，实际0.360748mm，余量−4.539252mm。demo45仅APP全开在frame40（grasp）要求3.113623mm，实际2.902561mm，余量−0.111062mm。demo49仅APP全开在frame71（postgrasp）要求3.164263mm，实际2.746590mm，余量−0.317672mm。因此APP修改可以在后续阶段触发整段淘汰。

## 单侧近零不等于整个Ry被禁止

demo9：全开APP±30°实际峰值−2.352373/+10.278632°，速度0.713414589rad/s；选中APP−30°/+0°实际−12.764383/+0°，速度0.709688252rad/s。全开与选中都合格。demo21、29也从主要正向释放切换为负向释放；demo38、46相反，保留正侧而关闭未被使用的负侧。

可固定其它搜索边界的直接对照：demo21第1/5查询只改APP正侧30→0°，速度0.700363042→0.695953635rad/s；demo29同样只改这一侧，0.665891028→0.658029781rad/s。它们的全开候选可行，正侧归零是有限预算控制回放的速度选择。

demo38最终与全开的速度只差约0.00000576%，实际正侧峰值11.66590195→11.66589534°，负侧原本就没有使用。选择器没有微小收益等价处理，仍会把带零边界的候选写为标签；这个零不提供该方向物理不可释放的证据。

## 全零参考模式的额外差异

冻结runner对完整trial判断是否所有边界为零：全零时zero_tolerance_nominal_reference=True、stage_handoff_policy=nominal_without_stage_offset；任一APP或EXEC边界非零则False、inherit_accepted_handoff。初始/阶段衔接捕获的参考分支不同。所有读取的原始trial都验证了该记录字段。

因此全零与任意非零的比较还包含参考策略变化，不能看成在完全相同参考下把容差连续缩到0的纯扫描。demo8/44/45/49连APP固定0、只开EXEC的第3初始候选也全部因净空失败。这进一步说明全局零回退不能单独解释成APP Ry本身无自由度。未新跑同参考对照，不能量化此模式差异贡献多少。

零基线还可能有负净空：demo44最小−28.459825mm，45为−12.072487mm，49为−13.273985mm。规则允许保留零基线已有负净空；这里只能解释为离线几何相对验收，不能把它等同于原物理场景的绝对碰撞安全性。

## 正常对照与结论边界

demo0：28/28候选合格，APP30/30°实际0/11.467058°；速度0.781011551→0.599242237rad/s。它有充足的相对净空余量，搜索可以保留较大边界并获得释放收益。与近零组的关键差异可在完整轨迹逐帧净空、释放方向及候选排名中直接核对，不能只用初始摆放图片判断。

结论：近零标签混合了整段净空过滤、单侧运动偏好、接近等价候选的数值排名以及全零参考模式差异。现有标签表示已搜索到的速度最优合格候选，不足以识别任务Ry的唯一或最大几何容差。所有候选和失败均保留，无补试和成败筛选。

## 本地源数据与复核

- 原冻结标注/请求/选样/审计/日志根：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107`；每demo来源`libero_spatial/task_00/demo_N/subtask_00/annotation.json`、`evaluations/<hash>/trial.json`及`zero_baselines/subtask_00/evaluations/<hash>/trial.json`；配置`annotation_request.json`、`authored_tolerances_deg.json`，日志`annotation.log`、`pipeline.log`与每demo的`search.log`；原独立审计`annotation_independent_final_audit.json`。

- 绑定标签的物理数据plan：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1/plan.json`。本次未从物理成败选择样本；原物理视频与逐回合记录地址沿用本目录README。无新视频、远端运行或模型版本。

- 本次脚本：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_spatial00_label_distribution/diagnose_pre_ry.py`；全47条概要及近零组完整查询过程：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_spatial00_label_distribution/pre_ry_diagnosis.json`；逐份原文件绝对路径/SHA：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_spatial00_label_distribution/pre_ry_diagnosis_sources.json`；验证：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_spatial00_label_distribution/pre_ry_diagnosis_verification.json`。只保存派生统计，未复制原始轨迹/视频。

- 冻结实现：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107/source_snapshot/label_franka_nominal_dsp.py`的search_subtask；`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107/source_snapshot/tasktol/tolerance/clearance_policy.py`的relative_feedback；`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107/source_snapshot/tasktol/counterfactual/runner.py`的zero_tolerance_nominal_reference；`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107/source_snapshot/tasktol/tolerance/release_state.py`的begin_stage。
