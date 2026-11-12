# 取消净空验收：既有候选重排对照

状态：2026-11-09，只读重排及原始指标SHA核对完成。沿用同一Spatial00的47条demo，每条28个原候选；未启动新优化/搜索/仿真/训练，未修改冻结请求、标签或原回放文件。编号为原始demo，不是闭环评测seed。

## 对照定义

原方案：原相对净空规则下合格候选中平均关节速度最低者。对照：取消桌面相对净空与绝对桌面接触验收，保留完整轨迹、非桌面运动学合格（含自碰撞）及有限速度；在相同候选中按(observed_velocity_rad_s, index)选最低者。原成功/失败记录全部保留，只有派生的候选选择改变。

1316个查询元数据绑定原独立审计SHA；去净空后1314个可比较，2个optimizer_infeasible仍排除。原757个合格、557个仅净空失败。候选改变28/47，新的28个均来自原净空失败候选。

## 变化条数

明显变化预先统一按至少一个相同阶段/带符号方向差值≥5°统计，标注和实际峰值分别计算。实际为已存候选离线回放的阶段/方向峰值，不是SO(3)总角度，也不是后续物理回放。

|范围|标注边界变化≥5°|实际释放峰值变化≥5°|
|---|---:|---:|
|APP+EXEC全部8方向|27/47（57.45%）|18/47（38.30%）|
|仅APP Ry正负2方向|18/47（38.30%）|9/47（19.15%）|

APP双侧≤1°由4条降为0条；至少一侧≤1°由15条变为13条。原双侧近零demo8/44/45/49均恢复>5°的实际APP释放。单侧归零仍普遍存在，因为边界还受速度目标、释放方向及有限候选选择影响。

## 速度收益

指标是平均关节运动速度，越低表示在相同动作时序下关节运动量越小；这里不改变轨迹时长，不将降速百分比解释成任务执行速度提升。各demo等权，主要百分比由两组均值之比计算。

|样本|原平均rad/s|取消净空后rad/s|相对原方案再降低|
|---|---:|---:|---:|
|全部47条|0.701074|0.679236|3.11%|
|原APP至少一侧近零的15条|0.757181|0.714168|5.68%|
|原APP双侧近零的4条|0.870980|0.726154|16.63%|
|换了候选的28条|0.727685|0.691027|5.04%|

相同47条零容差基线均值0.863956rad/s：原方案降低18.85%，去净空重排降低21.38%，增加2.53个百分点；相对原方案的额外降幅为3.11%，两种分母不能混用。

逐demo额外降幅中位数0.327%、均值2.697%；至少降低1%的15条，至少5%的7条，至少10%的5条。主要收益集中于少数受净空约束的样本。

## 收益最大的5条

|demo|原APP边界−/+°|去净空APP边界−/+°|原→新实际APP峰值−/+°|原→新平均速度rad/s|再降低|
|---|---:|---:|---:|---:|---:|
|demo_44|0.000 / 0.000|30.000 / 30.000|0.000 / 0.000 → 4.894 / 8.360|0.906283 → 0.718901|20.68%|
|demo_45|0.000 / 0.000|30.000 / 0.000|0.000 / 0.000 → 7.552 / 0.000|0.853932 → 0.717485|15.98%|
|demo_15|30.000 / 30.000|30.000 / 30.000|0.803 / 10.017 → 0.803 / 10.017|0.749834 → 0.630107|15.97%|
|demo_8|0.047 / 0.129|30.000 / 0.000|0.047 / 0.129 → 9.083 / 0.000|0.969682 → 0.817423|15.70%|
|demo_49|0.000 / 0.000|30.000 / 30.000|0.000 / 0.000 → 0.000 / 14.554|0.754024 → 0.650808|13.69%|

demo15的APP边界与实际APP峰值不变，主要变化在EXEC；不能将全部关节速度收益归因于APP Ry。demo44/45/49从全零参考模式转入非零继承参考模式，沿用原诊断中的协议差异，不把该收益冒充纯角度上限效应。

## 适用范围与来源

这是原净空感知BO已采样候选内的回顾性重排。若从头删除净空再搜索，后续查询点会改变，本报告没有测量该结果；本报告也没有新物理任务成功率。标注边界未写回正式数据集。

- 冻结候选根、请求、source_snapshot、日志和独立审计：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107`，每demo为`libero_spatial/task_00/demo_N/subtask_00/annotation.json`及`evaluations/<hash>/trial.json`。原始逐回合轨迹/视频、零参考与来源目录详见同目录README，未复制或改写。

- 原标签SHA绑定plan：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1/plan.json`。原始评价映射：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107/source_snapshot/benchmark_joint_pre_post_t1_60.py`的result_outcome，完整非桌面运动学与净空验收在该函数中分列。

- 只读入口：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_spatial00_label_distribution/rerank_without_clearance.py`；汇总：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_spatial00_label_distribution/without_clearance_summary.json`；逐demo/候选路径/SHA与实际峰值：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_spatial00_label_distribution/without_clearance_pairs.json`；逐项CSV：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_spatial00_label_distribution/without_clearance_pairs.csv`；原文件SHA清单：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_spatial00_label_distribution/without_clearance_sources.json`；统计复算：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_spatial00_label_distribution/without_clearance_verification.json`。
