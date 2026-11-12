# 同任务5条demo：搜索容差与实际释放的差异

状态：2026-11-06只读统计完成，来源哈希与统计复算通过。未启动新搜索、物理回放或策略测评，未改写原标签。分析对象是每任务按原demo编号选取的前5条成功重构轨迹，不是rollout seed0–4。

## 比较定义与范围

- 搜索结果：每个已选最佳候选的分轴、正负方向容差上限。实际释放：该候选原运动学回放记录的对应方向释放峰值，不等于上限，也不等于逐帧差异。
- 按同一语义子任务、APP/EXEC、轴及正负方向对齐。重复观察周期映射到同一语义子任务后，在每demo内取最大包络；这是统计汇总，不改写各周期标签。仅比较所有可用demo共同出现、语义表允许释放的方向，不将缺失阶段填零。原始周期与被排除的非共同方向保存在per_task_details.json。
- 每方向计算demo间最大值减最小值（极差），以及所有demo对的绝对差均值。先在任务内平均，再在任务间等权平均，单位均为度。禁用轴不参与可释放方向平均。
- 200条选样中199条有标签；Spatial04/demo_1基线不可用，故该任务仅4条。Long08/demo_4来自新零参考无偏置补标，其余198条保留原协议，Long08属于混合基线协议。
- Goal00与Goal05所有轴禁用容差，保留在40任务明细中，但没有可释放方向可比，不将其固定零上限用于稀释均值。主要方向统计覆盖38任务、189条标签、292个共同方向；全来源仍为40任务199条。

## 结果

|套件|参与方向均值的任务数|搜索上限平均极差|实际释放峰值平均极差|
|---|---:|---:|---:|
|Spatial|10|11.00°|6.70°|
|Object|10|13.93°|6.92°|
|Goal|8|14.32°|7.08°|
|Long|10|14.17°|10.80°|
|全部可释放任务|38|13.30°|7.92°|

平均两条demo之间，搜索上限相差6.63°、实际释放峰值相差3.86°。辅助指标：阶段整体SO(3)释放角峰值的平均极差10.43°，与分轴坐标峰值定义不同。

剔除缺一条标签的Spatial04和混合协议Long08，剩余38个完整同协议任务，其中36个允许释放、180条demo；搜索上限平均极差13.54°、实际峰值平均极差7.96°，平均两两差6.76°和3.88°。结论不依赖这两项特殊样本。

实际释放差异最大的任务是Long09（任务内方向平均极差20.23°）、Long02（18.37°）、Goal07（16.29°）。全部40任务见per_task_variation.md；每个方向的5条具体数值见per_task_details.json。

示例Spatial00：APP正Ry的搜索上限依demo_0–4为30.00/22.96/30.00/14.06/30.00°，实际峰值为11.47/10.66/12.55/6.65/14.75°，极差分别15.94°与8.10°。同任务EXEC负Ry的搜索上限15.00/6.41/15.00/0.00/15.00°，实际峰值均为0°。因此某些方向的边界变化很大，但并未被运动实际使用。292个共同方向中，41个同时满足搜索极差>5°而实际极差<1°。

## 解释与适用范围

当前差异不能仅归因于场景变化：demo间还存在不同的名义轨迹、抓取/阶段历史；搜索使用demo相关的固定随机种子，每块仅16次贝叶斯追加查询，包含有限预算和采样差异。比较的是搜索选中的候选边界，不是已证明的最大可行边界；结果说明同任务标签与实际运动并不高度一致，不能据此把一个demo标签直接复制到其余demo，也不能据此量化纯场景敏感性。本分析没有控制相同搜索初始设计和随机种子重新优化。

## 本地源数据地址与验证

- 分析脚本：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-06_libero_5demo_tolerance_variability/analyze.py`。
- 总统计：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-06_libero_5demo_tolerance_variability/summary.json`。
- 40任务表：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-06_libero_5demo_tolerance_variability/per_task_variation.md`。
- 逐方向、逐demo与原始周期、证据地址：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-06_libero_5demo_tolerance_variability/per_task_details.json`。
- 统计复算：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-06_libero_5demo_tolerance_variability/verification.json`。核对199个标签文件、全部选中最佳trial的SHA，292个共同方向极差及两两差复算一致。该验证是统计验证；候选运动学和净空完整验收沿用下述正式独立审计。
- 合并来源清单：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_zero_nominal_continuation_20261106/combined_200_source_manifest.json`，SHA256为`8f22961d8f0433a9baeb95c06c19195631863f2e3364bc05a416bde2b02d9aa2`。
- 原198条标签、配置、逐帧来源、候选轨迹及日志：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_dsp16_first5pertask_20261106`；该目录的`annotation_request.json`、`selection.json`、`annotation_independent_final_audit.json`、`pipeline.log`、`annotation.log`、`independent_audit.log`。每条结果位于`<suite>/task_XX/demo_N/`，最佳候选trial地址与SHA在per_task_details.json逐一列出。
- 新Long标签与缺基线Spatial状态：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_zero_nominal_continuation_20261106`；该目录的`annotation_request.json`、`selection.json`、`annotation_independent_final_audit.json`、`annotation.log`、`independent_audit.log`。
- 原重构视频/物理轨迹根目录：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261025_success10_v1/episodes`及`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261101_full_v1/episodes`；本次未复制或修改，精确来源由各prepared_input.json的provenance追溯。离线候选无新增物理视频。

本次不涉及远端数据或新进程；所有结果均在上述本地路径。
