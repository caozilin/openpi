# 搜索选中候选与语义上限全开对比

状态：只读统计与源文件SHA、原始反馈/逐帧净空及独立重计核对完成。更新时间：2026-11-07T18:32:22.474064+08:00。

来自已完整审计的1875条cache85批次，无新增搜索/回放/训练。24条零基线不可用在两组均排除；分母为1851条（40任务），共2275个原独立标注块。这里成功指全部标注块完整运动学可解且满足每帧min(5mm,c0)−0.1mm净空要求，不是LIBERO物理任务成功率。

“上限全开”指手工表当前各轴5°/15°/30°上限的两个方向同时全开，APP/PRE和EXEC/POST均取各自上限，严格轴仍为0；不是所有任务一律111/30°。该点原本就在每块结构化初始设计中。本次按prepared_input重新构造最大点并逐一匹配原始候选，未通过名称或查询序号猜测。

## 验收通过率

|套件|有效demo|搜索选中通过|上限全开通过|双方通过时速度降低|
|---|---:|---:|---:|---:|
|Spatial|455|455/455（100%）|298/455（65.49%）|0.81%|
|Object|493|493/493（100%）|369/493（74.85%）|0.36%|
|Goal|448|448/448（100%）|298/448（66.52%）|0.93%|
|Long|455|455/455（100%）|273/455（60.00%）|1.36%|
|合计|1851|1851/1851（100%）|1238/1851（66.88%）|0.85%|

完整运动学通过（暂不计净空）方面，全开为1818/1851（98.22%），搜索选中为1851/1851（100%）；含净空的完整验收通过率由66.88%变为100%，增加33.12个百分点。

|套件|运动学未完整通过|完整但净空未通过|合计未通过|
|---|---:|---:|---:|
|Spatial|4|153|157|
|Object|0|124|124|
|Goal|4|146|150|
|Long|25|157|182|
|合计|33|580|613|

同demo多个原标注块全部通过才计整条通过；若存在运动学未通过块，该demo归到运动学失败，其余完整但净空不满足归到净空失败。本批33条没有与其他块净空失败重叠；失败判定不等同于严格证明不可达。

## 同一批样本的速度比较

双方均验收通过的1238条：上限全开平均关节速度0.517966370rad/s，搜索选中0.513589378rad/s，降低0.8450%。其中1134条更快、104条相同、0条更慢（差异阈值1e-10rad/s）。按每demo内各独立块N−1速度样本加权，然后demo等权；不生成跨块差分，不混入失败截断速度。

若只要求运动学完整、把580条净空违规全开也纳入，共1818条同源配对：全开0.519568247rad/s，搜索0.530598658rad/s，搜索平均反而提高2.1230%。这表明主要收益是净空约束下的可接受性，不能宣传为相对全开速度大幅下降。两种速度结果分母/验收条件不同，不能混用。

搜索候选来自相同初始设计+16追加的样本内选择，其中已包含全开点；选中候选按可接受性过滤后取最低观测速度。因此这里100%是该已标注集合的搜索验收结果，不是独立未见样本的成功率，也不证明全局最优。整体1851/1875的标签覆盖率98.72%与本节条件通过率分开。

83条严格零域任务的两种方案实际相同；剔除后1768条中全开1155/1768（65.33%），双方通过1155条的速度降低0.9191%。Long02/demo40历史GP数值恢复仍单列，保留所有旧候选，只补缺失3查询；剔除该条后的对照也保存在JSON，不改原始结果。

## 全开通过率最低的任务

|套件/task（0起始）|上限全开通过|全开未通过|
|---|---:|---:|
|Goal / 07|3/48（6.25%）|45|
|Object / 08|4/50（8.00%）|46|
|Long / 02|7/50（14.00%）|43|
|Object / 06|8/49（16.33%）|41|
|Long / 07|13/50（26.00%）|37|
|Long / 01|14/50（28.00%）|36|
|Spatial / 06|15/47（31.91%）|32|
|Object / 01|16/49（32.65%）|33|

这些是终止/净空记录的统计，不替代逐帧因果诊断。

## 本地源数据地址与校验

- 原始run：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107`；其`annotation_independent_final_audit.json`及`annotation_request.json`是本次母集证据。
- 每条原始`prepared_input.json`、`subtask_*/annotation.json`和选中/全开`evaluations/*/trial.json`均在上述run内；精确绝对路径及SHA在`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/best_vs_full_episodes.json`与`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/best_vs_full_sources.json`。旧请求及原始结果完全不改。
- 汇总和40任务统计：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/best_vs_full_summary.json`。
- 只读统计脚本：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/compare_best_vs_full.py`；独立重计脚本与凭据：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/verify_best_vs_full.py`、`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/best_vs_full_verification.json`。
- 原四维零基线对照：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance/kinematic_gain_pairs.json`，本次已逐demo交叉核对搜索速度一致。

原批次审计冻结源码仍在run/source_snapshot。本次存储默认已更新只影响未来请求，统计直接核对原文件，不重跑需要当前rootSHA的原批次审计。
