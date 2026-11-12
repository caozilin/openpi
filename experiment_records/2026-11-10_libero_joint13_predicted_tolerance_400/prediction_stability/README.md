# 模型容差预测的动作块稳定性

只读快照：2026-11-10T17:33:46.216066+08:00。纳入已提交的238回合（成功与失败均含），43358帧；完整400测评及最终回合审计仍未结束。未执行额外推理、优化、物理回放或修改原始结果。

## 口径

本次模型horizon=10、replan=5；记录仅保存实际执行的前5帧，未保存每次完整10帧，更没有16帧。不能从这些记录判断未执行后半块。统计六方向有向容差，单位度。raw为原始模型预测；clipped为裁剪0–30°后、EMA之前的值，均不是最终恢复硬盒。

块内：完整5帧且phase/subtask及连续段一致；按方向计算总体标准差(ddof=0)和max-min极差。块间：先计算每块5帧均值，再比较同任务、同seed、同连续phase/subtask内相邻块的均值绝对差；跳过跨阶段块，不跨阶段回访或子任务连接。另统计每连续阶段内各块均值的极差及标准差。数值汇总采用块×6方向等权，连续阶段极差采用段×6方向等权；长段对相邻块统计权重更多。

完整同段块7583；跨阶段块排除1002，末尾不足5帧块排除168。主要看允许释放的APP/EXEC，ENG/DIS仍在JSON单列，但原协议强制不释放。

## APP/EXEC汇总

| 指标 | 原始预测：均值 / 中位数 / P95 | 裁剪预测：均值 / 中位数 / P95 |
|---|---:|---:|
| 块内标准差 | 0.167° / 0.069° / 0.217° | 0.143° / 0.050° / 0.205° |
| 块内极差 | 0.424° / 0.191° / 0.603° | 0.357° / 0.135° / 0.568° |
| 相邻块均值绝对差 | 2.286° / 0.467° / 11.652° | 2.134° / 0.286° / 11.543° |
| 连续阶段各块均值极差 | 7.413° / 2.808° / 28.786° | 6.933° / 2.262° / 28.259° |

APP/EXEC含6378块、5695相邻块对、627个至少含2完整块的连续阶段。93.01%块的六方向各自极差均≤1°，96.68%均≤2°；相邻块方向均值差>5°占12.53%。

| 方向 | 块内平均极差 | 相邻块均值差：均值 | 相邻块均值差：P95 | 连续阶段块均值平均极差 |
|---|---:|---:|---:|---:|
| rx- | 0.365° | 0.849° | 3.753° | 3.114° |
| rx+ | 0.308° | 1.326° | 7.293° | 4.179° |
| ry- | 0.488° | 3.820° | 16.739° | 12.271° |
| ry+ | 0.576° | 4.145° | 21.471° | 12.625° |
| rz- | 0.366° | 2.035° | 9.832° | 6.915° |
| rz+ | 0.441° | 1.539° | 7.088° | 5.373° |

## 解释与边界

已执行前5帧通常近似同一个容差值，但不同推理块并非一个固定阶段常量。块间变化明显大于块内，ry±最突出；块内也存在少量约0↔30°的真实跳变，不能宣称全部常数。原始预测在0–30°之外的帧×方向比例为35.60%；裁剪会使边界更显稳定，原始/裁剪结果因此分开。这里只量化数值一致性，不据此判定波动是预测噪声、必要闭环修正、或导致失败，也不评价EMA后有效硬盒。

## 本地源数据与复核

- 原始run：`/media/czl/sata/franka_my_code/openpi/data/libero_evaluation/pi05_joint13_successmix20_10000_predtol_hold20_40tasks_10seeds_20261110`；其episodes/<suite>/task_XX/seed_N/{result.json,trajectory.json}。
- 精确快照来源及SHA：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_predicted_tolerance_400/prediction_stability/sources.json`。
- 汇总：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_predicted_tolerance_400/prediction_stability/summary.json`。
- 逐块：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_predicted_tolerance_400/prediction_stability/blocks.csv`；相邻块：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_predicted_tolerance_400/prediction_stability/adjacent_block_pairs.csv`；连续阶段：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_predicted_tolerance_400/prediction_stability/continuous_segments.csv`。
- 最大变化及逐帧核对：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_predicted_tolerance_400/prediction_stability/largest_within_block_changes.json`、`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_predicted_tolerance_400/prediction_stability/large_change_examples.json`。
- 独立CSV算术/输入SHA复核：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_predicted_tolerance_400/prediction_stability/analysis_verification.json`。
- 分析脚本：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_predicted_tolerance_400/analyze_prediction_stability.py`。

所有原始源文件完成后再次逐项SHA核验不变；CSV用独立math.fsum复算各均值与JSON一致。无成败筛选或额外rollout。
