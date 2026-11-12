# 优先检查的全开失败与运动学损失样本

状态：2026-11-09，对已完成独立复核的全40任务对照做重点清单整理；无新增搜索或执行。整体失败率38/1875=2.03%，但L08任务失败24/45=53.33%，不能用整体低失败率替代该任务结论。以下编号为原始demo_N，不是闭环评测seed。两组均不筛桌面净空，完整求解及自碰撞条件保留。

## 全部38条失败，按任务集中程度排列

|任务|demo编号|失败类别|
|---|---|---|
|libero_10/08|0, 2, 4, 7, 8, 10, 11, 16, 17, 18, 20, 22, 24, 25, 27, 28, 29, 34, 36, 37, 39, 40, 43, 44|optimizer_infeasible|
|libero_goal/02|8, 21, 26, 36, 47|optimizer_infeasible|
|libero_spatial/01|3, 26|optimizer_infeasible|
|libero_spatial/07|13, 48|optimizer_infeasible|
|libero_10/07|4, 15|optimizer_infeasible|
|libero_10/05|15, 43|robot_self_collision|
|libero_10/06|3|optimizer_infeasible|

Long05/demo15在原块内frame229发生link1/link5碰撞，几何穿透0.572mm；demo43在frame239发生link2/link5碰撞，穿透5.473mm。该位置是原离线候选20Hz帧，非新物理回放证据。其余36条均达到IPOPT20次迭代上限；优先抽查L08/demo0（EXEC，frame249）、demo2（EXEC，frame157）、demo4（DIS，frame192），完整逐条阶段/帧/残差见full_failure_diagnosis.json。

## 速度代价最大的样本

筛选搜索相对全开降速>10%的全部14条。降幅=(全开−搜索)/全开，不能把该降幅直接称作以搜索为分母的“全开增加百分比”。正值为改善；速度/加速度越低越好。

|任务/demo|全开→搜索平均速度rad/s|搜索降速|搜索降低加速度|归一化σmin：全开→搜索|
|---|---:|---:|---:|---:|
|libero_10/04/demo_14|0.548631 → 0.411686|24.96%|22.01%|0.081472 → 0.633090|
|libero_10/05/demo_23|0.589863 → 0.478249|18.92%|55.89%|1.174152 → 1.360185|
|libero_goal/02/demo_49|0.768040 → 0.623806|18.78%|14.18%|0.192104 → 0.791378|
|libero_10/07/demo_10|0.564654 → 0.466769|17.34%|18.76%|0.333463 → 0.904375|
|libero_10/04/demo_28|0.536708 → 0.452283|15.73%|14.94%|0.472427 → 1.388026|
|libero_goal/02/demo_7|0.689343 → 0.587332|14.80%|24.13%|0.196498 → 0.617149|
|libero_spatial/07/demo_28|0.653871 → 0.563015|13.90%|31.68%|0.115446 → 0.808366|
|libero_goal/02/demo_22|0.816096 → 0.705036|13.61%|10.85%|0.321457 → 0.735510|
|libero_goal/02/demo_13|0.882881 → 0.772935|12.45%|18.35%|0.067042 → 0.350118|
|libero_10/07/demo_45|0.506601 → 0.444556|12.25%|9.99%|0.172097 → 0.870750|
|libero_spatial/07/demo_30|0.613531 → 0.538904|12.16%|22.01%|0.193945 → 0.613042|
|libero_goal/02/demo_27|0.771347 → 0.686739|10.97%|9.70%|0.387978 → 0.677048|
|libero_10/04/demo_12|0.431396 → 0.386404|10.43%|12.85%|0.373954 → 1.260942|
|libero_10/04/demo_9|0.442154 → 0.397385|10.13%|5.64%|0.547894 → 1.465790|

重点首看L04/demo14（双杯左右分放，速度改善24.96%，加速度改善22.01%）和L05/demo23（书入收纳架，速度改善18.92%，加速度改善55.89%）。

## 速度收益未覆盖的其他明显代价

|指标与样本|全开→搜索|同时的速度降幅|
|---|---:|---:|
|归一化σmin，libero_10/08/demo_30|0.007957 → 0.498871|6.65%|
|归一化σmin，libero_spatial/07/demo_34|0.055175 → 0.534878|9.81%|
|归一化σmin，libero_10/04/demo_14|0.081472 → 0.633090|24.96%|
|mY，libero_object/00/demo_47|0.050321 → 0.075474（+49.98%）|0.26%|
|mY，libero_10/07/demo_36|0.050513 → 0.074748（+47.98%）|6.84%|
|mY，libero_10/00/demo_45|0.051160 → 0.073970（+44.59%）|7.01%|

σ比值存在小分母放大，故这里同时列绝对值。L08/demo30的原始未归一化最小σ也从0.000769提高到0.052323；不是仅由归一化表达导致的巨大相对变化。O00/demo47速度只改善0.26%，mY却从0.050321到0.075474，说明速度收益小不能替代所有指标代价小。

## 精确来源与验证

- 主记录：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_alltasks_noclearance_full_vs_best/README.md`；40任务表：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_alltasks_noclearance_full_vs_best/PER_TASK.md`。
- 本清单及每条full/best原trial绝对路径/SHA：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_alltasks_noclearance_full_vs_best/focus_cases.json`；入口：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_alltasks_noclearance_full_vs_best/focus_cases.py`。
- 原始候选、请求、日志、视频来源、两个数据协议分层和完整复核均沿用主记录；全38失败证据：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_alltasks_noclearance_full_vs_best/full_failure_diagnosis.json`。输入摘要SHA在focus_cases.json绑定；原验证10416份来源SHA、4358份trial、2301块及40任务统计已通过。没有将截断失败轨迹纳入运动学损失比较，也没有改写正式标签。

