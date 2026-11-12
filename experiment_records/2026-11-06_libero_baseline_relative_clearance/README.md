# LIBERO逐帧相对零容差基线净空标注

状态：实现、小样本与独立审计通过。未重启200条批次，旧结果保留。

用户截图规则：req_t=min(5mm,c0_t)，g=min_t(c_t-req_t)，接受R=1且g>=-epsilon。epsilon默认1微米，可配置；原始绝对净空继续保存。净空GP学习g+epsilon、阈值0，其余目标/分类器/预算不变。

44项回归通过。Spatial00/demo4真实104帧，12初始化+1贝叶斯查询，13条观测独立审计通过；零点速度0.74229，最佳0.61707rad/s，最差逐帧g=+0.11942mm。此候选绝对最小净空仍为-3.59644mm，属于相对不恶化协议，不代表绝对碰撞安全。多子任务沿用同一录制初始化分块协议，基线覆盖全部帧一次，候选与基线严格对齐。

- MuJoCo主总结：[README.md](/media/czl/sata/franka_my_code/franka_mujoco/experiment_records/2026-11-06_libero_baseline_relative_clearance/README.md)
- 原始小样本run：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_pilot_20261106`
- 请求：run下`annotation_request.json`；独立审计：`annotation_independent_final_audit.json`；日志：`annotation.log`。
- 单回合输入/基线/候选/标签：run下`libero_spatial/task_00/demo_4/`。
- 原数据：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261101_full_v1/episodes/libero_spatial/00/demo_4`。
- 已测试源码及验证清单：MuJoCo主总结同目录`tested_source/`、`verification.json`。

默认应用于MuJoCo的LIBERO适配器；其他现有标注实验的绝对阈值保留。零基线不能完整求解时单列baseline_unavailable；旧50条Object桌面元数据适配错误仍需单独处理。


## Spatial00/demo4完整DSP16重测：已完成并独立审计通过

新独立目录：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_spatial00_demo4_dsp16_20261106`。

104帧，12初始化+16贝叶斯追加，共28候选；28条均完整运动学可解，8条通过逐帧相对净空。零点平均关节速度0.7422926673rad/s，最佳0.6063278405rad/s，降低18.3169%；最佳g=+0.0102685035mm，瓶颈帧87，绝对最小净空-3.765715593mm。按相对规则合格，不代表绝对碰撞安全。前13候选与先前pilot数值完全一致。原始数据与旧结果未改，零物理步；本次不是策略推理/任务成功率benchmark。

完整源数据与证据：上述目录`annotation_request.json`、`annotation_independent_final_audit.json`、`retest_summary.json`、`task_completion.json`、`annotation.log`；连续六向标签位于`libero_spatial/task_00/demo_4/tolerance_labels.json`。MuJoCo主实验记录已同步。


## PRE Ry释放与净空（最优候选只读检查）

PRE的Ry配置上限为正负30度，实际接受峰值约+14.7506度/负0.0570度。源帧43净空从31.5484降至27.9923mm，确有3.5561mm降低，但该帧要求只有5mm；PRE最紧帧49从7.2006升至7.8030mm，对5mm要求余量2.8030mm。全回合瓶颈为EXEC帧87，余量0.0102685mm。规则min(5mm,c0_t)允许原本高于5mm的帧降低至5mm，因此不能称每帧都不低于原名义轨迹。详细证据同目录pre_ry_clearance_diagnosis.json。未新增搜索/仿真或修改标注。
