# 优化器近期修改识别：EMA动作采样与损失坐标系对齐

状态：2026-11-06只读代码比较完成。未修改生产代码、原运行请求、标签或缓存，未启动新测试、搜索、训练或物理回放。本记录用于回答用户“修改一小部分优化器能否识别”。

比较基准为最后一次零容差无偏置补标保存的source_after，而非累计很多历史改动的Git HEAD。快照覆盖的Python源码中6个文件变化、182个未变化。IPOPT控制器、目标函数、运动学核心、RotationReleaseState与该快照逐字节相同。

主要变化：共享advance_reference_pose原先先在旧tolerance.frame采样动作EMA，再推进名义姿态；现在先推进名义姿态，再根据当前模式冻结容差系，最后执行同一个update_intent。投影仍为frame.T @ rotation_action，EMA幅值/时间常数/死区/归一化及释放损失公式不变，但实时坐标系下，动作EMA对应本帧损失与硬约束使用的轴。

相关入口同步：benchmark/environment传递frame mode；UI固定阶段描述符更新提前到动作采样前，并将模式传入共享入口；多步预测每节点沿用相同顺序；LIBERO先推进姿态并处理原朝向退化保护，再冻结坐标系并采样动作。counterfactual缓存版本84升级85，避免复用旧目标权重语义的轨迹。没有更换IPOPT、修改硬约束形式、carry公式或数值验收阈值。

影响范围：实时容差系发生旋转时，各释放轴的EMA惩罚权重可能变化，因此有容差的最优命令、搜索采样评价及最终标签都可能改变；固定不变的坐标系下，这个顺序修改本身不产生轴投影差异。旧199条标签和四项指标收益统计仍描述原冻结实现，不应直接写成新版实现的结果。此次未重标，也未授权额外运行。此前0容差不继承阶段偏置的修复已经包含在比较基准，不能误认为是本次新改动。

已有验证（读取用户工作区现存证据，并未在本轮重跑）：post_fix报告记载核心122项通过，扩展152项通过/3跳过/1项历史CSV缺失失败；四组两任务seed0物理小样本共712帧，EMA同轴对照误差为0；样本不代表LIBERO总体收益。已将该审计manifest中9个源码/锁文件SHA与当前工作区重新核对，全部匹配。

## 本地源数据地址

- 新代码与基准SHA、比较范围及现存审计匹配：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-06_optimizer_ema_frame_change/comparison.json`。
- 6个文件完整差异：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-06_optimizer_ema_frame_change/changes.diff`。
- 基准快照：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_zero_nominal_continuation_20261106/source_after`。
- 当前共享入口：`/media/czl/sata/franka_my_code/franka_mujoco/tasktol/tolerance/runtime.py`。
- 当前意图滤波：`/media/czl/sata/franka_my_code/franka_mujoco/tasktol/tolerance/state.py`。
- 当前原生LIBERO入口：`/media/czl/sata/franka_my_code/franka_mujoco/tasktol/simulation/libero.py`。
- 当前缓存版本：`/media/czl/sata/franka_my_code/franka_mujoco/tasktol/counterfactual/runner.py`。
- 用户已有修复后验证：`/media/czl/sata/franka_my_code/franka_mujoco/experiment_records/2026-11-06_ema_release_consistency/post_fix/report_zh.md`，同目录`manifest.json`、`summary.json`、`pytest.log`、`rollout.log`、`synthetic_frame_samples.json`及`raw/<task>/ema_<weight>/`逐帧轨迹和frame_audit.json。该记录未录制视频。
- 修复前证据：`/media/czl/sata/franka_my_code/franka_mujoco/experiment_records/2026-11-06_ema_release_consistency/report_zh.md`及同目录raw/、manifest.json、summary.json和日志。
- 历史LIBERO199条四维统计：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-06_libero_5demo_kinematic_gain/README.md`。

本次无新评测样本、模型或远端数据；未生成新的候选或视频。
