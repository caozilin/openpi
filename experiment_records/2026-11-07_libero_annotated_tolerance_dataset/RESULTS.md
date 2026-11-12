# 最新标注容差：原始LIBERO物理回放数据集

状态：已生成1851条，全部通过独立记录/标签/状态/双MP4及原生谓词审计；旧容差回放已删除并完成viewer替换。

1851次固定来源回放，1503条完整执行且最终原生任务成功，348条失败（含7条求解拒绝）；24条无有效标注来源未回放。所有成败结果保留，没有按失败重试。成功索引仅收录满足完整执行+最终原生谓词的样本，不把1851条处理完成冒称全成功。

|套件|已回放|最终原生任务成功|失败|
|---|---:|---:|---:|
|Spatial|455|378|77|
|Object|493|412|81|
|Goal|448|418|30|
|Long|455|295|160|

沿用同原始重构初态/物理模型/7维物理动作，共享当前IPOPT20/EMA10与原生20Hz/500Hz关节伺服。逐源帧使用已冻结离线标签的阶段、子任务和六方向边界；不是在线语义识别。源离线候选按独立块初始化，本次为连续含任务物体的物理回放，离线100%验收不代表物理100%成功。首个求解拒绝停止协议沿用原数据回放，不套用benchmark的20失败帧缓冲；不因结果改控制器、预算或标签。

双相机256×256、20Hz、旋转180°一次，普通H.264 MP4；同时保留trajectory.json、states.npz、solver_observations.json及report.json。未输出图像HDF5、LeRobot或500Hz伺服采样，未重复复制reference HDF5。未来格式转换应从这些原始记录及其来源引用读取，不将有损MP4冒充无损PNG。

首版只读pilot审计访问了错误的环境包装层属性，原错误日志保留在run/attempts/0001_audit_wrapper_attribute；仅修复审计读取路径后重新检查已有六条，未重跑回合。pilot5成功/1物理失败/基础设施错误0。

旧400固定30°容差及52条恢复复测的原始回放文件已按用户明确授权删除，小型原summary/plan/规则/删除清单保留于本记录retired_legacy_metadata。当前及历史viewer里的旧容差/恢复媒体和嵌入轨迹已清除；名义重构/原始HDF5/离线搜索与标签/benchmark评测不删除。旧目录只留小型规则文件/退役凭据，避免规则导入工具依赖失效。

## 本地源数据地址

- 新原始数据集：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1`；逐条`episodes/<suite>/<task_id>/demo_N/{main_rgb.mp4,wrist_rgb.mp4,trajectory.json,states.npz,solver_observations.json,report.json}`。
- 固定请求及所有来源：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1/plan.json`；源码：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1/source_snapshot`和source_snapshot_manifest.json；全量结果：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1/summary.json`；成功数据索引：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1/successful_sources.json`。
- 全量独立审计：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1/independent_final_audit.json`、independent_final_audit_state.json、auditing.log；完成凭据：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1/task_completion.json`。
- 母集标签：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107`及annotation_independent_final_audit.json；各标签SHA在新plan.json绑定。
- 生成/审计入口：`/media/czl/sata/franka_my_code/openpi/experiments/libero_tasktol_compat/regenerate_annotated_tolerance.py`、audit_annotated_tolerance_dataset.py；它们调用既有共享执行链，没有复制控制或物理循环。
- 监督/发布：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_annotated_tolerance_dataset/finish_pipeline.py`；状态日志：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1/pipeline_state.json`、pipeline.log、generation.log及logs/逐条日志；真实进程：process.json、processes/、pipeline_process.json。
- 替换删除凭据：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_annotated_tolerance_dataset/legacy_deletion_verification.json`；旧小型元数据：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_annotated_tolerance_dataset/retired_legacy_metadata`；viewer：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261101_full_v1/viewer`，发布证据：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1/viewer_publication_verification.json`。
- 不使用服务器、不训练、不格式转换、不上传。用户viewer服务不终止。

## 最终交付复核（2026-11-07T21:14:53.122554+08:00）

全部1851条处理及审计完成，物理任务成功1503/1851=81.20%，失败348条（含7条求解拒绝，另341条完整执行后最终谓词不成立）；基础设施错误0，24无有效标签来源排除。各套件成功率Spatial83.08%、Object83.57%、Goal93.30%、Long64.84%。成败均保留，成功索引1503条已与原生审计逐项核对。

发布和旧数据清理后，再次核对14808份回合文件SHA、1851份标签SHA、3702个MP4与完整解码审计的绑定、7404个viewer媒体/记录链接；所有新回合文件均未被替换清理改写。回合文件逻辑体积3441504647字节（约3.21GiB，不含索引/日志/源码快照）；视频普通H.264，未转换LeRobot。原两旧数据根已无回放文件，仅保留小型规则和退役凭据；历史viewer清除2276个旧容差/恢复媒体目录和2122份嵌入轨迹。原始名义重构、HDF5及搜索标注证据保留。

生成69335、监督87835、审计259383及viewer构建344919均已真实退出；execution/pipeline/independent_final_audit三锁释放。当前未启动额外viewer服务，不干预用户原有服务。15分钟监督libero已在全部完成后删除。

终止记录中失败最多的任务如下，仅表示结果/终止类别，不据此臆断抓取或接触的具体原因：

|套件/task（0起始）|成功/总数|失败|其中求解拒绝|
|---|---:|---:|---:|
|libero_10 / 01|12/50|38|0|
|libero_10 / 07|26/50|24|0|
|libero_10 / 00|26/49|23|0|
|libero_object / 00|28/50|22|0|
|libero_10 / 06|27/45|18|0|
|libero_spatial / 07|26/43|17|0|
|libero_object / 04|33/48|15|0|
|libero_10 / 09|32/46|14|2|

全40任务统计：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_annotated_tolerance_dataset/per_task_summary.json`；7条求解拒绝源：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_annotated_tolerance_dataset/solver_rejected_sources.json`；替换后全量SHA与进程/锁复核：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_annotated_tolerance_dataset/final_delivery_verification.json`；复核脚本：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_annotated_tolerance_dataset/verify_delivery.py`。

完整40任务逐项成功率见[逐任务表](PER_TASK_SUCCESS_RATES.md)，任务编号0起始，分母按实际有效标签回放数。

## 与旧30°组的同源比较

共同397条：旧283/397（71.28%）→新325/397（81.86%），提升10.58个百分点；51条恢复、9条退化。协议同时有EMA/阶段时序/边界变化，属于整套方案比较。[完整配对报告](LEGACY30_COMPARISON.md)。
