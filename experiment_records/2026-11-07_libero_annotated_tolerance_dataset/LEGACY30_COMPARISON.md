# 旧30°与最新标注容差：同源配对比较

状态：既有结果只读配对统计完成，无新增回放。旧组是最初400条对应语义mask启用轴±30°回放，不是全任务统一111，也不混入52条失败定向姿态恢复复测。

整体旧286/400=71.50%，新1503/1851=81.20%，增加9.70个百分点；总体来源数量不同，因此主比较取共同397条。每条reference HDF5、原动作、名义report与物理model SHA全部一致。

|套件|共同demo数|旧30°成功|新标注成功|提升（百分点）|
|---|---:|---:|---:|---:|
|Spatial|99|65/99（65.66%）|90/99（90.91%）|+25.25|
|Object|100|82/100（82.00%）|82/100（82.00%）|+0.00|
|Goal|98|88/98（89.80%）|93/98（94.90%）|+5.10|
|Long|100|48/100（48.00%）|60/100（60.00%）|+12.00|
|合计|397|283/397（71.28%）|325/397（81.86%）|+10.58|

共同397条：274条两组都成功、63条两组都失败、51条旧失败→新成功、9条旧成功→新失败，净增42成功。Spatial贡献25、Goal5、Long12，Object净成功数不变。未配对的旧3条（libero_goal/02/demo_20, libero_goal/02/demo_45, libero_spatial/04/demo_1）在旧组均成功，当前无有效标签，双方配对统计均排除。

这说明提升不只是总体样本范围变化。它衡量的是新版整套方法；旧组使用旧冻结共享实现、在线手指阶段判定和固定语义身份计划、启用轴统一±30°，新组使用新版EMA同轴采样、逐源帧标注阶段/子任务和搜索六方向边界（手工5/15/30°上限）。不能把全部提升单独归因于贝叶斯搜索。未为此启动同实现新对照。

## 本地源数据

- 旧保留摘要/plan：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_annotated_tolerance_dataset/retired_legacy_metadata/20261031_mask30_confirmed_v1/summary.json`、`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_annotated_tolerance_dataset/retired_legacy_metadata/20261031_mask30_confirmed_v1/plan.json`；已与同目录deleted_file_manifest.json中删除前SHA核对。旧回放已按用户要求删除，本次不声称重新读取或解码旧视频。
- 新原始结果：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1`；summary.json、plan.json及independent_final_audit.json。
- 397条配对结果/原动作与参考SHA/套件统计/协议差异：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_annotated_tolerance_dataset/legacy30_vs_annotated_paired.json`；只读脚本：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_annotated_tolerance_dataset/compare_legacy30.py`。
