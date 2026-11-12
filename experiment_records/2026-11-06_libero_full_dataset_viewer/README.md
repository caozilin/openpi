# 完整重构数据集可视化目录修复

状态：已完成并验证。8768已提供完整1875条目录。

旧页面仍指向20261025的400条、每任务10条目录，因而缺少后来重构的1475条。新目录与审计后的LeRobot来源索引严格匹配：1875条重构成功demo、40任务，保留原始demo编号。原2000个来源中的125条重构未成功，不属于本训练集；本次离线标注只选其中200条，标注结果不决定demo是否显示。

默认全部demo，新增本次200条/未找到合格标签/适配错误筛选，保留任务与demo双列、历史容差回放和释放图。原400视频缓存复用；其余直接导出原HDF5保存图像，重构轨迹与视频引用已完成记录，不执行仿真。旧目录保留备份，原8768服务入口继续使用。

本地源数据地址：

- 完整来源：`/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_full_20261101/meta/reconstruction_sources.json`
- 新viewer：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261101_full_v1/viewer`
- 目录与媒体核验：上述viewer下`full_catalog_verification.json`，浏览器验证后追加`qa/full_dataset_browser_checks.json`。
- 构建日志：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261101_full_v1/viewer_build.log`
- 离线标注状态来源：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_reconstructed_authoredcaps_dsp16_first5pertask_20261106/selection.json`及各demo状态文件；viewer下`annotation_status.json`为只读快照。

验证结果：1875条唯一demo与审计来源索引完全一致，全部媒体/轨迹文件存在。四套件各抽取一条旧目录缺失demo，真实HTTP轨迹与帧数匹配，8个原始/重构视频Range请求均206通过。浏览器实测Spatial00完整47条、Object00完整50条，新增Spatial00/demo1原始与重构视频readyState4无解码错误；200样本、无合格标签、适配错误筛选均保留原demo编号，控制台无错误。5项状态保留与前台CLI退出回归通过；Python编译和JavaScript语法通过。构建总控已正常退出，原始记录未改写，旧目录备份地址见full_catalog_verification.json。
