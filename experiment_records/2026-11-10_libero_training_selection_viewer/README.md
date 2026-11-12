# 数据集viewer：训练入选容差301条筛选

筛选与2×2布局均已实现并离线验证通过。保留训练数据、抽样、标签及全部源文件；仅新增viewer腕部派生MP4和展示代码。未启动任何HTTP服务或浏览器测试。

使用`python3 /media/czl/sata/franka_my_code/openpi/examples/libero/serve_mask_annotation_viewer.py --port 8768`前台启动，Ctrl+C退出。刷新页面，Demo范围选择“训练入选容差（301条）”。或打开`http://127.0.0.1:8768/index.html?dataset=libero_spatial%2F00&episode=demo_2&training=1`直接进入筛选模式。保留任务/demo双列；视频区域按下述2×2升级；当前demo不入选时自动跳到该任务首个入选demo，切换任务继续保留筛选。每条入选demo标明“训练入选容差”。不会把“有标签”或“物理成功”当成“已用于训练”。

筛选唯一依据：冻结replacement_request.json.entries中suite/task_id/demo精确成员关系，核对request.tasks.selected_demos及已发布viewer同源replay文件夹；禁止重新随机抽样。共301，Spatial75/Object85/Goal84/Long57；入选episode_index集合与真实conversion_manifest中trajectory_is_tolerance=1的集合完全一致。其余1574条名义观测在该筛选下隐藏，全部模式仍保留1875列表。

## 本地源数据地址

- 冻结训练请求：`/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109/replacement_request.json`，SHA 92e86f597f84b801299ba80482adf506ea1a989e8b94f7a4cec53619797d8337。
- 实际转换：`/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109/conversion_manifest.json`。
- 训练抽样清单：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/selection.json`。
- viewer：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261101_full_v1/viewer`；入选索引`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261101_full_v1/viewer/training_selection.json`；媒体保持原处。
- 规则和UI源码：`/media/czl/sata/franka_my_code/openpi/examples/libero/mask_viewer/training_selection.js`、`/media/czl/sata/franka_my_code/openpi/examples/libero/mask_viewer/full_dataset.js`；部署镜像在viewer/assets同名文件。
- 小型索引导出入口：`/media/czl/sata/franka_my_code/openpi/examples/libero/export_training_viewer_selection.py`。重建viewer后可执行该脚本，只重建training_selection.json，不复制媒体。
- 测试：`/media/czl/sata/franka_my_code/openpi/examples/libero/training_viewer_selection_test.py`、`/media/czl/sata/franka_my_code/openpi/examples/libero/mask_viewer/training_selection.test.cjs`；6项Python通过、Node301成员/跨任务demo隔离通过、JS语法通过。
- 验证凭据：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_training_selection_viewer/verification.json`。未做浏览器渲染检查，按用户要求不启动服务；现有用户服务无干预。


## 2×2主/腕部视角升级（2026-11-10，已完成）

按用户要求移除页面中的“原始HDF5视频”及其勾选、打开链接，保留源HDF5和已有原始数据。主区域左列无容差重构、右列标注容差；上排主视角、下排腕部视角。四路共用无容差主视频时间轴，支持暂停、逐帧、倍速；容差提前停止后双视角保持最后画面。无容差回放的腕部像素来自其本身recording.hdf5保存的pre-action图像，256×256、20Hz、旋转180°一次后普通H.264编码；不是原始示范HDF5，也不重新运行仿真。容差腕部MP4直接硬链接既有同源文件，不重复编码。

- 媒体导出入口：`/media/czl/sata/franka_my_code/openpi/examples/libero/prepare_rollout_wrist_videos.py`；6并行，只准备viewer派生媒体。
- 布局安装入口：`/media/czl/sata/franka_my_code/openpi/examples/libero/install_rollout_quad_viewer.py`；构建器`/media/czl/sata/franka_my_code/openpi/examples/libero/build_reconstructed_rollout_viewer.py`同步接入。
- 名义腕部源：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261101_full_v1/episodes/<suite>/<task_id>/demo_N/recording.hdf5`，数据集`data/demo_N/obs/eye_in_hand_rgb`。
- 容差腕部源：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1/episodes/<suite>/<task_id>/demo_N/wrist_rgb.mp4`。
- 派生媒体与逐项像素/MP4 SHA、帧数验证：viewer中的`rollouts/<suite>/<task_id>/demo_N/wrist_rgb.{mp4,json}`及`wrist_export_verification.json`（1875/1875完成，其中1851条有容差腕部，24条无标签保留左列），进度`wrist_export_state.json`。
- 同步逻辑：`/media/czl/sata/franka_my_code/openpi/examples/libero/mask_viewer/video_sync.js`及`tolerance_comparison.js`；同步/缺失/结束保持的真实控制器VM检查`video_sync.test.cjs`通过。
- HTML布局幂等性、真实3帧编码/旋转/源SHA保护测试与原筛选回归共8项通过：`/media/czl/sata/franka_my_code/openpi/examples/libero/rollout_quad_viewer_test.py`等。未启动HTTP服务或浏览器。


最终发布核对：1875条名义腕部MP4逐项SHA与帧数记录一致；1851条容差腕部与原回放为同inode硬链接；4个video按“名义主/容差主/名义腕/容差腕”排列，原始HDF5视频节点、勾选与打开链接均移除；301条冻结成员与真实训练请求保持一致。全部发布脚本路径及语法通过。8项Python回归、同步控制器VM、301条真实成员检查及既有释放显示回归均通过。导出进程exit0，用时845.73秒，未启动服务、未重跑仿真、未覆盖或删除源记录。

- 最终凭据：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_training_selection_viewer/quad_verification.json`。
- 只读复核入口：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_training_selection_viewer/verify_quad_layout.py`。
- 导出完成日志：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_training_selection_viewer/wrist_export.log`。

刷新已有页面即可使用；本次仅做离线代码、媒体、结构与同步控制检查，未做浏览器渲染检查。


## 四格高度收紧（2026-11-10）

用户反馈2×2方形过高，现将视频区固定为两行等高、总高`min(56dvh,560px)`（旧浏览器回退56vh），每格取消1:1容器限制。摄像头方形画面仍用contain等比完整显示，横向多余空间留黑，不拉伸或裁切。视频面板已移到内容区第一项，mask说明置于其后；任务/demo双列、301筛选、同步和所有媒体不变。不重新编码、不启动服务。

源码：`/media/czl/sata/franka_my_code/openpi/examples/libero/mask_viewer/rollout_comparison.css`、`/media/czl/sata/franka_my_code/openpi/examples/libero/install_rollout_quad_viewer.py`；已同步发布viewer同名CSS和index.html。现有2项布局/导出回归通过；真实发布HTML确认视频区第一、安装幂等、CSS源与发布一致；仅离线验证，未做浏览器渲染检查。凭据：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_training_selection_viewer/compact_layout_verification.json`。


## 最新布局：1×4单排（2026-11-10）

按用户最新选择改为一行四列，顺序为无容差主视角、标注容差主视角、无容差腕部、标注容差腕部。网格优先按4:1比例随可用宽度自然定高，最大高度为min(48dvh,420px)，每路保持contain完整图像。视频仍位于内容区第一项，筛选/同步/媒体不变。源码及发布CSS、三份说明脚本均已同步；四个video顺序及发布副本一致性、JS语法检查通过，未启动服务或浏览器。此前2×2验证为布局历史，不代表当前排布。

当前凭据：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_training_selection_viewer/single_row_verification.json`；布局源码仍为`/media/czl/sata/franka_my_code/openpi/examples/libero/mask_viewer/rollout_comparison.css`，发布目录仍为`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261101_full_v1/viewer`。


## 单排铺满主区域宽度（2026-11-10）

用户要求进一步放大，取消此前48dvh/420px高度限制，保持四列均分、总宽100%、按4:1自然定高；视频面板抵消内容区两侧18px（窄窗口12px）留白，并移除两侧边框，铺满右侧workspace宽度。仍保持等比完整画面及任务/demo双列；仅更新展示CSS，未修改视频、同步或数据，未启动服务。源码与发布CSS字节一致性和四列/取消限高/边距规则已离线核对，未做浏览器渲染检查。

源码：`/media/czl/sata/franka_my_code/openpi/examples/libero/mask_viewer/rollout_comparison.css`；发布：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261101_full_v1/viewer/assets/rollout_comparison.css`；凭据：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_training_selection_viewer/full_width_verification.json`。此前限高记录为历史版本。


## Spatial/07为什么训练入选仅5条（2026-11-10）

任务pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate的名义重构44条；本批标注容差物理回放43条，逐份原report核对26成功、17失败，demo_49未生成该批回放。替换规则的分母是26条成功容差候选，不是全部44条：round(26×0.2)=5。因此训练入选筛选显示demo_2、demo_23、demo_36、demo_45、demo_47，episode_index325/344/356/363/364；与冻结请求及真实转换manifest的trajectory_is_tolerance=1完全一致。并非只有5条容差回放；选择“全部重构demo”可查看全44条中的43条容差回放，包括失败记录。

本次仅只读核查，没有重新抽样、重放或改写数据。小型核对结果、原始43份report绝对路径与SHA、请求SHA见`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_training_selection_viewer/spatial07_selection_diagnosis.json`。冻结来源`/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109/replacement_request.json`及`conversion_manifest.json`；物理记录仍在`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1/episodes/libero_spatial/07/`。
