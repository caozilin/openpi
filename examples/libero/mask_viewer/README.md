# LIBERO · MuJoCo Dataset Studio

## 当前完整重构目录（2026-11-06）

当前默认展示审计通过的1875条重构成功demo，40任务，每任务显示实际成功数；原demo编号保留，不要求连续。125条重构未成功来源不属于这份训练数据集。目录严格匹配：
`/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_full_20261101/meta/reconstruction_sources.json`。

在仓库运行 `python3 examples/libero/serve_mask_annotation_viewer.py --port 8768`，前台持续运行，Ctrl+C退出。默认目录为`data/libero_tasktol_joint_servo/20261101_full_v1/viewer`。旧`--background`仅兼容，仍在前台。

任务/demo双列导航支持原编号搜索，以及“全部重构demo / 本次200条标注样本 / 未找到合格标签 / 适配错误”筛选。默认全部显示，未标注或无合格标签不会从数据集中消失。离线标注状态来自独立的`annotation_status.json`，与历史30°容差回放分开；历史视频缺失表示未做该条对照。

完整目录构建使用`build_reconstructed_rollout_viewer.py --source-index ...`，从已有重构记录和原HDF5观测导出，不执行物理重放。状态更新用`export_dataset_annotation_status.py --run <标注run> --viewer <viewer目录>`。

以下是旧400条语义标注页面的历史说明。

入口为同目录的 `index.html`（`report.html` 内容相同）。保留整个目录结构；页面、视频、轨迹和证据图片均在本地，无 CDN 或远程模型依赖。

## 打开页面

在导出目录运行以下命令，然后访问 <http://127.0.0.1:8768/index.html>：

```bash
python3 serve.py --port 8768
```

`--background`已停用，服务在当前终端持续运行，Ctrl+C结束。服务仅监听 `127.0.0.1`，支持 MP4 Range 请求和拖动定位。日志和 PID 文件属于可变运行记录，不纳入输出文件校验清单。

## 页面操作

- 左侧按 **4 任务集 → 每集 10 任务 → 每任务 10 MP4** 浏览，也可搜索任务指令。每个任务第一条保留原标注代表，其余 9 条从剩余成功示范中随机抽取，固定种子 `20261019`；原始 demo 编号保留。
- 顶部共享预测面板显示任务的有序子任务和 APP/EXEC mask。展开后可查看逐轴判定摘要、候选/框架/任务点因子、置信度、未知项、训练参考及实际查看过的主视角证据。这里记录可核查的依据与结论。
- 点击证据缩略图会切回原代表示范并定位对应帧。多子任务代表视频可随阶段自动切换子任务，也可手动选取。
- 视频支持播放、倍速、空格暂停及左右逐帧。点击阶段条或曲线定位；空间轨迹支持拖动旋转、滚轮缩放与视角重置。视频、当前帧、3D 轨迹和全部曲线共享时间游标。
- 轨迹包含世界系末端位置、旋转向量、7 关节角/速度、夹爪开口、源帧映射和原始 OSC_POSE 控制。可打开原始 MP4、轨迹 JSON，或导出当前轨迹 CSV。

## 与原 MuJoCo 可视化的关系

直接复用 `franka_mujoco/tasktol/data/viewer/static/` 的布局、CSS、canvas 曲线、空间投影/轨迹交互和播放联动。`assets/mujoco-studio.*` 为复用资产，`assets/adapter.*` 接入 LIBERO 数据和判定面板；上游文件指纹见 `viewer_manifest.json`。这是已记录视频与数值观测的查看器；没有启动新的 MuJoCo 物理回放。

必要适配包括：20 Hz、单 agentview、LIBERO 开口度 `q_left - q_right`、世界系 grip_site 位置、轴角旋转向量 rad、归一化控制值与观测位移的区分、pre-action 帧对齐，以及没有标称参考时的空值展示。

## 共享预测与证据范围

40 个原代表 MP4 和原 `mask_annotations.json` 保持不变。新增 360 条用于查看不同成功示范及轨迹；页面共享原任务预测，但这些补充示范没有逐条视觉复核、语义分段或逐帧 mask。新增示范只显示夹爪事件候选，逐帧语义标签为 null。共享显示不代表已证明所有抓取模式均满足同一预测。

数值轨迹仅用于本次新增的轨迹展示，不用于改写原视觉预测。`F_obs` 是基于观测姿态和已标定物理闭合轴的诊断代理，保持竖直 z 及论文退化保护；它不是动作积分标称参考。未提供标称目标或六方向数值边界时保留 null。关节速度由相邻保留观测按 20 Hz 差分，首帧未知；源帧空洞单独展示，不插入虚构停顿。

原实验的访问协议、已知偏差和不确定项分别保留在 `run_manifest.json`、`report.md`、`uncertainties.md`；逐任务记录在 `decision_records/`，关键帧在 `keyframes/` 和 `review/`。原证据快照可能与 v1 使用硬链接，修改时应先复制为独立文件，避免连带改变冻结结果。

## 构建和验证

在 openpi 仓库根目录运行：

```bash
.venv/bin/python examples/libero/build_mask_annotation_viewer.py
.venv/bin/python examples/libero/validate_mask_annotation_viewer.py
.venv/bin/python -m pytest -q examples/libero/mask_annotation_viewer_test.py
```

构建器默认读取冻结的 `20261019_mainview_v1`，输出 `20261019_viewer_10demos_v2`；可用 `--source-run`、`--output`、`--mujoco-viewer` 指定输入和位置。需要当前环境中的 numpy、h5py、Pillow、ffmpeg/ffprobe；静态页面服务只需要 Python 标准库。

`viewer_catalog.json` 记录全部 400 个示范、选择结果、媒体与轨迹 SHA256。`viewer_validation.json` 记录数据对齐、视频完整解码、文件链接和冻结标注的检查；`qa/browser_checks.json` 及页面截图记录浏览器交互验证。验证关注数据、媒体和适配正确性，不计算无独立真值的 mask 准确率。
# 数据集回放的容差释放

原始／无容差／容差三路回放的下方显示右侧已选容差轨迹的三轴有符号释放条、有效边界和同步时序曲线。切换“姿态恢复复测”时，释放显示同步切换来源。曲线可点击定位视频帧。

数值读取原始 `trajectory.json` 的 `rotation_release_rad`，不会用跟踪误差或旧导出中的默认零值代替。求解拒绝的候选不会覆盖上一接受释放；记录结束后保持最后记录，并将后续曲线留空。

已有数据集页面可运行 `python3 examples/libero/install_dataset_release_viewer.py` 更新静态资源，再刷新页面。重新构建带容差对比的数据集页面时，构建器也会自动安装组件。该操作不重建视频、不修改轨迹或评测配置。


## 当前2×2对照（2026-11-10）

完整重构viewer保留任务/demo双列和“训练入选容差（301条）”精确筛选。主视频区域左列无容差、右列标注容差，上排主视角、下排腕部视角，共用20Hz时间轴。页面不再展示原始HDF5视频；源文件不删除。名义腕部从自身recording.hdf5的保存像素导出，容差腕部复用已有MP4，不运行仿真。

已有viewer补齐媒体：`.venv/bin/python examples/libero/prepare_rollout_wrist_videos.py --viewer data/libero_tasktol_joint_servo/20261101_full_v1/viewer --workers 6`；媒体完成后执行`python3 examples/libero/install_rollout_quad_viewer.py data/libero_tasktol_joint_servo/20261101_full_v1/viewer`。两者均不启动服务。带容差对照的构建器已自动接入。
