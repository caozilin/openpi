# LIBERO-40 · MuJoCo Dataset Studio 适配版

入口为同目录的 `index.html`（`report.html` 内容相同）。保留整个目录结构；页面、视频、轨迹和证据图片均在本地，无 CDN 或远程模型依赖。

## 打开页面

在导出目录运行以下命令，然后访问 <http://127.0.0.1:8768/index.html>：

```bash
python3 serve.py --port 8768
```

也可使用 `python3 serve.py --port 8768 --background`。后台模式在 `qa/preview_server.json` 记录 PID、地址和目录，在 `qa/preview_server.log` 记录错误。结束后台服务时可向所记录的 PID 发送 SIGTERM。服务仅监听 `127.0.0.1`，支持 MP4 Range 请求和拖动定位。日志和 PID 文件属于可变运行记录，不纳入输出文件校验清单。

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
