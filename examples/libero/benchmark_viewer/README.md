# LIBERO Benchmark Studio

在本项目内复用 TaskTol Dataset Studio 的布局、播放器、时间轴、3D轨迹和曲线交互，直接读取本地评测已提交的回合记录。

```bash
.venv/bin/python examples/libero/serve_benchmark_viewer.py --port 8770
```

服务在当前终端持续运行，按 `Ctrl+C` 退出；收到 `SIGTERM` 或关闭终端也会退出并释放端口，不创建后台子进程。

也可沿用原可视化入口和8768端口：

```bash
python3 examples/libero/serve_mask_annotation_viewer.py --benchmark --port 8768
```

访问 <http://127.0.0.1:8770>（第二条命令对应8768）。默认扫描 `data/libero_evaluation/*/evaluation_request.json`，也可用 `--root` 指定评测根目录。服务只监听本地回环地址，只读文件。

打开后主区域首先显示双视角视频。左侧保持双列表：一列任务、一列seed；跨任务保留选中的seed。保留播放、倍速、逐帧、3D轨迹旋转和曲线定位。每3秒增量更新已保存回合列表，保留当前视频、播放位置、列表滚动和选择。前端不展示测评进度。

切换时优先更新播放器，再绘制可见轨迹图；屏幕外的曲线暂停绘制，滚动出现时自动更新。缓存最近24份回合数据和派生曲线，预取下一seed及下一任务，悬停seed时预热视频。两路播放器复用同一媒体URL及浏览器缓存，回合数据优先于后台视频预取。服务端回合读取直接定位文件，不等待全目录扫描，并缓存回合转换结果；已封存视频支持长期本地缓存和HTTP Range。快速连续切换使用请求序号，避免较慢响应覆盖当前选择。

视频使用客户端已保存的512×256双相机MP4，通过两路同步播放器分别裁出agentview和wrist，无需复制或重编码。播放器支持HTTP Range拖动定位。模型物理动作按米/世界旋转向量弧度显示，轨迹保持20Hz动作前观测；IK不可行的终止帧和失败回合均保留。视频封存、`result.json`原子提交后，回合才可回放。

`studio.js`和`studio.css`基于 `franka_mujoco/tasktol/data/viewer/static/`；原始布局、资产SHA记录在 `upstream_manifest.json`。`studio.js`调整初始化入口、回合请求优先级及切换期间播放的竞态处理。其余LIBERO适配集中在 `serve_benchmark_viewer.py`、`live.js`和`live.css`，没有修改正在运行的评测客户端、冻结控制器或推理副本。

验证：

```bash
.venv/bin/python -m pytest -q examples/libero/serve_benchmark_viewer_test.py
```

测试包括新提交记录自动可见、成功/失败保留、物理动作与终止帧映射、视频Range/HEAD、未完成回合及路径边界。

## 每任务10次与来源视图

目录优先显示“无容差 · 40任务×10次（含失败重测）”：40个任务，每个seed0–9各一个回合，结果来自已保存的 `combined_latest_400.json` 来源清单。原223成功、17条失败重测和新增160分别读取各自的原始文件，不复制、不覆盖评测数据。回合元数据中的“记录来源”显示实际run目录。

“新增无容差 · seed 6–9（160回合）”、原240组和17条失败重测也保留为独立目录。只接受请求中明确选择的任务/seed；部分评测不会补造缺失记录。新增seed和合并视图均支持双视频、Range拖动、轨迹与下载。连续失败保持帧显示“保持上一参考”和失败计数，终止原因区分20帧求解失败与步数上限。前端继续以视频为主、任务和seed双列表，不增加测评进度面板。

原标注入口的 `--background` 为旧命令兼容参数，现在仍以前台方式运行。查看旧重构数据时不加 `--benchmark`。
容差回合的视频下方提供与MuJoCo Studio同款的三轴有符号释放条，显示记录中的`rotation_release_rad`（阶段相对优化释放）、正负边界、当前phase、语义mask和有效mask。播放、暂停、拖动及切换seed时同步刷新；下方“容差释放时序”展示完整记录和游标。失败保持帧使用上一接受值，失败候选不混入已执行曲线；未记录的值显示“—”，无容差回合标明关闭。这里显示求解器接受的释放量，不把实际跟踪姿态误差冒充为释放量。

## 成功后继续1秒（新回合默认）

无容差与对应mask容差客户端首次命中原生成功条件后，继续原策略闭环20个控制帧，即1秒仿真时间。首次成功动作不占这20帧。若在原步数上限前命中成功，允许仅为这段延长越过原上限；未成功的回合仍按原上限结束。连续20帧求解失败的提前终止保持生效，发生在成功后时会明确记录延长中断。

`success`仍表示首次原生成功已命中，逐帧`success`保留每帧实际谓词。额外保存`first_success_frame`、`first_success_time_s`、`post_success_steps_completed`、`post_success_complete`和`final_native_success`，可以区分短暂成功与结束时仍然成功。视频、动作、状态和推理均记录整个延长段。独立审计分别恢复首次成功状态与最终状态进行原生谓词验证。

新请求须冻结`examples/libero/success_continuation.py`及新的客户端源码SHA，并显式写入：

```json
{"post_success_seconds": 1.0, "post_success_steps": 20, "success_criterion": "first_native_success"}
```

准备请求时可调用`success_continuation.request_fields()`获取默认字段。旧请求/旧回合不会静默恢复为新策略，也不覆盖已完成结果。此改动不自动启动评测。
