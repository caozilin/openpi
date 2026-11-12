# LIBERO 名义轨迹 → 当前 DSP 贝叶斯容差标注（2026-11-05）

适配和小样本验证完成；未开始全量 1875 条标注。

正式代码在 `/media/czl/sata/franka_my_code/franka_mujoco`：

- `annotate_libero_tolerance.py` 为前台 CLI。
- `tasktol/data/libero_annotation/` 分离只读数据源、记录几何、语义映射、准备流程、DSP 调用。
- `tasktol/counterfactual/stored.py` 为共享后端提供不可变的导入轨迹上下文，没有新增执行循环。
- 搜索直接使用当前 `label_franka_nominal_dsp.search_subtask`，执行仍调用唯一的 `run_counterfactual_trial`。

## 旧实现清理

按用户明确要求删除活动源码中的 `annotate_tolerance.py`、`annotate_custom_layouts.py`、
`tasktol/counterfactual/search.py` 和两份专属测试，移除旧公开导出。
依赖的 JSON I/O 与保存布局发现函数迁到 `tasktol/data/annotation_io.py`，
95 处脚本和 20 处测试导入已迁移。640 个活动 Python 文件检查未发现旧入口导入。
历史实验快照、结果和原始数据保留；没有将历史副本注册回当前运行路径。
删除清单及迁移记录见 `legacy_retirement_receipt.json`。

## 几何和搜索定义

只编译原记录的机器人子树，加由记录提取的桌面高度与有限 XY 范围数值几何。
不实例化完整 LIBERO 任务场景，不加载任务物体，不计算任务物理或成功谓词。
语义路由需要的物体位置仅从记录 XML/状态中做数值读取。
完整机器人碰撞几何（包括固定基座）参与桌面净空和自碰撞检查。
所有查询禁用 `mj_step`；114 个实际原始 trial 均为 0 物理步、20 Hz、无物体交互。

保留原动作、手指观测、名义参考、观测阶段和时间索引；独立复核动作增量可以重构连续名义参考。
当前贝叶斯协议为 PRE6+POST6 联合连续 DSP，结构化初始化后默认追加 16 次自适应查询。
选择整段平均关节速度最小且通过 5 mm 桌面净空等条件的完整已观测解。
标签保存该解连续六向上限，策略 `speed_best_feasible_search_caps_v1`，不取整、不外扩。
ENG/DIS 为零。语义表中的正数用作可释放轴，搜索域仍为 0–30°，不是将手工 5/15°当作搜索结果。

## 最终小样本

下表任务编号按代码从 0 开始，均为 `demo_0`。

| 任务 | 追加查询/子任务 | 实际总查询 | 合格观测 | 标签 |
|---|---:|---:|---:|---|
| Spatial 00 | 16（正式默认预算） | 28 | 28 | 已生成 |
| Spatial 04 | 2（集成验证预算） | 14 | 14 | 已生成 |
| Goal 03 | 2 | 14 | 0 | 未生成：桌面净空不足 |
| Long 06 | 2 | 28（两子任务） | 0 | 未生成：桌面净空不足 |
| Long 07 | 2 | 30（两子任务） | 0 | 未生成：桌面净空不足 |

最终有效输出仅为 `final_default16/` 与 `final_smoke2/`。
`pilot_v*`、`dsp_smoke_v1`、`dsp_final_*`、`dsp_verified_*` 是先前适配/修订验证记录，
其中旧搜索原型不能作为当前协议标签；这些记录没有写回数据集。

Goal 03 的 14 个查询整段最小净空最大也只有 2.657 mm，瓶颈位于夹爪手指，低于 5 mm。
Long 06/07 第一子任务均存在固定基座 `robot0_link0` 相对桌面高度几何 -16.792 mm 的最小距离，
第二子任务还有手指净空不足。这些是当前完整机器人/桌面定义下的实测拒绝原因。
没有排除基座、修改桌面范围或放宽门限来使其通过。
有限查询没有合格观测不代表已证明连续搜索域内不存在合格解。

## 验证

- 226 项相关测试通过：DSP/共享后端/适配 107，I/O 迁移 107，现有 LIBERO 环境接口 12。
- `compileall tasktol tests` 通过。
- 默认环境全测试收集得到 1943 项，另有两个模块因缺少 robosuite/h5py 无法导入；
  对应模块已在已有 LIBERO 环境运行，12 项全部通过，没有安装依赖。
- 独立核对 114 个原始 trial SHA、273 个不可变输入/源码/mesh 文件、最佳观测选择和逐帧标签。
- 独立审计 `final_independent_audit.json` 的 `complete=true`，范围仅这五条小样本。
- 所有本次 CLI、搜索子进程与审计进程已退出；未启动推理、训练、全量标注或修改用户 viewer。

用法见 MuJoCo 项目的 `tasktol/data/libero_annotation/README.md`。
无合格标签与基础设施错误分开记录，`processing_complete` 不等同于 `all_labeled`。
