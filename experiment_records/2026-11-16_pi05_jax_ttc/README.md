# π0.5 JAX TTC 实现与本地验证

状态：**实现与本地验证已完成**，2026-11-16。本轮未启动正式模型训练、任务成功率测评、物理机器人控制或长期服务器操作。验证范围为合成数组、小型真实动作专家、测试用优化器与 checkpoint，以及本机 WebSocket；不据此推导任务成功率、运动性能或 GPU 时延。

## 目的、模型与协议

当前 `task_tolerance` 分支为原始 π0.5 添加可选 JAX training-time RTC（TTC），默认关闭；同时适配仓库内 `openpi-client`、LIBERO 和 simple_client 示例。保留用户既有 `cmd.txt`、训练预设及其他实验记录修改。

用户指定：总训练步数 T，已完成更新数 t；前20% α=0，中间40%线性增加，最后40% α=1。每样本从 `(1-α)δ0+αq_target` 采样，`q_target=(0.10,0.05,0.80,0.05)`。使用恢复的 `TrainState.step`，不重置课程；验证固定采用目标分布。保留既有时间/噪声 RNG、动作监督维度、分组权重与来源权重。有效后缀按全局 token 数平均。

模型不新增参数，adaRMSNorm 允许逐 token 时间；前缀干净时间为0。模型采样逐步固定已知前缀。checkpoint 保存课程元数据，服务自动恢复；普通预热阶段 checkpoint 不广告已接受 TTC 训练。

客户端默认普通推理；开启后以一个后台请求承接前 d 步，接管时跳过已执行部分。推理超过预算则在承诺前缀边界等待；重置排空旧响应。先验在模型归一化空间传递，实际控制动作独立返回。带 DeltaActions 的状态重锚坐标目前明确拒绝，需要专用适配。

未来动作与容差为独立去噪块，一起进入动作专家、分别输出；两块共享前缀长度 d，各自使用前 d 步先验。本轮提供通用前缀/flow 工具与命名块传输/客户端承接接口，**未实现双块容差网络**。

## 样本与验证范围

- 课程：T=100，t=0/19/20/40/60/99；恢复 step40 后 α=0.5；预热 checkpoint 完成步20/21不启用条件推理、22开始启用。
- 延迟采样：seed42，100000个独立样本；d=0/1/2/3计数9975/4994/80069/4962，占比9.975%/4.994%/80.069%/4.962%。这些比例是采样频率，不是任务成功率。
- LeRobot 数值对照：固定 commit `781d3daccc64e8ce3e3d0910e64b2b0180551059`，仅抽取并运行已检查的 `FlowConvention`、`make_flow_matching_inputs`、`_reduce_training_rtc_loss`，不安装新 LeRobot。seed42；FP32/BF16 × D=7/32 × 延迟向量全0、0/1/2/3、全3，共12组。输入噪声、时间条件与后缀损失均通过，最大 flow 绝对差2.384185791015625e-7。
- 模型：保持真实 π0.5 时间 MLP、Gemma 注意力、adaRMSNorm、KV cache 与输出投影，缩减宽度32、1层、词表32、动作D4/H5；视觉塔替换为均值加 Dense。初始化seed7，动作seed8，主要验证key42。覆盖 JIT、参数梯度、前缀固定采样和输出变换。
- 旧源码对照：从本次修改前 Git commit 提取原 pi0.py/gemma.py，FP32和BF16各一组；参数树、旧参数 load、TTC关闭时损失/输入梯度/采样逐位一致。未加载官方大模型 checkpoint。
- 分片：两个CPU设备，设备间有效token数不同，验证全局分母与梯度；不代表已测真实多GPU训练性能。
- 客户端：合成多块先验18控制步连续对齐；超时、d=0、重置、异常和非法预算。真实本机 WebSocket 中运行10控制步并重置，同时验证普通请求结构不变；不执行环境物理步进。
- 优化器/恢复：真实 train_step 与 Orbax API，小型测试模型；保存/恢复 step40，验证课程继续和 assets/ttc.json。原有训练入口、TaskTol采样、服务资产和消息序列化回归一起运行。

## 当前验证结果

| 验证 | 已核对结果 | 证据 |
|---|---|---|
| 完整集合最终代码复跑 | 112 passed，1 skipped（单CPU不运行双设备检查） | `final_tests_v2.log` |
| 主集合首轮 | 109 passed，1 skipped（单CPU不运行双设备检查） | `final_tests.log` |
| 双CPU分片专项 | 1 passed，27 deselected；补齐上述跳过项 | `sharding_tests.log` |
| 后续预热服务边界与相关回归 | 11 passed；与主集合有重叠，不相加为独立样本数 | `warmup_serving_tests.log` |
| FP32/BF16旧源码兼容对照 | 两组全部通过 | `legacy_comparison.json` |
| 固定LeRobot函数数值对照 | 12/12组通过 | `lerobot_comparison.json` |
| 静态检查 | Ruff通过，git diff --check通过 | `lint.log` |

最终完整集合复跑112项通过；单CPU跳过的唯一双设备检查已在双CPU专项通过。两轮完整集合和专项之间有重叠，不累加为独立实验次数。所有测试断言的范围可由测试源码及 SHA 清单核对。无任务/seed组合的成功率分母，无关键失败任务、无正式回合重测或筛选。

## 本地绝对路径

- 操作和实现规格：`/media/czl/sata/franka_my_code/openpi/docs/ttc_jax_design_20261116_zh.md`。
- 原始验证目录：`/media/czl/sata/franka_my_code/openpi/analysis/ttc_jax_20261116`。
- 完整测试命令：`/media/czl/sata/franka_my_code/openpi/analysis/ttc_jax_20261116/verification_commands.sh`。
- 配置、源码 SHA、版本、日志清单及临时 checkpoint 清单：`/media/czl/sata/franka_my_code/openpi/analysis/ttc_jax_20261116/verification_manifest.json`。
- 测试日志：上述原始验证目录中的 `final_tests.log`、`final_tests_v2.log`、`sharding_tests.log`、`warmup_serving_tests.log`、`pipeline_regression_unsandboxed.log`、`client_tests.log`、`lint.log`。
- 课程和采样统计：`/media/czl/sata/franka_my_code/openpi/analysis/ttc_jax_20261116/curriculum_audit.json`。
- 旧源码与比对：上述验证目录中的 `baseline_commit.txt`、`baseline_pi0.py`、`baseline_gemma.py`、`verify_legacy.py`、`legacy_comparison.json`、`legacy_comparison.log`。
- 外部参考冻结来源：上述验证目录中的 `lerobot_sources.json`、`lerobot_pi05_modeling_pi05.py`、`lerobot_common_flow_matching.py`、`verify_lerobot.py`、`lerobot_comparison.json`、`lerobot_comparison.log`。
- 测试源码：`/media/czl/sata/franka_my_code/openpi/src/openpi/models/rtc_test.py`、`/media/czl/sata/franka_my_code/openpi/scripts/train_ttc_test.py`、`/media/czl/sata/franka_my_code/openpi/src/openpi/policies/policy_ttc_test.py`、`/media/czl/sata/franka_my_code/openpi/src/openpi/serving/ttc_test.py`、`/media/czl/sata/franka_my_code/openpi/packages/openpi-client/src/openpi_client/ttc_action_chunk_broker_test.py`。
- 小型 Orbax 测试 checkpoint：位于 `/tmp/pytest-of-czl/pytest-*/test_real_train_step_restored_0/checkpoint/40`，最后核对的**精确绝对路径和20文件SHA**见 verification_manifest.json 的 temporary_checkpoint 字段。pytest 临时产物以后可能自动清理，未复制为持久模型；日志和清单保存在仓库验证目录。
- 正式训练配置/请求、正式 checkpoint、逐回合结果、轨迹与视频：**未生成**。本轮没有远端实验产物。

## 异常与修正历史

1. 初始 warmup 梯度逐位断言观察到最大7.450581e-9的 FP32 舍入差；条件分支改变XLA反向融合。warmup 梯度使用明确的 rtol=1e-6/atol=1e-8；TTC关闭相对旧源码仍逐位对照，不放宽该兼容要求。
2. 沙箱内 Orbax 异步写入停在事件循环；诊断日志 `pipeline_retry.log`，相关测试进程已定向中断。原回归另遇 `/home/czl/.cache/swanlab` 只读错误，保留 `regression_tests.log`。改用 /tmp 缓存目录，并在获准的本地测试执行环境中运行异步/回环测试后通过；没有更改生产 Orbax 异步逻辑或共享环境。
3. 初版回环测试 listener 使用库默认1MiB上限，导致合成float图像请求过大；保留 `websocket_tests.log`。测试 listener 改为与正式服务相同的 max_size=None 后，最终完整集合真实SDK往返通过；正式服务原本已设置该值，未修改。
4. 最终审查补充预热 checkpoint 服务边界，增加完成步20/21/22测试并通过。测试失败/中断不作为正式训练失败或任务失败统计，也不混写为首次全量测评。

## 结论适用范围

代码与本地验证支持：原始 π0.5 可选 JAX TTC、指定课程、模型条件采样、checkpoint配置承接和仓库内客户端选项；旧普通路径在已测试的小模型FP32/BF16范围兼容。实际任务性能、端到端部署延迟分布、GPU吞吐、官方大checkpoint恢复和未来独立容差网络仍需后续相应验证，本轮不声称这些结果。
