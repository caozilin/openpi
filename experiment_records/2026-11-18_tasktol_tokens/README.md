# TaskTol 六维容差 token 与双块 TTC

状态：本地 CPU 代码验证已完成，最终回归 97/97 通过。固定动作输入并扰动容差噪声与先验，动作逐值一致；扰动动作噪声会改变后缀容差预测。两路输入/输出投影梯度、TTC 前缀钳制与两块在异步取帧时的对齐均通过专项断言。尚无闭环成功率或部署耗时结论。

实现基于 `task_tolerance` 分支 `eeab65a` 与工作区已有 π0.5 TTC。模型数据接口沿用 joint13 归一化后补齐 32 维的格式。动作专家内部使用 H 个动作 token 和 H 个容差 token，容差读取动作，两路使用独立噪声、相同 flow 时间和 TTC 延迟。两组预测在一次专家前向中计算。

验证使用 CPU，模拟两个 JAX 设备。新模型专项采用合成观测、两条样本、H=5、32 宽单层 Gemma 和替换的视觉塔；模型初始化 key 为 7，方向性测试噪声 key 为 55、先验 key 为 12。测试显式激活原本为零的注意力门，再检查双向干预结果。权重恢复测试以合成参数树替代磁盘读取，经过真实的权重合并和模型恢复入口。异步重规划使用仓库 SDK 和进程内真实 JAX 策略，连续消费 7 个控制步。无正式训练、仿真或机器人回合，故无成功率和失败任务分母。

- 原始验证目录 `/media/czl/sata/franka_my_code/openpi/analysis/tasktol_tokens_20261118`。
- 验证命令与源码清单 `/media/czl/sata/franka_my_code/openpi/analysis/tasktol_tokens_20261118/verify.sh`、`/media/czl/sata/franka_my_code/openpi/analysis/tasktol_tokens_20261118/sha256.txt`。
- 专项断言 `/media/czl/sata/franka_my_code/openpi/src/openpi/models/tasktol_test.py`，训练与服务开关 `/media/czl/sata/franka_my_code/openpi/docs/ttc_jax_design_20261116_zh.md`。
- 首轮专项日志 `/media/czl/sata/franka_my_code/openpi/analysis/tasktol_tokens_20261118/unit.log`，5 项通过，1 项 CLI 测试因测试命令缺必填 `--exp-name` 失败。补齐测试参数后，阶段回归 96 项通过，见 `/media/czl/sata/franka_my_code/openpi/analysis/tasktol_tokens_20261118/regression.log`。
- 最终回归日志 `/media/czl/sata/franka_my_code/openpi/analysis/tasktol_tokens_20261118/final.log`。不同轮次存在重叠，分别报告。
- 正式训练请求、checkpoint、逐回合轨迹与视频、GPU 延迟测量均待生成。
