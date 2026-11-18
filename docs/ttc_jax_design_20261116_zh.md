# π0.5 JAX TTC：训练、模型采样与仓库内客户端

2026-11-16。实现已接入，本地验证详见 [实验记录](../experiment_records/2026-11-16_pi05_jax_ttc/README.md)。本轮没有正式训练、任务成功率测评或机器人运行。

## 使用

现有训练配置默认关闭 TTC。对任一原始 π0.5 的 JAX 训练配置增加：

```bash
uv run scripts/train.py pi05_libero --exp-name pi05_libero_ttc --model.ttc.enabled
```

省略 `--model.ttc.enabled` 即沿用普通训练。T 使用该配置的 `num_train_steps`，不是学习率计划中的 `decay_steps`。目标概率默认就是 `(0.10, 0.05, 0.80, 0.05)`，可显式指定：

```bash
uv run scripts/train.py pi05_libero --exp-name pi05_libero_ttc --model.ttc.enabled \
  --model.ttc.target-delay-probabilities 0.10 0.05 0.80 0.05
```

TTC checkpoint 会在 `assets/ttc.json` 中保存课程设置、总步数和已完成步数。使用原训练配置名启动服务即可自动读取 TTC 设置：

```bash
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir checkpoints/pi05_libero/pi05_libero_ttc/30000
```

普通训练预热阶段保存的 checkpoint 尚未接受正长度前缀训练，服务时自动保留普通策略模式；该边界按最后实际完成的更新核对。上述路径是训练完成后的示例，当前未生成。不能把原始 π0.5 checkpoint 改一个配置开关就当作已训练的 TTC；没有 TTC 元数据的旧 checkpoint 仍按原逻辑服务。

LIBERO 客户端增加可选开关：

```bash
uv run examples/libero/main.py --ttc --ttc-delay-steps 2 --replan-steps 5
```

默认不启用 TTC。`examples/simple_client/main.py` 也提供相同三个参数，适合接口调试；其随机观测和无控制节拍的循环不能用于证明真实部署延迟或任务性能。

其他仓库内客户端可以直接复用：

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from openpi_client.ttc_action_chunk_broker import TTCActionChunkBroker

client = WebsocketClientPolicy("localhost", 8000)
policy = TTCActionChunkBroker(client, replan_steps=5, delay_steps=2)
try:
    # 每次 infer 对应一个控制步；使用最新观测，返回一帧动作。
    action = policy.infer(observation)["actions"]
    # 执行 action；下一控制步继续 infer。
    # 回合结束时调用 policy.reset()。
finally:
    policy.close()
```

`delay_steps` 是本次推理预先承诺继续执行的动作步数，默认 2，可在训练支持范围内选择。约束为 `0 <= d <= max_delay`、`max(d,1) <= replan_steps` 和 `replan_steps+d <= action_horizon`。

## 用户指定的课程

使用 checkpoint 恢复的 `TrainState.step` 作为 t，即本次更新前已完成的梯度更新数；总步数为 T。

- t/T <= 0.2 时 α=0，动作全部取 d=0，执行普通训练前向分支。
- 0.2 < t/T < 0.6 时 α=(t/T-0.2)/0.4。
- t/T >= 0.6 时 α=1。

每个样本独立从 `q_t=(1-α)δ0+αq_target` 采样，`q_target=(0.10,0.05,0.80,0.05)`。例如中点 t/T=0.4 时概率为 `(0.55,0.025,0.40,0.025)`。最后40%依然有10%的样本 d=0。

断点恢复不重置 t。续训时保持原 T 和课程配置；调整 T 会改变剩余课程位置，应作为协议变更记录。启用 TTC 后，验证损失始终按目标分布 α=1 计算，避免随训练课程变化而改变验证口径；关闭时保留原验证行为。

保留原有预处理、时间 Beta 分布、高斯噪声和三个随机 key 流。延迟采样使用额外派生 key，不改变原流。记录 `ttc/alpha`、平均 d 和各 d 的实际批次占比。

## 模型与损失

对动作 A、噪声 ε 和单样本 flow 时间 t，构造前缀 mask `P[b,h]=(h<d[b])`，并令：

```text
τ = where(P, 0, t[:,None])
xτ = τ[...,None] * ε + (1-τ[...,None]) * A
uτ = ε - A
```

openpi 的 0 是干净端，与论文展示的时间方向相反。前缀来自当前观测对应的同一真实动作块，无需数据集新增上一块字段。后缀仍可关注前缀，不增加因果注意力限制。

`embed_suffix` 与 Gemma adaRMSNorm 同时接受样本级和 token 级条件；不新增可训练参数、不改变参数名或投影形状。采样开始及每次 Euler 更新后固定前缀，其 flow 时间始终为 0。

损失沿用 `action_loss_groups` 和逐样本物理动作权重，然后只对有效后缀归约：`sum(valid * loss) / sum(valid)`。返回值保留 `[B,H]` 接口，训练入口仍取均值。分组损失使用相同时间分母；批次来源占比和样本权重均值保留原 batch 分母。分片分母基于全局数组。

这里对齐 LeRobot 默认的有效元素平均，并非先对每条样本平均后再对 batch 平均。TTC 开关本身不会把原先监督32维的配置改成只监督7维；对照时必须使用相同维度范围。

## 客户端和传输

客户端首次推理同步获取完整块；之后每隔 `replan_steps` 个控制步发起一个后台请求，执行旧块中已承诺的前 d 步。新块与请求时观测对齐，接管后跳过已执行的 d 步，不重复执行它们。

如果推理早于 d 步完成，仍执行完承诺前缀再切换；如果超出 d 步尚未完成，则在边界等待，不继续消费未承诺动作。一次只有一个请求。回合重置排空旧响应，后台异常向调用方传播，不隐式重试。

WebSocket 透传保留字段 `_ttc`：

```text
请求：{version:1, space:"model_normalized", delay:d,
       prefixes:{actions:[H,D]}}
响应：ttc:{version:1, space:"model_normalized", blocks:{actions:[H,D]}}
```

首次 d=0 可发送空 prefixes。传回的先验在模型输出裁维、反归一化和机器人输出变换之前提取，包含完整模型维度；机器人实际执行的仍是常规 `actions` 字段。服务端校验版本、长度、块名、维度和有限值，握手广告训练支持范围。普通请求不携带该字段，返回结构保持原样。

当前线协议要求动作坐标不依赖本次观测重新锚定，适用于本仓库未额外做 DeltaActions 的 LIBERO/Franka 配置。服务端对带 `DeltaActions` 的 TTC 配置明确拒绝，避免把旧观测下的相对动作当作新观测下的前缀。其他状态相关坐标变换也需要专用适配。

## TaskTol 动作 token 与六维容差 token

对已有 JAX π0.5 joint13 训练命令添加 `--model.tolerance-tokens` 即可启用。训练数据和归一化统计使用同一套 7+6 标签。动作块与容差块各有 H 个 token，各自采独立高斯噪声，共享每个样本的 flow 时间。序列按「上下文、H 个动作、H 个容差」排列。容差能关注整个动作块，动作无法关注容差块。每个 Euler 步执行一次共享专家前向，两组速度分别由原动作投影和新增六维投影给出。新增投影在从 pi05_base 初始化时随机初始化，恢复双块模型时读取已训练权重。

示例（数据 repo_id 和对应的 joint13 统计需与实际训练集一致）：

```bash
uv run scripts/train.py pi05_franka_mujoco_joint32 \
  --exp-name tasktol_tokens_ttc --data.repo-id caozilin/franka_nominal80_tolerance20 \
  --model.tolerance-tokens --model.ttc.enabled
```

仅训练独立容差 token 时去掉 `--model.ttc.enabled`，并另取实验名。容差来源动作权重沿用所选 preset；上例基础配置为 1，需要 4× 时添加 `--data.tolerance-trajectory-action-loss-weight 4`。开启 TTC 后，两路共用同一个 d，前 d 步固定且不计损失。两路物理输出分别为 `actions[H,7]` 和 `rotation_tolerance[H,6]`。容差按原数据顺序和单位反归一化。

服务时选择训练使用的 preset 和 checkpoint，训练时覆盖了 repo_id 的配置还需传 `--policy.asset-id` 指定该数据统计。服务入口从权重识别双块模型，从 TTC 元数据识别可用课程。开启 TTC 后，客户端以同一 offset 移动两组归一化先验。每个控制步只调用一次 `step = broker.infer(obs)`，同时读取 `step["actions"]` 和 `step["rotation_tolerance"]`。

## 参考与验证边界

- [论文 v1](https://arxiv.org/html/2512.05964v1)，第 IV 节；实机模型为 π0.6，本项目复现方法在 π0.5 上的适配。
- [LeRobot π0.5](https://github.com/huggingface/lerobot/blob/781d3daccc64e8ce3e3d0910e64b2b0180551059/src/lerobot/policies/pi05/modeling_pi05.py) 和 [flow 工具](https://github.com/huggingface/lerobot/blob/781d3daccc64e8ce3e3d0910e64b2b0180551059/src/lerobot/policies/common/flow_matching.py)，固定 commit `781d3daccc64e8ce3e3d0910e64b2b0180551059`。课程为用户指定扩展。

验证包括小型真实 JAX π0.5 动作专家的 JIT、参数梯度、KV cache 与采样；修改前源码的 FP32/BF16 兼容对照；固定 LeRobot 函数的数值对照；两设备 CPU 分片归约；真实训练入口/Orbax 恢复；实际本机 WebSocket SDK 往返；客户端异步时序、异常与重置。

轻量模型缩减了 Gemma 宽度、深度和词表，替换视觉塔，未加载官方大模型 checkpoint。测试不能替代正式训练、GPU 性能测量或闭环成功率评测。所有命令、通过范围和异常历史在实验记录中归档。
