# 单机七卡训练代码审计

日期：2026-10-28。已终止的正式续训没有重新启动。本次仅执行小模型数值审计，结束后七卡利用率为0%，显存约44–59MiB。

## 结论

**未发现单机七卡的数据切分、全局loss平均或梯度聚合错误。当前场景应继续使用官方JAX多卡核心，不需要自行增加DDP、手工all-reduce或额外除以设备数。**

这次检查包括：官方源码快照对照、本地与服务器源码哈希核对，以及真实七张3090上的单卡/七卡数值比较。数值比较使用小模型和合成数据，但调用项目实际的 `train_step`、分组loss、优化器、sharding及 `TorchDataLoader`。没有对完整π0.5模型做长期单卡/七卡等价实验。

## 官方代码对照

下载的上游快照位于本目录 `upstream_20261028/`，来源为 `https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/`。比较结果在 `upstream_comparison.json`。

| 核查位置 | 结果 |
|---|---|
| `src/openpi/training/sharding.py` | 与官方逐字节一致，SHA256相同 |
| `src/openpi/training/optimizer.py` | 与官方逐字节一致，SHA256相同 |
| `scripts/train.py:init_train_state` | AST完全相同 |
| main中的mesh、data_sharding、replicated_sharding、ptrain_step | 对应赋值AST全部相同 |
| `train_step`主要差异 | 改用带分组指标的loss接口，`has_aux=True`，记录分项loss；全局mean、参数更新和EMA主体保持一致 |
| 数据加载器主要差异 | 本地数据/Arrow/schema检查和有限验证迭代支持；JAX batch分配和make_array分片方式保留 |

服务器的sharding、train.py、optimizer和data_loader哈希与本地审计版本一致，因此本次审查针对实际训练代码。

## 全局batch与loss

1. `batch_size`为全局batch。`local_batch_size = batch_size // jax.process_count()` 中的local指进程，当前单机单进程为84，之后才分到七张GPU，并不是每张GPU84。
2. `fsdp_devices=1`生成7×1 mesh：batch轴7，fsdp轴1。模型参数和Adam状态复制，batch沿七卡切分。
3. 全局loss为 `jnp.mean(chunked_loss)`。chunked_loss的全局形状为[B,H]，其反向传播由JAX全局数组语义生成所需通信。代码没有再手工乘除7。
4. 分组loss先在对应动作维度取均值；容差轨迹权重[B]扩展为[B,1]，只乘物理动作项，随后按全局B和H求均值。没有按设备分别归一化权重。
5. 随机数由全局step折叠，噪声按全局动作形状生成。当前是一个进程执行全局程序，不能套用“每卡一个独立Python进程”的DDP随机数处理方式。

## 真实七张3090的数值检查

脚本：`audit_multigpu.py`。结果：`multigpu_gpu_audit.json`。运行环境JAX版本和设备列表保存在JSON中。

每组固定相同初始参数、优化器计数、全局数据和随机key，分别使用1张和7张GPU，执行3次实际 `train_step`。包含13维/32维、全局batch84/28、SGD/AdamW、混合1/4倍样本权重以及随机噪声。诊断使用恒定1e-3学习率以便观察参数差异，这只作用于小模型，不是正式训练的学习率调整。

| 案例 | 最大loss绝对差 | 最大参数绝对差 |
|---|---:|---:|
| FP32、13维、batch84、SGD | 2.38e-7 | 1.46e-11 |
| FP32、32维、batch84、AdamW | 2.38e-7 | 7.45e-9 |
| BF16矩阵乘、32维、batch28、AdamW | 4.77e-7 | 2.18e-5 |
| FP32、13维、batch84、AdamW、随机噪声 | 2.38e-7 | 2.61e-8 |

实际数据加载器的样本ID检查结果：全局84个ID在七卡分别为0–11、12–23、24–35、36–47、48–59、60–71、72–83。每卡12个，合并后无遗漏、无重复。

编译后的七卡程序含all-reduce通信，FP32案例中并非各卡独立更新后碰巧输出近似loss。

### BF16精度检查的解释

首次使用严格逐元素判据时，BF16案例在Adam状态一个224元素张量中的1个元素超出 `rtol=.005, atol=3e-5`，最大绝对差1.416e-4；原始记录保存在 `gpu_bf16_initial_elementwise_check.log`，没有删除。

进一步计算各张量整体相对L2误差，参数最大约0.00328%，优化器状态最大约0.38846%，指标最大约0.09731%。BF16下不同batch分片会改变矩阵梯度部分和的量化、累加顺序，不能要求逐元素高精度一致。后续检查保留绝对误差，同时使用1%的张量相对L2界限；FP32仍保留原严格逐元素判据。

这些结果符合低精度数值差异的量级，没有呈现7倍或1/7倍的梯度缩放。它们不能证明完整模型长训练轨迹相同，也不能单独解释当前闭环表现。

显式读取 `.lower()` 编译文本时，项目的运行时类型检查器不接受JAX的ArgInfo占位对象。因此脚本仅在HLO检查环节临时关闭类型检查；实际训练调用保留正常类型检查。该问题不是普通训练调用的七卡错误。

## 应保留的实现与需要调整的配置

- 单机七卡继续单进程运行官方JAX训练脚本，`fsdp_devices=1`。
- 全局batch必须能被7整除，例如28、56、84；不要把全局batch误当成每卡batch。
- 不要额外给loss除以7，不要再手工增加梯度平均，否则会改变已正确的全局梯度。
- 模型有效监督维度、预训练动作投影的形状兼容性、全局batch、更新次数、学习率日程、数据和4倍轨迹权重，仍是这次实验需要单独控制的因素。
- 目前没有代码证据要求重写多卡核心。更有价值的下一步是固定模型与训练配置，用相同全局batch比较不同卡数的短程训练或验证结果。
