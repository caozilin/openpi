# 基于观测与动作的第三容差专家：VLA 文献调研

检索日期：2026-11-10。范围：2025–2026 年公开 VLA / World-Action 论文，重点核对独立专家、动作条件、控制参数与逐维约束。本文为文献调研，没有启动训练、测评或数据处理实验。文中的结构建议是设计推断，尚无本项目实验验证。

## 结论

已经存在非常接近的结构先例：**CompVLA 用独立 Compliance Expert，基于 VLM 上下文和动作专家预测的参考轨迹，生成控制参数；T-Rex 的第三专家明确复用视觉语言和动作两部分 KV。** 因而“再加一个读观测与动作的专家”不能单独作为新颖性主张。

本轮核对的论文中，未发现明确预测 `(rx−, rx+, ry−, ry+, rz−, rz+)` 六个非对称旋转容差边界、再交给下游姿态优化器使用的同构方法。这是检索结果的边界，不是不存在相关工作的证明。

本项目六维标签的含义来自既有训练记录：单位 rad，表示搜索容差边界，不是六自由度位姿，也不是实际释放量。依据：[现有 joint13 记录](/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/README.md)。

## 最接近的两篇

### CompVLA

标题：*CompVLA: A Variable Compliance Vision-Language-Action Model for Contact-rich Manipulation*。2026-09-20 首发，v2 更新于 2026-11-06；项目页标注 CoRL 2026。

它在 π0 上增加与动作专家同结构的柔顺性专家。动作分支先生成 TCP twist 与夹爪命令，积分为参考轨迹；柔顺性分支结合该轨迹及 VLM KV，输出六维对角刚度和六维虚拟位移。两个分支使用 flow matching，新增专家复制动作专家权重初始化。训练时轨迹条件来自真值，推理时来自预测。

重要连接细节：动作信息通过显式轨迹条件输入；论文附录 A.1 将相对位姿和夹爪编码为每步 13 维条件，放入柔顺性输入的 padding 通道。不能将它描述成“第三分支直接读取动作分支逐层 KV”。

与本设想的共同点是将运动生成与控制参数生成拆开；差别是刚度/虚拟位移描述受力响应，旋转容差描述任务允许的姿态偏离范围。

来源：[论文 §4.2 / 附录 A.1](https://arxiv.org/html/2609.23614v2)、[项目页](https://compvla.github.io/)。项目页的代码链接 `https://github.com/compvla/CompVLA` 在本轮访问返回 404，因此上述实现细节按论文核对，未由公开代码独立确认。

### T-Rex

标题：*T-Rex: Tactile-Reactive Dexterous Manipulation*。2026-06-15 首发，核对 v2。

三个 MoT 专家分别负责视觉语言/未来表示、动作去噪、触觉驱动的动作细化。动作专家完成前段去噪后，触觉专家继续处理；附录 B 明确给出其上下文缓存：

\[
\mathrm{KV}_{\mathrm{split}}
= [\mathrm{KV}^{\mathrm{lat}}\mid \mathrm{KV}^{\mathrm{act}}_{\mathrm{split}}].
\]

动作部分在中间流时间重新编码，第三专家读取该缓存并结合实时触觉继续生成动作。它是“第三专家读取观测与动作缓存”的直接结构先例，但预测目标仍是动作，不是独立容差。

来源：[论文 §4 / 附录 B](https://arxiv.org/html/2606.17055v2)。

## 其他相关论文及区别

| 论文与首发时间 | 已核对的方法 | 与容差第三专家的关系 |
|---|---|---|
| [Bee](https://arxiv.org/html/2609.27450v1)，2026-09-23 | Correction Model 以状态与 VLA 动作提案为条件，预测逐维修正分布；方差调节 RL 约束松紧 | 与“根据当前动作决定哪些维度可以放松”接近；方差表达人工纠正的一致性，不是已验证任务容差；不是第三个 MoT 专家 |
| [AffordanceVLA](https://arxiv.org/html/2606.06155v2)，2026-06-04 | Understanding / Affordance / Action 三专家；动作读前两者，affordance 分支不读动作 | 三专家与结构化几何表示的先例，条件方向与本设想不同 |
| [PLaW-VLA](https://arxiv.org/html/2610.12285v1)，2026-10-08 | VLM / 潜在世界模型 / 动作三专家；预测未来 latent 供动作使用 | 展示三专家的因果注意力组织；新增分支输出未来状态，不是控制参数 |
| [CompliantVLA-adaptor](https://arxiv.org/html/2601.15541v2)，2026-01-21 | 外接 VLM 结合视觉、语言及力反馈调节刚度/阻尼，与 VLA 轨迹一起驱动 VIC | 控制参数自适应的先例；并非共享主干下的第三专家 |
| [ForceVLA2](https://arxiv.org/abs/2603.15169)，2026-03-16 | 力语义提示与动作专家中的 Cross-Scale MoE，输出运动与力控制目标 | 同时学习动作和物理控制量，但 MoE 不等同于独立容差 Transformer 分支 |
| [TA-VLA](https://arxiv.org/html/2509.07962v1)，2025-09-09 | 比较扭矩输入位置、历史编码，联合预测未来动作与扭矩 | 可作为“共享动作生成器增加物理量输出”的相关基线，不是容差专家 |
| [Motus](https://arxiv.org/abs/2512.13030)，2025-12-15 | 理解、视频生成、动作三个 MoT 专家 | 进一步说明三专家本身已有先例；目标不是容差 |
| [Wiring Matters](https://arxiv.org/html/2610.06318v1)，2026-10-05 | 研究 affordance 辅助头的梯度隔离、注入位置与桥接初始化 | 提醒新增监督的连接方式需要对照；其结论来自特定 affordance/QwenPI 设置，不能直接归因到本项目 |

以上主要结论核对论文正文或作者项目页；Motus 与 ForceVLA2 在本轮主要核对摘要/作者概述。没有把第三方综述中的结构描述作为唯一结论依据。

## 代码公开状态复核（2026-11-10）

进一步检查作者项目页、GitHub 实际目录及部分源码，不能将所有论文归为“未开源”：

| 方法 | 本轮核实状态 | 可直接检查的证据 |
|---|---|---|
| CompVLA | 官方代码链接返回 404，当前无法获取 | [作者项目页](https://compvla.github.io/) 的 Code 链接指向 `compvla/CompVLA`；404 不能区分尚未公开、私有或链接失效 |
| T-Rex | 实质代码已公开，MIT | [官方仓库](https://github.com/ZhuoyangLiu2005/T-Rex) 包含 `qwen_vla/modeling_qwen3vl_mot.py`、`modeling_vla.py`、训练及推理入口；已读取 `modeling_vla.py`，可见缓存复用实现。README 说明 main 提供 post-training/inference，full-pipeline 分支提供预训练/中训代码；完整训练语料不随代码发布 |
| AffordanceVLA | 模型及训练代码已公开，仓库含许可证 | [官方仓库](https://github.com/Skywalker-yqz/AffordanceVLA) 实际包含 `src/models/paligemma_with_expert.py`、`modeling_affordance.py`、三个 affordance decoder 与 `src/train.py`；并非仅 README |
| PLaW-VLA | 模型及训练代码已公开，Apache-2.0；官方预训练权重待发布 | [官方仓库](https://github.com/RainyRobo/PLaW-VLA) 实际包含 `src/plawvla/models_pytorch/world_model_pytorch.py`、`pi0_pytorch.py`、`gemma_pytorch.py` 与 Stage I–III 入口；README 明确区分当前基座初始化训练与论文权重 |
| Bee | 项目页标注 Code Coming Soon | [作者项目页](https://genie-universal-algorithm.github.io/BEE/) |

本轮 GitHub 目录快照：AffordanceVLA 默认分支树 SHA `3bb77f1a11a05781427458575c6873c147a8a0ab`；PLaW-VLA 默认分支树 SHA `5fb8f32cba29e47f6838ebf572e7bf036282a68b`。T-Rex 已读文件 `qwen_vla/modeling_vla.py` 的 blob SHA 为 `4ea39759bd46602431ebd475ed611bc64ccd0283`。

此处“代码已公开”表示已确认存在实质实现，不代表本机完成依赖安装、训练或复现。权重、数据与代码的公开范围分别判断。

## 对本项目设想的设计含义

用户提出的条件关系可以写成：

\[
A\sim p_\theta(A\mid O,L),\qquad
C\sim p_\phi(C\mid O,L,A),
\]

其中 `A` 是动作或参考轨迹，`C` 是六个旋转方向的容差边界。该分解是对设想的形式化，不是这些论文已经验证了本项目的目标。

两种实现需分别讨论：

1. **先预测动作，再预测容差**：参考 CompVLA，将完整预测动作/相对轨迹作为显式条件。若训练只用真值动作，需评估预测动作造成的分布差异。
2. **第三专家逐层读图文与动作 KV**：T-Rex 提供双来源缓存参考。需要定义动作 KV 对应哪个流时间；动作尚在去噪时，其特征随流时间变化，不能把某一步动作缓存当作整个动作生成过程恒定有效的上下文。若先固定一条动作轨迹再编码，才可讨论复用该固定轨迹的缓存。

本项目的六维是三条旋转轴的正负边界，并非笛卡尔六自由度。标签中的逐轴范围也不能自动解释为六个边界任意组合都可行的联合区域，联合姿态仍受下游优化与验收协议约束。

研究贡献应落在“学习动作相关的非对称任务容差及其有效使用”上。是否需要完整 Transformer 专家，应与同头 joint13、独立小头、仅观测容差头和动作条件容差分支做匹配对照；这些是未来验证建议，本轮没有运行这些实验。

## 检索边界与复核地址

检索覆盖 `vision-language-action / VLA` 与 `third expert / three experts / tolerance / pose tolerance / orientation / constraint / compliance / stiffness / force expert` 的组合，并沿 CompVLA 的相关工作扩展。排除了天文 Very Large Array、推理压缩误差容忍度等同名命中。

- 本地既有论文笔记只读检索目录：`/media/czl/sata/ai-paper-research-vault/20_Research/Papers/`。
- 本地既有方法背景：`/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training/README.md`。
- 本调研文档：`/media/czl/sata/franka_my_code/openpi/docs/vla_tolerance_expert_literature_20261110_zh.md`。
- 未生成本项目训练/测评结果、轨迹或视频；本文不形成新增实验成功率结论。
