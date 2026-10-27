# 面向原示范state复现的动作重构：算法调研

日期：2026-10-24。本轮仅核对论文、作者页面和官方代码，并结合现有实验作方法选择；未启动新重录批次、安装依赖或修改控制器。

## 需求与结论

目标是：保持原速20Hz、原夹爪命令、现有严格IPOPT/完整七维关节参考以及LIBERO原生物理，通过重构前六维动作，使整段连续执行的实际state尽可能接近原示范同一时刻的state，并完成任务。主要比较机器人q/dq、末端位姿和任务相关物体位姿/速度；不能仅以末端误差或最终任务成功替代完整轨迹评价。

已有成熟可用的算法类别。最小改动的入口是迭代参考学习IRLC；与完整state目标最一致的是示范引导的闭环轨迹优化，可采用CEM或Predictive Sampling。未核实到专门为“LIBERO原OSC→本机严格IPOPT/B10”提供直接即用迁移器的工作。以下将文献已有证据与本项目适配建议分开。

## 1. IRLC：最贴近保留控制器、只改参考的需求

**Iterative Reference Learning for Cartesian Impedance Control of Robot Manipulators，IROS 2024**，Julian M. Salt Ducaju、Björn Olofsson、Rolf Johansson。

该方法根据前一次执行的实际运动修改笛卡尔参考，不替换底层阻抗控制器；作者在Panda插孔及卡扣装配任务中验证。论文给出模型假设下的收敛条件，不能直接将其证明用于含硬约束IPOPT、奇异点和不同接触模式的本项目。论文主体关注末端跟踪，并非完整机器人—物体state的联合复现。

来源：[作者机构页面](https://portal.research.lu.se/en/publications/iterative-reference-learning-for-cartesian-impedance-control-of-r/)、[全文](https://lup.lub.lu.se/search/files/206302733/IROS24.pdf)，DOI 10.1109/IROS58592.2024.10801796。本次未找到可核验的官方可执行代码仓库。

**本项目适配建议，非论文现成保证：** 从现有全局投影参考开始，每次从相同初始条件连续执行，按同一时刻的源实际末端误差更新参考，再作可达性/构型正则处理，重新差分。平移的示意形式为`r_next = Project(r + eta*filtered_demo_minus_actual_error)`，旋转需用SO(3)复合；该表达不是逐字复刻原论文更新律。需要小更新、平滑、失败回退/接受检查，避免将几何可达性投影误当闭环收敛保证。此处回退指离线拒收一次更新，不是在线回退OSC。

机器人冗余构型和物体state不能直接作为六维末端误差相加；若完整state不改善，应转入具有显式state目标的优化，而不是继续调末端学习增益。

## 2. I2RLC：接触示范重构后再训练的直接先例

**Refinement of Accelerated Demonstrations via Incremental Iterative Reference Learning Control for Fast Contact-Rich Imitation Learning**，Koki Yamane等，2026-04-18 arXiv预印本，2604.16850。

在擦拭、插孔等实机接触任务中，反复执行并利用实际轨迹误差修正参考，再用修正后数据训练ACT。它说明参考重构能够用于后续模仿学习，不只是轨迹控制演示。原工作包含逐级加速；本项目固定原速即可借鉴参考更新部分，不需要照搬速度调度。其末端及空间相似性评价不等于本项目要求的逐时刻完整state复现。

来源：[全文III-C、III-D与实验](https://arxiv.org/html/2604.16850v1)。本次未核实到官方算法代码；不能表述为已获开源即用方案或已确认会议录用。

## 3. PhysicsGen：完整state目标最接近的接触数据生成先例

**Physics-Driven Data Generation for Contact-Rich Manipulation via Trajectory Optimization，RSS 2025**，Lujie Yang等。

用示范初始化轨迹，在物理仿真中优化机器人及物体轨迹，采用CEM求解，目标包含state跟踪、控制输入及终端代价，对物体跟踪赋更高权重。工作包含双Panda等接触操作，仿真主要使用Drake，并非LIBERO。论文的目标允许机器人运动为物体成功而调整，不是要求原机器人全部state逐位复现。

来源：[RSS正式记录](https://www.roboticsproceedings.org/rss21/p053.html)、[全文IV-B及附录X-A](https://arxiv.org/html/2502.20382v2)、[作者项目页](https://lujieyang.github.io/physicsgen/)。核对时项目页仍标注Code (Coming Soon)，不可声称整套代码可直接下载运行。

**本项目适配建议：** 保留现有LIBERO/IPOPT执行链作为真实候选评价器，借鉴其“示范初始化＋state代价＋采样优化”的结构，不迁移到Drake，也不换为直接力矩/关节动作绕开IPOPT。

## 4. MuJoCo MPC / Predictive Sampling：已有官方开源的优化器参考

**Predictive Sampling: Real-time Behaviour Synthesis with MuJoCo**，Taylor Howell等，2022。MJPC包含Predictive Sampling、iLQG、梯度下降等shooting规划器，有官方仓库。采样法的核心是给已有动作轨迹增加扰动、真实前向模拟、按代价保留更好候选，不依赖对接触求解析导数。

来源：[论文](https://arxiv.org/html/2212.00541)、[官方仓库](https://github.com/google-deepmind/mujoco_mpc)、[官方Python采样实现](https://github.com/google-deepmind/mujoco_mpc/blob/main/python/mujoco_mpc/mjx/predictive_sampling.py)。官方Python示例直接把动作写入MJX ctrl；不能在不改rollout的情况下当作本项目的参考动作重构器。cyipopt/robosuite也不能直接塞入JAX jit/vmap。

**本项目适配建议：** 优先复用算法结构和低维参考修正参数化，直接调用现有Python客户端做候选rollout；只有证明计算瓶颈需要时，才考虑更复杂的并行实现。任何“实时”结论都需重新计入内层IPOPT成本；离线数据重构无需声称硬实时。

## 5. 其他相关路线为何不是当前首选

- [BCO，IJCAI 2018](https://www.ijcai.org/proceedings/2018/0687.pdf)：通过目标机器人交互学习逆动力学，从state转移推断动作。方向相关，但我们已经有可调用的精确执行环境，当前没有必要先训练另一个网络；其单步逆模型也不自动解决长程接触误差、累积参考和控制器内部状态。
- [MimicGen，CoRL 2023](https://mimicgen.github.io/)：适合复用示范生成不同场景的数据，但主要目的不是在同场景同时间复现原完整state。本次不把它列作该问题的直接算法替代。
- [van den Berg等，ICRA 2010](https://rll.eecs.berkeley.edu/surgical/icra10/)：较早使用示范和迭代学习提高实际轨迹执行表现的实机先例；现代IRLC更贴近接触及参考接口。

## 本项目应采用的优化建模

物理state记为`s`；另将独立累积参考、控制器历史/夹爪内部积分状态、求解器必要历史等记为`h`。闭环状态为`z=(s,h)`。用`F`表示完整的“参考累积→IPOPT→B10→25个原生物理步”，不是仅调用`mj_step`的裸动力学。

```text
min over reconstructed arm actions a[0:T-1]:
  sum_t state_distance(s[t], source_s[t])
        + lambda * correction_size(a[t], initial_a[t])
        + mu * action_roughness(a, t)
  + terminal_state_distance(s[T], source_s[T])

subject to:
  z[t+1] = F(z[t], a[t], source_gripper[t])
  identical initial condition and 50ms command grid
  strict IPOPT feasible; original joint/physics constraints
```

`state_distance`需按单位和任务意义分块归一化：关节位置/速度、末端平移和SO(3)旋转、物体/抽屉等相关自由度的位置/速度。不能直接对原state扁平向量做未加权欧氏距离，不能相减四元数或把仿真时间作为优化误差。任务成功单独验收；不使用DTW/时间重对齐掩盖原速时序误差。

离线候选可以从当前rollout状态复制分支比较，但必须同时恢复`h`，不只恢复MuJoCo qpos/qvel。不能每个样本都强制从原示范下一state开始然后拼接成“闭环成功”。最终验收必须从单一初始状态完整执行。优化器可在影子环境试动作，但正式客户端只读最终保存动作，不读源未来state。

## 对现有10/11方案的判断与建议顺序

现有全局动作意图投影＋下一state构型正则是很好的**初始化**，因为原速11条中10条成功且无拒解。但它的目标是逐帧几何拟合，并未直接最小化仿真后的完整state误差。成功样本的末端均值1.66mm不能代替q/dq及物体state评价；失败长任务也必须纳入总体代价和报告。

建议先做两种冻结算法的小范围对照：

1. 原速投影初始化＋IRLC参考更新，作为最小工程成本的基线；统一应用于全部示范，按实际state指标决定是否接受更新。
2. 相同初始化＋参考空间短窗CEM/Predictive Sampling，直接评价完整闭环state代价。只优化前六维参考修正，固定夹爪，使用少量修正节点并插值，不立即把整段每一维都设为独立搜索变量。

IRLC本身仅针对末端误差时不能宣称实现完整state最优；若只有一种方法要围绕用户的严格目标实现，选第二种建模，第一种作低成本比较或预热。候选评价必须经过同一IPOPT链，不能用更快的直接关节/力矩仿真代替后再期待等效。

两者额外开销主要在离线重构试录。最终保存动作长度仍为原N-1，训练数据和部署不需要加倍，也不要求部署时运行外层搜索。内层求解拒绝、计算预算及接触分支变化仍需按本机结果评价；无现有论文保证所有原state在更受限接口下都可精确复现。
