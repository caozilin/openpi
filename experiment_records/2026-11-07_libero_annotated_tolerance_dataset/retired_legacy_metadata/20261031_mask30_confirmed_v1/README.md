# LIBERO-40：确认mask的30°闭环对照

本批次对既有400条成功重构轨迹各执行一次，不做新的动作修正、候选搜索或成功筛选。原始400条成功基线保留在 ../20261025_success10_v1。

- `confirmed_masks.json`：2026-10-31用户最终确认的40任务mask；轴顺序xF/yF/zF，正负边界统一30°，ENG/DIS严格。
- `plan.json`：400条来源、相同初态/模型/动作哈希、逐示范语义身份映射；Goal/03打开抽屉至接近抓碗均为PRE010，POST000；Goal/05全零。
- `episodes/{suite}/{task_id}/{demo}/`：所有结果的视频、逐帧轨迹、物理状态、500Hz伺服采样、求解诊断。失败结果作为对照保留，不加入原成功训练集。
- `records/`、`summary.json`：逐条结果与任务成功计数；完整可行执行且最终原生谓词成立才算成功。
- `validation.json`：完成后的400条媒体解码、动作/初态/执行链验证、关节运动、各阶段释放、求解耗时和拒绝诊断。
- `off_check_records/`：容差关闭路径的独立复现，物理状态及500Hz伺服记录与原基线逐项相同。
- `source_snapshot_manifest.json`：177份执行源码/资产快照与哈希；共享代码实际取既有20261025冻结运行时，保持与基线一致。

执行链直接复用 LiberoTaskTolSession → 默认IPOPT → 共享关节插值/JointReferenceTracker → Panda位置伺服；原生LIBERO500Hz物理、20Hz动作、原夹爪和原速。默认IPOPT设置和目标函数不变，不包含原始关节轨迹惩罚。

语义子任务身份由已保存的配对基线计划提供，重复抓取保留对象身份。APP/ENG/EXEC/DIS由当前手指反馈、20Hz/连续3帧稳定的共享分类器判定；不是VLA任务识别实验。POST000禁止该阶段新增释放，允许继承PRE已接受偏差。30°是可用边界上限，不是强制偏转30°，也不是已验证的安全幅值。

可视化使用现有8768服务：左原始存储图像，中无容差重构，右30°容差闭环。共同时间轴按帧号对齐，提前停止时右侧保留最后动作前观测并标明停止。原始视频保留额外终帧。轨迹图默认是中间基线，容差逐帧详情和CSV单独展示。

刷新页面数据：
```
python3 examples/libero/build_reconstructed_rollout_viewer.py --tolerance-run data/libero_tasktol_joint_servo/20261031_mask30_confirmed_v1
```

恢复未完成回放须使用既有LIBERO虚拟环境及EGL设置，执行 `experiments/libero_tasktol_compat/rollout_confirmed_tolerance400.py --resume --workers 6`。程序检查冻结源码哈希，避免中途混用实现。

## 本次完成结果

400/400完成；286成功（71.5%），114失败；其中110完整执行后任务未完成，4次可行性拒绝（均迭代20次，非CPU超时）。Spatial66/100，Object82/100，Goal90/100，Long48/100。

400个容差视频全部逐帧解码通过；同动作、同初态、同模型、默认优化器及关节伺服一致性验证通过。源码哈希在执行结束时保持一致。

求解核心墙钟最大56.835ms；4个pilot首次调用的包装层存在9.8–10.8秒启动阻塞，核心solve仅4.0–5.2ms，详见latency_diagnostics.json。没有CPU预算超时，不代表硬实时20Hz。

启用轴的优化解最大容差越界约4.05e-8度；实际动力学姿态最大越界5.01度，不能把优化可行与实际零误差混为一谈。全零mask的20条均成功，零容差约束路径与直接OFF路径有约1e-9rad量级数值差，接触敏感轨迹的实际状态并非逐比特相同；详见validation.json的zero_mask_bitwise_checks。

## 成功样本的四维运动学比较

见 [配对统计表](successful_kinematics_comparison.md)、successful_kinematics_comparison.json及逐条CSV。同一286条成功示范，完整20Hz优化命令域，按示范等权。速度降低19.95%，加速度降低28.17%，mY降低6.44%，每轨迹最小归一化σmin均值提高23.20%。20条全零mask样本属于数值近似不变，其余266条中速度263条改善、加速度262条改善、mY79条改善、σmin202条改善。
