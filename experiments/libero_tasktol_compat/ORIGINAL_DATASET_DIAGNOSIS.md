# 原始 LIBERO 示范采样的目标可达性诊断

原始示范确实包含不可达的瞬时 OSC 位姿目标；但当前三个失败采样的来源不同。不可达的是虚拟目标，并非原始实际关节轨迹。以下只诊断三条选定示范，不是整个数据集的比例统计。

| 任务 | 原始demo/采样（从0计数） | 对应重录采样 | 原始目标结论 | 证据 |
|---|---|---:|---|---|
| Spatial04，抽屉黑碗放到盘子 | demo_22 / 105 | 105 | 可达 | 14个初值中10个找到合格逆解，独立MuJoCo FK验证通过 |
| Goal02，酒瓶放到柜顶 | demo_10 / 75 | 75 | 固定姿态下不可达 | 所需最小肩腕距离724.187mm，大于几何上界719.354mm |
| Long08，两把摩卡壶放到炉灶 | demo_17 / 373 | 371 | 固定姿态下不可达 | 所需最小肩腕距离729.637mm，大于几何上界719.354mm |

这纠正了把重录结果直接归因于原始数据的范围错误。Spatial04/105原始目标可达，重录后对应目标不可达。该采样重录状态相对原始状态有6.169mm末端位置差、约1.666度姿态差，因此即使动作完全相同，OSC反馈相对目标也不同。重录目标违反同一几何上界约0.435mm；扣除验收容差影响仍违反约0.367mm。

## 原始数据和目标重建

直接读取原始HDF5的states、actions、obs、env_args和demo保存的model_file XML，仅替换失效资产地址，几何参数不改。使用原配置和原生OperationalSpaceController.set_goal，从原始pre-action状态重建目标。三处采样均有非零旋转动作，目标不依赖未知的初始零旋转保持历史。

动作与重录文件逐元素一致；Long08重录跳过两个no-op，所以371对应原始373。原始obs/joint_states[j-1]与states[j]关节角逐元素一致，确认state是动作前、obs是动作后。

原始末端观测保留最后一个2ms物理步的运动学缓存：将state关节角回退dt*qvel后，三条完整示范的位置/姿态观测均匹配到约1e-15。原生OSC设目标前会forward，因此必须使用state即时FK，不能直接用缓存的obs/ee_pos加动作。

obs/ee_ori是hand body姿态，OSC内部是control site姿态，已作固定旋转转换，没有把约90度坐标差当成跟踪误差。原始和当前模型同关节角的FK位置差小于1e-15m，固定姿态偏移约2.18e-7rad也已转换。

## 与优化器分支无关的几何证书

S为joint2原点、E为joint4原点、W为joint6原点。由原始模型刚体链：

```text
|SE| = sqrt(0.316² + 0.0825²) = 0.326591870689 m
|EW| = sqrt(0.384² + 0.0825²) = 0.392762332715 m
|SW| <= |SE| + |EW|          = 0.719354203404 m
```

目标位置与姿态固定后，joint7原点H和link7姿态R7确定。把joint7放宽为可以转完整一圈，W落在半径0.088m的圆上。计算S到圆的最小距离d_min；若这个最小值已经超过0.719354m，任何joint7角度、任何IK分支都无法满足目标。原有关节限位只会进一步缩小可达集合。

令圆心相对S的轴向分量为z、平面内距离为rho：

```text
d_min = sqrt((rho - 0.088)² + z²)
```

Goal02/75违反上界4.833mm，Long08/373违反10.282mm。生产验收为位置各分量1e-5m、旋转误差各分量1e-4rad；把这些全部向有利于可达方向放宽，最多改变该界约0.068mm。两处仍违反4.765mm、10.214mm，因此结论不依赖“多初值没找到解”。

Spatial04/105的可达证据是实际关节解：第一个解位置最大分量误差约4.44e-12m、旋转最大分量误差约1.12e-11rad，且满足全部关节限位。完整7维关节解见audit.json。

每个模型100个随机关节构型验证几何链恒等式，误差小于5e-15；每个模型另100个接近伸展边界、但仍在生产验收范围内的扰动目标，没有误判不可达。Spatial04的10个可行解另以原生MuJoCo FK独立验收。

## 原始实际运动与邻近采样

原文件动作后观测相对该动作目标的误差：

| 采样 | 位置误差范数 | 姿态误差角 |
|---|---:|---:|
| Spatial04/105 | 37.028mm | 4.595度 |
| Goal02/75 | 24.307mm | 1.745度 |
| Long08/373 | 24.800mm | 2.611度 |

这些观测保留上述一个物理步的缓存时序，不冒充最终qpos的精确FK。原始运动没有逐帧精确达到所有虚拟目标。三个demo文件末尾均标注done=1/reward=1；这是原有数据集标签，并非本次新增的成功复测。

沿三条原始示范检查，该几何证书还检出：Spatial04/demo_22原始136–148（13帧）；Goal02/demo_10原始74–87（14帧）；Long08/demo_17原始371–393（23帧）。这只是该必要条件能证明不可达的帧数，其他帧未违反此条件不等于已证明可达。

## 留存与复现

脚本：[audit_original_osc_samples.py](audit_original_osc_samples.py)。用现有锁定LIBERO解释器，设置MUJOCO_GL=egl、OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1、PYTHONDONTWRITEBYTECODE=1运行；输出目录须尚不存在。

正式结果：[original_dataset_target_audit_v2](results/original_dataset_target_audit_v2)。每任务保存原始XML、仅替换资产地址的XML、原始图片及便于查看的180度旋转图片、全部原始目标序列、几何证书、每个初值及候选、采样与模型哈希。summary.json含完整结果，validation.json含额外验证。没有物理步进，没有修改源HDF5或执行控制器。

动作前原始图片：[Spatial04的obs104](results/original_dataset_target_audit_v2/libero_spatial/04/raw_obs_0104_display.png)、[Goal02的obs74](results/original_dataset_target_audit_v2/libero_goal/02/raw_obs_0074_display.png)、[Long08的obs372](results/original_dataset_target_audit_v2/libero_10/08/raw_obs_0372_display.png)。

旧original_dataset_target_audit_v1的观测姿态对照混用hand/site坐标，只供调试，正式结论以v2为准；incomplete目录都是未通过验证的中间运行。
