"""Summarize measured control/contact ordering and same-state experiments."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
r = ROOT/'results'
a = np.load(r/'drift_old_default/physics_trace.npz')
b = np.load(r/'drift_new_default/physics_trace.npz')
arm_error = np.max(np.abs(a['qpos_after'][:,:7]-b['qpos_after'][:,:7]), axis=1)
i = int(np.flatnonzero(arm_error > 1e-8)[0])
force_delta = float(np.max(np.abs(a['constraint_force'][i,:7]-b['constraint_force'][i,:7])))
ctrl_before = float(np.max(np.abs(a['ctrl'][:i+1,:7]-b['ctrl'][:i+1,:7])))
contacts_a = [json.loads(s) for s in (r/'drift_old_default/robot_contacts.jsonl').read_text().splitlines()]
contacts_b = [json.loads(s) for s in (r/'drift_new_default/robot_contacts.jsonl').read_text().splitlines()]
base = np.load(r/'baseline_libero_10_00/observations.npz')
profiles = []
for name,label in [('final_libero_10_00','新版本默认'),('drift_no_native','只关闭 Native CCD'),('drift_no_multiccd','只关闭 MultiCCD'),('drift_legacy_collision','同时关闭两项 CCD 开关')]:
    data = json.loads((r/name/'report.json').read_text())
    z = np.load(r/name/'observations.npz')
    error = np.linalg.norm(base['eef_position']-z['eef_position'], axis=1)
    profiles.append(dict(directory=name, label=label, success=data['task_success'], max_eef_difference_m=float(error.max())))
old = np.load(r/'drift_same_old/same_state.npz')
same = []
for name in ['drift_same_new','drift_same_legacy']:
    new = np.load(r/name/'same_state.npz')
    same.append(dict(directory=name, inputs_identical=all(np.array_equal(old[k],new[k]) for k in ['qpos','qvel','ctrl']),
        bias_force_max_difference=float(np.max(np.abs(old['bias']-new['bias']))),
        actuator_force_max_difference=float(np.max(np.abs(old['actuator']-new['actuator']))),
        arm_constraint_torque_max_difference_nm=float(np.max(np.abs(old['constraint'][:7]-new['constraint'][:7]))),
        arm_acceleration_max_difference_rad_s2=float(np.max(np.abs(old['qacc'][:7]-new['qacc'][:7])))))
source_model_old = np.load(r/'drift_same_old/compiled_model.npz')
source_model_new = np.load(r/'drift_same_new/compiled_model.npz')
physical_parameters = {}
for key in ['body_mass','body_inertia','geom_friction','geom_solref','geom_solimp','geom_margin','geom_gap','dof_damping','dof_armature','actuator_gainprm','actuator_biasprm','actuator_gear','actuator_ctrlrange','actuator_forcerange']:
    physical_parameters[key] = float(np.max(np.abs(source_model_old[key]-source_model_new[key])))
result = dict(conclusion='TaskTol/IPOPT excluded; divergence originates in native contact simulation before OSC command divergence. Exact single underlying MuJoCo implementation change not isolated.',
    first_arm_divergence=dict(physics_trace_index=i, processed_frame=int(a['frame'][i]), substep_in_control_frame=int(i%25), sim_time_s=float(a['time'][i]),
        arm_constraint_torque_difference_nm=force_delta, arm_torque_max_difference_before_and_at_event_nm=ctrl_before,
        old_contacts=contacts_a[i]['after'], new_contacts=contacts_b[i]['after']),
    same_state_experiments=same, compiled_physical_parameter_max_differences=physical_parameters, profiles=profiles,
    optimizer_imported_in_trace_cases={name:json.loads((r/name/'diagnostic.json').read_text())['tasktol_optimizer_imported']
        for name in ['drift_old_default','drift_new_default','drift_no_native','drift_no_multiccd','drift_legacy_collision','drift_same_old','drift_same_new','drift_same_legacy']},
    limitation='CCD switch experiments did not restore the old trajectory. Mesh-local frame representation changes do not by themselves prove shape changes. No optimizer tuning or production environment changes were made.')
(ROOT/'drift_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
rows = '\n'.join('| '+x['label']+' | '+('成功' if x['success'] else '失败')+' | '+format(x['max_eef_difference_m']*1000,'.3f')+' |' for x in profiles)
text = f'''# 长任务回放漂移诊断 · 2026-10-20

**已排除 TaskTol/IPOPT 优化器。** 所有新增诊断均使用 `--skip-tasktol`，没有导入优化器，更没有调用求解或执行容差目标。原生 LIBERO OSC 继续执行完全相同的示范动作。

## 分叉顺序

1. 首次明显的机器人状态分叉位于物理追踪索引 {i}，处理帧 {int(a['frame'][i])} 内的第二个物理子步（索引从 0 开始）。接触对为 `orange_juice_1_g1` 与 `gripper0_finger1_collision`，即手指先碰到旁边的橙汁盒。
2. 截至这个子步，双方关节力矩最大差仅 {ctrl_before:.3g} N·m；先出现的关节约束力矩差为 {force_delta:.6f} N·m。力矩分叉发生在下一个物理子步，是 OSC 对已经不同的状态作出的反馈。
3. 第 52 个处理帧的末端差异约 1.17 mm，第 53 帧约 4.62 mm。完整 285 步示范的最大末端差异约 60 mm；它不是某一步施加了 6 cm 偏移。
4. 后续仍按原示范继续运动，没有针对抓取失败重新规划；旧环境抓住并搬运目标，新环境未完成相同抓取，轨迹差异随后扩大。

## 同状态实验

将旧环境在首次分叉处的完整 qpos、qvel 和 ctrl 原样放入两个引擎的独立 MjData 副本，统一清零外力和 warmstart，只调用原生 `mj_forward`：

- 输入数组逐元素完全相同；偏置力、执行器力也完全相同。
- 新默认环境的最大关节接触约束力矩差为 {same[0]['arm_constraint_torque_max_difference_nm']:.8f} N·m。
- 同时关闭两项 CCD 开关后，仍有 {same[1]['arm_constraint_torque_max_difference_nm']:.8f} N·m 的差异。

这将差异定位到原生接触仿真链，而非策略动作、优化器或 OSC 首先输出了不同控制量。尚未将底层版本差异收敛到一项具体算法变更。

## 单因素/组合开关验证

| 新环境设置 | 原长任务是否成功 | 相对旧环境最大末端差 mm |
|---|---|---:|
{rows}

MuJoCo 3.2.3 实际默认关闭 Native CCD/MultiCCD，3.10 默认开启；但关闭开关并不足以恢复旧版本结果。因此不能把某一项开关当成已确认根因或直接作为修复。

## 已排除与记录

- 同一 BDDL、初始状态、动作、20 Hz、夹爪命令和资产；质量矩阵桥接使用原生公式。
- 已比较的质量、惯性、摩擦、solref/solimp、关节阻尼、执行器增益等编译参数一致。
- 编译后部分 mesh 的局部 geom_quat/geom_size 表示不同，但恢复到同一父坐标系后，顶点云近乎一致；不能把局部表示差异直接解释为物体形状被改大。
- 原始数据：[机器可读结论](drift_summary.json)，物理子步追踪在 `results/drift_old_default/physics_trace.npz` 和 `results/drift_new_default/physics_trace.npz`；接触对逐步记录在相邻的 `robot_contacts.jsonl`。
- 所有新实验视频、初始/中间/末帧及模型选项均留存在对应 `results/drift_*` 目录。

后续应优先固定 LIBERO 物理执行语义，再接入共享 TaskTol 核心。当前没有通过调优化器参数、改动作或修改正式环境来掩盖漂移。
'''
(ROOT/'DRIFT_DIAGNOSIS.md').write_text(text)
assert all(not v for v in result['optimizer_imported_in_trace_cases'].values())
assert all(s['inputs_identical'] and s['bias_force_max_difference']==0 and s['actuator_force_max_difference']==0 for s in same)
assert ctrl_before < 1e-8 and force_delta > 1e-3
print(json.dumps({'first_arm_divergence':result['first_arm_divergence'], 'same_state':same, 'profile_count':len(profiles)}, ensure_ascii=False))
