"""Explain small APP Ry bounds using existing observations and raw trials only."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math

HERE=Path(__file__).resolve().parent
ROOT=Path('/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107')
PLAN=Path('/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1/plan.json')
sources=[]


def read(path, expected=None):
    content=Path(path).read_bytes()
    sha=hashlib.sha256(content).hexdigest()
    if expected is not None:
        assert sha==expected,path
    sources.append(dict(path=str(path),sha256=sha))
    return json.loads(content)


def raw_detail(row, policy):
    raw=read(row['raw_trial_path'],row['raw_trial_sha256'])
    e=raw['result']['evaluation']
    assert e['physics_steps']==0 and not e['objects_considered']
    result=dict(query_index=row['query_index'],initial=row['query_index']<12,
        pre_ry_deg=row['actual_pre_caps_deg'][2:4],post_caps_deg=row['actual_post_caps_deg'],
        success=row['success'],reason=row['reason'],solver_pass=row['solver_pass'],
        clearance_pass=row['clearance_pass'],observed_velocity_rad_s=row['observed_velocity_rad_s'],
        qualified_velocity_rad_s=row['velocity_rad_s'],
        raw_trial_path=row['raw_trial_path'],raw_trial_sha256=row['raw_trial_sha256'],
        zero_nominal_reference=e['zero_tolerance_nominal_reference'],
        stage_handoff_policy=e['stage_handoff_policy'])
    pre=e['sqp_subtask_stage_release_extrema_deg'][0]['pregrasp']
    result['actual_pre_ry_peak_deg']=[pre['negative'][1],pre['positive'][1]]
    if row['trajectory_complete']:
        samples=e['robot_table_clearance']['samples']
        assert len(samples)==len(policy['required_clearance_m'])
        margins=[s['distance_m']-r+policy['epsilon_m'] for s,r in zip(samples,policy['required_clearance_m'])]
        i=min(range(len(margins)),key=margins.__getitem__)
        assert math.isclose(margins[i],row['clearance_margin_m'],abs_tol=1e-12)
        assert policy['source_frame_indices'][i]==row['clearance_bottleneck_source_frame']
        result.update(clearance_margin_mm=margins[i]*1000,
            bottleneck_source_frame=policy['source_frame_indices'][i],bottleneck_phase=samples[i]['phase'],
            baseline_at_bottleneck_mm=policy['baseline_clearance_m'][i]*1000,
            required_at_bottleneck_mm=policy['required_clearance_m'][i]*1000,
            candidate_at_bottleneck_mm=samples[i]['distance_m']*1000,
            minimum_clearance_mm=e['robot_table_clearance']['distribution']['minimum_m']*1000,
            absolute_minimum_pair=e['robot_table_clearance']['minimum'])
    return result


def main():
    plan=read(PLAN)
    audit=read(ROOT/'annotation_independent_final_audit.json',plan['audit_sha256'])
    entries=sorted([e for e in plan['entries'] if e['task']['suite']=='libero_spatial'
                    and e['task']['task_id']==0],key=lambda e:int(e['job']['demo'][5:]))
    tasks=[]
    for entry in entries:
        folder=Path(entry['labels']).parent
        labels=read(entry['labels'],entry['labels_sha256'])
        bound=next(r for r in audit['results'] if r['suite']=='libero_spatial' and r['task_id']==0
                   and r['demo']==entry['job']['demo'])
        a=read(folder/'subtask_00/annotation.json')
        obs=a['observations'];best=a['best']
        assert a['complete'] and a['additional_queries']==16
        assert len(obs)==28 and a['design']['unique_initial_count']==12
        for o in obs:
            assert bound['raw_trial_sha256'][o['raw_trial_path']]==o['raw_trial_sha256']
        feasible=[o for o in obs if o['success']]
        assert best['query_index']==min(feasible,key=lambda o:(o['velocity_rad_s'],o['index']))['query_index']
        pre=best['actual_pre_caps_deg'][2:4]
        assert pre==labels['subtasks'][0]['pregrasp'][2:4]
        small=[j for j in range(2) if pre[j]<=1.]
        alternatives=[]
        for j in small:
            good=[o for o in feasible if o['actual_pre_caps_deg'][2+j]>1.]
            alt=min(good,key=lambda o:(o['velocity_rad_s'],o['index'])) if good else None
            alternatives.append(dict(side='negative' if j==0 else 'positive',larger_feasible_count=len(good),
               best_larger_query=alt['query_index'] if alt else None,
               best_larger_velocity=alt['velocity_rad_s'] if alt else None,
               selected_gain_over_larger_pct=(100*(alt['velocity_rad_s']-best['velocity_rad_s'])/alt['velocity_rad_s']) if alt else None,
               best_larger_pre_ry_deg=alt['actual_pre_caps_deg'][2:4] if alt else None))
        needed=set(range(28)) if small or entry['job']['demo']=='demo_0' else {0,1,2,3,best['query_index']}
        details={str(i):raw_detail(obs[i],a['clearance_policy']) for i in sorted(needed)}
        zero=details['0'];full=details['1'];selected=details[str(best['query_index'])]
        assert zero['zero_nominal_reference'] and zero['stage_handoff_policy']=='nominal_without_stage_offset'
        for i,v in details.items():
            if int(i)!=0:
                assert not v['zero_nominal_reference'] and v['stage_handoff_policy']=='inherit_accepted_handoff'
        tasks.append(dict(demo=entry['job']['demo'],selected_pre_ry_deg=pre,
            selected_actual_pre_peak_deg=selected['actual_pre_ry_peak_deg'],
            small_sides=[v['side'] for v in alternatives],both_small=len(small)==2,
            observations=28,initial=12,adaptive=16,feasible=len(feasible),
            outcome_counts=dict(Counter(o['reason'] for o in obs)),
            selected_query=best['query_index'],selected_velocity=best['velocity_rad_s'],
            full_pass=full['success'],full_pre_only_pass=details['2']['success'],
            full_gain_pct=100*(obs[1]['observed_velocity_rad_s']-best['velocity_rad_s'])/obs[1]['observed_velocity_rad_s'],
            alternatives_for_small_sides=alternatives,details=details,
            source_annotation=str(folder/'subtask_00/annotation.json')))
    small=[t for t in tasks if t['small_sides']]
    no_larger=[t for t in small if not any(v['larger_feasible_count'] for v in t['alternatives_for_small_sides'])]
    has_larger=[t for t in small if any(v['larger_feasible_count'] for v in t['alternatives_for_small_sides'])]
    summary=dict(n=47,threshold_deg=1.,small_either_side=len(small),both_small=sum(t['both_small'] for t in tasks),
       one_side_small=sum(len(t['small_sides'])==1 for t in tasks),
       no_feasible_larger_on_small_sides=[t['demo'] for t in no_larger],
       feasible_larger_on_small_sides=[t['demo'] for t in has_larger],
       only_zero_feasible=[t['demo'] for t in tasks if t['feasible']==1 and t['selected_query']==0],
       full_pass_among_small=[t['demo'] for t in small if t['full_pass']],
       raw_trials_read=sum(len(t['details']) for t in tasks),
       observation_metadata_bound_to_historical_audit=sum(t['observations'] for t in tasks),
       original_run=str(ROOT),new_search_or_rollout=False)
    (HERE/'pre_ry_diagnosis.json').write_text(json.dumps(dict(summary=summary,tasks=tasks),indent=2)+'\n')
    (HERE/'pre_ry_diagnosis_sources.json').write_text(json.dumps(dict(files=sources),indent=2)+'\n')
    lines=['# APP Ry接近零：原搜索过程诊断','',
       '状态：2026-11-09，只读诊断和来源/逐帧净空复算完成。沿用同一47条demo与2026-11-07冻结协议，无新训练、搜索或回放。近零定义为某侧边界≤1°；每条有12个结构化初始点+16个追加点，联合搜索APP 2维与EXEC 6维。候选不是逐轴独立求最大可行值。','',
       f"47条中{len(small)}条至少一侧近零，其中4条双侧近零，11条仅单侧。",'',
       '## 验收规则与选择过程','',
       '每帧要求 c_candidate(t) ≥ min(5mm,c_zero(t))−0.1mm，全程通过才可进入速度排名。完整可解与相对净空分别验收，不能只看整段最小净空。选中行严格按(velocity_rad_s, observation index)取最小；没有为大容差加偏好，也没有对微小速度差设置等价阈值。最终将整组候选的APP/EXEC边界直接写成标签。','',
       '## 全部近零样本','',
       '|demo|选中APP Ry−/+°|实际APP峰值−/+°|合格/查询|近零侧有>1°的合格候选|选中较该侧最佳较大候选降速|',
       '|---|---:|---:|---:|---|---:|']
    for t in small:
        pair=lambda x:' / '.join(f'{v:.4f}' for v in x)
        alt=t['alternatives_for_small_sides']
        verdict='；'.join(v['side']+': '+str(v['larger_feasible_count']) for v in alt)
        gain=' / '.join('—' if v['selected_gain_over_larger_pct'] is None else f"{v['selected_gain_over_larger_pct']:.8f}%" for v in alt)
        lines.append(f"|{t['demo']}|{pair(t['selected_pre_ry_deg'])}|{pair(t['selected_actual_pre_peak_deg'])}|{t['feasible']}/28|{verdict}|{gain}|")
    lines+=['',
      '有该侧较大合格候选的9条：'+', '.join(summary['feasible_larger_on_small_sides'])+'。近零是现有候选中的速度选择；其中也有EXEC同时改变的联合配置，不能全部当作仅改变一个APP方向的因果比较。', '',
      '未找到该侧较大合格候选的6条：'+', '.join(summary['no_feasible_larger_on_small_sides'])+'。这表示28次记录中没有找到，不代表数学上不存在。','',
      '## 双侧近零的4条：约束过滤','',
      '|demo|全开APP实际峰值−/+°|全开净空余量mm|全开仅APP净空余量mm|最终APP边界−/+°|合格数|',
      '|---|---:|---:|---:|---:|---:|']
    for t in tasks:
        if not t['both_small']:continue
        full=t['details']['1'];preonly=t['details']['2']
        pair=lambda x:' / '.join(f'{v:.4f}' for v in x)
        lines.append(f"|{t['demo']}|{pair(full['actual_pre_ry_peak_deg'])}|{full['clearance_margin_mm']:.6f}|{preonly['clearance_margin_mm']:.6f}|{pair(t['selected_pre_ry_deg'])}|{t['feasible']}/28|")
    lines += ['',
      'demo44/45/49：第0个全零候选合格，另外27个均完整可解但相对净空不合格，零是原第0候选被保留。demo8：只有全零和第19查询合格，最终选择第19，APP=0.046718/0.129195°，实际峰值几乎贴住两侧上限；速度1.034235→0.969682rad/s，降低6.24%。','',
      'demo8全开在源frame36（grasp）零参考净空8.202539mm，因此要求5mm；候选3.978892mm，加0.1mm容忍后余量−0.921108mm，被淘汰。其整段最小距离1.233774mm反而高于零参考1.104836mm，但不是发生于同一帧，不能据此判为合格。','',
      'demo44仅APP全开/EXEC严格零，在frame34（pregrasp）要求5mm，实际0.360748mm，余量−4.539252mm。demo45仅APP全开在frame40（grasp）要求3.113623mm，实际2.902561mm，余量−0.111062mm。demo49仅APP全开在frame71（postgrasp）要求3.164263mm，实际2.746590mm，余量−0.317672mm。因此APP修改可以在后续阶段触发整段淘汰。','',
      '## 单侧近零不等于整个Ry被禁止','',
      'demo9：全开APP±30°实际峰值−2.352373/+10.278632°，速度0.713414589rad/s；选中APP−30°/+0°实际−12.764383/+0°，速度0.709688252rad/s。全开与选中都合格。demo21、29也从主要正向释放切换为负向释放；demo38、46相反，保留正侧而关闭未被使用的负侧。','',
      '可固定其它搜索边界的直接对照：demo21第1/5查询只改APP正侧30→0°，速度0.700363042→0.695953635rad/s；demo29同样只改这一侧，0.665891028→0.658029781rad/s。它们的全开候选可行，正侧归零是有限预算控制回放的速度选择。','',
      'demo38最终与全开的速度只差约0.00000576%，实际正侧峰值11.66590195→11.66589534°，负侧原本就没有使用。选择器没有微小收益等价处理，仍会把带零边界的候选写为标签；这个零不提供该方向物理不可释放的证据。','',
      '## 全零参考模式的额外差异','',
      '冻结runner对完整trial判断是否所有边界为零：全零时zero_tolerance_nominal_reference=True、stage_handoff_policy=nominal_without_stage_offset；任一APP或EXEC边界非零则False、inherit_accepted_handoff。初始/阶段衔接捕获的参考分支不同。所有读取的原始trial都验证了该记录字段。','',
      '因此全零与任意非零的比较还包含参考策略变化，不能看成在完全相同参考下把容差连续缩到0的纯扫描。demo8/44/45/49连APP固定0、只开EXEC的第3初始候选也全部因净空失败。这进一步说明全局零回退不能单独解释成APP Ry本身无自由度。未新跑同参考对照，不能量化此模式差异贡献多少。','',
      '零基线还可能有负净空：demo44最小−28.459825mm，45为−12.072487mm，49为−13.273985mm。规则允许保留零基线已有负净空；这里只能解释为离线几何相对验收，不能把它等同于原物理场景的绝对碰撞安全性。','',
      '## 正常对照与结论边界','',
      'demo0：28/28候选合格，APP30/30°实际0/11.467058°；速度0.781011551→0.599242237rad/s。它有充足的相对净空余量，搜索可以保留较大边界并获得释放收益。与近零组的关键差异可在完整轨迹逐帧净空、释放方向及候选排名中直接核对，不能只用初始摆放图片判断。','',
      '结论：近零标签混合了整段净空过滤、单侧运动偏好、接近等价候选的数值排名以及全零参考模式差异。现有标签表示已搜索到的速度最优合格候选，不足以识别任务Ry的唯一或最大几何容差。所有候选和失败均保留，无补试和成败筛选。','',
      '## 本地源数据与复核','',
      '- 原冻结标注/请求/选样/审计/日志根：`'+str(ROOT)+'`；每demo来源`libero_spatial/task_00/demo_N/subtask_00/annotation.json`、`evaluations/<hash>/trial.json`及`zero_baselines/subtask_00/evaluations/<hash>/trial.json`；配置`annotation_request.json`、`authored_tolerances_deg.json`，日志`annotation.log`、`pipeline.log`与每demo的`search.log`；原独立审计`annotation_independent_final_audit.json`。','',
      '- 绑定标签的物理数据plan：`'+str(PLAN)+'`。本次未从物理成败选择样本；原物理视频与逐回合记录地址沿用本目录README。无新视频、远端运行或模型版本。','',
      '- 本次脚本：`'+str(HERE/'diagnose_pre_ry.py')+'`；全47条概要及近零组完整查询过程：`'+str(HERE/'pre_ry_diagnosis.json')+'`；逐份原文件绝对路径/SHA：`'+str(HERE/'pre_ry_diagnosis_sources.json')+'`；验证：`'+str(HERE/'pre_ry_diagnosis_verification.json')+'`。只保存派生统计，未复制原始轨迹/视频。','',
      '- 冻结实现：`'+str(ROOT/'source_snapshot/label_franka_nominal_dsp.py')+'`的search_subtask；`'+str(ROOT/'source_snapshot/tasktol/tolerance/clearance_policy.py')+'`的relative_feedback；`'+str(ROOT/'source_snapshot/tasktol/counterfactual/runner.py')+'`的zero_tolerance_nominal_reference；`'+str(ROOT/'source_snapshot/tasktol/tolerance/release_state.py')+'`的begin_stage。','']
    (HERE/'PRE_RY_DIAGNOSIS.md').write_text('\n'.join(lines))
    verified=dict(complete=True,metadata_observations_bound=summary['observation_metadata_bound_to_historical_audit'],
        raw_trials_sha_checked=summary['raw_trials_read'],n=47,
        verified=['all best selections','12+16 query counts','label/plan/audit SHA binding',
                  'per-frame relative clearance and bottleneck','zero/nonzero handoff policy'],
        original_data_modified=False,new_search_or_rollout=False)
    (HERE/'pre_ry_diagnosis_verification.json').write_text(json.dumps(verified,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
