"""Reselect existing candidates without table-clearance acceptance; no rollouts."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics as st

HERE=Path(__file__).resolve().parent
ROOT=Path('/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107')
PLAN=Path('/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1/plan.json')
INDICES=[2,3,6,7,8,9,10,11]
UNITS=['APP Ry-','APP Ry+','EXEC Rx-','EXEC Rx+','EXEC Ry-','EXEC Ry+','EXEC Rz-','EXEC Rz+']
sources=[]


def read(path,expected=None):
    b=Path(path).read_bytes();sha=hashlib.sha256(b).hexdigest()
    if expected is not None:assert sha==expected,path
    sources.append(dict(path=str(path),sha256=sha))
    return json.loads(b)


def metric(row):
    raw=read(row['raw_trial_path'],row['raw_trial_sha256'])
    result=raw['result'];e=result['evaluation'];k=result['trajectory_kinematics']
    assert e['trajectory_complete'] and e['kinematic_feasible_without_table']
    assert e['physics_steps']==0 and not e['objects_considered']
    assert math.isclose(k['mean_joint_velocity_rad_s'],row['observed_velocity_rad_s'],abs_tol=1e-12)
    full=row['actual_pre_caps_deg']+row['actual_post_caps_deg']
    ext=e['sqp_subtask_stage_release_extrema_deg'][0]
    all_peaks=[ext[phase][side][axis] for phase in ['pregrasp','postgrasp']
               for axis in range(3) for side in ['negative','positive']]
    return dict(query_index=row['query_index'],caps_deg=[full[j] for j in INDICES],
        actual_peak_deg=[all_peaks[j] for j in INDICES],
        mean_joint_velocity_rad_s=k['mean_joint_velocity_rad_s'],
        original_clearance_pass=row['clearance_pass'],
        original_reason=row['reason'],clearance_margin_mm=row['clearance_margin_m']*1000,
        raw_trial=str(row['raw_trial_path']),raw_sha256=row['raw_trial_sha256'],
        stage_handoff_policy=e['stage_handoff_policy'])


def aggregate(rows):
    old=st.fmean(t['old']['mean_joint_velocity_rad_s'] for t in rows)
    new=st.fmean(t['new']['mean_joint_velocity_rad_s'] for t in rows)
    zero=st.fmean(t['zero']['mean_joint_velocity_rad_s'] for t in rows)
    return dict(n=len(rows),old_mean_velocity_rad_s=old,new_mean_velocity_rad_s=new,
        zero_mean_velocity_rad_s=zero,additional_velocity_reduction_pct=100*(1-new/old),
        old_reduction_vs_zero_pct=100*(1-old/zero),new_reduction_vs_zero_pct=100*(1-new/zero),
        mean_per_demo_additional_reduction_pct=st.fmean(t['additional_reduction_pct'] for t in rows),
        median_per_demo_additional_reduction_pct=st.median(t['additional_reduction_pct'] for t in rows))


def main():
    plan=read(PLAN)
    audit=read(ROOT/'annotation_independent_final_audit.json',plan['audit_sha256'])
    entries=sorted([e for e in plan['entries'] if e['task']['suite']=='libero_spatial' and e['task']['task_id']==0],
                   key=lambda e:int(e['job']['demo'][5:]))
    pairs=[];outcomes=Counter();eligible_total=0
    for entry in entries:
        path=Path(entry['labels']).parent/'subtask_00/annotation.json'
        a=read(path);labels=read(entry['labels'],entry['labels_sha256'])
        assert a['complete'] and len(a['observations'])==28 and a['additional_queries']==16
        bound=next(r for r in audit['results'] if r['suite']=='libero_spatial' and r['task_id']==0
                   and r['demo']==entry['job']['demo'])
        obs=a['observations']
        for row in obs:
            assert bound['raw_trial_sha256'][row['raw_trial_path']]==row['raw_trial_sha256']
            outcomes[row['reason']]+=1
        old=min((r for r in obs if r['success']),key=lambda r:(r['velocity_rad_s'],r['index']))
        assert old['query_index']==a['best']['query_index']
        eligible=[r for r in obs if r['trajectory_complete'] and r['solver_pass'] and r['other_pass']
                  and r.get('observed_velocity_rad_s') is not None
                  and math.isfinite(r['observed_velocity_rad_s'])]
        eligible_total+=len(eligible)
        new=min(eligible,key=lambda r:(r['observed_velocity_rad_s'],r['index']))
        assert new['observed_velocity_rad_s']<=old['velocity_rad_s']
        old_m,new_m,zero_m=metric(old),metric(new),metric(obs[0])
        assert old_m['caps_deg'][:2]==labels['subtasks'][0]['pregrasp'][2:4]
        diffs={name:[abs(x-y) for x,y in zip(old_m[field],new_m[field])]
               for name,field in [('caps','caps_deg'),('actual','actual_peak_deg')]}
        pairs.append(dict(demo=entry['job']['demo'],source_annotation=str(path),
            old=old_m,new=new_m,zero=zero_m,
            selected_candidate_changed=old['query_index']!=new['query_index'],
            max_cap_change_deg=max(diffs['caps']),max_actual_change_deg=max(diffs['actual']),
            max_pre_cap_change_deg=max(diffs['caps'][:2]),max_pre_actual_change_deg=max(diffs['actual'][:2]),
            additional_reduction_pct=100*(old['velocity_rad_s']-new['observed_velocity_rad_s'])/old['velocity_rad_s'],
            original_both_pre_near_zero=max(old_m['caps_deg'][:2])<=1,
            original_either_pre_near_zero=min(old_m['caps_deg'][:2])<=1,
            new_both_pre_near_zero=max(new_m['caps_deg'][:2])<=1,
            new_either_pre_near_zero=min(new_m['caps_deg'][:2])<=1,
            eligible_candidates=len(eligible)))
    assert len(pairs)==47
    counts=dict(candidates_changed=sum(t['selected_candidate_changed'] for t in pairs))
    for name in ['max_cap_change_deg','max_actual_change_deg','max_pre_cap_change_deg','max_pre_actual_change_deg']:
        counts[name+'_ge5']=dict(count=sum(t[name]>=5 for t in pairs),
                                demos=[t['demo'] for t in pairs if t[name]>=5])
    for threshold in [.1,1,5,10]:
        counts['velocity_reduction_ge'+str(threshold)+'pct']=dict(
            count=sum(t['additional_reduction_pct']>=threshold for t in pairs),
            demos=[t['demo'] for t in pairs if t['additional_reduction_pct']>=threshold])
    for field in ['original_both_pre_near_zero','original_either_pre_near_zero','new_both_pre_near_zero','new_either_pre_near_zero']:
        counts[field]=dict(count=sum(t[field] for t in pairs),demos=[t['demo'] for t in pairs if t[field]])
    summary=dict(scope='libero_spatial/task_00, all 47 demos, all 28 existing candidates per demo',
        changed_definition='At least one aligned signed phase/direction differs by >=5 degrees',
        ignored='Both baseline-relative table clearance and absolute table contact acceptance',
        retained='Complete trajectory, non-table kinematic admissibility including robot self-collision, finite speed',
        sample_weight='Each demo equal; 8 aligned signed units, exclude authored strict axes',
        query_metadata_count=47*28,eligible_without_clearance=eligible_total,
        ineligible_without_clearance=47*28-eligible_total,original_outcome_counts=dict(outcomes),
        counts=counts,all=aggregate(pairs),
        old_either_pre_near_zero=aggregate([t for t in pairs if t['original_either_pre_near_zero']]),
        old_both_pre_near_zero=aggregate([t for t in pairs if t['original_both_pre_near_zero']]),
        candidate_changed=aggregate([t for t in pairs if t['selected_candidate_changed']]),
        top5=[t['demo'] for t in sorted(pairs,key=lambda t:-t['additional_reduction_pct'])[:5]],
        no_new_trials=True,
        limitation='Retrospective selection in a candidate set collected by clearance-aware BO; not a new no-clearance optimization, not physical task success.')
    (HERE/'without_clearance_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (HERE/'without_clearance_pairs.json').write_text(json.dumps(dict(units=UNITS,pairs=pairs),indent=2)+'\n')
    (HERE/'without_clearance_sources.json').write_text(json.dumps(dict(files=sources),indent=2)+'\n')
    with (HERE/'without_clearance_pairs.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['demo','old_query','new_query','old_velocity','new_velocity','additional_reduction_pct',
            'max_cap_change_deg','max_actual_change_deg','max_pre_cap_change_deg','max_pre_actual_change_deg']+
            ['old '+u+' cap_deg' for u in UNITS]+['new '+u+' cap_deg' for u in UNITS]+
            ['old '+u+' peak_deg' for u in UNITS]+['new '+u+' peak_deg' for u in UNITS])
        for t in pairs:
            w.writerow([t['demo'],t['old']['query_index'],t['new']['query_index'],
                t['old']['mean_joint_velocity_rad_s'],t['new']['mean_joint_velocity_rad_s'],t['additional_reduction_pct'],
                t['max_cap_change_deg'],t['max_actual_change_deg'],t['max_pre_cap_change_deg'],t['max_pre_actual_change_deg']]+
                t['old']['caps_deg']+t['new']['caps_deg']+t['old']['actual_peak_deg']+t['new']['actual_peak_deg'])
    lines=['# 取消净空验收：既有候选重排对照','',
      '状态：2026-11-09，只读重排及原始指标SHA核对完成。沿用同一Spatial00的47条demo，每条28个原候选；未启动新优化/搜索/仿真/训练，未修改冻结请求、标签或原回放文件。编号为原始demo，不是闭环评测seed。','',
      '## 对照定义','',
      '原方案：原相对净空规则下合格候选中平均关节速度最低者。对照：取消桌面相对净空与绝对桌面接触验收，保留完整轨迹、非桌面运动学合格（含自碰撞）及有限速度；在相同候选中按(observed_velocity_rad_s, index)选最低者。原成功/失败记录全部保留，只有派生的候选选择改变。','',
      f"1316个查询元数据绑定原独立审计SHA；去净空后1314个可比较，2个optimizer_infeasible仍排除。原757个合格、557个仅净空失败。候选改变28/47，新的28个均来自原净空失败候选。",'',
      '## 变化条数','',
      '明显变化预先统一按至少一个相同阶段/带符号方向差值≥5°统计，标注和实际峰值分别计算。实际为已存候选离线回放的阶段/方向峰值，不是SO(3)总角度，也不是后续物理回放。','',
      '|范围|标注边界变化≥5°|实际释放峰值变化≥5°|','|---|---:|---:|',
      '|APP+EXEC全部8方向|27/47（57.45%）|18/47（38.30%）|',
      '|仅APP Ry正负2方向|18/47（38.30%）|9/47（19.15%）|','',
      'APP双侧≤1°由4条降为0条；至少一侧≤1°由15条变为13条。原双侧近零demo8/44/45/49均恢复>5°的实际APP释放。单侧归零仍普遍存在，因为边界还受速度目标、释放方向及有限候选选择影响。','',
      '## 速度收益','',
      '指标是平均关节运动速度，越低表示在相同动作时序下关节运动量越小；这里不改变轨迹时长，不将降速百分比解释成任务执行速度提升。各demo等权，主要百分比由两组均值之比计算。','',
      '|样本|原平均rad/s|取消净空后rad/s|相对原方案再降低|','|---|---:|---:|---:|']
    for title,key in [('全部47条','all'),('原APP至少一侧近零的15条','old_either_pre_near_zero'),('原APP双侧近零的4条','old_both_pre_near_zero'),('换了候选的28条','candidate_changed')]:
        z=summary[key]
        lines.append(f"|{title}|{z['old_mean_velocity_rad_s']:.6f}|{z['new_mean_velocity_rad_s']:.6f}|{z['additional_velocity_reduction_pct']:.2f}%|")
    z=summary['all']
    lines+=['',f"相同47条零容差基线均值{z['zero_mean_velocity_rad_s']:.6f}rad/s：原方案降低{z['old_reduction_vs_zero_pct']:.2f}%，去净空重排降低{z['new_reduction_vs_zero_pct']:.2f}%，增加2.53个百分点；相对原方案的额外降幅为3.11%，两种分母不能混用。",'',
      '逐demo额外降幅中位数0.327%、均值2.697%；至少降低1%的15条，至少5%的7条，至少10%的5条。主要收益集中于少数受净空约束的样本。','',
      '## 收益最大的5条','',
      '|demo|原APP边界−/+°|去净空APP边界−/+°|原→新实际APP峰值−/+°|原→新平均速度rad/s|再降低|',
      '|---|---:|---:|---:|---:|---:|']
    pair=lambda v:' / '.join(f'{x:.3f}' for x in v)
    for t in sorted(pairs,key=lambda t:-t['additional_reduction_pct'])[:5]:
        o,n=t['old'],t['new']
        lines.append(f"|{t['demo']}|{pair(o['caps_deg'][:2])}|{pair(n['caps_deg'][:2])}|{pair(o['actual_peak_deg'][:2])} → {pair(n['actual_peak_deg'][:2])}|{o['mean_joint_velocity_rad_s']:.6f} → {n['mean_joint_velocity_rad_s']:.6f}|{t['additional_reduction_pct']:.2f}%|")
    lines+=['',
      'demo15的APP边界与实际APP峰值不变，主要变化在EXEC；不能将全部关节速度收益归因于APP Ry。demo44/45/49从全零参考模式转入非零继承参考模式，沿用原诊断中的协议差异，不把该收益冒充纯角度上限效应。','',
      '## 适用范围与来源','',
      '这是原净空感知BO已采样候选内的回顾性重排。若从头删除净空再搜索，后续查询点会改变，本报告没有测量该结果；本报告也没有新物理任务成功率。标注边界未写回正式数据集。','',
      '- 冻结候选根、请求、source_snapshot、日志和独立审计：`'+str(ROOT)+'`，每demo为`libero_spatial/task_00/demo_N/subtask_00/annotation.json`及`evaluations/<hash>/trial.json`。原始逐回合轨迹/视频、零参考与来源目录详见同目录README，未复制或改写。','',
      '- 原标签SHA绑定plan：`'+str(PLAN)+'`。原始评价映射：`'+str(ROOT/'source_snapshot/benchmark_joint_pre_post_t1_60.py')+'`的result_outcome，完整非桌面运动学与净空验收在该函数中分列。','',
      '- 只读入口：`'+str(HERE/'rerank_without_clearance.py')+'`；汇总：`'+str(HERE/'without_clearance_summary.json')+'`；逐demo/候选路径/SHA与实际峰值：`'+str(HERE/'without_clearance_pairs.json')+'`；逐项CSV：`'+str(HERE/'without_clearance_pairs.csv')+'`；原文件SHA清单：`'+str(HERE/'without_clearance_sources.json')+'`；统计复算：`'+str(HERE/'without_clearance_verification.json')+'`。','']
    (HERE/'WITHOUT_CLEARANCE.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
