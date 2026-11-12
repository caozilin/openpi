"""Read-only within-task variation of searched caps and realized release."""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics as st

HERE=Path(__file__).resolve().parent
MANIFEST=Path('/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_zero_nominal_continuation_20261106/combined_200_source_manifest.json')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
mean_or_none=lambda values:st.fmean(values) if values else None
fmt=lambda value:'—' if value is None else f'{value:.3f}'

def stats(values):
    values=list(values)
    assert values and all(math.isfinite(v) for v in values)
    pairs=[abs(a-b) for i,a in enumerate(values) for b in values[i+1:]]
    return dict(n=len(values),mean=st.fmean(values),std_population=st.pstdev(values),
                minimum=min(values),maximum=max(values),span=max(values)-min(values),
                mean_pairwise_abs_difference=st.fmean(pairs) if pairs else 0.)

def main():
    manifest=read(MANIFEST)
    groups=defaultdict(list);evidence=[];missing=[]
    for item in manifest['episodes']:
        key=(item['suite'],item['task_id'])
        if item['status']!='labeled':
            missing.append({k:item[k] for k in ('suite','task_id','demo','status')});continue
        folder=Path(item['source_run'])/item['suite']/f"task_{item['task_id']:02d}"/item['demo']
        labels=folder/'tolerance_labels.json';assert sha(labels)==item['labels_sha256']
        prepared=read(folder/'prepared_input.json')
        units={};stages={};cycles=[]
        for i,subtask in enumerate(prepared['subtasks']):
            semantic=prepared['provenance']['source_segmentation'][i]['authored_subtask_index']
            path=folder/f'subtask_{i:02d}'/'annotation.json';annotation=read(path)
            best=annotation['best'];assert best is not None
            rawpath=Path(best['raw_trial_path']);assert sha(rawpath)==best['raw_trial_sha256']
            assert item['raw_trial_sha256'][str(rawpath)]==best['raw_trial_sha256']
            raw=read(rawpath);e=raw['result']['evaluation'];assert e['trajectory_complete'] and e['kinematic_feasible_without_table']
            assert len(e['sqp_subtask_stage_release_extrema_deg'])==1
            extrema=e['sqp_subtask_stage_release_extrema_deg'][0]
            counts=e['sqp_subtask_stage_release_sample_counts'][0]
            angle=e['sqp_rotation_release']['subtasks'][0]
            cycle=dict(index=i,authored_subtask=semantic,raw_trial=str(rawpath),raw_sha256=best['raw_trial_sha256'],
                pre_caps_deg=best['actual_pre_caps_deg'],post_caps_deg=best['actual_post_caps_deg'],
                release_extrema_deg=extrema,so3_peaks=angle)
            cycles.append(cycle)
            for phase,phasekey,capkey in [('APP','pregrasp','actual_pre_caps_deg'),('EXEC','postgrasp','actual_post_caps_deg')]:
                if counts[phasekey]==0:continue
                limits=subtask[phase+'_caps_deg']
                if any(limits):
                    stagekey=f'semantic{semantic}/{phase}'
                    stages[stagekey]=max(stages.get(stagekey,0.),angle[phasekey]['peak_angle_deg'])
                for axis,name in enumerate(('Rx','Ry','Rz')):
                    if limits[axis]<=0:continue
                    for sideindex,side in enumerate(('negative','positive')):
                        unit=f'semantic{semantic}/{phase}/{name}{"-" if sideindex==0 else "+"}'
                        cap=best[capkey][2*axis+sideindex];actual=extrema[phasekey][side][axis]
                        assert math.isfinite(cap) and math.isfinite(actual) and actual>=0
                        current=units.setdefault(unit,dict(search_cap_deg=0.,actual_peak_deg=0.,authored_cap_deg=limits[axis]))
                        assert current['authored_cap_deg']==limits[axis]
                        current['search_cap_deg']=max(current['search_cap_deg'],cap)
                        current['actual_peak_deg']=max(current['actual_peak_deg'],actual)
        groups[key].append(dict(demo=item['demo'],episode_index=item['source_episode_index'],
            reference_protocol=item['reference_protocol'],units=units,stage_so3_peaks_deg=stages,cycles=cycles))
        evidence.append(dict(case=f"{key[0]}/{key[1]:02d}/{item['demo']}",labels=str(labels),labels_sha256=item['labels_sha256'],prepared=str(folder/'prepared_input.json')))
    tasks=[]
    for (suite,task),demos in sorted(groups.items()):
        common=set.intersection(*(set(d['units']) for d in demos))
        allunits=set.union(*(set(d['units']) for d in demos));units={}
        for unit in sorted(common):
            caps=stats(d['units'][unit]['search_cap_deg'] for d in demos)
            actual=stats(d['units'][unit]['actual_peak_deg'] for d in demos)
            limits={d['units'][unit]['authored_cap_deg'] for d in demos};assert len(limits)==1
            units[unit]=dict(search=caps,actual=actual,authored_cap_deg=next(iter(limits)),
                per_demo={d['demo']:d['units'][unit] for d in demos})
        if not units:
            assert not allunits, 'No common observed stages: requires explicit treatment'
        commonstages=set.intersection(*(set(d['stage_so3_peaks_deg']) for d in demos))
        stage_stats={s:stats(d['stage_so3_peaks_deg'][s] for d in demos) for s in sorted(commonstages)}
        tasks.append(dict(suite=suite,task_id=task,demos=[d['demo'] for d in demos],n=len(demos),
            mixed_protocol=len({d['reference_protocol'] for d in demos})>1,
            common_direction_units=len(units),omitted_noncommon_units=sorted(allunits-common),
            mean_search_cap_span_deg=mean_or_none([v['search']['span'] for v in units.values()]),
            mean_actual_peak_span_deg=mean_or_none([v['actual']['span'] for v in units.values()]),
            mean_search_pairwise_difference_deg=mean_or_none([v['search']['mean_pairwise_abs_difference'] for v in units.values()]),
            mean_actual_pairwise_difference_deg=mean_or_none([v['actual']['mean_pairwise_abs_difference'] for v in units.values()]),
            mean_stage_so3_peak_span_deg=mean_or_none([v['span'] for v in stage_stats.values()]),
            max_search_cap_span_deg=max((v['search']['span'] for v in units.values()),default=None),
            max_actual_peak_span_deg=max((v['actual']['span'] for v in units.values()),default=None),
            units=units,stage_so3_peak_stats=stage_stats,raw_demos=demos))
    def aggregate(selected):
        total=len(selected);episodes=sum(t['n'] for t in selected)
        selected=[t for t in selected if t['common_direction_units']]
        return dict(tasks=total,episodes=episodes,eligible_tasks=len(selected),eligible_episodes=sum(t['n'] for t in selected),
            mean_search_cap_span_deg=st.fmean(t['mean_search_cap_span_deg'] for t in selected),
            mean_actual_peak_span_deg=st.fmean(t['mean_actual_peak_span_deg'] for t in selected),
            median_task_search_cap_span_deg=st.median(t['mean_search_cap_span_deg'] for t in selected),
            median_task_actual_peak_span_deg=st.median(t['mean_actual_peak_span_deg'] for t in selected),
            mean_search_pairwise_difference_deg=st.fmean(t['mean_search_pairwise_difference_deg'] for t in selected),
            mean_actual_pairwise_difference_deg=st.fmean(t['mean_actual_pairwise_difference_deg'] for t in selected),
            mean_stage_so3_peak_span_deg=st.fmean(t['mean_stage_so3_peak_span_deg'] for t in selected))
    values=[v for t in tasks for v in t['units'].values()]
    summary=dict(complete=True,checked_at=datetime.now().astimezone().isoformat(),
        manifest=str(MANIFEST),manifest_sha256=sha(MANIFEST),task_weighting='equal weight per task; equal weight per commonly observed eligible signed direction within task',
        comparison='max-minus-min across available demos, in degrees; only nonzero authored eligible directions; actual=observed signed coordinate peak',
        repeated_cycle_rule='Within a demo, max across observed cycles mapped to the same authored semantic subtask and phase; envelope statistics, not a rewritten label',
        sample_definition='First5 successful original demo IDs per task, not rollout seeds0to4; report.seed is not independent sample ID',
        all40=aggregate(tasks),clean_complete38=aggregate([t for t in tasks if t['n']==5 and not t['mixed_protocol']]),
        suites={s:aggregate([t for t in tasks if t['suite']==s]) for s in sorted({t['suite'] for t in tasks})},
        missing=missing,mixed_protocol_tasks=[(t['suite'],t['task_id']) for t in tasks if t['mixed_protocol']],
        comparable_signed_units=len(values),
        search_span_gt5_actual_span_lt1_units=sum(v['search']['span']>5 and v['actual']['span']<1 for v in values),
        search_span_gt10_units=sum(v['search']['span']>10 for v in values),
        actual_span_gt10_units=sum(v['actual']['span']>10 for v in values),
        strict_tasks=[(t['suite'],t['task_id']) for t in tasks if not t['common_direction_units']],
        tasks_with_search_mean_span_lt1=sum(t['mean_search_cap_span_deg']<1 for t in tasks if t['common_direction_units']),
        tasks_with_actual_mean_span_lt1=sum(t['mean_actual_peak_span_deg']<1 for t in tasks if t['common_direction_units']),
        caveat='Different demo-specific BO seeds, trajectories, grasp histories and a finite query budget confound pure scene sensitivity; no new search or rollout conducted.')
    HERE.mkdir(parents=True,exist_ok=True)
    (HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (HERE/'per_task_details.json').write_text(json.dumps(dict(tasks=tasks,evidence=evidence),ensure_ascii=False,indent=2)+'\n')
    lines=['# 同任务5条demo：搜索上限与实际释放差异','',
        '差异单位：度。每个共同出现的可释放方向先计算demo间最大值−最小值，再在任务内取平均；禁用轴不参与平均。实际释放取逐阶段观测到的带符号坐标峰值。重复观察周期按同语义子任务/阶段取最大包络。','',
        '|套件|任务|有效demo数|搜索边界平均极差|实际释放峰值平均极差|阶段SO(3)峰值平均极差|备注|',
        '|---|---:|---:|---:|---:|---:|---|']
    for t in tasks:
        note='混合基线协议' if t['mixed_protocol'] else '1条缺标签' if t['n']<5 else ''
        if not t['common_direction_units']:note='全部禁用容差，不纳入可释放方向均值'
        lines.append(f"|{t['suite']}|{t['task_id']:02d}|{t['n']}|{fmt(t['mean_search_cap_span_deg'])}|{fmt(t['mean_actual_peak_span_deg'])}|{fmt(t['mean_stage_so3_peak_span_deg'])}|{note}|")
    (HERE/'per_task_variation.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    for metric in ['mean_search_cap_span_deg','mean_actual_peak_span_deg']:
        print(metric,[(t['suite'],t['task_id'],round(t[metric],3)) for t in sorted((t for t in tasks if t['common_direction_units']),key=lambda t:t[metric],reverse=True)[:7]])

if __name__=='__main__':main()
