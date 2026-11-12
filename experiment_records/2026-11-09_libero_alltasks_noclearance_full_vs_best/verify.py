"""Independent source hashes, candidate selection, block pooling and CSV stats."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics as st

HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
rows=read(HERE/'episodes.json');summary=read(HERE/'summary.json')
source_files=read(HERE/'sources.json')['files']
for item in source_files:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
print(f"checked {len(source_files)} original source file hashes",flush=True)
fields=dict(velocity=('mean_joint_velocity_rad_s',1),acceleration=('mean_joint_acceleration_rad_s2',2),
            jerk=('mean_joint_jerk_rad_s3',3),mY=('mean_manipulability',0),
            sigma_min=('minimum_sigma_min_velocity_normalized',None),
            sigma_min_full=('minimum_sigma_min_full',None),travel=('total_joint_travel_rad',None))
cache={};blocks_checked=0
for row in rows:
    pools={'full':[],'best':[]};ns=[]
    for block in row['blocks']:
        ann=read(block['annotation_path']);obs=ann['observations']
        prepared=read(Path(block['annotation_path']).parents[1]/'prepared_input.json')
        sub=prepared['subtasks'][block['index']]
        maximum=[]
        for phase in ['APP','EXEC']:
            for value in sub[phase+'_caps_deg']:maximum.extend([value,value])
        full=next(r for r in obs if r['actual_pre_caps_deg']+r['actual_post_caps_deg']==maximum)
        allowed=[r for r in obs if r['solver_pass'] is True and r['other_pass'] is True
                 and r['trajectory_complete'] is True and r['observed_velocity_rad_s'] is not None]
        ix=lambda r:r.get('query_index',r.get('index',0))
        best=min(allowed,key=lambda r:(r['observed_velocity_rad_s'],ix(r)))
        assert full['raw_trial_path']==block['full']['path']
        assert best['raw_trial_path']==block['best']['path']
        ns.append(block['expected_frames'])
        for name,selected in [('full',full),('best',best)]:
            p=selected['raw_trial_path']
            if p not in cache:
                raw=read(p)['result'];e=raw['evaluation']
                valid=bool(e['trajectory_complete'] and e['kinematic_feasible_without_table'])
                cache[p]=(valid,raw['trajectory_kinematics'])
            valid,k=cache[p]
            assert valid==block[name]['complete']
            pools[name].append(k if valid else None)
        blocks_checked+=1
    for arm in ['full','best']:
        if any(v is None for v in pools[arm]):
            assert row[arm+'_metrics'] is None
            continue
        for metric,(field,offset) in fields.items():
            values=[v[field] for v in pools[arm]]
            if metric.startswith('sigma_min'):value=min(values)
            elif metric=='travel':value=sum(values)
            else:
                weights=[max(0,n-offset) for n in ns]
                value=sum(v*w for v,w in zip(values,weights))/sum(weights)
            assert math.isclose(value,row[arm+'_metrics'][metric],rel_tol=1e-11,abs_tol=1e-12)
print(f"reselected and recomputed {blocks_checked} independent blocks",flush=True)
with (HERE/'per_demo.csv').open() as f:csvrows=list(csv.DictReader(f))
assert len(csvrows)==1875
paired=[r for r in csvrows if r['full_complete']=='True']
assert len(paired)==1837
for metric in fields:
    sign=-1 if metric in ['velocity','acceleration','jerk','travel'] else 1
    full=[float(r['full_'+metric]) for r in paired]
    best=[float(r['best_'+metric]) for r in paired]
    gains=[sign*100*(b-a)/a for a,b in zip(full,best) if a>1e-12]
    target=summary['all']['metrics'][metric]
    assert math.isclose(st.fmean(full),target['full_mean'],abs_tol=1e-12)
    assert math.isclose(st.fmean(best),target['best_mean'],abs_tol=1e-12)
    assert math.isclose(st.median(gains),target['paired_gain_pct']['median'],abs_tol=1e-10)
    for threshold in [.01,.1,.5,1,2,5,10]:
        assert sum(g<=threshold for g in gains)==target['cumulative_gain_le_pct'][str(threshold)]['count']
    assert sum(g< -1 for g in gains)==target['regressed_gt1pct']
for task in summary['tasks']:
    subset=[r for r in paired if r['suite']==task['suite'] and int(r['task_id'])==task['task_id']]
    assert len(subset)==task['paired_episodes']
    a=st.fmean(float(r['full_velocity']) for r in subset);b=st.fmean(float(r['best_velocity']) for r in subset)
    assert math.isclose(100*(a-b)/a,task['metrics']['velocity']['improvement_pct_of_means'],abs_tol=1e-10)
result=dict(complete=True,source_hashes_checked=len(source_files),unique_raw_trials_reparsed=len(cache),
    episodes_checked=1875,paired_episodes=1837,independent_blocks_checked=blocks_checked,tasks_checked=40,
    checks=['independent candidate selection from authored endpoints','all seven metric pooling rules',
            'CSV mean/median/threshold and regression counts','40-task speed summary','all original SHA unchanged'],
    new_search_or_rollout=False,original_data_modified=False)
(HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
