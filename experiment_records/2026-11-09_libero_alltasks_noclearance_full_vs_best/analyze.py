"""Compare existing full semantic endpoints to speed-best no-clearance candidates.

No simulation/search imports. Keep the original 1851 and supplemental 24 strata.
"""
from collections import Counter, defaultdict
from pathlib import Path
import ast
import csv
import hashlib
import json
import math
import statistics as st
import time

import numpy as np

HERE=Path(__file__).resolve().parent
BASE=Path('/media/czl/sata/franka_my_code/franka_mujoco/analysis_results')
RUNS=[('original1851',BASE/'libero_relative_clearance_eps01_ema85_full1875_20261107'),
      ('supplement24',BASE/'libero_missing24_baseline_observation_20261109')]
TASKMAP=Path('/media/czl/sata/franka_my_code/openpi/third_party/libero/libero/libero/benchmark/libero_suite_task_map.py')
METRICS={
 'velocity':('mean_joint_velocity_rad_s',1,-1),
 'acceleration':('mean_joint_acceleration_rad_s2',2,-1),
 'jerk':('mean_joint_jerk_rad_s3',3,-1),
 'mY':('mean_manipulability',0,1),
 'sigma_min':('minimum_sigma_min_velocity_normalized',None,1),
 'sigma_min_full':('minimum_sigma_min_full',None,1),
 'travel':('total_joint_travel_rad',None,-1)}
SOURCES={};TRIAL_CACHE={}


def read(path,expected=None):
    path=Path(path);b=path.read_bytes();sha=hashlib.sha256(b).hexdigest()
    if expected is not None:assert sha==expected,path
    SOURCES[str(path)]=sha
    return json.loads(b)


def write(name,value):
    (HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def index(row):return row.get('query_index',row.get('index',0))
def eligible(row):
    return bool(row['trajectory_complete'] and row['solver_pass'] and row['other_pass']
                and row.get('observed_velocity_rad_s') is not None
                and math.isfinite(row['observed_velocity_rad_s']))


def load_trial(row,expected_frames,audit_hashes):
    path=row['raw_trial_path'];expected=row['raw_trial_sha256']
    assert audit_hashes[path]==expected
    if path in TRIAL_CACHE:return TRIAL_CACHE[path]
    raw=read(path,expected)
    assert raw['point']==row['actual_pre_caps_deg']+row['actual_post_caps_deg']
    for key,value in raw['row'].items():assert row[key]==value,(path,key)
    r=raw['result'];e=r['evaluation'];k=r['trajectory_kinematics']
    assert e['physics_steps']==0 and e['control_rate_hz']==20 and not e['objects_considered']
    assert e['expected_frames']==expected_frames
    assert not r['termination_reason'].startswith('runner_error:')
    complete=bool(e['trajectory_complete'] and e['kinematic_feasible_without_table'])
    assert complete==eligible(row)
    metrics=None
    if complete:
        assert k['samples']==expected_frames
        assert k['mean_joint_velocity_rad_s']==row['observed_velocity_rad_s']
        metrics={name:float(k[field]) for name,(field,_,_) in METRICS.items()}
        assert all(math.isfinite(v) and v>=0 for v in metrics.values())
    result=dict(path=path,sha256=expected,query_index=index(row),complete=complete,
                samples=k['samples'],metrics=metrics,reason=row['reason'],
                termination_reason=r['termination_reason'],original_clearance_pass=row['clearance_pass'],
                stage_handoff_policy=e['stage_handoff_policy'],
                hard_failures=e.get('hard_failure_reasons',[]),
                declared_caps_deg=raw['point'])
    TRIAL_CACHE[path]=result
    return result


def combine(blocks,key):
    if not all(b[key]['complete'] for b in blocks):return None
    out={}
    for name,(_,offset,_) in METRICS.items():
        vals=[b[key]['metrics'][name] for b in blocks]
        if name.startswith('sigma_min'):out[name]=min(vals)
        elif name=='travel':out[name]=math.fsum(vals)
        else:
            weights=[max(0,b[key]['samples']-offset) for b in blocks]
            assert sum(weights)>0
            out[name]=math.fsum(v*w for v,w in zip(vals,weights))/sum(weights)
    return out


def combine_initial_speed(blocks):
    if any(b['initial_best_velocity'] is None for b in blocks):return None
    weights=[b['best']['samples']-1 for b in blocks]
    return math.fsum(b['initial_best_velocity']*w for b,w in zip(blocks,weights))/sum(weights)


def quantiles(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return None
    return dict(n=len(a),mean=float(a.mean()),std_population=float(a.std()),
        **{k:float(np.quantile(a,q)) for k,q in [('min',0),('p05',.05),('p25',.25),
                    ('median',.5),('p75',.75),('p90',.9),('p95',.95),('p99',.99),('max',1)]})


def metric_summary(rows,name):
    sign=METRICS[name][2]
    full=[r['full_metrics'][name] for r in rows];best=[r['best_metrics'][name] for r in rows]
    mean_full=st.fmean(full);mean_best=st.fmean(best)
    valid=[(a,b) for a,b in zip(full,best) if a>1e-12]
    pct=[sign*100*(b-a)/a for a,b in valid]
    absolute=[b-a for a,b in zip(full,best)]
    return dict(n=len(rows),full_mean=mean_full,best_mean=mean_best,
        improvement_pct_of_means=sign*100*(mean_best-mean_full)/mean_full if mean_full>1e-12 else None,
        paired_gain_pct=quantiles(pct),absolute_delta_best_minus_full=quantiles(absolute),
        percent_valid_count=len(valid),percent_excluded_near_zero=len(rows)-len(valid),
        denominator_le1e_minus6_count=sum(a<=1e-6 for a in full),
        improved_gt1pct=sum(v>1 for v in pct),regressed_gt1pct=sum(v< -1 for v in pct),
        abs_change_le1pct=sum(abs(v)<=1 for v in pct),
        cumulative_gain_le_pct={str(t):dict(count=sum(v<=t for v in pct),
          fraction=sum(v<=t for v in pct)/len(pct)) for t in [1e-6,.01,.1,.5,1,2,5,10]},
        all_four_quantile_values_are_demo_equal=True)


def summarize(allrows):
    paired=[r for r in allrows if r['full_complete']]
    result=dict(total_episodes=len(allrows),paired_episodes=len(paired),
        full_incomplete_episodes=len(allrows)-len(paired),
        best_complete_episodes=sum(r['best_metrics'] is not None for r in allrows),
        strict_zero_episodes=sum(r['strict_zero'] for r in allrows),metrics={})
    if not paired:return result
    result['metrics']={name:metric_summary(paired,name) for name in METRICS}
    fv=st.fmean(r['full_metrics']['velocity'] for r in paired)
    iv=st.fmean(r['initial_best_velocity'] for r in paired)
    bv=st.fmean(r['best_metrics']['velocity'] for r in paired)
    added=[100*(r['initial_best_velocity']-r['best_metrics']['velocity'])/r['initial_best_velocity'] for r in paired]
    result['initial_vs_adaptive']=dict(full_mean_velocity=fv,initial_best_mean_velocity=iv,
        all_best_mean_velocity=bv,initial_reduction_vs_full_pct=100*(1-iv/fv),
        adaptive_reduction_vs_initial_pct=100*(1-bv/iv),
        adaptive_paired_gain_pct=quantiles(added),
        adaptive_gain_le_0_1pct=sum(v<=.1 for v in added),adaptive_gain_le_1pct=sum(v<=1 for v in added),
        all_best_blocks_initial_episodes=sum(all(b['best_initial'] for b in r['blocks']) for r in paired),
        any_adaptive_selected_episodes=sum(any(not b['best_initial'] for b in r['blocks']) for r in paired),
        all_full_selected_episodes=sum(all(b['same_point_full_best'] for b in r['blocks']) for r in paired))
    return result


def main():
    start=time.monotonic()
    tree=ast.parse(TASKMAP.read_text());names=ast.literal_eval(tree.body[0].value)
    SOURCES[str(TASKMAP)]=hashlib.sha256(TASKMAP.read_bytes()).hexdigest()
    rows=[];excluded=[];runinfo=[];observation_count=0
    for stratum,root in RUNS:
        audit=read(root/'annotation_independent_final_audit.json')
        request=read(root/'annotation_request.json');read(root/'selection.json')
        assert audit['complete'] and audit['adapter_errors']==0
        assert audit['per_frame_labels_verified'] and audit['zero_physics_verified']
        runinfo.append(dict(stratum=stratum,root=str(root),audit_sha256=SOURCES[str(root/'annotation_independent_final_audit.json')],
            original_request_sha256=SOURCES[str(root/'annotation_request.json')],
            baseline_protocol='require feasible zero reference' if stratum=='original1851' else 'baseline observation allows held rejected states; candidate unchanged'))
        for item in audit['results']:
            if item['status']!='labeled':
                excluded.append(dict(stratum=stratum,**{k:item[k] for k in ['suite','task_id','demo','status']}));continue
            folder=root/item['suite']/f"task_{item['task_id']:02d}"/item['demo']
            prepared=read(folder/'prepared_input.json',item['prepared_input_sha256'])
            read(folder/'tolerance_labels.json',item['labels_sha256'])
            blocks=[]
            for i,sub in enumerate(prepared['subtasks']):
                ap=folder/f'subtask_{i:02d}'/'annotation.json';a=read(ap)
                assert a['complete']
                obs=a['observations'];observation_count+=len(obs)
                maximum=[v for phase in ['APP','EXEC'] for v in sub[phase+'_caps_deg'] for _ in range(2)]
                initial_count=a.get('design',{}).get('unique_initial_count',len(obs))
                if any(maximum):
                    assert a['design']['maximum_caps_deg']==maximum
                    assert maximum in a['design']['initial_caps_deg']
                    assert a['additional_queries']==16
                else:assert a['strict_authored_task']
                for o in obs:assert item['raw_trial_sha256'][o['raw_trial_path']]==o['raw_trial_sha256']
                matching=[o for o in obs if o['actual_pre_caps_deg']+o['actual_post_caps_deg']==maximum]
                assert len(matching)==1
                fullrow=matching[0];assert index(fullrow)<initial_count
                valid=[o for o in obs if eligible(o)];assert valid
                bestrow=min(valid,key=lambda o:(o['observed_velocity_rad_s'],index(o)))
                initial=[o for o in valid if index(o)<initial_count]
                initial_best=min(initial,key=lambda o:(o['observed_velocity_rad_s'],index(o))) if initial else None
                n=sum(f['subtask_index']==i for f in prepared['frames'])
                full=load_trial(fullrow,n,item['raw_trial_sha256'])
                best=load_trial(bestrow,n,item['raw_trial_sha256']);assert best['complete']
                if full['complete']:assert best['metrics']['velocity']<=full['metrics']['velocity']+1e-12
                blocks.append(dict(index=i,expected_frames=n,maximum_caps_deg=maximum,
                    full=full,best=best,initial_count=initial_count,queries=len(obs),
                    eligible_count=len(valid),initial_best_velocity=initial_best['observed_velocity_rad_s'] if initial_best else None,
                    initial_best_query=index(initial_best) if initial_best else None,
                    best_initial=index(bestrow)<initial_count,
                    same_point_full_best=bestrow['raw_trial_path']==fullrow['raw_trial_path'],
                    annotation_path=str(ap),annotation_sha256=SOURCES[str(ap)]))
            fullmetrics=combine(blocks,'full');bestmetrics=combine(blocks,'best')
            rows.append(dict(key=f"{item['suite']}/{item['task_id']:02d}/{item['demo']}",
                 stratum=stratum,suite=item['suite'],task_id=item['task_id'],demo=item['demo'],
                 task_name=names[item['suite']][item['task_id']],source_folder=prepared['source']['folder'],
                 full_complete=fullmetrics is not None,full_metrics=fullmetrics,best_metrics=bestmetrics,
                 initial_best_velocity=combine_initial_speed(blocks),blocks=blocks,
                 strict_zero=all(not any(b['maximum_caps_deg']) for b in blocks),
                 gp_recovered=(item['suite'],item['task_id'],item['demo'])==('libero_10',2,'demo_40')))
            if len(rows)%100==0:
                print(f"processed {len(rows)}/1875; unique trials {len(TRIAL_CACHE)}; elapsed {time.monotonic()-start:.0f}s",flush=True)
    keys=[r['key'] for r in rows]
    assert len(keys)==len(set(keys))==1875
    tasks=[]
    for (suite,task),subset in sorted(_groups(rows).items()):
        tasks.append(dict(suite=suite,task_id=task,task_name=subset[0]['task_name'],
             stratum_counts=dict(Counter(r['stratum'] for r in subset)),**summarize(subset)))
    assert len(tasks)==40
    paired=[r for r in rows if r['full_complete']]
    total_saving=math.fsum(r['full_metrics']['velocity']-r['best_metrics']['velocity'] for r in paired)
    all_result=summarize(rows)
    grand_gain=all_result['metrics']['velocity']['improvement_pct_of_means']
    for task in tasks:
        subset=[r for r in paired if (r['suite'],r['task_id'])==(task['suite'],task['task_id'])]
        saving=math.fsum(r['full_metrics']['velocity']-r['best_metrics']['velocity'] for r in subset)
        task['velocity_absolute_saving_sum']=saving
        task['share_of_total_velocity_saving']=saving/total_saving
        rest=[r for r in rows if (r['suite'],r['task_id'])!=(task['suite'],task['task_id'])]
        task['overall_velocity_gain_if_task_removed_pct']=summarize(rest)['metrics']['velocity']['improvement_pct_of_means']
    ranked_tasks=sorted(tasks,key=lambda t:-t['velocity_absolute_saving_sum'])
    contribution={}
    for k in [1,3,5,10]:
        remove={(t['suite'],t['task_id']) for t in ranked_tasks[:k]}
        rest=[r for r in rows if (r['suite'],r['task_id']) not in remove]
        contribution[str(k)]=dict(tasks=[f"{t['suite']}/{t['task_id']:02d}" for t in ranked_tasks[:k]],
           saving_share=sum(t['share_of_total_velocity_saving'] for t in ranked_tasks[:k]),
           remaining_summary=summarize(rest))
    gain=lambda r:100*(1-r['best_metrics']['velocity']/r['full_metrics']['velocity'])
    mean_gain=st.fmean(gain(r) for r in paired)
    total_ss=math.fsum((gain(r)-mean_gain)**2 for r in paired)
    between_ss=0.
    for subset in _groups(paired).values():
        between_ss+=len(subset)*(st.fmean(gain(r) for r in subset)-mean_gain)**2
    top_episodes=sorted(paired,key=lambda r:-(r['full_metrics']['velocity']-r['best_metrics']['velocity']))
    tail_saving={str(p):dict(n=max(1,math.ceil(p*len(paired))),share=math.fsum(
       r['full_metrics']['velocity']-r['best_metrics']['velocity'] for r in top_episodes[:max(1,math.ceil(p*len(paired)))])/total_saving)
       for p in [.01,.05,.1]}
    output=dict(complete=True,definition='Both arms ignore all table-clearance acceptance; retain complete non-table kinematic admissibility; speed-best among all existing queries, including initialization.',
      protocol='20Hz, IPOPT20/EMA10/cache85; original disjoint blocks, no new execution or search; source strata retained',
      metrics=METRICS,runs=runinfo,all=all_result,
      strata={s:summarize([r for r in rows if r['stratum']==s]) for s,_ in RUNS},
      suites={s:summarize([r for r in rows if r['suite']==s]) for s in sorted({r['suite'] for r in rows})},
      excluding_strict=summarize([r for r in rows if not r['strict_zero']]),
      excluding_gp_recovery=summarize([r for r in rows if not r['gp_recovered']]),
      task_equal_mean_velocity_ratio_gain_pct=st.fmean(t['metrics']['velocity']['improvement_pct_of_means'] for t in tasks),
      tasks=tasks,top_task_contribution=contribution,
      fraction_velocity_gain_variance_between_tasks=between_ss/total_ss,
      top_episode_saving_shares=tail_saving,
      full_incomplete=[dict(key=r['key'],stratum=r['stratum'],task_name=r['task_name'],
          failures=[dict(index=b['index'],path=b['full']['path'],reason=b['full']['reason'],
              termination=b['full']['termination_reason'],hard_failures=b['full']['hard_failures'])
              for b in r['blocks'] if not b['full']['complete']]) for r in rows if not r['full_complete']],
      original_exclusions_replaced_by_supplement=excluded,
      candidate_observations_audit_bound=observation_count,blocks=sum(len(r['blocks']) for r in rows),
      unique_trials_sha_verified=len(TRIAL_CACHE),
      top20_velocity_gain=[dict(key=r['key'],task_name=r['task_name'],gain_pct=gain(r),
         full=r['full_metrics'],best=r['best_metrics']) for r in sorted(paired,key=lambda r:-gain(r))[:20]],
      caveat='Candidate sets were collected by clearance-aware BO. This is a retrospective comparison, not re-optimization without clearance, a physical success test, or a four-objective optimum.')
    write('episodes.json',rows);write('summary.json',output)
    write('sources.json',dict(files=[dict(path=p,sha256=s) for p,s in SOURCES.items()]))
    with (HERE/'per_demo.csv').open('w') as f:
        columns=['key','stratum','suite','task_id','demo','full_complete','strict_zero','initial_best_velocity']
        for name in METRICS:columns += ['full_'+name,'best_'+name,'gain_pct_'+name]
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in rows:
            row={k:r[k] for k in columns[:8]}
            for name,(_,_,sign) in METRICS.items():
                full=r['full_metrics'][name] if r['full_complete'] else None;best=r['best_metrics'][name]
                row['full_'+name]=full;row['best_'+name]=best
                row['gain_pct_'+name]=sign*100*(best-full)/full if full is not None and full>1e-12 else None
            writer.writerow(row)
    print(json.dumps(dict(total=all_result['total_episodes'],paired=all_result['paired_episodes'],
      full_failures=all_result['full_incomplete_episodes'],velocity=all_result['metrics']['velocity'],
      initial_vs_adaptive=all_result['initial_vs_adaptive'],
      top_task_names=[(t['suite'],t['task_id'],t['task_name'],t['metrics']['velocity']['improvement_pct_of_means']) for t in ranked_tasks[:8]]),ensure_ascii=False,indent=2),flush=True)


def _groups(rows):
    groups=defaultdict(list)
    for row in rows:groups[(row['suite'],row['task_id'])].append(row)
    return groups


if __name__=='__main__':main()
