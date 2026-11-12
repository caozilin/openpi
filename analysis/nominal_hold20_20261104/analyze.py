"""Read committed traces; no rollout, solver, policy or artifact mutation."""
import collections
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'data/libero_evaluation/pi05_reconstructed_10000_nominal_failed_retry_hold20_20261104'
OUT=Path(__file__).resolve().parent
request=json.loads((RUN/'evaluation_request.json').read_text())
baseline=Path(request['baseline_run'])
bursts=[];episodes=[];statuses=collections.Counter();oldstatuses=collections.Counter()
def errors(row, side, target):
 p=row[side]
 return dict(position_m=float(np.linalg.norm(np.array(p['position_m'])-target['position_m'])),
 rotation_rad=float((Rotation.from_rotvec(target['rotation_vector_rad']).inv()*Rotation.from_rotvec(p['rotation_vector_rad'])).magnitude()),
 joint_max_rad=float(np.abs(np.array(p['joint_position_rad'])-row['joint_target_rad']).max()))
for path in sorted(RUN.glob('episodes/*/task_*/seed_*/result.json')):
 result=json.loads(path.read_text());rows=json.loads(path.with_name('trajectory.json').read_text())
 oldrows=json.loads((baseline/path.parent.relative_to(RUN)/'trajectory.json').read_text())
 if not oldrows[-1]['optimizer']['feasible']:oldstatuses[oldrows[-1]['optimizer']['status']]+=1
 episode={k:result[k] for k in ['suite','task_id','seed','success','termination','total_solver_failures','maximum_solver_failure_streak']}
 episodes.append(episode)
 i=0
 while i<len(rows):
  if not rows[i]['solver_failed']:i+=1;continue
  start=i
  while i<len(rows) and rows[i]['solver_failed']:
   statuses[rows[i]['optimizer']['status']]+=1;i+=1
  end=i-1
  target=rows[start]['nominal_target']
  recovered=i<len(rows) and not rows[i]['solver_failed']
  pre=errors(rows[start],'pre_action',target);post=errors(rows[end],'post_action',target)
  event={**{k:episode[k] for k in ['suite','task_id','seed']},'episode_success':result['success'],'start_step_1based':start+1,'end_step_1based':end+1,'failure_frames':end-start+1,'hold_seconds':(end-start+1)*.05,'recovered':recovered,'pre_errors_to_held_reference':pre,'post_errors_to_held_reference':post,'reference_exactly_constant':all(x['nominal_target']==target for x in rows[start:i]),'recovery_uses_same_inference_chunk_as_first_failure':bool(recovered and i//5==start//5),'recovery_uses_same_inference_chunk_as_last_failure':bool(recovered and i//5==end//5),'model_inferences_since_first_failed_action_before_recovery':(i//5-start//5) if recovered else None}
  if recovered:
   event['recovery_action']=rows[i]['action'];event['first_failed_action']=rows[start]['action']
   event['recovery_candidate_distance_from_first_rejected_m']=float(np.linalg.norm(np.array(rows[i]['nominal_target']['position_m'])-rows[start]['rejected_nominal_target']['position_m']))
  bursts.append(event)
recovered=[e for e in bursts if e['recovered']]
def summarize(xs):
 return {'count':len(xs),'length_histogram':dict(sorted(collections.Counter(e['failure_frames'] for e in xs).items())),
 'same_first_chunk_recovery':sum(e['recovery_uses_same_inference_chunk_as_first_failure'] for e in xs),
 'same_last_chunk_recovery':sum(e['recovery_uses_same_inference_chunk_as_last_failure'] for e in xs),
 'error_decreases':{k:sum(e['post_errors_to_held_reference'][k]<e['pre_errors_to_held_reference'][k] for e in xs) for k in ['position_m','rotation_rad','joint_max_rad']},
 'median_pre_errors':{k:float(np.median([e['pre_errors_to_held_reference'][k] for e in xs])) for k in ['position_m','rotation_rad','joint_max_rad']},
 'median_post_errors':{k:float(np.median([e['post_errors_to_held_reference'][k] for e in xs])) for k in ['position_m','rotation_rad','joint_max_rad']}}
report={'scope':'Descriptive trace analysis, not causal ablation. First action chunks differ from original run. No new simulation or inference.',
'episodes':episodes,'success_with_solver_failures':sum(e['success'] and e['total_solver_failures']>0 for e in episodes),'success_without_solver_failures':sum(e['success'] and e['total_solver_failures']==0 for e in episodes),
'all_bursts':summarize(bursts),'recovered_bursts':summarize(recovered),'recovered_bursts_in_successful_episodes':summarize([e for e in recovered if e['episode_success']]),'failed_solver_statuses':dict(statuses),'original_failed_solver_statuses':dict(oldstatuses),'bursts':bursts}
(OUT/'trace_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['episodes','bursts']},ensure_ascii=False,indent=2))
print('Examples:')
for x in recovered:
 if x['failure_frames']>=3:print(json.dumps(x,ensure_ascii=False))
