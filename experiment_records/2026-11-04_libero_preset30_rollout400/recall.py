"""Recall formal 400-vs-400 rollout and recompute recorded motion statistics."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import statistics as st
import numpy as np

H=Path(__file__).resolve().parent
ROOT=Path('/media/czl/sata/franka_my_code/openpi/data/libero_evaluation')
NOM=ROOT/'pi05_reconstructed_10000_nominal_hold20_formal_seed0to5_20261104'
TOL=ROOT/'pi05_reconstructed_10000_tol30_confirmedmask_hold20_40tasks_10seeds_20261104'
sources=[]
def read(p):
    b=Path(p).read_bytes();sources.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest()))
    return json.loads(b)
comparison=read(NOM/'formal_nominal_vs_tolerance.json')
assert comparison['complete'] and not comparison['targeted_failed_retests_included']
for root in [NOM,TOL]:
    read(root/'evaluation_request.json');assert read(root/'evaluation_independent_final_audit.json')['complete']
read(NOM/'combined_formal_400.json');read(NOM/'formal_400_audit.json')
read(TOL/'confirmed_mask400_report.json')

def motion(path):
    result=read(path);traj=read(Path(path).with_name('trajectory.json'))
    assert result['complete'] and traj and len(traj)==result['steps']
    assert result['control_hz']==20 and result['physics_hz']==500
    assert all(r['physics_steps']==25 for r in traj)
    series={}
    for name,field in [('actual','post_action'),('reference','joint_target_rad')]:
        q=np.array([r[field]['joint_position_rad'] if name=='actual' else r[field] for r in traj])
        assert q.shape==(len(traj),7) and np.isfinite(q).all()
        diffs=[np.diff(q,n=i,axis=0)/(.05**i) for i in [1,2,3]]
        series[name]={k:float(np.linalg.norm(d,axis=1).mean()) for k,d in zip(['velocity','acceleration','jerk'],diffs)}
        series[name]['travel']=float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum())
    series['duration_s']=len(traj)*.05
    return series

rows=[]
for pair in comparison['pairs']:
    n,t=motion(pair['nominal_source']),motion(pair['tolerance_source'])
    rows.append(dict(suite=pair['suite'],task_id=pair['task_id'],seed=pair['seed'],
                     outcome=pair['paired_outcome'],nominal=n,tolerance=t,
                     nominal_source=pair['nominal_source'],tolerance_source=pair['tolerance_source']))

def summary(sub):
    output=dict(n=len(sub),metrics={})
    for series in ['actual','reference']:
        output['metrics'][series]={}
        for key in ['velocity','acceleration','jerk','travel']:
            a=st.fmean(r['nominal'][series][key] for r in sub)
            b=st.fmean(r['tolerance'][series][key] for r in sub)
            output['metrics'][series][key]=dict(nominal_mean=a,tolerance_mean=b,reduction_pct=100*(1-b/a))
    output['duration_s']={arm:st.fmean(r[arm]['duration_s'] for r in sub) for arm in ['nominal','tolerance']}
    return output

assert len(rows)==400 and len({(r['suite'],r['task_id'],r['seed']) for r in rows})==400
counts=dict(Counter(r['outcome'] for r in rows))
assert counts==comparison['paired_counts']
output=dict(complete=True,rollouts_tolerance=400,nominal_formal_comparators=400,
            formal_nominal_successes=380,tolerance_successes=370,success_delta_percentage_points=-2.5,
            pairs=counts,all400=summary(rows),both_success=summary([r for r in rows if r['outcome']=='both_success']),
            definition='20Hz joint-space L2 finite differences over saved post-action physical q, and separately command joint targets; no initial padding, no cross-episode differences. Entire recorded episode, episode equal. Not task duration speed.',
            limitations='Closed-loop groups produce different actions and lengths; successful intersection is conditional. No mY or sigma FK recomputation and no new rollout.')
(H/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
(H/'per_pair.json').write_text(json.dumps(rows,indent=2)+'\n')
(H/'sources.json').write_text(json.dumps(dict(files=sources),indent=2)+'\n')
print(json.dumps(output,indent=2))
